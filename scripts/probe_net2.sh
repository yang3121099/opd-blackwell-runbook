#!/usr/bin/env bash
# PROBE-NET2 (runbook [v2.5.9h]; Runming via the reviewer 12:1x UTC 10-21: keep pushing on the network, multi-connection
# downloads (aria2) were tried before and the night may be faster; download directly in a controlled time, without outside
# help): is the node's proxy limited per connection or per host / in total? CPU lane, read-only network, fixed URLs.
#  1. the first shard of Qwen3-32B from huggingface.co, hf-mirror.com and ModelScope's API, each with 1 / 4 / 16 / 32
#     (PN2_CONNS) parallel Range connections for PN2_SECS (120) s, connection counts outermost so the three sources are
#     measured close in time (scripts/par_range.py: standard library, data discarded): bytes, MB/s, HTTP codes, failed
#     requests; the 1-connection row is the baseline against PROBE-NET
#  2. if aria2c is on the node: aria2c -x 16 -s 16 --max-tries=1 --timeout=30 on each source for the same time (bytes
#     written, sparse file, removed after)
#  3. the slow background pull (fetch_pass's fetch_model.py ... --task FETCH-*) is paused while the probe measures: SIGSTOP
#     on its own processes and their children only, SIGCONT at the end whatever happens (trap; the pids are kept in
#     paused_pids, and a later run first resumes any left there); events pulls_paused / pulls_resumed
# Every run appends one line to history.jsonl (kept across reruns: rerun renames .done and report.json only); report.json =
# every run + conclusions: the best source / connection count / tool and its MB/s, per source how the rate scales from 1 to
# 32 connections (per_connection >= 8x, per_host_or_total <= 2x, partial between), the night gain (best MB/s at night,
# 22:00-08:00 Beijing, over the best by day) once both exist, the hours for Qwen3-VL-8B-Thinking (17.5 GB) and Qwen3-32B
# (65.5 GB) at the best rate, and the next step (>= 1 MB/s: a parallel Range mode for fetch_model.py; >= 3 MB/s: also
# Qwen3-VL-8B-Thinking into #fetch; both on the reviewer's A1). Private: logs/<node>/probe_net2/<task>; an event; .done.
# Reruns at 15:00 and 19:00 UTC (`rerun PROBE-NET2`). ~25 min a run (30 with aria2c).
# Usage: bash probe_net2.sh cpu
set -uo pipefail
T=PROBE-NET2; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_NET2 $(date -u +%T) $*" | tee -a $LOG; }
SECS=${PN2_SECS:-120}; CONNS=${PN2_CONNS:-1 4 16 32 64}; PY=${PN2_PY:-python3}   # [v2.5.9p] 64 too
MS=https://modelscope.cn/api/v1
declare -a SOURCES=(
  "hf|https://huggingface.co/Qwen/Qwen3-32B/resolve/main/model-00001-of-00017.safetensors"
  "hf_mirror|https://hf-mirror.com/Qwen/Qwen3-32B/resolve/main/model-00001-of-00017.safetensors"
  "modelscope|$MS/models/Qwen/Qwen3-32B/repo?Revision=master&FilePath=model-00001-of-00017.safetensors")
# [v2.5.9i] (ops-lead's A1 of 9h) not while result uploads time out (the proxy is shared and results go first, as in fetch_pass):
# checked every PN2_UPLOAD_WAIT_STEP (300 s) up to PN2_UPLOAD_CHECKS (24) times, then measured anyway
for _w in $(seq 0 ${PN2_UPLOAD_CHECKS:-24}); do
  { [ -n "$(find $OPD_ROOT/logs/heartbeat.err -mmin -60 2> /dev/null)" ] && tail -20 $OPD_ROOT/logs/heartbeat.err | grep -q UPLOAD_TIMEOUT; } || break
  [ $_w = 0 ] && step "result uploads timed out in the last hour: waiting"
  [ $_w = ${PN2_UPLOAD_CHECKS:-24} ] && { step "uploads still timing out after the wait: measuring anyway"; break; }
  sleep ${PN2_UPLOAD_WAIT_STEP:-300}
done
# 3. the slow pull: resume what an earlier, killed run may have left stopped, then pause the current one
[ -s $D/paused_pids ] && { kill -CONT $(cat $D/paused_pids) 2> /dev/null; step "resumed pids left by an earlier run: $(cat $D/paused_pids)"; }
: > $D/paused_pids
tree() { local p; for p in "$@"; do echo $p; tree $(pgrep -P $p); done; }
PAUSED=$(tree $(pgrep -f "fetch_model\.py .* --task FETCH-") | sort -u | tr '\n' ' '); PAUSED=${PAUSED% }
resume() {
  [ -n "$PAUSED" ] || return 0
  kill -CONT $PAUSED 2> /dev/null; : > $D/paused_pids; step "slow pull resumed ($PAUSED)"
  python3 $RUNBOOK/scripts/control.py event $T pulls_resumed "$PAUSED" > /dev/null 2>&1; PAUSED=""; }
trap resume EXIT; trap 'exit 1' INT TERM HUP
if [ -n "$PAUSED" ]; then
  echo "$PAUSED" > $D/paused_pids; kill -STOP $PAUSED 2> /dev/null; step "slow pull paused ($PAUSED)"
  python3 $RUNBOOK/scripts/control.py event $T pulls_paused "$PAUSED" > /dev/null 2>&1
fi
# 1. the grid
UTC=${PN2_UTC:-$(date -u +%FT%TZ)}; RUN=$D/run.jsonl; : > $RUN   # PN2_UTC: the tests' clock for the day / night slot
step "start: sources ${#SOURCES[@]}, connections $CONNS, $SECS s each"
for n in $CONNS; do
  for s in "${SOURCES[@]}"; do
    name=${s%%|*}; url=${s#*|}
    timeout $((${SECS%.*} + 90)) $PY $RUNBOOK/scripts/par_range.py "$url" $n $SECS --source $name > $D/one.json 2> $D/err.txt \
      && [ -s $D/one.json ] && cat $D/one.json >> $RUN \
      || python3 -c "import json, sys; print(json.dumps({'tool': 'par_range', 'source': sys.argv[1], 'conns': int(sys.argv[2]), 'secs': float(sys.argv[3]), 'bytes': 0, 'MBps': 0.0, 'codes': {}, 'failed': 0, 'error': 'tool: ' + sys.argv[4][:150]}))" \
           $name $n $SECS "$(tail -c 300 $D/err.txt | tr '\n' ' ')" >> $RUN
    step "$name x$n: $(tail -1 $RUN | python3 -c 'import json, sys; v = json.load(sys.stdin); print(v["MBps"], "MB/s, codes", v.get("codes"), "failed", v.get("failed"), v.get("error") or "")')"
  done
done
# 2. aria2c, if present
if A=$(command -v aria2c); then
  for s in "${SOURCES[@]}"; do
    name=${s%%|*}; url=${s#*|}; rm -rf $D/aria2; mkdir -p $D/aria2
    # [v2.5.9p] the rate from aria2c's own result table (avg speed; SIGINT makes it print one), not from the file's size: on
    # the work disk files are not sparse, so writing the 16th segment at 15/16 of the file allocates everything before it
    timeout -s INT ${SECS%.*} $A -x 16 -s 16 -k 4M --max-tries=1 --timeout=30 --file-allocation=none --allow-overwrite=true \
      --auto-file-renaming=false --summary-interval=0 -d $D/aria2 -o f.bin "$url" > $D/aria2.out 2>&1; rm -rf $D/aria2
    python3 - $name $SECS $D/aria2.out >> $RUN <<'PYA'
import json, re, sys
name, secs, out = sys.argv[1], float(sys.argv[2]), open(sys.argv[3], errors="ignore").read()
unit = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}
m = re.findall(r"^[0-9a-f]{6}\|[A-Z]+ *\| *([0-9.]+)(B|KiB|MiB|GiB)/s\|", out, re.M)
bps = float(m[-1][0]) * unit[m[-1][1]] if m else 0.0
print(json.dumps({"tool": "aria2c", "source": name, "conns": 16, "secs": secs, "bytes": int(bps * secs), "MBps": round(bps / 1e6, 4),
                  "error": None if m else "no result table: " + " ".join(out.split())[-150:]}))
PYA
    step "aria2c $name x16: $(tail -1 $RUN | python3 -c 'import json, sys; print(json.load(sys.stdin)["MBps"], "MB/s")')"
  done
else step "aria2c not on the node: standard-library connections only"; fi
resume
# the report: this run into history.jsonl, conclusions over every run
python3 - "$D" "$UTC" <<'PY'
import json, os, sys
d, utc = sys.argv[1:3]
rows = [json.loads(l) for l in open(os.path.join(d, "run.jsonl")) if l.strip()]
bj = (int(utc[11:13]) + 8) % 24
run = {"utc": utc, "beijing_hour": bj, "slot": "night" if bj >= 22 or bj < 8 else "day", "method": 2, "rows": rows}   # [v2.5.9p] method 2
with open(os.path.join(d, "history.jsonl"), "a") as f:
    f.write(json.dumps(run) + "\n")
hist = [json.loads(l) for l in open(os.path.join(d, "history.jsonl")) if l.strip()]
# [v2.5.9p] runs before method 2 measured aria2c by the size of a non-sparse file (the 12:31 UTC run of 10-21: 31 MB/s on
# every source): their aria2c rows are void and left out of every conclusion; their standard-library rows stay
void = [{"utc": h["utc"], "source": r["source"]} for h in hist if h.get("method", 1) < 2 for r in h["rows"] if r.get("tool") == "aria2c"]
for h in hist:
    if h.get("method", 1) < 2:
        h["rows"] = [r for r in h["rows"] if r.get("tool") != "aria2c"]
best_of = lambda rs: max(rs, key=lambda r: r.get("MBps") or 0) if rs else None
def scaling(rs):
    out = {}
    for src in sorted({r["source"] for r in rs if r.get("tool") == "par_range"}):
        by = {r["conns"]: r.get("MBps") or 0 for r in rs if r.get("tool") == "par_range" and r["source"] == src}
        lo, hi = by.get(min(by)), by.get(max(by))
        k = "no_data" if not hi else "per_connection" if lo and hi / lo >= 8 else "per_host_or_total" if lo and hi / lo <= 2 else "partial"
        out[src] = {"MBps_by_conns": {str(c): by[c] for c in sorted(by)}, "ratio_max_over_1": round(hi / lo, 2) if lo else None, "limit": k}
    return out
b = best_of(rows); ball = best_of([r for h in hist for r in h["rows"]])
slot_best = {s: best_of([r for h in hist if h["slot"] == s for r in h["rows"]]) for s in ("day", "night")}
gain = None
if slot_best["day"] and slot_best["night"] and slot_best["day"].get("MBps"):
    gain = round(slot_best["night"]["MBps"] / slot_best["day"]["MBps"], 2)
hours = lambda mbps, gb: round(gb * 1e9 / (mbps * 1e6) / 3600, 1) if mbps else None
top = (ball or {}).get("MBps") or 0
nxt = ("parallel Range mode for fetch_model.py and Qwen3-VL-8B-Thinking into #fetch (reviewer A1)" if top >= 3 else
       "parallel Range mode for fetch_model.py (reviewer A1)" if top >= 1 else "no change: below 1 MB/s with every combination")
c = {"best_source": b and b["source"], "best_parallel": b and b["conns"], "best_tool": b and b.get("tool"), "MBps": b and b.get("MBps"),
     "best_any_run": ball and {"source": ball["source"], "conns": ball["conns"], "tool": ball.get("tool"), "MBps": ball.get("MBps")},
     "scaling": scaling(rows), "night_gain": gain,
     "slot_best_MBps": {s: (v or {}).get("MBps") for s, v in slot_best.items()},
     "hours_vl8b_17.5GB": hours(top, 17.5), "hours_32b_65.5GB": hours(top, 65.5), "next": nxt,
     "void_rows": void, "void_note": "aria2c rows of runs before method 2 (sized by a non-sparse file) are void and left out" if void else None}
json.dump({"latest": run, "runs": len(hist), "history": hist, "conclusions": c}, open(os.path.join(d, "report.json"), "w"), indent=1)
print(f"best {c['MBps']} MB/s ({c['best_source']} x{c['best_parallel']}, {c['best_tool']}) in this {run['slot']} run; "
      f"night gain {gain}; {c['hours_vl8b_17.5GB']} h for VL-8B at the best rate; next: {nxt}")
PY
[ -f $D/report.json ] || { step "report failed"; exit 1; }
summary=$(python3 -c "import json, sys; c = json.load(open(sys.argv[1]))['conclusions']; print(f\"best {c['MBps']} MB/s ({c['best_source']} x{c['best_parallel']}); night gain {c['night_gain']}; next: {c['next']}\")" $D/report.json)
mkdir -p $D/up; cp $D/report.json $D/history.jsonl $D/probe.log $D/up/
python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_net2/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
python3 $RUNBOOK/scripts/control.py event $T probe_net2_done "$summary" > /dev/null 2>&1
step "done; $summary"; touch $D/.done
