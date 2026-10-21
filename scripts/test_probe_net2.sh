#!/usr/bin/env bash
# [v2.5.9h] Stub tests of probe_net2.sh (no network: par_range.py and aria2c are stubs; a fake slow pull of our own is
# paused and resumed, so run it on a machine without training, as the other process tests). Usage: bash
# scripts/test_probe_net2.sh -> PROBE_NET2_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root $T/bin $T/x; cd $T
  cp $SRC/scripts/probe_net2.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  # par_range: huggingface.co capped at 0.04 MB/s in total, hf-mirror 0.05 MB/s per connection, ModelScope refuses (403);
  # FAKE_SCALE scales every rate, FAKE_FAIL=<source>:<conns> makes that call crash; each call records the fake pull's state
  cat > $T/rb/scripts/par_range.py <<'P'
import json, os, subprocess, sys
url, conns, secs, src = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[5]
st = subprocess.run(["ps", "-o", "stat=", "-p", os.environ["FAKE_FETCH_PID"]], capture_output=True, text=True).stdout.strip()
open(os.environ["CALLS"], "a").write(f"par {src} {conns} fetch={st[:1]}\n")
if os.environ.get("FAKE_FAIL") == f"{src}:{conns}":
    sys.exit("Traceback: boom")
k = float(os.environ.get("FAKE_SCALE", "1"))
if src == "modelscope":
    r = {"conns": conns, "secs": secs, "bytes": 0, "MBps": 0.0, "codes": {"403": 1}, "failed": 0, "error": "resolve http 403"}
else:
    mbps = k * (0.04 if src == "hf" else 0.05 * conns)
    r = {"conns": conns, "secs": secs, "bytes": int(mbps * 1e6 * secs), "MBps": round(mbps, 4), "codes": {"206": conns}, "failed": 0, "error": None}
print(json.dumps({"tool": "par_range", "source": src, **r}))
P
  command -v timeout > /dev/null || { printf '#!/usr/bin/env bash\nshift; exec "$@"\n' > $T/bin/timeout; chmod +x $T/bin/timeout; }
  # the slow pull: a fetch_model.py of our own with a child, as fetch_pass starts it
  printf 'import subprocess, time\np = subprocess.Popen(["sleep", "300"])\ntime.sleep(300)\n' > $T/x/fetch_model.py
  python3 $T/x/fetch_model.py Qwen/X '' $T/x/d --task FETCH-test > /dev/null 2>&1 & FP=$!; sleep 1
  export CALLS=$T/calls PATH=$T/bin:$PATH FAKE_FETCH_PID=$FP PN2_SECS=1 PN2_UTC=2026-10-21T12:30:00Z; : > $CALLS
}
teardown() { kill $(pgrep -P $FP) $FP 2> /dev/null; wait $FP 2> /dev/null; cd $SRC; rm -rf $T; }   # only what setup started
r() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-NET2/report.json "$1"; }
# 1. a day run: the grid, the limits, the best rate, the pull paused during and resumed after
setup
bash $T/rb/scripts/probe_net2.sh cpu > /dev/null 2>&1
ok "12 measurements (3 sources x 1/4/16/32), connection counts outermost" '[ "$(r "len(v[\"latest\"][\"rows\"])")" = 12 ] && [ "$(grep "^par " $CALLS | head -3 | cut -d" " -f2-3 | tr "\n" ,)" = "hf 1,hf_mirror 1,modelscope 1," ]'
ok "per-connection limit on hf-mirror, total limit on huggingface.co, no data from ModelScope" '[ "$(r "\" \".join(v[\"conclusions\"][\"scaling\"][s][\"limit\"] for s in (\"hf_mirror\", \"hf\", \"modelscope\"))")" = "per_connection per_host_or_total no_data" ]'
ok "best: hf-mirror x32 at 1.6 MB/s; VL-8B in 3.0 h; next step the parallel Range mode" '[ "$(r "\" \".join(str(x) for x in (v[\"conclusions\"][\"best_source\"], v[\"conclusions\"][\"best_parallel\"], v[\"conclusions\"][\"MBps\"], v[\"conclusions\"][\"hours_vl8b_17.5GB\"]))")" = "hf_mirror 32 1.6 3.0" ] && [[ "$(r "v[\"conclusions\"][\"next\"]")" == "parallel Range mode for fetch_model.py (reviewer A1)" ]]'
ok "the slow pull is stopped while measuring, running again after; events; no pids left" '! grep "^par " $CALLS | grep -vq "fetch=T" && [ "$(ps -o stat= -p $FP | cut -c1)" != T ] && [ "$(ps -o stat= -p $(pgrep -P $FP) | cut -c1)" != T ] && grep -q "^control event PROBE-NET2 pulls_paused" $CALLS && grep -q "^control event PROBE-NET2 pulls_resumed" $CALLS && [ ! -s $T/root/runs/PROBE-NET2/paused_pids ]'
ok "day slot, no night gain yet; aria2c absent noted; report private; event; .done" '[ "$(r "\" \".join(str(x) for x in (v[\"latest\"][\"slot\"], v[\"conclusions\"][\"night_gain\"]))")" = "day None" ] && grep -q "aria2c not on the node" $T/root/runs/PROBE-NET2/probe.log && grep -q "^remote put $T/root/runs/PROBE-NET2/up logs/abcd1234/probe_net2/PROBE-NET2" $CALLS && grep -q "^control event PROBE-NET2 probe_net2_done best 1.6 MB/s (hf_mirror x32)" $CALLS && [ -f $T/root/runs/PROBE-NET2/.done ]'
# 2. the rerun at night (rerun renames .done and report.json): twice the rate -> night gain 2.0, VL-8B into #fetch next
mv $T/root/runs/PROBE-NET2/.done $T/root/runs/PROBE-NET2/.done.prev_1; mv $T/root/runs/PROBE-NET2/report.json $T/root/runs/PROBE-NET2/report.json.prev_1
FAKE_SCALE=2 PN2_UTC=2026-10-21T19:00:00Z bash $T/rb/scripts/probe_net2.sh cpu > /dev/null 2>&1
ok "night rerun: two runs in the history, night gain 2.0, 3.2 MB/s -> VL-8B into #fetch" '[ "$(r "\" \".join(str(x) for x in (v[\"runs\"], v[\"latest\"][\"slot\"], v[\"conclusions\"][\"night_gain\"], v[\"conclusions\"][\"MBps\"]))")" = "2 night 2.0 3.2" ] && [[ "$(r "v[\"conclusions\"][\"next\"]")" == *"Qwen3-VL-8B-Thinking into #fetch"* ]]'
teardown
# 3. a crashing measurement is a row with its error; aria2c rows when it is on the node; a pull left stopped by a killed run is resumed first
setup
printf '#!/usr/bin/env bash\nd=""; o=""; while [ $# -gt 0 ]; do case $1 in -d) d=$2; shift;; -o) o=$2; shift;; esac; shift; done\nhead -c 2000000 /dev/zero > $d/$o\n' > $T/bin/aria2c; chmod +x $T/bin/aria2c
mkdir -p $T/root/runs/PROBE-NET2; echo $FP > $T/root/runs/PROBE-NET2/paused_pids; kill -STOP $FP
FAKE_FAIL=hf:4 bash $T/rb/scripts/probe_net2.sh cpu > /dev/null 2>&1
ok "a crashed call is a row with the tool error, the grid goes on" '[[ "$(r "[x for x in v[\"latest\"][\"rows\"] if x[\"source\"] == \"hf\" and x[\"conns\"] == 4][0][\"error\"]")" == "tool: "*boom* ]] && [ "$(r "len([x for x in v[\"latest\"][\"rows\"] if x[\"tool\"] == \"par_range\"])")" = 12 ]'
ok "aria2c x16 per source from the bytes written" '[ "$(r "len([x for x in v[\"latest\"][\"rows\"] if x[\"tool\"] == \"aria2c\"])")" = 3 ] && python3 -c "import json, sys; v = json.load(open(sys.argv[1])); a = [x for x in v[\"latest\"][\"rows\"] if x[\"tool\"] == \"aria2c\"]; sys.exit(0 if all(1.9 < x[\"MBps\"] < 2.2 for x in a) else 1)" $T/root/runs/PROBE-NET2/report.json && [ ! -d $T/root/runs/PROBE-NET2/aria2 ]'
ok "pids an earlier killed run left stopped are resumed before measuring" 'grep -q "resumed pids left by an earlier run: $FP" $T/root/runs/PROBE-NET2/probe.log && ! grep "^par " $CALLS | grep -vq "fetch=T"'
teardown
# 4. [v2.5.9i] result uploads timed out in the last hour: the probe waits (here 1 check of 0 s), then measures anyway
setup
mkdir -p $T/root/logs; echo "UPLOAD_TIMEOUT runs/X 600 s" > $T/root/logs/heartbeat.err
PN2_UPLOAD_CHECKS=1 PN2_UPLOAD_WAIT_STEP=0 bash $T/rb/scripts/probe_net2.sh cpu > /dev/null 2>&1
ok "uploads timing out: waits, then measures anyway" 'grep -q "result uploads timed out in the last hour: waiting" $T/root/runs/PROBE-NET2/probe.log && grep -q "measuring anyway" $T/root/runs/PROBE-NET2/probe.log && [ "$(r "len(v[\"latest\"][\"rows\"])")" = 12 ]'
teardown
[ $fails = 0 ] && echo "PROBE_NET2_TEST PASS ($n checks)" || echo "PROBE_NET2_TEST FAIL ($fails of $n)"
