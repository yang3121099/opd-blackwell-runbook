#!/usr/bin/env bash
# PROBE-STOPMERGE (runbook [v2.5.9l], Q-038): the gate of the stop-merged arm E10b-P5stopmerge.
#  1. the unit tests of code/patches/test_opd_stopmerge.py on the node (the arm's python, with torch, a plain runner, no GPU),
#     with GOPD_VERL = the installed verl package: both hooks' anchors are found once, the patched texts compile and a
#     second application changes nothing; the merge and the stop-set log-prob against a direct log_softmax
#  2. a 3-step smoke of E10b-P5stopmerge on ONE GPU (SMOKE=1: no checkpoint, upload, validation; nvidia-smi every 5 s):
#     exit 0, no OOM, peak < 260 GB; the smoke's config naming the stop merge; every step in train.log with
#     critic/score/mean in [0, 1]; stopmerge.jsonl (one line per reference micro-batch) with responses that have a box,
#     student stops merged after it, no merge at or before a box (violations 0), the merged log-prob never below the
#     teacher's own (gain >= 0: the stop set holds the student's stop); the installed files holding the hooks once.
# report.json (per step: rows, share with a box, merges per row, mean gain, the teacher's stop mass after the box), to
# logs/<node>/probe_stopmerge/<task> (private), an event, .done. ~25 min on one GPU.
# [v2.5.9z6] PSM_UNIT_ONLY=1 (task PROBE-STOPMERGE-UNIT, the CPU lane): step 1 only, no GPU, in runs/PROBE-STOPMERGE-UNIT (the gate's
# own report is not touched): the unit-test line "N passed, F failed, S skipped" (a skip fails the gate too) and hooks.txt,
# the installed files' hook lines with the guard each one runs under (OPD_STOPMERGE=1 or the batch key it sets), and [v2.5.9z8]
# every line with opd_box_end in the installed verl tree and $RUNBOOK/code (the writes marked).
# [v2.5.9z66] (Q-059) PSM_PROBE=<task> PSM_ARM=<arm> PSM_MAXRESP=<n>: the same gate for another stop-merged arm (PROBE-SMERGE-8K:
# E10b-P5smerge-8k, its smoke's config must name max_response_length 8192; PSM_PY = that arm's python, venv-fa)
# Usage: bash probe_stopmerge.sh <gpu>   |   PSM_UNIT_ONLY=1 bash probe_stopmerge.sh cpu
set -uo pipefail
G=$1; T=PROBE-STOPMERGE; [ "${PSM_UNIT_ONLY:-0}" = 1 ] && T=PROBE-STOPMERGE-UNIT; [ -n "${PSM_PROBE:-}" ] && T=$PSM_PROBE; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv   # [v2.5.9z11] this run's memory samples only (a rerun keeps the directory: PROBE-STOPMERGE's 4th peak was the 2nd run's)
step() { echo "PROBE_STOPMERGE $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    vals = [float(v.split(",")[1]) for v in open(os.path.join(d, "mem.csv")) if "," in v]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
p = os.path.join(d, "checks.json")
if os.path.exists(p):
    r["checks"] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/unit_tests.txt $D/hooks.txt $D/up/ 2> /dev/null
  [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  [ -f $D/smoke/stopmerge.jsonl ] && cp $D/smoke/stopmerge.jsonl $D/up/
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_stopmerge/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_stopmerge_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
# 1. unit tests with the python the arm trains in (P5's venv)
PY=${PSM_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
VD=$($PY -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('verl').origin))" 2> /dev/null)
[ -n "$VD" ] && [ -f "$VD/workers/actor/dp_actor.py" ] || finish FAIL "verl's package not found by $PY"
# [v2.5.9z6] GOPD_SRC = G-OPD's source tree (the trainings run from it; its .orig when its files carry the hooks already)
GS=${GOPD_SRC:-$OPD_ROOT/G-OPD/verl/verl}; [ -f $GS/workers/actor/dp_actor.py ] || GS=$VD
step "start gpu=$G unit tests with $PY (GOPD_VERL $VD, GOPD_SRC $GS)"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_VERL=$VD GOPD_SRC=$GS $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback, unittest
sys.path.insert(0, ".")
n = fails = skips = 0
mod = importlib.import_module("test_opd_stopmerge")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_opd_stopmerge":
        n += 1
        try:
            f()
        except unittest.SkipTest as e:   # [v2.5.9z6] counted, and it fails the gate: a test that did not run is not a pass
            skips += 1; print("SKIPPED", name, e)
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
try:
    import torch  # noqa: F401  (the torch tests return early without it: on the node they must run)
except ImportError:
    fails += 1; print("FAILED torch not importable: the merge and stop-set tests did not run")
print(f"{n - fails - skips} passed, {fails} failed, {skips} skipped")
sys.exit(1 if fails or skips or not n else 0)
PY
); rc=$?
# [v2.5.9z6] the hook lines of the files the trainings import (the installed package and the source tree), each with its guard
python3 - "$VD" "$GS" "$RUNBOOK/code" > $D/hooks.txt 2>&1 <<'PY'
import os, re, sys
mark = "# [opd-variants patch v2] [v2.5.9l] stop-merged OPD"
seen = set()
for d in sys.argv[1:3]:
    for rel in ("trainer/ppo/ref_input_utils.py", "workers/actor/dp_actor.py"):
        f = os.path.realpath(os.path.join(d, rel))
        if f in seen or not os.path.exists(f):
            continue
        seen.add(f); lines = open(f).read().splitlines()
        hits = [i for i, x in enumerate(lines) if mark in x]
        print(f"== {rel} ({'the installed package' if d == sys.argv[1] else 'the source tree'}): {len(hits)} hook lines")
        for i in hits:
            print(f"{i + 1}: {lines[i].strip()}")
            if not lines[i].lstrip().startswith(("if ", "_sm_on")):   # the hook's guard is the next line
                print(f"{i + 2}: {lines[i + 1].strip()}")
print(f"files: {len(seen)} distinct")
# [v2.5.9z8] (ops-lead 10-21) every line holding the batch key opd_box_end in the installed verl tree(s) and the runbook's code
# (tree:file:line:text, relative paths), and the writes among them (an assignment to the key): the 6 actor hooks are guarded by the
# key, so they are no-ops unless the OPD_STOPMERGE branch is its only writer
write = re.compile(r"""opd_box_end['"]\]\s*=[^=]""")
walked = set()
for label, root in (("verl", sys.argv[1]), ("verl-src", sys.argv[2]), ("runbook/code", sys.argv[3])):
    rr = os.path.realpath(root)
    if rr in walked or not os.path.isdir(rr):
        continue
    walked.add(rr); hits = writes = 0
    print(f"-- opd_box_end in {label}")
    for base, dirs, files in os.walk(rr, followlinks=True):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for fn in sorted(files):
            f = os.path.join(base, fn)
            if fn.endswith((".pyc", ".so")) or os.path.getsize(f) > 5 << 20:
                continue
            for i, line in enumerate(open(f, errors="ignore"), 1):
                if "opd_box_end" in line:
                    w = bool(write.search(line)); hits += 1; writes += w
                    print(f"{label}:{os.path.relpath(f, rr)}:{i}:{' WRITE' if w else ''} {line.strip()}")
    print(f"-- {label}: {hits} lines, {writes} writes")
PY
step "unit tests: $(tail -1 $D/unit_tests.txt); hooks: $(grep -c '^== ' $D/hooks.txt) files listed"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
[ "${PSM_UNIT_ONLY:-0}" = 1 ] && finish PASS "unit tests only (no smoke): $(tail -1 $D/unit_tests.txt)"
# 2. the 3-step smoke
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PSM_WAIT_STEP:-10}
done
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PSM_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G ${PSM_ARM:-E10b-P5stopmerge} > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $OPD_ROOT/runs/$T-smoke/stopmerge.jsonl $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PSM_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
why=$(python3 - "$D" "${PSM_STEPS:-3}" "$VD" <<'PY'
import json, os, re, sys
d, nsteps, vd = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sm = os.path.join(d, "smoke"); why = []
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
if "stop-merged after the first box" not in rd(os.path.join(sm, "config.yaml")):
    why.append("the smoke's config does not name the stop merge")
mr = os.environ.get("PSM_MAXRESP", "")   # [v2.5.9z66] an 8K arm: its window really is 8K
if mr and f"max_response_length {mr}" not in rd(os.path.join(sm, "config.yaml")):
    why.append(f"the smoke's config does not name max_response_length {mr}")
scores = {int(s): float(v) for s, v in re.findall(r"(?:^|[ )\]])step:(\d+) - .*?critic/score/mean:([-0-9.e]+)", rd(os.path.join(sm, "train.log")))}
if sorted(scores) != list(range(1, nsteps + 1)):
    why.append(f"train.log steps with a score {sorted(scores)}, expected 1..{nsteps}")
if any(not 0.0 <= v <= 1.0 for v in scores.values()):
    why.append(f"a score outside [0, 1]: {scores}")
rows = [json.loads(x) for x in rd(os.path.join(sm, "stopmerge.jsonl")).splitlines() if x.strip()]
tot = lambda k: sum(r.get(k) or 0 for r in rows)
if len(rows) < nsteps:
    why.append(f"stopmerge.jsonl has {len(rows)} lines, expected >= {nsteps} (one per reference micro-batch)")
if tot("rows_box") == 0:
    why.append("no response with a complete box was found")
if tot("merged") == 0:
    why.append("no student stop after a box was merged")
if tot("violations"):
    why.append(f"{tot('violations')} merges at or before a box")
gmin = [r["gain_min"] for r in rows if r.get("gain_min") is not None]
if gmin and min(gmin) < -1e-4:
    why.append(f"a merged log-prob below the teacher's own (gain {min(gmin)})")
mark = "# [opd-variants patch v2] [v2.5.9l] stop-merged OPD"
hooks = {f: rd(os.path.join(vd, f)).count(mark) for f in ("trainer/ppo/ref_input_utils.py", "workers/actor/dp_actor.py")}
if hooks != {"trainer/ppo/ref_input_utils.py": 1, "workers/actor/dp_actor.py": 6}:
    why.append(f"the installed hooks {hooks}, expected 1 and 6")
summ = {"lines": len(rows), "rows": tot("rows"), "rows_box_share": round(tot("rows_box") / max(tot("rows"), 1), 4),
        "merged_per_row": round(tot("merged") / max(tot("rows"), 1), 4),
        "gain_mean": round(sum((r["gain_mean"] or 0) * r["merged"] for r in rows) / max(tot("merged"), 1), 4),
        "stop_mass_after_box_mean": round(sum(r["stop_mass_after_box_mean"] for r in rows if r.get("stop_mass_after_box_mean") is not None)
                                          / max(1, sum(1 for r in rows if r.get("stop_mass_after_box_mean") is not None)), 4)}
json.dump({"score_by_step": scores, "stopmerge": summ, "hooks": hooks}, open(os.path.join(d, "checks.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "unit tests ok; ${PSM_STEPS:-3}-step smoke ok, peak ${peak} MiB; $(python3 -c "import json, sys; s = json.load(open(sys.argv[1]))['stopmerge']; print(f\"{s['rows_box_share']:.0%} of responses with a box, {s['merged_per_row']} merges per response, mean gain {s['gain_mean']}, teacher stop mass after the box {s['stop_mass_after_box_mean']}\")" $D/checks.json)"
