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
# Usage: bash probe_stopmerge.sh <gpu>
set -uo pipefail
G=$1; T=PROBE-STOPMERGE; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
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
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/unit_tests.txt $D/up/ 2> /dev/null
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
step "start gpu=$G unit tests with $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_VERL=$VD $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_opd_stopmerge")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_opd_stopmerge":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
try:
    import torch  # noqa: F401  (the torch tests return early without it: on the node they must run)
except ImportError:
    fails += 1; print("FAILED torch not importable: the merge and stop-set tests did not run")
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
# 2. the 3-step smoke
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PSM_WAIT_STEP:-10}
done
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PSM_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10b-P5stopmerge > $D/smoke.out 2>&1; rc=$?
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
