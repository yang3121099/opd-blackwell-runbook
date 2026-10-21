#!/usr/bin/env bash
# PROBE-ROUTE (runbook v2.5.9x; Q-041 / Q-042, the reviewer's gate): the gate of E10b-P5route and E10b-P5mix.
#  1. the unit tests of code/patches/test_opd_route.py on the node (P5's python, with torch: the sampler-order test runs too),
#     GOPD_REF_INPUT_UTILS = the installed ref_input_utils.py (the hook's anchor found once, the patched file compiles);
#  2. PROBE-DIFF17's labels_17.jsonl present, 6400 training prompts; the mix coin over them gives a thinking share in [0.45, 0.55];
#  3. a 3-step smoke of E10b-P5route on ONE GPU, then per step (route.jsonl, one call per step): the reference inputs built
#     once for the whole batch (rows = batch = 128: one teacher pass per sample), think + plain = rows, no unknown prompt (the
#     sampler order of PROBE-DIFF17 is the trainer's), every decision = its label (r >= 0.5 -> plain, else think); the smoke's
#     config names the routing. report.json: per step think / plain / unknown, the steps' timing; to logs/<node>/probe_route/
#     (private), an event, .done. PASS = all of it. Usage: bash probe_route.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-ROUTE; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
LAB=${ROUTE_LABELS:-$OPD_ROOT/runs/PROBE-DIFF17/labels_17.jsonl}
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_ROUTE $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k in ("route_check", "mix_check"):
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/unit_tests.txt $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  [ -f $D/smoke/route.jsonl ] && cp $D/smoke/route.jsonl $D/up/
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_route >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_route_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
# 1. unit tests with the python the arm trains in
PY=${PRT_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
RU=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ref_input_utils').origin)" 2> /dev/null)
[ -n "$RU" ] && [ -f "$RU" ] || finish FAIL "verl's ref_input_utils.py not found by $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_REF_INPUT_UTILS=$RU $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_opd_route")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_opd_route":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
import os
try:
    import torch  # noqa: F401  (the sampler-order test returns early without it: on the node it must run)
except ImportError:
    if os.environ.get("PRT_REQUIRE_TORCH", "1") == "1":
        fails += 1; print("FAILED torch not importable: the sampler-order test did not run")
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
# 2. the labels and the mix coin over them
[ -f "$LAB" ] || finish FAIL "no PROBE-DIFF17 labels ($(basename $LAB))"
python3 - "$LAB" "$D/mix_check.json" "${OPD_SEED:-1234}" "$RUNBOOK/code/patches" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[4]); import opd_route as R
lab = R.load_labels(sys.argv[1]); seed = int(sys.argv[3])
share = sum(R.coin(k, seed) for k in lab) / max(1, len(lab))
json.dump({"train_labels": len(lab), "mix_think_share": round(share, 4), "easy_share": round(sum(1 for r in lab.values() if r >= 0.5) / max(1, len(lab)), 4)}, open(sys.argv[2], "w"))
PY
python3 -c "import json, sys; m = json.load(open(sys.argv[1])); sys.exit(0 if m['train_labels'] == int(sys.argv[2]) and 0.45 <= m['mix_think_share'] <= 0.55 else 1)" $D/mix_check.json ${PRT_TRAIN_LABELS:-6400} \
  || finish FAIL "labels / mix coin: $(cat $D/mix_check.json)"
# 3. the smoke
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PRT_WAIT_STEP:-10}
done
step "start smoke gpu=$G"
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PRT_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10b-P5route > $D/smoke.out 2>&1; rc=$?
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $OPD_ROOT/runs/$T-smoke/route.jsonl $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
grep -q "template per prompt: route" $D/smoke/config.yaml 2> /dev/null || finish FAIL "the smoke's config does not name the routing"
python3 - "$LAB" "$D/smoke/route.jsonl" "$D/route_check.json" "${PRT_STEPS:-3}" "${PRT_TBS:-128}" "$RUNBOOK/code/patches" <<'PY'
import json, os, sys
lab_p, log_p, out, steps, tbs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, sys.argv[6]); import opd_route as R
lab = R.load_labels(lab_p)
calls = [json.loads(x) for x in open(log_p)] if os.path.exists(log_p) else []
why, per = [], []
if len(calls) != steps:
    why.append(f"{len(calls)} reference builds for {steps} steps")
for c in calls:
    bad = [d for d in c["decisions"] if d[1] != lab.get(d[0]) or d[2] != (d[1] is None or d[1] < 0.5)]
    per.append({"call": c["call"], "rows": c["rows"], "think": c["think"], "plain": c["plain"], "unknown": c["unknown"], "mismatch": len(bad)})
    if c["rows"] != tbs or c["batch"] != tbs:
        why.append(f"call {c['call']}: {c['rows']} rows / batch {c['batch']}, expected {tbs} (one teacher pass per sample)")
    if c["think"] + c["plain"] != c["rows"]:
        why.append(f"call {c['call']}: think + plain != rows")
    if c["unknown"]:
        why.append(f"call {c['call']}: {c['unknown']} prompts without a label (the sampler order differs)")
    if bad:
        why.append(f"call {c['call']}: {len(bad)} decisions not as labelled")
json.dump({"per_call": per, "why": why}, open(out, "w"), indent=1)
print("; ".join(why))
PY
WHY=$(python3 -c "import json, sys; print('; '.join(json.load(open(sys.argv[1]))['why']))" $D/route_check.json 2> /dev/null || echo "the route checks did not run")
[ -z "$WHY" ] || finish FAIL "route: $WHY"
finish PASS "${PRT_STEPS:-3}-step smoke ok; think/plain per step $(python3 -c "import json; print(' '.join(f\"{c['think']}/{c['plain']}\" for c in json.load(open('$D/route_check.json'))['per_call']))"); mix think share $(python3 -c "import json; print(json.load(open('$D/mix_check.json'))['mix_think_share'])")"
