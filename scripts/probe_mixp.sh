#!/usr/bin/env bash
# PROBE-MIXP (runbook [v2.5.9z38]; Q-060, the reviewer's pre-registration 10-22): the gate of E10b-P5mix75 / -P5mix90, the CPU lane.
# The arms are E10b-P5mix's recipe (PROBE-ROUTE passed it with a 3-step smoke) with the coin at p = 0.75 / 0.9 (OPD_ROUTE_P, a
# threshold coin in opd_route.coin); what is new is checked here, without a GPU:
#  1. the unit tests of code/patches/test_opd_route.py with the python the arms train in (torch: the sampler-order test runs), as
#     PROBE-ROUTE runs them; any failure, or no torch, is a FAIL (nothing counts as skipped)
#  2. PROBE-DIFF17's labels_17.jsonl, 6400 training prompts; for each p in MIXP_PS (0.75 0.9) the coin's thinking share over them
#     in [p - 0.05, p + 0.05], and the prompts thinking at the smaller p a subset of those at the larger one
# report.json: the unit-test line, per p the share; to logs/<node>/probe_mixp/ (private), an event, .done. Usage: bash probe_mixp.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=PROBE-MIXP; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
LAB=${ROUTE_LABELS:-$OPD_ROOT/runs/PROBE-DIFF17/labels_17.jsonl}
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_MIXP $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
p = os.path.join(d, "mix_check.json")
if os.path.exists(p):
    r["mix_check"] = json.load(open(p))
try:
    r["unit_tests"] = open(os.path.join(d, "unit_tests.txt")).read().strip().splitlines()[-1]
except (OSError, IndexError):
    pass
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/unit_tests.txt $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_mixp >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_mixp_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
# 1. unit tests with the python the arms train in
PY=${PMX_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
RU=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ref_input_utils').origin)" 2> /dev/null)
[ -n "$RU" ] && [ -f "$RU" ] || finish FAIL "verl's ref_input_utils.py not found by $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_REF_INPUT_UTILS=$RU $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, os, sys, traceback
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
try:
    import torch  # noqa: F401  (the sampler-order test returns early without it: here it must run)
except ImportError:
    if os.environ.get("PMX_REQUIRE_TORCH", "1") == "1":
        fails += 1; print("FAILED torch not importable: the sampler-order test did not run")
print(f"{n - fails} passed, {fails} failed, 0 skipped")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
# 2. the coin at each p over the 6400 training labels
[ -f "$LAB" ] || finish FAIL "no PROBE-DIFF17 labels ($(basename $LAB))"
python3 - "$LAB" "$D/mix_check.json" "${OPD_SEED:-1234}" "$RUNBOOK/code/patches" ${MIXP_PS:-0.75 0.9} <<'PY'
import json, sys
sys.path.insert(0, sys.argv[4]); import opd_route as R
lab = R.load_labels(sys.argv[1]); seed = int(sys.argv[3]); ps = sorted(float(x) for x in sys.argv[5:])
sets = {p: {k for k in lab if R.coin(k, seed, p)} for p in ps}
out = {"train_labels": len(lab), "seed": seed, "share": {str(p): round(len(sets[p]) / max(1, len(lab)), 4) for p in ps},
       "nested": all(sets[a] <= sets[b] for a, b in zip(ps, ps[1:]))}
json.dump(out, open(sys.argv[2], "w"), indent=1); print("MIX_CHECK", json.dumps(out))
PY
python3 -c "import json, sys; m = json.load(open(sys.argv[1])); sys.exit(0 if m['train_labels'] == int(sys.argv[2]) and m['nested'] and all(float(p) - 0.05 <= s <= float(p) + 0.05 for p, s in m['share'].items()) else 1)" $D/mix_check.json ${PMX_TRAIN_LABELS:-6400} \
  || finish FAIL "labels / coin at p: $(cat $D/mix_check.json | tr '\n' ' ')"
finish PASS "unit tests $(tail -1 $D/unit_tests.txt); thinking share $(python3 -c "import json, sys; m = json.load(open(sys.argv[1])); print(', '.join(f'p={p}: {s}' for p, s in m['share'].items()))" $D/mix_check.json) over $(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['train_labels'])" $D/mix_check.json) labels, nested"
