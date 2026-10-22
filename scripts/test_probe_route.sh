#!/usr/bin/env bash
# [v2.5.9x] Stub tests of probe_route.sh (no GPU, network or HF): the real unit tests against a fake installed ref_input_utils.py,
# 6400 fake training labels, a fake 3-step smoke whose route.jsonl agrees with the labels or not.
# Usage: bash scripts/test_probe_route.sh -> PROBE_ROUTE_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/rb/code/analysis $T/root/runs/PROBE-DIFF17 $T/bin $T/py/verl/trainer/ppo
  cp $SRC/scripts/probe_route.sh $T/rb/scripts/
  cp $SRC/code/patches/opd_route.py $SRC/code/patches/route_patch.py $SRC/code/patches/stopmerge_patch.py $SRC/code/patches/test_opd_route.py $T/rb/code/patches/
  cp $SRC/code/analysis/diff_labels.py $T/rb/code/analysis/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  : > $T/py/verl/__init__.py; : > $T/py/verl/trainer/__init__.py; : > $T/py/verl/trainer/ppo/__init__.py
  # the installed ref_input_utils.py (the P5 patch in): the function shape route_patch.py looks for
  printf "def prepare_ref_model_inputs(\n    batch,\n    ref_tokenizer,\n    apply_chat_template_kwargs=None,\n):\n    batch_size = len(batch)\n    for i in range(batch_size):\n        messages = list(batch[i])\n        import os as _os\n        _think = _os.environ.get('OPD_REF_THINK') == '1'\n        _kw = apply_chat_template_kwargs\n    return batch\n" > $T/py/verl/trainer/ppo/ref_input_utils.py
  python3 - $T/root/runs/PROBE-DIFF17/labels_17.jsonl $T/rb/code/patches <<'PY'
import json, sys
sys.path.insert(0, sys.argv[2]); import opd_route as R
with open(sys.argv[1], "w") as f:
    for i in range(6400):
        f.write(json.dumps({"split": "train", "pos": i, "key": R.key([{"role": "user", "content": f"p{i}"}]), "k": i % 9, "n": 8, "r": (i % 9) / 8, "mean_len": 1.0, "trunc": 0}) + "\n")
    for i in range(500):
        f.write(json.dumps({"split": "math500", "pos": i, "key": R.key([{"role": "user", "content": f"m{i}"}]), "k": 4, "n": 8, "r": 0.5, "mean_len": 1.0, "trunc": 0}) + "\n")
PY
  # the smoke: three steps of 128 prompts (p0..p383); FAKE_ROUTE = ok | unknown | wrong | half
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\nteacher: /m/Qwen3-8B (ref; think-template=1; param_offload=1)${FAKE_NOTE-; template per prompt: route (opd_route.py; ...)}\n${FAKE_INV_NOTE:-}" > $d/config.yaml
for i in 1 2 3; do echo "(TaskRunner pid=1) step:$i - timing_s/step:300" >> $d/train.log; done
python3 - $d/route.jsonl <<'PY'
import json, os, sys
mode = os.environ.get("FAKE_ROUTE", "ok"); rows = 64 if mode == "half" else 128; inv = os.environ.get("FAKE_INV") == "1"   # [v2.5.9z45] the reversed rule
with open(sys.argv[1], "w") as f:
    for c in range(3):
        dec = []
        for j in range(rows):
            i = c * 128 + j
            r = (i % 9) / 8; k = __import__("hashlib").sha1(f"p{i}".encode()).hexdigest()[:16]
            if mode == "unknown" and j == 5:
                k, r = "deadbeefdeadbeef", None
            think = r is None or (r >= 0.5 if inv else r < 0.5)
            if mode == "wrong" and j == 7:
                think = not think
            dec.append([k, r, think])
        rec = {"call": c + 1, "batch": rows, "rows": rows, "think": sum(d[2] for d in dec), "plain": sum(not d[2] for d in dec),
               "unknown": sum(d[1] is None for d in dec), "decisions": dec}
        if inv and os.environ.get("FAKE_NO_INVLOG") != "1":
            rec["invert"] = True
        f.write(json.dumps(rec) + "\n")
PY
exit ${FAKE_RC:-0}
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PRT_WAIT_STEP=0 PRT_PY=python3 PRT_REQUIRE_TORCH=0 PYTHONPATH=$T/py; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-ROUTE/report.json "$1"; }
setup; bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1
ok "PASS: unit tests, labels + mix coin, a 3-step smoke of E10b-P5route, think/plain per step, upload, event, .done" '[ "$(res result)" = PASS ] && grep -q "passed, 0 failed" $T/root/runs/PROBE-ROUTE/unit_tests.txt && grep -q "^run_arm 4 E10b-P5route SMOKE=1 STEPS=3 PROBE_TASK=PROBE-ROUTE-smoke$" $CALLS && [[ "$(res reason)" == *"think/plain per step "*"/"*" mix think share 0."* ]] && grep -q "^remote put $T/root/runs/PROBE-ROUTE/up logs/abcd1234/probe_route" $CALLS && grep -q "^control event PROBE-ROUTE probe_route_pass" $CALLS && [ -f $T/root/runs/PROBE-ROUTE/.done ] && [ -f $T/root/runs/PROBE-ROUTE/up/route.jsonl ]'
ok "the installed file got no hook from the unit tests (they patch copies only)" '! grep -q "per-prompt teacher template" $T/py/verl/trainer/ppo/ref_input_utils.py'
rm -rf $T
check() { ok "$1" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"'"$2"'"* ]]'; }
setup; FAKE_ROUTE=unknown bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1; check "a prompt without a label is FAIL" "without a label"; rm -rf $T
setup; FAKE_ROUTE=wrong bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1; check "a decision against its label is FAIL" "not as labelled"; rm -rf $T
setup; FAKE_ROUTE=half bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1; check "64 rows for a 128 batch is FAIL" "expected 128"; rm -rf $T
setup; FAKE_NOTE="" bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1; check "a config without the routing is FAIL" "does not name the routing"; rm -rf $T
setup; FAKE_RC=1 bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1; check "a failed smoke is FAIL" "smoke exited 1"; rm -rf $T
setup; rm $T/root/runs/PROBE-DIFF17/labels_17.jsonl; bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1; check "no labels is FAIL" "no PROBE-DIFF17 labels"; ok "no smoke without labels" '! grep -q "^run_arm" $CALLS'; rm -rf $T
setup; printf "def prepare_ref_model_inputs(\n    batch,\n):\n    return batch\n" > $T/py/verl/trainer/ppo/ref_input_utils.py; bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1
check "an installed file without the P5 anchor fails the unit tests" "unit tests"; rm -rf $T
unset PYTHONPATH
# [v2.5.9z45] (Q-063) PROBE-ROUTE-INV: the same probe for E10b-P5route-inv, the decisions by the reversed rule, its own report / events
INVENV="PRT_TASK=PROBE-ROUTE-INV PRT_ARM=E10b-P5route-inv PRT_INVERT=1"
resi() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-ROUTE-INV/report.json "$1"; }
setup; env $INVENV FAKE_INV=1 FAKE_INV_NOTE="route_invert: easy (r >= 0.5) -> thinking teacher" bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1
ok "[v2.5.9z45] PROBE-ROUTE-INV: PASS with the reversed decisions; the smoke of E10b-P5route-inv; logs/<node>/probe_route_inv, probe_route_inv_pass; PROBE-ROUTE untouched" '[ "$(resi result)" = PASS ] && grep -q "^run_arm 4 E10b-P5route-inv SMOKE=1 STEPS=3 PROBE_TASK=PROBE-ROUTE-INV-smoke$" $CALLS && grep -q "^remote put $T/root/runs/PROBE-ROUTE-INV/up logs/abcd1234/probe_route_inv$" $CALLS && grep -q "^control event PROBE-ROUTE-INV probe_route_inv_pass " $CALLS && [ ! -e $T/root/runs/PROBE-ROUTE/report.json ]'
rm -rf $T
setup; env $INVENV FAKE_INV_NOTE="route_invert: x" bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1
ok "[v2.5.9z45] PROBE-ROUTE-INV with route's own decisions is FAIL (decisions not as labelled; no invert mark)" '[ "$(resi result)" = FAIL ] && [[ "$(resi reason)" == *"decisions not as labelled"*"logged invert=False, expected True"* ]]'; rm -rf $T
setup; env $INVENV FAKE_INV=1 FAKE_NO_INVLOG=1 FAKE_INV_NOTE="route_invert: x" bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1
ok "[v2.5.9z45] reversed decisions without the log's invert mark is FAIL" '[ "$(resi result)" = FAIL ] && [[ "$(resi reason)" == *"logged invert=False, expected True"* ]] && [[ "$(resi reason)" != *"not as labelled"* ]]'; rm -rf $T
setup; env $INVENV FAKE_INV=1 bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1
ok "[v2.5.9z45] a config without route_invert: is FAIL" '[ "$(resi result)" = FAIL ] && [[ "$(resi reason)" == *"does not name the reversed routing"* ]]'; rm -rf $T
setup; FAKE_INV=1 bash $T/rb/scripts/probe_route.sh 4 > /dev/null 2>&1
ok "[v2.5.9z45] PROBE-ROUTE (not inverted) with reversed decisions stays FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"decisions not as labelled"*"logged invert=True, expected False"* ]]'; rm -rf $T
[ $fails = 0 ] && echo "PROBE_ROUTE_TEST PASS ($n checks)" || echo "PROBE_ROUTE_TEST FAIL ($fails of $n)"
