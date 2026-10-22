#!/usr/bin/env bash
# [v2.5.9z38] Stub tests of probe_mixp.sh (PROBE-MIXP, the CPU lane; no GPU, network or HF): the real unit tests of test_opd_route.py
# against a fake installed ref_input_utils.py (as test_probe_route.sh), 6400 fake training labels; the coin's share at p = 0.75 / 0.9.
# Without torch here the runner counts one failure unless PMX_REQUIRE_TORCH=0 (on the node torch is there and required).
# Usage: bash scripts/test_probe_mixp.sh -> PROBE_MIXP_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/rb/code/analysis $T/root/runs/PROBE-DIFF17 $T/py/verl/trainer/ppo
  cp $SRC/scripts/probe_mixp.sh $T/rb/scripts/
  cp $SRC/code/patches/opd_route.py $SRC/code/patches/route_patch.py $SRC/code/patches/stopmerge_patch.py $SRC/code/patches/test_opd_route.py $T/rb/code/patches/
  cp $SRC/code/analysis/diff_labels.py $T/rb/code/analysis/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  : > $T/py/verl/__init__.py; : > $T/py/verl/trainer/__init__.py; : > $T/py/verl/trainer/ppo/__init__.py
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
  export CALLS=$T/calls PMX_PY=python3 PMX_REQUIRE_TORCH=0 PYTHONPATH=$T/py; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-MIXP/report.json "$1"; }
mixok() { python3 - $T/root/runs/PROBE-MIXP/report.json <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))['mix_check']
sys.exit(0 if m['train_labels'] == 6400 and m['nested'] and all(abs(m['share'][p] - float(p)) <= 0.05 for p in ('0.6', '0.75', '0.9')) else 1)   # [v2.5.9z47] 0.6 too
PY
}
setup; bash $T/rb/scripts/probe_mixp.sh > /dev/null 2>&1
ok "PASS: the unit tests (0 failed, 0 skipped), the coin at 0.75 / 0.9 within 0.05 and nested, upload, event, .done" '[ "$(res result)" = PASS ] && grep -q "passed, 0 failed, 0 skipped$" $T/root/runs/PROBE-MIXP/unit_tests.txt && mixok && grep -q "^remote put $T/root/runs/PROBE-MIXP/up logs/abcd1234/probe_mixp$" $CALLS && grep -q "^control event PROBE-MIXP probe_mixp_pass" $CALLS && [ -f $T/root/runs/PROBE-MIXP/.done ]'
rm -rf $T
check() { ok "$1" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"'"$2"'"* ]]'; }
setup; PMX_REQUIRE_TORCH=1 bash $T/rb/scripts/probe_mixp.sh > /dev/null 2>&1; check "no torch (required on the node) is FAIL, not a skip" "unit tests: "; ok "the missing torch named" 'grep -q "^FAILED torch not importable" $T/root/runs/PROBE-MIXP/unit_tests.txt'; rm -rf $T
setup; PMX_TRAIN_LABELS=6401 bash $T/rb/scripts/probe_mixp.sh > /dev/null 2>&1; check "a label count off is FAIL" "labels / coin at p"; rm -rf $T
setup; MIXP_PS="0.75 0.2" bash $T/rb/scripts/probe_mixp.sh > /dev/null 2>&1; ok "a share check at p = 0.2 still passes (0.2 +- 0.05), shares written per p" '[ "$(res result)" = PASS ] && grep -q "0.2" $T/root/runs/PROBE-MIXP/mix_check.json'; rm -rf $T
setup; rm $T/root/runs/PROBE-DIFF17/labels_17.jsonl; bash $T/rb/scripts/probe_mixp.sh > /dev/null 2>&1; check "no labels is FAIL" "no PROBE-DIFF17 labels"; rm -rf $T
[ $fails = 0 ] && echo "PROBE_MIXP_TEST PASS ($n checks)" || echo "PROBE_MIXP_TEST FAIL ($fails of $n)"
