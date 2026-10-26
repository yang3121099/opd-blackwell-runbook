#!/usr/bin/env bash
# [v2.5.9z89] Stub tests of diag_vlm_trunc.sh (CPU only): the six inputs, the wait while one is missing, report / upload / event / .done.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/analysis $T/root/runs
cp $SRC/scripts/diag_vlm_trunc.sh $T/rb/scripts/; cp $SRC/code/analysis/vlm_trunc_classify.py $T/rb/code/analysis/
echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
export CALLS=$T/calls; : > $CALLS
REC='{"finish_reason": "length", "token_ids": [1, 2, 3], "text": "\\\\boxed{2} more"}'
for f in EVAL-e15-base/e15_base2B_mathvista_T0.6_mathvista_no_think_t0.6.jsonl EVAL-e15-base2/e15_base2B_mathvision_T0_mathvision_no_think_t0.0.jsonl \
         EVAL-e15-base2/e15_base2B_mathverse_T0_mathverse_no_think_t0.0.jsonl EVAL-e15-vlm-s50/e15_P5_s50_mathvista_T0.6_mathvista_no_think_t0.6.jsonl \
         EVAL-e15-vlm-s50/e15_P5_s50_mathvision_T0_mathvision_no_think_t0.0.jsonl; do
  mkdir -p $T/root/runs/$(dirname $f); echo "$REC" > $T/root/runs/$f
done
out=$(bash $T/rb/scripts/diag_vlm_trunc.sh 2>&1); rc=$?
ok "a missing input waits, no attempt" '[ $rc = 1 ] && [[ "$out" == "WAIT_INPUTS DIAG-VLM-TRUNC: runs/EVAL-e15-vlm-s50/e15_P5_s50_mathverse_T0_mathverse_no_think_t0.0.jsonl not on this node" ]] && [ ! -f $T/root/runs/DIAG-VLM-TRUNC/.done ]'
echo "$REC" > $T/root/runs/EVAL-e15-vlm-s50/e15_P5_s50_mathverse_T0_mathverse_no_think_t0.0.jsonl
out=$(bash $T/rb/scripts/diag_vlm_trunc.sh 2>&1); rc=$?
ok "six inputs classified; report, upload, event, .done" '[ $rc = 0 ] && [ -f $T/root/runs/DIAG-VLM-TRUNC/.done ] && [ "$(python3 -c "import json; r = json.load(open(\"$T/root/runs/DIAG-VLM-TRUNC/report.json\")); print(r[\"result\"], len(r[\"inputs\"]), r[\"inputs\"][\"s50_mathverse\"][\"counts\"][\"box_more\"])")" = "PASS 6 1" ] && grep -q "^remote put .*/up logs/abcd1234/diag_vlm_trunc$" $CALLS && grep -q "^control event DIAG-VLM-TRUNC diag_vlm_trunc_done base_mathvista loop 0.0%" $CALLS'
rm -rf $T
[ $fails = 0 ] && echo "DIAG_VLM_TRUNC_TEST PASS ($n checks)" || echo "DIAG_VLM_TRUNC_TEST FAIL ($fails of $n)"
