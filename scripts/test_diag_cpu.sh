#!/usr/bin/env bash
# [v2.5.9z99] Stub tests of diag_cpu.sh (CPU only, no tokenizer): an unknown name refused, the wait while an input is missing,
# DIAG-T32-TAILS and DIAG-VERIFY-ONSET end to end (report, upload to logs/<node>/<task>, event, .done).
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/analysis $T/root/runs
cp $SRC/scripts/diag_cpu.sh $T/rb/scripts/; cp $SRC/code/analysis/zero_gpu_diag.py $SRC/code/analysis/vlm_trunc_classify.py $T/rb/code/analysis/
echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb STUDENT_17B=none DIAG_PY=python3" > $T/rb/scripts/env.sh
printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
export CALLS=$T/calls; : > $CALLS
out=$(bash $T/rb/scripts/diag_cpu.sh DIAG-NOPE 2>&1); rc=$?
ok "an unknown name is refused" '[ $rc = 2 ] && [[ "$out" == "unknown diagnostic DIAG-NOPE" ]]'
out=$(bash $T/rb/scripts/diag_cpu.sh DIAG-T32-TAILS 2>&1); rc=$?
ok "a missing input waits, no attempt" '[ $rc = 1 ] && [[ "$out" == "WAIT_INPUTS DIAG-T32-TAILS: runs/EVAL-e10b-P5t32/e10b_P5t32_s50_math500_T0.6_math500_no_think_t0.6.jsonl not on this node" ]] && [ ! -f $T/root/runs/DIAG-T32-TAILS/.done ]'
mkdir -p $T/root/runs/EVAL-e10b-P5t32
python3 -c "
import json
with open('$T/root/runs/EVAL-e10b-P5t32/e10b_P5t32_s50_math500_T0.6_math500_no_think_t0.6.jsonl', 'w') as f:
    for i in range(30):
        ids = list(range(10)) + [1, 2, 3] * 30 if i % 2 else list(range(100))
        f.write(json.dumps({'pid': f'math500-{i}', 'sample': 0, 'n_tokens': len(ids), 'finish_reason': 'length', 'token_ids': ids, 'text': 'x' * 40}) + '\n')
"
out=$(bash $T/rb/scripts/diag_cpu.sh DIAG-T32-TAILS 2>&1); rc=$?
ok "t32 tails: 30 truncated, 20 drawn; report, upload, event, .done" '[ $rc = 0 ] && [ -f $T/root/runs/DIAG-T32-TAILS/.done ] && [ "$(python3 -c "import json; r = json.load(open(\"$T/root/runs/DIAG-T32-TAILS/report.json\")); print(r[\"result\"], r[\"truncated\"], len(r[\"drawn\"]), r[\"counts\"][\"loop\"])")" = "PASS 30 20 15" ] && grep -q "^remote put .*/up logs/abcd1234/diag-t32-tails$" $CALLS && grep -q "^control event DIAG-T32-TAILS diag_done truncated 30 of 30: loop 50.0%" $CALLS'
out=$(bash $T/rb/scripts/diag_cpu.sh DIAG-PROBE-GATE 2>&1); rc=$?
ok "probe gate waits for its readings" '[ $rc = 1 ] && [[ "$out" == "WAIT_INPUTS DIAG-PROBE-GATE: runs/EVAL-budA17a/"* ]]'
for f in E11b-P5/e11b_P5_math500_T0.6_math500_no_think_t0.6.jsonl EVAL-tail/e10b_P5tail_s50_math500_T0.6_math500_no_think_t0.6.jsonl; do
  mkdir -p $T/root/runs/$(dirname $f)
  python3 -c "
import json
with open('$T/root/runs/$f', 'w') as f:
    for i in range(120):
        f.write(json.dumps({'pid': f'math500-{i}', 'text': 'a' * 60 + ' Let me verify. ' + 'b' * 30 + ' \\\\boxed{1}'}) + '\n')
"
done
out=$(bash $T/rb/scripts/diag_cpu.sh DIAG-VERIFY-ONSET 2>&1); rc=$?
ok "verify onset: 100 each, the gate; report, upload, event" '[ $rc = 0 ] && [ "$(python3 -c "import json; r = json.load(open(\"$T/root/runs/DIAG-VERIFY-ONSET/report.json\")); print(r[\"result\"], r[\"gate\"], r[\"readings\"][\"p5s150\"][\"samples\"])")" = "PASS True 100" ] && grep -q "^control event DIAG-VERIFY-ONSET diag_done gate PASS" $CALLS'
out=$(bash $T/rb/scripts/diag_cpu.sh DIAG-T32-TAILS 2>&1)
ok "done once: no second run" '[[ "$out" == "TASK_ALREADY_DONE DIAG-T32-TAILS" ]]'
rm -rf $T
[ $fails = 0 ] && echo "DIAG_CPU_TEST PASS ($n checks)" || echo "DIAG_CPU_TEST FAIL ($fails of $n)"
