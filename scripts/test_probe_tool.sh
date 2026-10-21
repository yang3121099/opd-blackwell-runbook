#!/usr/bin/env bash
# [v2.5.9b] Stub tests of probe_tool.sh (no GPU, network, vLLM or HF). Usage: bash scripts/test_probe_tool.sh -> PROBE_TOOL_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/agent/data $T/root/runs $T/bin; cd $T   # nothing the stubs write lands in the caller's directory
  cp $SRC/scripts/probe_tool.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb STUDENT_17B=/m/Qwen3-1.7B TEACHER=/m/Qwen3-8B" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK TOOL_DATA=$TOOL_DATA" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\n${FAKE_CFG:-data: tool calls (tools), reward tool_grader.py, prompts <= 2048}\n" > $d/config.yaml
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.4 - timing_s/step:30" >> $d/train.log; done
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  # the venv python: '-' = the unit-test runner; prep_tools_parquet.py and tool_eval.py write what the real ones write
  cat > $T/bin/fakepy <<'S3'
#!/usr/bin/env bash
echo "py $*" >> $CALLS
case "$1" in
  -) cat > /dev/null; [ "${FAKE_UNIT_RC:-0}" = 0 ] && { echo "8 passed, 0 failed"; exit 0; }; echo "7 passed, 1 failed"; exit 1;;
  prep_tools_parquet.py) [ "${FAKE_PREP_RC:-0}" = 0 ] || { echo "AssertionError: sample w1: the teacher renders the system-message form differently"; exit 1; }
     : > $3_train.parquet; : > $3_val.parquet; echo "student prompt (300 tokens)" > $3_sample.txt; echo "TOOLS_PREP_OK 10 samples, train 400 rows, val 10 rows";;
  tool_eval.py) out=$(echo "$*" | sed -n 's/.*--out \([^ ]*\).*/\1/p'); s=$(echo "$*" | grep -q Qwen3-8B && echo 80.0 || echo 35.5)
     echo '{"budgets": {"512": {"score": 10.0}, "4096": {"score": '$s'}}}' > $out; echo "TOOL_EVAL";;
esac
S3
  chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho 0\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PT_WAIT_STEP=0 PT_PY=$T/bin/fakepy; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-TOOL/report.json "$1"; }
# 1. PASS: unit tests with the real template, prep with the 1.7B / 8B tokenizers x40, the smoke, the two evaluations
setup
bash $T/rb/scripts/probe_tool.sh 2 > /dev/null 2>&1
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-TOOL/.done ] && [[ "$(res reason)" == *"student 35.5 / teacher 80.0 at 4096"* ]]'
ok "prep with the real tokenizers x40; smoke of E10b-P5tool on the prep prefix" 'grep -q "^py prep_tools_parquet.py data/synthetic_tools.jsonl $T/root/runs/PROBE-TOOL/tools /m/Qwen3-1.7B /m/Qwen3-8B --repeat 40$" $CALLS && grep -q "^run_arm 2 E10b-P5tool SMOKE=1 STEPS=3 PROBE_TASK=PROBE-TOOL-smoke TOOL_DATA=$T/root/runs/PROBE-TOOL/tools$" $CALLS'
ok "student evaluated without thinking, teacher with" 'grep -q "^py tool_eval.py --model /m/Qwen3-1.7B .* --mode no_think$" $CALLS && grep -q "^py tool_eval.py --model /m/Qwen3-8B .* --mode think$" $CALLS'
ok "report: evaluations and smoke scores; upload; event" '[ "$(res smoke_score_per_step)" = "[0.4, 0.4, 0.4]" ] && grep -q "^remote put $T/root/runs/PROBE-TOOL/up logs/abcd1234/probe_tool/PROBE-TOOL" $CALLS && grep -q "^control event PROBE-TOOL probe_tool_pass" $CALLS'
rm -rf $T
# 2. unit tests fail; 3. prep refuses a template mismatch: FAIL before the smoke
setup
FAKE_UNIT_RC=1 bash $T/rb/scripts/probe_tool.sh 2 > /dev/null 2>&1
ok "unit-test failure is FAIL, nothing else runs" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"1 failed"* ]] && ! grep -q "prep_tools" $CALLS'
rm -rf $T
setup
FAKE_PREP_RC=1 bash $T/rb/scripts/probe_tool.sh 2 > /dev/null 2>&1
ok "a prep mismatch is FAIL, no smoke" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"renders the system-message form differently"* ]] && ! grep -q "^run_arm" $CALLS'
rm -rf $T
# 4. smoke OOM; 5. a step without a score; 6. a config without the tool data
setup
FAKE_OOM=1 bash $T/rb/scripts/probe_tool.sh 2 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"exited"* || "$(res reason)" == *"out of memory"* ]]'
rm -rf $T
setup
FAKE_LOGSTEPS=2 bash $T/rb/scripts/probe_tool.sh 2 > /dev/null 2>&1
ok "a missing step is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"2 steps with a score, expected 3"* ]]'
rm -rf $T
setup
FAKE_CFG="data: math (dapo_train)" bash $T/rb/scripts/probe_tool.sh 2 > /dev/null 2>&1
ok "a smoke on other data is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"does not name the tool data"* ]]'
rm -rf $T
cd $SRC
[ $fails = 0 ] && echo "PROBE_TOOL_TEST PASS ($n checks)" || echo "PROBE_TOOL_TEST FAIL ($fails of $n)"
