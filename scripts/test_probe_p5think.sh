#!/usr/bin/env bash
# Stub tests of v2.5.7r probe_p5think.sh (no GPU, network or HF). Usage: bash scripts/test_probe_p5think.sh -> PROBE_P5THINK_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin
  cp $SRC/scripts/probe_p5think.sh $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb TEACHER=$T/m8" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n"); os.makedirs(sys.argv[3], exist_ok=True); open(os.path.join(sys.argv[3], "config.json"), "w").write("{}")\n' > $T/rb/scripts/fetch_model.py
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK ROLLOUT_DUMP=$ROLLOUT_DUMP" >> $CALLS
mkdir -p $ROLLOUT_DUMP; for k in 1 2 3 4 5 6 7 8 9 10; do o="<think>hmm</think> boxed 2"; [ $k -gt ${FAKE_OPEN:-10} ] && o="no think"; echo "{\"output\": \"$o\"}" >> $ROLLOUT_DUMP/1.jsonl; done
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; : > $d/config.yaml
for i in $(seq 1 $SMOKE_STEPS); do echo "(TaskRunner pid=1) step:$i - timing_s/gen:100 - timing_s/update_actor:150 - timing_s/step:$([ $i = 1 ] && echo 600 || echo 330) - response_length/mean:3000" >> $d/train.log; done
sleep ${FAKE_SLEEP:-0}; [ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-150000}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  printf '#!/usr/bin/env bash\necho "fakepy $*" >> $CALLS; cat > /dev/null\n[ "${FAKE_TPL:-ok}" = ok ] && echo "{\\"ok\\": true}" > $3 || echo "{\\"ok\\": false}" > $3\n' > $T/fakepy; chmod +x $T/fakepy
  export CALLS=$T/calls PATH=$T/bin:$PATH PTH_WAIT_STEP=0 PTH_PY=$T/fakepy; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-P5THINK/report.json "$1"; }
# 1. PASS: templates ok, a 3-step smoke of E10b-P5think with rollouts dumped, peak / timing / projection / <think> shares
setup
FAKE_MEM=0 bash $T/rb/scripts/probe_p5think.sh 3 > /dev/null 2>&1 &   # an empty GPU for the start check ...
P=$!; wait $P
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-P5THINK/.done ]'
ok "templates checked with the teacher's tokenizer" 'grep -q "^fakepy - $T/m8 " $CALLS && [ "$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])" $T/root/runs/PROBE-P5THINK/report.json template ok)" = True ]'
ok "3-step smoke of E10b-P5think, rollouts dumped, <think> shares" 'grep -q "^run_arm 3 E10b-P5think SMOKE=1 STEPS=3 PROBE_TASK=PROBE-P5THINK-smoke ROLLOUT_DUMP=$T/root/runs/PROBE-P5THINK/rollouts" $CALLS && [ "$(python3 -c "import json, sys; r = json.load(open(sys.argv[1]))[sys.argv[2]]; print(r[\"open_think\"], r[\"close_think\"], r[\"responses\"])" $T/root/runs/PROBE-P5THINK/report.json rollouts_step1)" = "1.0 1.0 10" ]'
ok "steps and projection" '[ "$(res steps)" = 3 ] && [ "$(res projected_hours_150)" = 13.8 ]'
ok "report to logs/<node>/probe_p5think, event" 'grep -q "^remote put $T/root/runs/PROBE-P5THINK/up logs/abcd1234/probe_p5think" $CALLS && grep -q "^control event PROBE-P5THINK probe_p5think_pass" $CALLS'
rm -rf $T
# 2. out of memory: FAIL (a result; the arms are not queued)
setup
FAKE_MEM=0 FAKE_OOM=1 bash $T/rb/scripts/probe_p5think.sh 3 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && grep -q "^control event PROBE-P5THINK probe_p5think_fail" $CALLS'
rm -rf $T
# 3. a peak at or above 260 GB: FAIL
setup
FAKE_MEM=500 FAKE_SLEEP=1 PTH_PEAK_MIB=100 bash $T/rb/scripts/probe_p5think.sh 3 > /dev/null 2>&1
ok "peak over the limit is FAIL" '[ "$(res result)" = FAIL ] && python3 -c "import json, sys; assert \"peak\" in json.load(open(sys.argv[1]))[\"reason\"]" $T/root/runs/PROBE-P5THINK/report.json'
rm -rf $T
# 4. responses that do not open with <think> (the template did not reach the rollout): FAIL
setup
FAKE_MEM=0 FAKE_OPEN=5 bash $T/rb/scripts/probe_p5think.sh 3 > /dev/null 2>&1
ok "<think> share below 90% is FAIL" '[ "$(res result)" = FAIL ] && grep -q "open with <think>" $T/root/runs/PROBE-P5THINK/report.json'
rm -rf $T
# 5. a template that keeps <think> in the thinking render: FAIL before any smoke
setup
FAKE_TPL=bad FAKE_MEM=0 bash $T/rb/scripts/probe_p5think.sh 3 > /dev/null 2>&1
ok "template check failure is FAIL, no smoke" '[ "$(res result)" = FAIL ] && ! grep -q "^run_arm" $CALLS'
rm -rf $T
[ $fails = 0 ] && echo "PROBE_P5THINK_TEST PASS ($n checks)" || echo "PROBE_P5THINK_TEST FAIL ($fails of $n)"
