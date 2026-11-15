#!/usr/bin/env bash
# Stub tests of v2.5.7v probe_p5self.sh (no GPU, network or HF). Usage: bash scripts/test_probe_p5self.sh -> PROBE_P5SELF_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin
  cp $SRC/scripts/probe_p5self.sh $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n"); os.makedirs(sys.argv[3], exist_ok=True); open(os.path.join(sys.argv[3], "config.json"), "w").write("{}")\n' > $T/rb/scripts/fetch_model.py
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS TEACHER_14B=${TEACHER_14B:-} PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; S=/m/Qwen3-1.7B; [ $2 = E10d-P5self ] && S=/m/Qwen3-4B   # [v2.5.8e] student / teacher lines
[ $2 = E10c-P5self ] && S=/m/Qwen3-0.6B   # [v2.5.9v]
printf "task: $2\nstudent: $S (no_think)\nteacher: ${FAKE_TEACHER:-$S} (ref; think-template=1; param_offload=1)\n" > $d/config.yaml
for i in $(seq 1 $SMOKE_STEPS); do echo "(TaskRunner pid=1) step:$i - timing_s/gen:100 - timing_s/update_actor:150 - timing_s/step:$([ $i = 1 ] && echo 600 || echo 330) - response_length/mean:3000" >> $d/train.log; done
# [z163] the teacher-suffix arms: the per-call log and the student prompts of the rollout dump, by FAKE_TS (ok / none / partial / leak)
if [[ $2 == E10b-P5selfp* ]]; then
  [ "${FAKE_TS:-ok}" = none ] || for c in 1 2 3; do sf=64; [ "${FAKE_TS:-ok}" = partial ] && [ $c = 2 ] && sf=63; echo "{\"call\": $c, \"batch\": 64, \"rows\": 64, \"suffixed\": $sf}" >> $d/teacher_suffix.jsonl; done
  mkdir -p $ROLLOUT_DUMP; q="What is 1+1?"; [ "${FAKE_TS:-ok}" = leak ] && q="What is 1+1? $(cat $RUNBOOK/code/train/teacher_suffix/p2.txt)"
  for i in 1 2 3; do echo "{\"input\": \"user $q assistant\", \"output\": \"x\"}" >> $ROLLOUT_DUMP/$i.jsonl; done
fi
sleep ${FAKE_SLEEP:-0}; [ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
SH
  # [z158i] one line per GPU of an -i list, as the real nvidia-smi
  printf '#!/usr/bin/env bash\nn=1; p=""; for a in "$@"; do [ "$p" = -i ] && n=$(echo "$a" | tr , "\\n" | grep -c .); p=$a; done\nfor k in $(seq $n); do echo ${FAKE_MEM:-150000}; done\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PSE_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-P5SELF/report.json "$1"; }
# 1. PASS: a 3-step smoke of E10b-P5self, peak / timing / projection reported, nothing fetched
setup
FAKE_MEM=0 bash $T/rb/scripts/probe_p5self.sh 3 > /dev/null 2>&1 &   # an empty GPU for the start check ...
P=$!; wait $P
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-P5SELF/.done ]'
ok "nothing fetched" '! grep -q "^fetch" $CALLS'
ok "3-step smoke of E10b-P5self" 'grep -q "^run_arm 3 E10b-P5self SMOKE=1 STEPS=3 TEACHER_14B= PROBE_TASK=PROBE-P5SELF-smoke" $CALLS'
ok "steps and projection" '[ "$(res steps)" = 3 ] && [ "$(res projected_hours_150)" = 13.8 ]'
ok "report to logs/<node>/probe_p5self, event" 'grep -q "^remote put $T/root/runs/PROBE-P5SELF/up logs/abcd1234/probe_p5self" $CALLS && grep -q "^control event PROBE-P5SELF probe_p5self_pass" $CALLS'
rm -rf $T
# 2. out of memory: FAIL (a result; the arms are not queued)
setup
FAKE_MEM=0 FAKE_OOM=1 bash $T/rb/scripts/probe_p5self.sh 3 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && grep -q "^control event PROBE-P5SELF probe_p5self_fail" $CALLS'
rm -rf $T
# 3. a peak at or above 260 GB: FAIL
setup
FAKE_MEM=500 FAKE_SLEEP=1 PSE_PEAK_MIB=100 bash $T/rb/scripts/probe_p5self.sh 3 > /dev/null 2>&1
ok "peak over the limit is FAIL" '[ "$(res result)" = FAIL ] && python3 -c "import json, sys; assert \"peak\" in json.load(open(sys.argv[1]))[\"reason\"]" $T/root/runs/PROBE-P5SELF/report.json'
rm -rf $T
# 4. [v2.5.8e] 4b: PROBE-P5SELF-4B, a 3-step smoke of E10d-P5self, its own report path
res4() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-P5SELF-4B/report.json "$1"; }
setup
FAKE_MEM=0 bash $T/rb/scripts/probe_p5self.sh 3 4b > /dev/null 2>&1
ok "[v2.5.8e] 4b PASS: E10d-P5self smoke, report to probe_p5self_4b" '[ "$(res4 result)" = PASS ] && grep -q "^run_arm 3 E10d-P5self SMOKE=1 STEPS=3 TEACHER_14B= PROBE_TASK=PROBE-P5SELF-4B-smoke" $CALLS && grep -q "^remote put $T/root/runs/PROBE-P5SELF-4B/up logs/abcd1234/probe_p5self_4b" $CALLS && grep -q "^control event PROBE-P5SELF-4B probe_p5self_pass" $CALLS && [ ! -e $T/root/runs/PROBE-P5SELF ]'
rm -rf $T
# 5. [v2.5.8e] a teacher that is not the student itself: FAIL
setup
FAKE_MEM=0 FAKE_TEACHER=/m/Qwen3-8B bash $T/rb/scripts/probe_p5self.sh 3 4b > /dev/null 2>&1
ok "[v2.5.8e] teacher != student is FAIL" '[ "$(res4 result)" = FAIL ] && [[ "$(res4 reason)" == *"is not the student itself"* ]]'
rm -rf $T
# 6. [v2.5.9v] 06b: PROBE-P5SELF-06B, a 3-step smoke of E10c-P5self, its own report path; a teacher other than the student FAILs
res6() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-P5SELF-06B/report.json "$1"; }
setup
FAKE_MEM=0 bash $T/rb/scripts/probe_p5self.sh 3 06b > /dev/null 2>&1
ok "[v2.5.9v] 06b PASS: E10c-P5self smoke, report to probe_p5self_06b, event" '[ "$(res6 result)" = PASS ] && grep -q "^run_arm 3 E10c-P5self SMOKE=1 STEPS=3 TEACHER_14B= PROBE_TASK=PROBE-P5SELF-06B-smoke" $CALLS && grep -q "^remote put $T/root/runs/PROBE-P5SELF-06B/up logs/abcd1234/probe_p5self_06b" $CALLS && grep -q "^control event PROBE-P5SELF-06B probe_p5self_pass" $CALLS && [ -f $T/root/runs/PROBE-P5SELF-06B/.done ] && [ ! -e $T/root/runs/PROBE-P5SELF/report.json ]'
rm -rf $T
setup
FAKE_MEM=0 FAKE_TEACHER=/m/Qwen3-1.7B bash $T/rb/scripts/probe_p5self.sh 3 06b > /dev/null 2>&1
ok "[v2.5.9v] 06b: a 1.7B teacher for the 0.6B student is FAIL" '[ "$(res6 result)" = FAIL ] && [[ "$(res6 reason)" == *"is not the student itself"* ]]'
# [z158i] (Q-095 B's gate on 4 GPUs, 11-14 21:50Z: the empty check read four lines and never passed) a GPU list: the fullest GPU decides
rm -rf $T; setup; FAKE_MEM=0 bash $T/rb/scripts/probe_p5self.sh 3,4,5,6 > /dev/null 2>&1
ok "[z158i] a 4-GPU list: the start check passes, the smoke runs on 3,4,5,6, PASS, a single-number peak" '[ "$(res result)" = PASS ] && grep -q "^run_arm 3,4,5,6 E10b-P5self SMOKE=1" $CALLS && [ -z "$(cut -d, -f2 $T/root/runs/PROBE-P5SELF/mem.csv | grep -v -E "^[0-9]+$")" ]'
rm -rf $T; setup; FAKE_MEM=5000 PSE_WAIT_STEP=0 bash $T/rb/scripts/probe_p5self.sh 3,4,5,6 > /dev/null 2>&1
ok "[z158i] a 4-GPU list with a busy GPU: FAIL not empty" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"not empty after 180 s"* ]]'
rm -rf $T
# 9. [z163] p2: PROBE-P5SELFP2, a 3-step smoke of E10b-P5selfp2 with a rollout dump; PASS only when every teacher call suffixed every row
#    and no student prompt contains the suffix
resp() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-P5SELFP2/report.json "$1"; }
for case in ok none partial leak; do
  setup; mkdir -p $T/rb/code/train/teacher_suffix; cp $SRC/code/train/teacher_suffix/p2.txt $T/rb/code/train/teacher_suffix/
  FAKE_MEM=0 FAKE_TS=$case bash $T/rb/scripts/probe_p5self.sh 3 p2 > /dev/null 2>&1
  case $case in
    ok) ok "[z163] p2 PASS: E10b-P5selfp2 smoke with a dump, 3 calls x 64 rows suffixed, 3 prompts clean" '[ "$(resp result)" = PASS ] && grep -q "^run_arm 3 E10b-P5selfp2 SMOKE=1 STEPS=3 TEACHER_14B= PROBE_TASK=PROBE-P5SELFP2-smoke" $CALLS && [[ "$(resp reason)" == *"OK 3 teacher calls, 192 rows all suffixed; 3 student prompts without it"* ]] && grep -q "^remote put $T/root/runs/PROBE-P5SELFP2/up logs/abcd1234/probe_p5selfp2" $CALLS';;
    none) ok "[z163] p2 FAIL: no teacher_suffix.jsonl" '[ "$(resp result)" = FAIL ] && [[ "$(resp reason)" == *"the hook did not run"* ]]';;
    partial) ok "[z163] p2 FAIL: a call that did not suffix every row" '[ "$(resp result)" = FAIL ] && [[ "$(resp reason)" == *"did not suffix every row"* ]]';;
    leak) ok "[z163] p2 FAIL: a student prompt with the suffix" '[ "$(resp result)" = FAIL ] && [[ "$(resp reason)" == *"student prompts contain the teacher suffix"* ]]';;
  esac
  rm -rf $T
done
[ $fails = 0 ] && echo "PROBE_P5SELF_TEST PASS ($n checks)" || echo "PROBE_P5SELF_TEST FAIL ($fails of $n)"
