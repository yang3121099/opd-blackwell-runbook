#!/usr/bin/env bash
# [v2.5.9u] Stub tests of probe_polish.sh (no GPU, network or HF). Usage: bash scripts/test_probe_polish.sh -> PROBE_POLISH_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin
  cp $SRC/scripts/probe_polish.sh $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  # the smoke: config with the start and the teacher as run_arm.sh writes them, three steps, the lengths falling
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
S=${FAKE_START:-/r/runs/E10c-P5-fix4/ckpt/global_step_150/actor/huggingface_bf16}
printf "task: $2\nstudent: $S (no_think)\nteacher: ${FAKE_TEACHER:-Qwen/Qwen3-0.6B} (ref; think-template=${FAKE_THINK:-1}; param_offload=1)${FAKE_NOTE-; polish: the 0.6B base on its thinking template, from E10c-P5-fix4 s150}\n" > $d/config.yaml
for i in $(seq 1 $SMOKE_STEPS); do echo "(TaskRunner pid=1) step:$i - timing_s/gen:100 - timing_s/update_actor:50 - timing_s/step:$([ $i = 1 ] && echo 300 || echo 180) - response_length/mean:$((9000 - 1000 * i))" >> $d/train.log; done
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PPO_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-POLISH/report.json "$1"; }
setup
bash $T/rb/scripts/probe_polish.sh 2 > /dev/null 2>&1
ok "PASS: a 3-step smoke of E10c-P5-polish" '[ "$(res "v[\"result\"]")" = PASS ] && grep -q "^run_arm 2 E10c-P5-polish SMOKE=1 STEPS=3 PROBE_TASK=PROBE-POLISH-smoke$" $CALLS && [ -f $T/root/runs/PROBE-POLISH/.done ]'
ok "steps, 20-step projection (180 s a step: 1.0 h), lengths per step" '[ "$(res "(v[\"steps\"], v[\"projected_hours_20\"], v[\"response_length\"])")" = "(3, 1.0, [8000.0, 7000.0, 6000.0])" ]'
ok "report to logs/<node>/probe_polish, event" 'grep -q "^remote put $T/root/runs/PROBE-POLISH/up logs/abcd1234/probe_polish" $CALLS && grep -q "^control event PROBE-POLISH probe_polish_pass" $CALLS'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_START=/m/Qwen3-0.6B bash $T/rb/scripts/probe_polish.sh 2 > /dev/null 2>&1; check "a start other than P5-fix4 s150 is FAIL" "is not E10c-P5-fix4's s150 save"; cd $SRC; rm -rf $T
setup; FAKE_TEACHER=Qwen/Qwen3-8B bash $T/rb/scripts/probe_polish.sh 2 > /dev/null 2>&1; check "a teacher other than the 0.6B base is FAIL" "is not the 0.6B base on its thinking template"; cd $SRC; rm -rf $T
setup; FAKE_THINK=0 bash $T/rb/scripts/probe_polish.sh 2 > /dev/null 2>&1; check "the teacher off its thinking template is FAIL" "is not the 0.6B base on its thinking template"; cd $SRC; rm -rf $T
setup; FAKE_NOTE="" bash $T/rb/scripts/probe_polish.sh 2 > /dev/null 2>&1; check "a config without the polish note is FAIL" "is not the 0.6B base on its thinking template"; cd $SRC; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_polish.sh 2 > /dev/null 2>&1; check "OOM is FAIL" "smoke exited"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_POLISH_TEST PASS ($n checks)" || echo "PROBE_POLISH_TEST FAIL ($fails of $n)"
