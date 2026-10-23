#!/usr/bin/env bash
# [v2.5.9z66] Stub tests of probe_w8k.sh (PROBE-P5-8K; no GPU, network or HF). Usage: bash scripts/test_probe_w8k.sh -> PROBE_W8K_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin; cd $T
  cp $SRC/scripts/probe_w8k.sh $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\nstudent_template: non-thinking, max_response_length ${FAKE_MAXRESP:-8192}\nattention: flash_attention_2 (use_remove_padding=${FAKE_RMPAD:-True})\n" > $d/config.yaml
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
  echo "(TaskRunner pid=1) step:$i - critic/score/mean:${FAKE_SCORE:-0.55} - response_length/mean:3100.0 - response_length/clip_ratio:0.05 - timing_s/gen:300 - timing_s/step:600" >> $d/train.log
done
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PW8_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-P5-8K/report.json "$1"; }
# 1. PASS: the 3-step smoke of E10b-P5-8k, report, upload, event
setup; bash $T/rb/scripts/probe_w8k.sh 3 > /dev/null 2>&1
ok "PASS" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-P5-8K/.done ]'
ok "the smoke = 3 steps of E10b-P5-8k in PROBE-P5-8K-smoke" 'grep -q "^run_arm 3 E10b-P5-8k SMOKE=1 STEPS=3 PROBE_TASK=PROBE-P5-8K-smoke$" $CALLS'
ok "report: config checks, 3 steps, projected hours for 50 steps" '[ "$(res "v[\"config\"]")" = "{'"'"'max_response_length_8192'"'"': True, '"'"'remove_padding'"'"': True}" ] && [ "$(res "v[\"steps\"]")" = 3 ] && [ "$(res "v[\"projected_hours_50\"]")" = 8.3 ]'
ok "report upload and event" 'grep -q "^remote put $T/root/runs/PROBE-P5-8K/up logs/abcd1234/probe_w8k" $CALLS && grep -q "^control event PROBE-P5-8K probe_w8k_pass" $CALLS'
cd $SRC; rm -rf $T
# 2. the failure modes, one at a time
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_MAXRESP=4096 bash $T/rb/scripts/probe_w8k.sh 3 > /dev/null 2>&1; check "a 4K window is FAIL" "max_response_length 8192"; cd $SRC; rm -rf $T
setup; FAKE_RMPAD=False bash $T/rb/scripts/probe_w8k.sh 3 > /dev/null 2>&1; check "no remove-padding is FAIL" "remove-padding"; cd $SRC; rm -rf $T
setup; FAKE_LOGSTEPS=2 bash $T/rb/scripts/probe_w8k.sh 3 > /dev/null 2>&1; check "a missing step is FAIL" "expected 1..3"; cd $SRC; rm -rf $T
setup; FAKE_SCORE=1.4 bash $T/rb/scripts/probe_w8k.sh 3 > /dev/null 2>&1; check "a score outside [0, 1] is FAIL" "a score outside [0, 1]"; cd $SRC; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_w8k.sh 3 > /dev/null 2>&1; check "OOM is FAIL" "smoke exited"; cd $SRC; rm -rf $T
setup; FAKE_MEM=2000 bash $T/rb/scripts/probe_w8k.sh 3 > /dev/null 2>&1; check "a busy GPU is FAIL before any smoke" "not empty after 180 s"; ok "no smoke on a busy GPU" '! grep -q "^run_arm" $CALLS'; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_W8K_TEST PASS ($n checks)" || echo "PROBE_W8K_TEST FAIL ($fails of $n)"
