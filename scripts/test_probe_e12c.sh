#!/usr/bin/env bash
# [v2.5.8a] Stub tests of probe_e12c.sh (no GPU, network or HF). Usage: bash scripts/test_probe_e12c.sh -> PROBE_E12C_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin
  cp $SRC/scripts/probe_e12c.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_e12.sh <<'SH'
echo "run_e12 $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; printf 'task: E12c-P0\nstart: /models/Qwen3-0.6B\n' > $d/config.yaml
for i in $(seq 1 ${FAKE_STEPS:-$SMOKE_STEPS}); do echo "(TaskRunner pid=1) step:$i - timing_s/step:120 - response_length/mean:900" >> $d/train.log; done
sleep ${FAKE_SLEEP:-0}; [ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit ${FAKE_RC:-0}
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-150000}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PEC_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-E12C/report.json "$1"; }
# 1. PASS: a 3-step smoke of E12c-P0 in its own directory; report, upload, event
setup
FAKE_MEM=0 bash $T/rb/scripts/probe_e12c.sh 3 > /dev/null 2>&1
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-E12C/.done ]'
ok "3-step smoke of E12c-P0" 'grep -q "^run_e12 3 E12c-P0 SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12C-smoke" $CALLS'
ok "last step and start reported" '[ "$(res last_step)" = 3 ] && [ "$(res start)" = /models/Qwen3-0.6B ]'
ok "report to logs/<node>/probe_e12c, event" 'grep -q "^remote put $T/root/runs/PROBE-E12C/up logs/abcd1234/probe_e12c" $CALLS && grep -q "^control event PROBE-E12C probe_e12c_pass" $CALLS'
bash $T/rb/scripts/probe_e12c.sh 3 > $T/again 2>&1
ok "done once" 'grep -q "TASK_ALREADY_DONE PROBE-E12C" $T/again && [ $(grep -c "^run_e12" $CALLS) = 1 ]'
rm -rf $T
# 2. out of memory: FAIL
setup
FAKE_MEM=0 FAKE_OOM=1 bash $T/rb/scripts/probe_e12c.sh 3 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && grep -q "^control event PROBE-E12C probe_e12c_fail" $CALLS'
rm -rf $T
# 3. exit 0 but stopped before step 3: FAIL
setup
FAKE_MEM=0 FAKE_STEPS=2 bash $T/rb/scripts/probe_e12c.sh 3 > /dev/null 2>&1
ok "stopped at step 2 is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"step 2 of 3"* ]]'
rm -rf $T
# 4. a peak at or above the limit: FAIL
setup
FAKE_MEM=500 FAKE_SLEEP=1 PEC_PEAK_MIB=100 bash $T/rb/scripts/probe_e12c.sh 3 > /dev/null 2>&1
ok "peak over the limit is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"peak"* ]]'
rm -rf $T
[ $fails = 0 ] && echo "PROBE_E12C_TEST PASS ($n checks)" || echo "PROBE_E12C_TEST FAIL ($fails of $n)"
