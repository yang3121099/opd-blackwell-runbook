#!/usr/bin/env bash
# Stub tests of v2.5.7p probe_t14.sh (no GPU, network or HF). Usage: bash scripts/test_probe_t14.sh -> PROBE_T14_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin
  cp $SRC/scripts/probe_t14.sh $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n"); os.makedirs(sys.argv[3], exist_ok=True); open(os.path.join(sys.argv[3], "config.json"), "w").write("{}")\n' > $T/rb/scripts/fetch_model.py
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS TEACHER_14B=$TEACHER_14B PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; : > $d/config.yaml
for i in $(seq 1 $SMOKE_STEPS); do echo "(TaskRunner pid=1) step:$i - timing_s/gen:100 - timing_s/update_actor:150 - timing_s/step:$([ $i = 1 ] && echo 600 || echo 330) - response_length/mean:3000" >> $d/train.log; done
sleep ${FAKE_SLEEP:-0}; [ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-150000}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PT14_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-T14/report.json "$1"; }
# 1. PASS: the 14B teacher fetched to the share, a 3-step smoke of E10b-P5-t14 with it, peak / timing / projection reported
setup
FAKE_MEM=0 bash $T/rb/scripts/probe_t14.sh 3 > /dev/null 2>&1 &   # an empty GPU for the start check ...
P=$!; wait $P
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-T14/.done ]'
ok "14B fetched to the share" 'grep -q "^fetch Qwen/Qwen3-14B  $T/root/models/Qwen3-14B --task PROBE-T14" $CALLS && [ -f $T/root/models/Qwen3-14B/.fetch_ok ]'
ok "3-step smoke of E10b-P5-t14 with it" 'grep -q "^run_arm 3 E10b-P5-t14 SMOKE=1 STEPS=3 TEACHER_14B=$T/root/models/Qwen3-14B PROBE_TASK=PROBE-T14-smoke" $CALLS'
ok "steps and projection" '[ "$(res steps)" = 3 ] && [ "$(res projected_hours_150)" = 13.8 ]'
ok "report to logs/<node>/probe_t14, event" 'grep -q "^remote put $T/root/runs/PROBE-T14/up logs/abcd1234/probe_t14" $CALLS && grep -q "^control event PROBE-T14 probe_t14_pass" $CALLS'
rm -rf $T
# 2. out of memory: FAIL (a result; the arms are not queued)
setup
FAKE_MEM=0 FAKE_OOM=1 bash $T/rb/scripts/probe_t14.sh 3 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && grep -q "^control event PROBE-T14 probe_t14_fail" $CALLS'
rm -rf $T
# 3. a peak at or above 260 GB: FAIL
setup
FAKE_MEM=500 FAKE_SLEEP=1 PT14_PEAK_MIB=100 bash $T/rb/scripts/probe_t14.sh 3 > /dev/null 2>&1
ok "peak over the limit is FAIL" '[ "$(res result)" = FAIL ] && python3 -c "import json, sys; assert \"peak\" in json.load(open(sys.argv[1]))[\"reason\"]" $T/root/runs/PROBE-T14/report.json'
rm -rf $T
[ $fails = 0 ] && echo "PROBE_T14_TEST PASS ($n checks)" || echo "PROBE_T14_TEST FAIL ($fails of $n)"
