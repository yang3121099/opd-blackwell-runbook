#!/usr/bin/env bash
# Stub tests of v2.5.7b probe_4b.sh (no GPU, network or HF). Usage: bash scripts/test_probe4b.sh -> PROBE4B_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin
  cp $SRC/scripts/probe_4b.sh $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n"); os.makedirs(sys.argv[3], exist_ok=True); open(os.path.join(sys.argv[3], "config.json"), "w").write("{}")\n' > $T/rb/scripts/fetch_model.py
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS STUDENT_4B=$STUDENT_4B PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; : > $d/config.yaml
for i in $(seq 1 $SMOKE_STEPS); do echo "(TaskRunner pid=1) step:$i - timing_s/gen:100 - timing_s/update_actor:150 - timing_s/step:$([ $i = 1 ] && echo 600 || echo 330) - response_length/mean:3000" >> $d/train.log; done
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-150000}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH P4_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-4B/report.json "$1"; }
# 1. PASS: the 4B student fetched to the share, a 10-step smoke of E10d-P5 with it, peak / timing / projection reported
setup
FAKE_MEM=0 bash $T/rb/scripts/probe_4b.sh 3 > /dev/null 2>&1 &   # an empty GPU for the start check ...
P=$!; wait $P
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-4B/.done ]'
ok "4B fetched to the share" 'grep -q "^fetch Qwen/Qwen3-4B  $T/root/models/Qwen3-4B --task PROBE-4B" $CALLS && [ -f $T/root/models/Qwen3-4B/.fetch_ok ]'
ok "10-step smoke of E10d-P5 with it" 'grep -q "^run_arm 3 E10d-P5 SMOKE=1 STEPS=10 STUDENT_4B=$T/root/models/Qwen3-4B PROBE_TASK=PROBE-4B-smoke" $CALLS'
ok "steps and projection" '[ "$(res steps)" = 10 ] && [ "$(res projected_hours_150)" = 13.8 ]'
ok "report to logs/<node>/probe_4b, event" 'grep -q "^remote put $T/root/runs/PROBE-4B/up logs/abcd1234/probe_4b" $CALLS && grep -q "^control event PROBE-4B probe_4b_pass" $CALLS'
rm -rf $T
# 2. out of memory: FAIL (a result; the arms are not queued)
setup
FAKE_MEM=0 FAKE_OOM=1 bash $T/rb/scripts/probe_4b.sh 3 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && grep -q "^control event PROBE-4B probe_4b_fail" $CALLS'
rm -rf $T
[ $fails = 0 ] && echo "PROBE4B_TEST PASS ($n checks)" || echo "PROBE4B_TEST FAIL ($fails of $n)"
