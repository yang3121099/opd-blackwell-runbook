#!/usr/bin/env bash
# [v2.5.9z68] Stub tests of probe_8b.sh (PROBE-8B; no GPU, network or HF). Usage: bash scripts/test_probe_8b.sh -> PROBE_8B_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin; cd $T
  cp $SRC/scripts/probe_8b.sh $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\nstudent: /m/models/Qwen3-8B (no_think)\nteacher: ${FAKE_TEACHER:-/m/models/Qwen3-8B} (ref; think-template=${FAKE_THINK:-1}; param_offload=1)\ngpu: NVIDIA B300 SXM6 AC x${FAKE_NG:-4} ($1)\n" > $d/config.yaml
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
  echo "(TaskRunner pid=1) step:$i - critic/score/mean:${FAKE_SCORE:-0.7} - response_length/mean:900.0 - response_length/clip_ratio:0.01 - timing_s/ref:120 - timing_s/step:900" >> $d/train.log
  echo '{"mode": "count", "call": '$i', "batch": 128, "rows": '${FAKE_ROWS:-128}', "think": '${FAKE_ROWS:-128}', "plain": 0}' >> $d/route.jsonl
done
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  # nvidia-smi: memory.used per GPU; with index: "<i>, <mib>" lines for the GPUs asked
  cat > $T/bin/nvidia-smi <<'S3'
#!/usr/bin/env bash
ids=""; p=""; for a in "$@"; do [ "$p" = -i ] && ids=$a; p=$a; done
for i in $(echo $ids | tr , ' '); do case "$*" in *index,memory.used*) echo "$i, ${FAKE_MEM:-0}";; *) echo "${FAKE_MEM:-0}";; esac; done
S3
  chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH P8B_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-8B/report.json "$1"; }
# 1. PASS: the 3-step smoke of E10e-P5self on the four GPUs, the teacher's rows counted, report, upload, event
setup; bash $T/rb/scripts/probe_8b.sh 4,5,6,7 > /dev/null 2>&1
ok "PASS" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-8B/.done ]'
ok "the smoke = 3 steps of E10e-P5self on 4,5,6,7" 'grep -q "^run_arm 4,5,6,7 E10e-P5self SMOKE=1 STEPS=3 PROBE_TASK=PROBE-8B-smoke$" $CALLS'
ok "report: 3 teacher calls of 128 rows, projected hours for 50 steps" '[ "$(res "len(v[\"checks\"][\"teacher_calls\"])")" = 3 ] && [ "$(res "v[\"projected_hours_50\"]")" = 12.5 ]'
ok "report upload and event" 'grep -q "^remote put $T/root/runs/PROBE-8B/up logs/abcd1234/probe_8b" $CALLS && grep -q "^control event PROBE-8B probe_8b_pass" $CALLS'
cd $SRC; rm -rf $T
# 2. the failure modes, one at a time
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; bash $T/rb/scripts/probe_8b.sh 4,5 > /dev/null 2>&1; check "two GPUs is FAIL" "needs four GPUs"; ok "no smoke on two GPUs" '! grep -q "^run_arm" $CALLS'; cd $SRC; rm -rf $T
setup; FAKE_ROWS=64 bash $T/rb/scripts/probe_8b.sh 4,5,6,7 > /dev/null 2>&1; check "64 teacher rows for 128 samples is FAIL" "teacher rows are not 128"; cd $SRC; rm -rf $T
setup; FAKE_TEACHER=/m/models/Qwen3-1.7B bash $T/rb/scripts/probe_8b.sh 4,5,6,7 > /dev/null 2>&1; check "a teacher other than the student is FAIL" "both student and teacher"; cd $SRC; rm -rf $T
setup; FAKE_THINK=0 bash $T/rb/scripts/probe_8b.sh 4,5,6,7 > /dev/null 2>&1; check "the teacher off its thinking template is FAIL" "thinking template"; cd $SRC; rm -rf $T
setup; FAKE_NG=2 bash $T/rb/scripts/probe_8b.sh 4,5,6,7 > /dev/null 2>&1; check "a smoke on two GPUs is FAIL" "four GPUs"; cd $SRC; rm -rf $T
setup; FAKE_LOGSTEPS=2 bash $T/rb/scripts/probe_8b.sh 4,5,6,7 > /dev/null 2>&1; check "a missing step is FAIL" "expected 1..3"; cd $SRC; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_8b.sh 4,5,6,7 > /dev/null 2>&1; check "OOM is FAIL" "smoke exited"; cd $SRC; rm -rf $T
setup; FAKE_MEM=2000 bash $T/rb/scripts/probe_8b.sh 4,5,6,7 > /dev/null 2>&1; check "busy GPUs is FAIL before any smoke" "not empty after 180 s"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_8B_TEST PASS ($n checks)" || echo "PROBE_8B_TEST FAIL ($fails of $n)"
