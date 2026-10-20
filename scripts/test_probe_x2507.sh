#!/usr/bin/env bash
# [v2.5.8i] Stub tests of probe_x2507.sh (no GPU, network or HF). Usage: bash scripts/test_probe_x2507.sh -> PROBE_X2507_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin; cd $T   # [v2.5.8i] nothing the stubs write can land in the caller's directory
  cp $SRC/scripts/probe_x2507.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb STUDENT_4B=/m/Qwen3-4B" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; RT=0; TT=/hf/models--Qwen--Qwen3-4B-Thinking-2507/snapshots/r; [ $2 = E10d-X2 ] && { RT=1; TT=/m/Qwen3-4B; }
printf "task: $2\nstudent: ${FAKE_STUDENT:-/hf/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/r} (no_think)\nteacher: $TT (ref; think-template=$RT; param_offload=1)\n" > $d/config.yaml
for i in 1 2 3; do echo "(TaskRunner pid=1) step:$i - timing_s/step:300" >> $d/train.log; done
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  # the template check's python: records its arguments, writes templates.json with FAKE_PROBLEM as the only problem
  printf '#!/usr/bin/env bash\necho "py $*" >> $CALLS; cat > /dev/null\nP=${FAKE_PROBLEM:-}; [ -n "$P" ] && P="\\"$P\\""\necho "{\\"problems\\": [$P], \\"ref_think\\": $4}" > $5\n' > $T/bin/fakepy; chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PXT_WAIT_STEP=0 PXT_PY=$T/bin/fakepy; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-X2507-$1/report.json "$2"; }
# 1. X1 PASS: the template check with the 2507 pair and think-template 0, then the smoke
setup
bash $T/rb/scripts/probe_x2507.sh 3 X1 > /dev/null 2>&1
ok "X1 PASS" '[ "$(res X1 result)" = PASS ] && [ -f $T/root/runs/PROBE-X2507-X1/.done ]'
ok "X1: templates checked for Instruct-2507 / Thinking-2507 without the P5 tag" 'grep -q "^py - Qwen/Qwen3-4B-Instruct-2507 Qwen/Qwen3-4B-Thinking-2507 0 " $CALLS && grep -q "^run_arm 3 E10d-X1 SMOKE=1 STEPS=3 PROBE_TASK=PROBE-X2507-X1-smoke" $CALLS'
ok "X1: report, upload, event" 'grep -q "^remote put $T/root/runs/PROBE-X2507-X1/up logs/abcd1234/probe_x2507/PROBE-X2507-X1" $CALLS && grep -q "^control event PROBE-X2507-X1 probe_x2507_pass" $CALLS'
rm -rf $T
# 2. X2 PASS: the hybrid 4B teacher with the P5 tag
setup
bash $T/rb/scripts/probe_x2507.sh 3 X2 > /dev/null 2>&1
ok "X2 PASS with ref_think 1 and the hybrid teacher" '[ "$(res X2 result)" = PASS ] && grep -q "^py - Qwen/Qwen3-4B-Instruct-2507 /m/Qwen3-4B 1 " $CALLS'
rm -rf $T
# 3. a template problem (e.g. a doubled <think>): FAIL before any smoke
setup
FAKE_PROBLEM="the teacher prompt holds 2 <think>" bash $T/rb/scripts/probe_x2507.sh 3 X1 > /dev/null 2>&1
ok "template problem is FAIL, no smoke" '[ "$(res X1 result)" = FAIL ] && [[ "$(res X1 reason)" == *"holds 2 <think>"* ]] && ! grep -q "^run_arm" $CALLS'
rm -rf $T
# 4. the smoke ran another student: FAIL; 5. OOM: FAIL
setup
FAKE_STUDENT=/m/Qwen3-4B bash $T/rb/scripts/probe_x2507.sh 3 X1 > /dev/null 2>&1
ok "wrong student in the smoke is FAIL" '[ "$(res X1 result)" = FAIL ] && [[ "$(res X1 reason)" == *"not Instruct-2507"* ]]'
rm -rf $T
setup
FAKE_OOM=1 bash $T/rb/scripts/probe_x2507.sh 3 X2 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res X2 result)" = FAIL ] && [[ "$(res X2 reason)" == *"out of memory"* || "$(res X2 reason)" == *"exited"* ]]'
rm -rf $T
setup
bash $T/rb/scripts/probe_x2507.sh 3 X3 > /dev/null 2>&1; rc=$?
ok "unknown arm exits 2" '[ $rc = 2 ]'
rm -rf $T
cd $SRC
[ $fails = 0 ] && echo "PROBE_X2507_TEST PASS ($n checks)" || echo "PROBE_X2507_TEST FAIL ($fails of $n)"
