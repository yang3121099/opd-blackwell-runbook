#!/usr/bin/env bash
# [v2.5.8b] Stub tests of probe_e12var.sh kl (no GPU, network or HF): PASS needs the start as the reference (config and
# log), a step-1 KL loss near 0 with a positive one later, and the reward equal to the verifier score.
# Usage: bash scripts/test_probe_e12var.sh -> PROBE_E12VAR_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin
  cp $SRC/scripts/probe_e12var.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_e12.sh <<'SH'
echo "run_e12 $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; S=/runs/E10b-P5/ckpt/global_step_150/actor/huggingface_bf16
printf "task: E12b-P5-kl\nstart: $S\nalgo: GRPO, ${FAKE_ALGO:-KL loss 0.01 low_var_kl to the start (reference = $S)}\n" > $d/config.yaml
echo "actor_rollout_ref.model.path=$S ref.model.path=${FAKE_REF:-$S}" >> $d/train.log
set -- ${FAKE_KL:-0.0001 0.002 0.004}; i=0
for k in "$@"; do i=$((i + 1)); r=0.5; [ "${FAKE_RW_STEP:-0}" = $i ] && r=0.7
  echo "(TaskRunner pid=1) step:$i - actor/kl_loss:$k - critic/score/mean:0.5 - critic/rewards/mean:$r - timing_s/step:300" >> $d/train.log; done
exit 0
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PEV_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-E12B-KL/report.json "$1"; }
# 1. PASS: reference = the start, step-1 KL ~0 and positive later, reward = score
setup
bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-E12B-KL/.done ]'
ok "3-step smoke of E12b-P5-kl" 'grep -q "^run_e12 2 E12b-P5-kl SMOKE=1 STEPS=3 PROBE_TASK=PROBE-E12B-KL-smoke" $CALLS'
ok "report: KL by step, the reference, reward vs score; upload and event" '[[ "$(res kl_loss_by_step)" == *"0.0001"* ]] && [[ "$(res reward_vs_score)" == *"0.5"* ]] && grep -q "^remote put $T/root/runs/PROBE-E12B-KL/up logs/abcd1234/probe_e12var/PROBE-E12B-KL" $CALLS && grep -q "^control event PROBE-E12B-KL probe_e12var_pass" $CALLS'
rm -rf $T
# 2. the reference is a teacher: a large step-1 KL loss is FAIL
setup
FAKE_KL="0.41 0.40 0.39" bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "step-1 KL too large is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"the reference is not the start"* ]]'
rm -rf $T
# 3. a reward that is not the verifier score is FAIL
setup
FAKE_RW_STEP=2 bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "reward != score is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"reward differs from the verifier score at steps 2"* ]]'
rm -rf $T
# 4. no positive KL loss after step 1 (the loss is not on) is FAIL
setup
FAKE_KL="0 0 0" bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "no KL after step 1 is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"no positive KL loss after step 1"* ]]'
rm -rf $T
# 5. the config does not name the start as the reference, and the log has the start once: FAIL
setup
FAKE_ALGO="KL 0" FAKE_REF=/models/Qwen3-8B bash $T/rb/scripts/probe_e12var.sh 2 kl > /dev/null 2>&1
ok "reference not the start in config / log is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"does not name the start"* && "$(res reason)" == *"appears 1x"* ]]'
rm -rf $T
# 6. an unknown variant is refused
setup
bash $T/rb/scripts/probe_e12var.sh 2 foo > /dev/null 2>&1; rc=$?
ok "unknown variant exits 2" '[ $rc = 2 ] && [ ! -e $T/root/runs/PROBE-E12B-FOO ]'
rm -rf $T
[ $fails = 0 ] && echo "PROBE_E12VAR_TEST PASS ($n checks)" || echo "PROBE_E12VAR_TEST FAIL ($fails of $n)"
