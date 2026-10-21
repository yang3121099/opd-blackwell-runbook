#!/usr/bin/env bash
# [v2.5.9a] Stub tests of probe_2gpu.sh (no GPU, network or HF). Usage: bash scripts/test_probe_2gpu.sh -> PROBE_2GPU_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
step_line() { echo "(TaskRunner pid=1) step:$1 - actor/pg_loss:0.01 - actor/grad_norm:0.2 - critic/score/mean:0.7 - response_length/mean:10000 - response_length/clip_ratio:0.2 - timing_s/gen:$(($2 * 3 / 4)) - timing_s/update_actor:$(($2 / 5)) - timing_s/step:$2 - perf/total_num_tokens:5000000"; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs/PROBE-E12B-16K-smoke $T/bin; cd $T   # nothing the stubs write lands in the caller's directory
  cp $SRC/scripts/probe_2gpu.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  for s in 1 2 3; do step_line $s ${FAKE_REF_STEP:-1500} >> $T/root/runs/PROBE-E12B-16K-smoke/train.log; done
  # the two-GPU smoke: its config names the GPUs, train.log has the steps; it lasts long enough for one memory sample
  cat > $T/rb/scripts/run_e12.sh <<'S2'
echo "run_e12 $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\ngpus: ${FAKE_CFG_GPUS:-$1} (n_gpus_per_node 2)\n" > $d/config.yaml
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do step_line $i ${FAKE_STEP:-1000} >> $d/train.log; done
sleep 1
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  # nvidia-smi: the first two calls (the empty-GPU checks) see 0 MiB, later ones FAKE_MEM, FAKE_IDLE_GPU stays at 0
  printf '#!/usr/bin/env bash\nn=$(cat $NVC 2>/dev/null || echo 0); n=$((n + 1)); echo $n > $NVC\ng=$(echo "$*" | sed -n "s/.*-i \\([0-9]*\\).*/\\1/p")\n[ $n -le 2 ] && { echo 0; exit 0; }\n[ "$g" = "${FAKE_IDLE_GPU:-x}" ] && echo 0 || echo ${FAKE_MEM:-30000}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export -f step_line
  export CALLS=$T/calls NVC=$T/nvc PATH=$T/bin:$PATH P2G_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); [v := v.get(k, {}) if isinstance(v, dict) else '' for k in sys.argv[2].split('.')]; print(v)" $T/root/runs/PROBE-2GPU/report.json "$1"; }
# 1. PASS: the smoke of E12b-P5-16k on 4,5; speedup 1.5 (1500 / 1000 s per step), worth it
setup
bash $T/rb/scripts/probe_2gpu.sh 4,5 > /dev/null 2>&1
ok "PASS on two GPUs" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-2GPU/.done ] && grep -q "^run_e12 4,5 E12b-P5-16k SMOKE=1 STEPS=3 PROBE_TASK=PROBE-2GPU-smoke$" $CALLS'
ok "speedup 1.5 per step and per token, worth it; means of both smokes" '[ "$(res speedup)" = 1.5 ] && [ "$(res speedup_per_token)" = 1.5 ] && [ "$(res worth_it)" = True ] && [ "$(res one_gpu.mean.step_s)" = 1500.0 ] && [ "$(res two_gpu.mean.step_s)" = 1000.0 ]'
ok "reaped on both GPUs, report upload, event" 'grep -q "^control reap 4 PROBE-2GPU" $CALLS && grep -q "^control reap 5 PROBE-2GPU" $CALLS && grep -q "^remote put $T/root/runs/PROBE-2GPU/up logs/abcd1234/probe_2gpu/PROBE-2GPU" $CALLS && grep -q "^control event PROBE-2GPU probe_2gpu_pass" $CALLS'
rm -rf $T
# 2. a small speedup still passes (a correct run), but is not worth it
setup
FAKE_STEP=1400 bash $T/rb/scripts/probe_2gpu.sh 4,5 > /dev/null 2>&1
ok "speedup 1.07: PASS, not worth it" '[ "$(res result)" = PASS ] && [ "$(res worth_it)" = False ]'
rm -rf $T
# 3. one GPU only; 4. no one-GPU smoke to compare with: FAIL before any smoke
setup
bash $T/rb/scripts/probe_2gpu.sh 4 > /dev/null 2>&1
ok "a single GPU is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"needs two GPUs"* ]] && ! grep -q "^run_e12" $CALLS'
rm -rf $T
setup
rm -rf $T/root/runs/PROBE-E12B-16K-smoke; bash $T/rb/scripts/probe_2gpu.sh 4,5 > /dev/null 2>&1
ok "no one-GPU smoke is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"no one-GPU smoke"* ]]'
rm -rf $T
# 5. a GPU left unused; 6. OOM; 7. a step missing; 8. the config names other GPUs
setup
FAKE_IDLE_GPU=5 bash $T/rb/scripts/probe_2gpu.sh 4,5 > /dev/null 2>&1
ok "an unused GPU is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"gpu 5 peak 0 MiB: not used"* ]]'
rm -rf $T
setup
FAKE_OOM=1 bash $T/rb/scripts/probe_2gpu.sh 4,5 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"exited"* || "$(res reason)" == *"out of memory"* ]]'
rm -rf $T
setup
FAKE_LOGSTEPS=2 bash $T/rb/scripts/probe_2gpu.sh 4,5 > /dev/null 2>&1
ok "a missing step is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"expected 1..3"* ]]'
rm -rf $T
setup
FAKE_CFG_GPUS=4 bash $T/rb/scripts/probe_2gpu.sh 4,5 > /dev/null 2>&1
ok "a config without both GPUs is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"does not name both GPUs"* ]]'
rm -rf $T
cd $SRC
[ $fails = 0 ] && echo "PROBE_2GPU_TEST PASS ($n checks)" || echo "PROBE_2GPU_TEST FAIL ($fails of $n)"
