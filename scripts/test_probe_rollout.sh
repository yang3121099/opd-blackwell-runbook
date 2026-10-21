#!/usr/bin/env bash
# [v2.5.9i] Stub tests of probe_rollout.sh (no GPU, network or HF): the variant's VLLM_UTIL reaches run_e12.sh through
# PROBE_ENV over control.env, the per-step timing and the comparison with U60, and the FAIL cases.
# Usage: bash scripts/test_probe_rollout.sh -> PROBE_ROLLOUT_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin; cd $T
  cp $SRC/scripts/probe_rollout.sh $T/rb/scripts/
  # env.sh as on the node: control.env (here VLLM_UTIL=0.6, a node-wide setting) first, then PROBE_ENV wins
  echo "VLLM_UTIL=0.6" > $T/root/control.env
  cat > $T/rb/scripts/env.sh <<E
export OPD_ROOT=$T/root RUNBOOK=$T/rb
if [ -f \$OPD_ROOT/control.env ]; then source \$OPD_ROOT/control.env; fi
$(sed -n '/^for _kv in \${PROBE_ENV:-}/p' $SRC/scripts/env.sh)
E
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  # run_e12.sh: the generation time scales with 0.6 / VLLM_UTIL (more KV, more sequences at once); FAKE_LEN_K scales the lengths
  cat > $T/rb/scripts/run_e12.sh <<'S'
source "$(dirname "$0")/env.sh"
U=${VLLM_UTIL:-0.6}; echo "run_e12 $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK VLLM_UTIL=$U" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; printf "task: $2\nvariant: 8k\n" > $d/config.yaml
echo "{'actor_rollout_ref': {'rollout': {'gpu_memory_utilization': ${FAKE_CFG_UTIL:-$U}, 'enforce_eager': False}}}" >> $d/train.log
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
  gen=$(python3 -c "print(round(500 * 0.6 / $U + (60 if $i == 1 else 0), 1))"); upd=145
  len=$(python3 -c "print(round(7000 * ${FAKE_LEN_K:-1}))")
  echo "(TaskRunner pid=1) step:$i - actor/pg_loss:0.01 - critic/score/mean:0.6 - response_length/mean:$len - response_length/clip_ratio:0.58 - timing_s/gen:$gen - timing_s/update_actor:$upd - timing_s/step:$(python3 -c "print($gen + $upd + 40)") - perf/max_memory_reserved_gb:219.5 - perf/throughput:5300" >> $d/train.log
done
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S
  # nvidia-smi: the first query (the empty-GPU check) 0 MiB, then FAKE_MEM
  printf '#!/usr/bin/env bash\nf=%s/nvs; c=$(cat $f 2>/dev/null || echo 0); echo $((c + 1)) > $f; [ $c = 0 ] && echo 0 || echo ${FAKE_MEM:-180000}\n' $T > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PR_WAIT_STEP=0; : > $CALLS
}
r() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-ROLLOUT-$1/report.json "$2"; }
# 1. U60 then U85: the variant's setting over control.env, the step means after the warm-up, U85 against U60
setup
bash $T/rb/scripts/probe_rollout.sh 3 U60 > /dev/null 2>&1
ok "U60 PASS: run_e12 got VLLM_UTIL 0.6, a 3-step smoke of E12b-P5-8k" '[ "$(r U60 "v[\"result\"]")" = PASS ] && grep -q "^run_e12 3 E12b-P5-8k SMOKE=1 STEPS=3 PROBE_TASK=PROBE-ROLLOUT-U60-smoke VLLM_UTIL=0.6$" $CALLS'
ok "per-step rows and the mean after step 1 (warm-up left out)" '[ "$(r U60 "len(v[\"by_step\"])")" = 3 ] && [ "$(r U60 "v[\"mean_after_step1\"][\"timing_s/gen\"]")" = 500.0 ] && [ "$(r U60 "v[\"by_step\"][\"1\"][\"timing_s/gen\"]")" = 560.0 ]'
ok "report private, event, .done; no comparison for U60 itself" 'grep -q "^remote put $T/root/runs/PROBE-ROLLOUT-U60/up logs/abcd1234/probe_rollout/PROBE-ROLLOUT-U60" $CALLS && grep -q "^control event PROBE-ROLLOUT-U60 probe_rollout_pass" $CALLS && [ -f $T/root/runs/PROBE-ROLLOUT-U60/.done ] && [ "$(r U60 "\"vs_U60\" in v")" = False ]'
rm -f $T/nvs; bash $T/rb/scripts/probe_rollout.sh 5 U85 > /dev/null 2>&1
ok "U85: VLLM_UTIL 0.85 reaches run_e12 although control.env says 0.6" 'grep -q "PROBE_TASK=PROBE-ROLLOUT-U85-smoke VLLM_UTIL=0.85$" $CALLS && [ "$(r U85 "v[\"result\"]")" = PASS ]'
ok "U85 against U60: faster (step 0.78x, gen 0.71x), comparable lengths and scores" '[ "$(r U85 "\" \".join(str(x) for x in (v[\"vs_U60\"][\"step_ratio\"], v[\"vs_U60\"][\"gen_ratio\"], v[\"vs_U60\"][\"faster\"], v[\"vs_U60\"][\"comparable\"]))")" = "0.785 0.706 True True" ]'
rm -f $T/nvs; FAKE_LEN_K=1.3 bash $T/rb/scripts/probe_rollout.sh 6 U75 > /dev/null 2>&1
ok "U75 with 30% longer responses: not comparable" '[ "$(r U75 "\" \".join(str(x) for x in (v[\"vs_U60\"][\"length_rel_diff\"], v[\"vs_U60\"][\"comparable\"]))")" = "0.3 False" ]'
cd $SRC; rm -rf $T
# 2. FAIL: out of memory; a config without the variant's setting; a missing step; a peak over 260 GB; an unknown variant
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_rollout.sh 1 U85 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(r U85 "v[\"result\"]")" = FAIL ]'
cd $SRC; rm -rf $T
setup; FAKE_CFG_UTIL=0.6 bash $T/rb/scripts/probe_rollout.sh 1 U85 > /dev/null 2>&1
ok "a smoke whose config does not show the variant's setting is FAIL" '[[ "$(r U85 "v[\"reason\"]")" == *"does not show gpu_memory_utilization 0.85"* ]]'
cd $SRC; rm -rf $T
setup; FAKE_LOGSTEPS=2 bash $T/rb/scripts/probe_rollout.sh 1 U75 > /dev/null 2>&1
ok "a missing step is FAIL" '[[ "$(r U75 "v[\"reason\"]")" == *"2 steps with timing, expected 3"* ]]'
cd $SRC; rm -rf $T
setup; FAKE_MEM=270000 bash $T/rb/scripts/probe_rollout.sh 1 U85 > /dev/null 2>&1
ok "a peak over 260 GB is FAIL" '[[ "$(r U85 "v[\"reason\"]")" == *"peak 270000 MiB"* ]]'
bash $T/rb/scripts/probe_rollout.sh 1 U95 > /dev/null 2>&1; rc=$?
ok "an unknown variant exits 2" '[ $rc = 2 ]'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_ROLLOUT_TEST PASS ($n checks)" || echo "PROBE_ROLLOUT_TEST FAIL ($fails of $n)"
