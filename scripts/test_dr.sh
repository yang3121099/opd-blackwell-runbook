#!/usr/bin/env bash
# Stub tests of v2.5.7 disaster recovery (no GPU, network or HF): dr_sync.sh mirrors the newest resume state (no HF
# weights), replaces the older one, respects the cap and clears the mirror when the task is done; dr_restore copies it
# back into an empty CKPT_ROOT and never over a local state; run_e12.sh saves every 25 steps with the resume state, and
# takes a start the container disk lost from the share (P1) or the result repository (E10b starts).
# Usage: bash scripts/test_dr.sh -> DR_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/base/root/runs $T/base/root/venv/bin $T/ck; : > $T/base/root/venv/bin/activate
  cp $SRC/scripts/dr.sh $SRC/scripts/dr_sync.sh $SRC/scripts/run_e12.sh $T/rb/scripts/
  cat > $T/rb/scripts/env.sh <<EOF
export OPD_ROOT=$T/base/root RUNBOOK=$T/rb CKPT_ROOT=$T/ck STUDENT_17B=$T/m17 STUDENT_4B=$T/m4 STUDENT_06B=Qwen/Qwen3-0.6B
has_weights() { ls "\$1"/*.safetensors > /dev/null 2>&1; }
ckpt_link() { local t=\$(basename "\$1"); mkdir -p "\$CKPT_ROOT/\$t/ckpt"; [ -e "\$1/ckpt" ] || ln -s "\$CKPT_ROOT/\$t/ckpt" "\$1/ckpt"; }
EOF
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["repo"] and print("me/pub")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/fetch_model.py
  export CALLS=$T/calls; : > $CALLS
  R=$T/base/root/runs/E10b-X; mkdir -p $R; (source $T/rb/scripts/env.sh; ckpt_link $R)
  DRD=$T/base/opd-dr/dr/E10b-X
}
step() {   # <n>: a complete resume state at step n (+ HF weights), latest = n
  mkdir -p $R/ckpt/global_step_$1/actor/huggingface; echo pt > $R/ckpt/global_step_$1/actor/model_world_size_1_rank_0.pt
  echo w > $R/ckpt/global_step_$1/actor/huggingface/model.safetensors; echo d > $R/ckpt/global_step_$1/data.pt; echo $1 > $R/ckpt/latest_checkpointed_iteration.txt; }
sync1() { DR_SYNC_SEC=0 DR_START_GRACE=0 bash $T/rb/scripts/dr_sync.sh $R > $T/sync.out 2>&1; }   # one round (no training running)
# 1. mirror, replace, one copy per task, no HF weights
setup
step 25; sync1
ok "step 25 mirrored without HF weights" '[ -f $DRD/global_step_25/actor/model_world_size_1_rank_0.pt ] && [ -f $DRD/global_step_25/data.pt ] && [ ! -e $DRD/global_step_25/actor/huggingface ] && [ "$(cat $DRD/latest)" = 25 ]'
ok "event dr_mirrored" 'grep -q "^control event E10b-X dr_mirrored step 25" $CALLS'
step 50; sync1
ok "step 50 replaces 25" '[ -d $DRD/global_step_50 ] && [ ! -e $DRD/global_step_25 ] && [ ! -e $DRD/global_step_50.tmp ] && [ "$(cat $DRD/latest)" = 50 ]'
ok "loop ends without a training, mirror kept" 'grep -q "DR_SYNC_EXIT E10b-X" $T/sync.out && [ -d $DRD/global_step_50 ]'
# 2. cap: nothing written over DR_CAP_GB
step 75; DR_CAP_GB=0 sync1
ok "over the cap: no mirror, event dr_cap" '[ "$(cat $DRD/latest)" = 50 ] && grep -q "^control event E10b-X dr_cap" $CALLS'
# 3. restore into an empty CKPT_ROOT; a local resume state wins
rm -rf $T/ck/E10b-X/ckpt/*; mkdir -p $R/ckpt
( source $T/rb/scripts/env.sh; source $T/rb/scripts/dr.sh; dr_restore $R ) > $T/r1 2>&1
ok "restored step 50" 'grep -q "DR_RESTORED E10b-X step 50" $T/r1 && [ -f $R/ckpt/global_step_50/actor/model_world_size_1_rank_0.pt ] && [ "$(cat $R/ckpt/latest_checkpointed_iteration.txt)" = 50 ]'
step 60; ( source $T/rb/scripts/env.sh; source $T/rb/scripts/dr.sh; dr_restore $R ) > $T/r2 2>&1
ok "a local state is never overwritten" '! grep -q DR_RESTORED $T/r2 && [ "$(cat $R/ckpt/latest_checkpointed_iteration.txt)" = 60 ]'
( source $T/rb/scripts/env.sh; source $T/rb/scripts/dr.sh; DR=0 dr_restore $R; DR=0 dr_on ) > /dev/null 2>&1; rc=$?
ok "DR=0 is off" '[ $rc != 0 ]'
# 4. done: the mirror is deleted
echo "finished: now" >> $R/config.yaml; sync1
ok "done task: mirror cleared" '[ ! -e $DRD ] && grep -q "^control event E10b-X dr_cleared" $CALLS'
ok "done task: local resume files deleted, weights kept" '! ls $R/ckpt/global_step_*/actor/*.pt > /dev/null 2>&1 && [ -f $R/ckpt/global_step_60/actor/huggingface/model.safetensors ]'
rm -rf $T
# 5. run_e12.sh: every 25 steps with the resume state (DR on), as before with DR=0; GC=0; starts from the share / result repo
setup
mkdir -p $T/m17 $T/base/root/venv-fa/bin; : > $T/base/root/venv-fa/bin/activate; d1=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1); d2=$(DR=0 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1)
d3=$(GC=0 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1); d5=$(GC=0 E12B_GC=1 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1)
ok "E12b DR: save 25 with resume state" '[[ "$d1" == *"steps=100 save=25 contents=['"'"'model'"'"','"'"'optimizer'"'"','"'"'extra'"'"','"'"'hf_model'"'"']"*"gc=True"* ]]'
ok "E12b DR=0: as before" '[[ "$d2" == *"save=100 contents=['"'"'hf_model'"'"']"* ]] && [[ "$d3" == *"gc=True"* ]] && [[ "$d5" == *"gc=False"* ]]'
mkdir -p $T/base/root/runs/SFT-P1 $T/base/opd-dr/keep/SFT-P1/ckpt/global_step_283/huggingface; echo 283 > $T/base/root/runs/SFT-P1/final_step
: > $T/base/opd-dr/keep/SFT-P1/ckpt/global_step_283/huggingface/model.safetensors
d4=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P1 2>&1 | tail -1)
d6=$(VENV=fa DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1); d7=$(VENV=fa E12B_FA=1 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1)
d8=$(SMOKE=1 PROBE_TASK=FA-SPEED-E12-fa_gc0 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1)   # v2.5.7c
mkdir -p $T/base/root/runs/E10b-P5-s44/ckpt/global_step_150/actor/huggingface_bf16; : > $T/base/root/runs/E10b-P5-s44/ckpt/global_step_150/actor/huggingface_bf16/model.safetensors
d9=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5-s44 2>&1 | tail -1)   # v2.5.7n
ok "v2.5.7n E12b-P5-s44: 100 steps from E10b-P5-s44; v2.5.7u: uploads s25 / s50 / s75 / s100, others the last step" '[[ "$d9" == "DRYRUN E12b-P5-s44 "*"E10b-P5-s44"*"steps=100 "*"upload=25,50,75,100 "* && "$d1" == *"upload=100 "* ]]'
# v2.5.7o: E12d from the 4B student / E10d-P5, E12b-P2-s43; both on venv-fa (without VENV / E12B_FA), E12d at vLLM 0.5
mkdir -p $T/base/root/runs/E10d-P5/ckpt/global_step_150/actor/huggingface_bf16 $T/base/root/runs/E10b-P2-s43/ckpt/global_step_150/actor/huggingface_bf16
: > $T/base/root/runs/E10d-P5/ckpt/global_step_150/actor/huggingface_bf16/model.safetensors; : > $T/base/root/runs/E10b-P2-s43/ckpt/global_step_150/actor/huggingface_bf16/model.safetensors
o1=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12d-P0 2>&1 | tail -1); o2=$(VLLM_UTIL=0.7 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12d-P5 2>&1 | tail -1)
o3=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P2-s43 2>&1 | tail -1); o4=$(E12D_VLLM_UTIL=0.45 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12d-P0 2>&1 | tail -1)
ok "v2.5.7o E12d-P0: base 4B, 100 steps, GC on, venv-fa, vLLM 0.5" '[[ "$o1" == "DRYRUN E12d-P0 start=$T/m4 steps=100 "*"gc=True venv=venv-fa "*"vllm=0.5 var=none"* ]]'
ok "v2.5.7o E12d-P5: from E10d-P5 s150, vLLM 0.5 whatever the node VLLM_UTIL; E12D_VLLM_UTIL sets it; v2.5.7u: mid-RL uploads (E12d-P0 last step only)" '[[ "$o2" == "DRYRUN E12d-P5 start=$T/base/root/runs/E10d-P5/ckpt/global_step_150/actor/huggingface_bf16 steps=100 "*"venv=venv-fa "*"upload=25,50,75,100 vllm=0.5 var=none"* && "$o4" == *"upload=100 vllm=0.45 var=none"* ]]'
ok "v2.5.7o E12b-P2-s43: from E10b-P2-s43, 100 steps, venv-fa, vLLM as E12b" '[[ "$o3" == "DRYRUN E12b-P2-s43 start=$T/base/root/runs/E10b-P2-s43/ckpt/global_step_150/actor/huggingface_bf16 steps=100 "*"venv=venv-fa "*"vllm=0.6 var=none"* ]] && [[ "$d9" == *"venv=venv "* ]]'
rm -rf $T/base/root/runs/E10d-P5; bash $T/rb/scripts/run_e12.sh 0 E12d-P5 > $T/e5 2>&1
ok "v2.5.7o E12d-P5 start fetched from the result repository (runs/E10d-P5)" 'grep -q "^fetch me/pub runs/E10d-P5/global_step_150/actor/huggingface $T/ck/adhoc_models/result --task E12d-P5" $CALLS'
ok "E12b: venv-fa only with E12B_FA=1" '[[ "$d6" == *"venv=venv "* && "$d7" == *"venv=venv-fa "* ]]'
ok "E12b smoke: own dir, 3 steps, no save" '[[ "$d8" == "DRYRUN FA-SPEED-E12-fa_gc0 "*"steps=3 save=-1 "* ]] && [ ! -e $T/ck/FA-SPEED-E12-fa_gc0 ]'
ok "P1 start from the share when the container disk lost it" '[[ "$d4" == *"start=$T/base/opd-dr/keep/SFT-P1/ckpt/global_step_283/huggingface "* ]]'
bash $T/rb/scripts/run_e12.sh 0 E12b-P4lowlr > $T/e4 2>&1
ok "E10b start fetched from the result repository" 'grep -q "^fetch me/pub runs/E10b-P4lowlr/global_step_150/actor/huggingface $T/ck/adhoc_models/result --task E12b-P4lowlr" $CALLS'
# [v2.5.8a] E12b-P5self from E10b-P5self step 50 (100 steps); E12c-P0 / -P5 (0.6B) gated by PROBE-E12C, P0 from the local
# 0.6B copy (models/, else the HF cache, else the hub id), P5 from E10c-P5-fix4 step 150 (or the result repository)
mkdir -p $T/base/root/runs/E10b-P5self/ckpt/global_step_50/actor/huggingface_bf16; : > $T/base/root/runs/E10b-P5self/ckpt/global_step_50/actor/huggingface_bf16/model.safetensors
a1=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5self 2>&1 | tail -1)
mkdir -p $T/base/root/runs/E10b-P5self-s43/ckpt/global_step_50/actor/huggingface_bf16; : > $T/base/root/runs/E10b-P5self-s43/ckpt/global_step_50/actor/huggingface_bf16/model.safetensors
a2=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5self-s43 2>&1 | tail -1)
ok "[v2.5.8e] E12b-P5self-s43: from E10b-P5self-s43 s50, 100 steps, as E12b-P5self" '[[ "$a2" == "DRYRUN E12b-P5self-s43 start=$T/base/root/runs/E10b-P5self-s43/ckpt/global_step_50/actor/huggingface_bf16 steps=100 "*"venv=venv "*"upload=100 vllm=0.6 var=none"* ]]'
ok "[v2.5.8a] E12b-P5self: from E10b-P5self s50, 100 steps, upload the last step, venv and vLLM as E12b-P5" '[[ "$a1" == "DRYRUN E12b-P5self start=$T/base/root/runs/E10b-P5self/ckpt/global_step_50/actor/huggingface_bf16 steps=100 "*"venv=venv "*"upload=100 vllm=0.6 var=none"* ]]'
c1=$(HF_HOME=$T/hf DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12c-P0 2>&1 | tail -1)
ok "[v2.5.8a] E12c-P0 refused before PROBE-E12C passed" '[[ "$c1" == "PROBE-E12C has not passed: E12c-P0 not started" ]] && [ ! -e $T/base/root/runs/E12c-P0/config.yaml ]'
c2=$(HF_HOME=$T/hf SMOKE=1 PROBE_TASK=PROBE-E12C-smoke DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12c-P0 2>&1 | tail -1)
ok "[v2.5.8a] the PROBE-E12C smoke is exempt: 3 steps, no save; the hub id when no local copy exists" '[[ "$c2" == "DRYRUN PROBE-E12C-smoke start=Qwen/Qwen3-0.6B steps=3 save=-1 "* ]]'
mkdir -p $T/base/root/runs/PROBE-E12C $T/base/root/models/Qwen3-0.6B; echo '{"result": "PASS"}' > $T/base/root/runs/PROBE-E12C/report.json; : > $T/base/root/models/Qwen3-0.6B/model.safetensors
c3=$(HF_HOME=$T/hf DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12c-P0 2>&1 | tail -1)
ok "[v2.5.8a] E12c-P0 after a PASS: the local 0.6B copy, 100 steps, the last step uploaded" '[[ "$c3" == "DRYRUN E12c-P0 start=$T/base/root/models/Qwen3-0.6B steps=100 "*"upload=100 vllm=0.6 var=none"* ]] && grep -q "^start: $T/base/root/models/Qwen3-0.6B$" $T/base/root/runs/E12c-P0/config.yaml'
mkdir -p $T/base/root/runs/E10c-P5-fix4/ckpt/global_step_150/actor/huggingface_bf16; : > $T/base/root/runs/E10c-P5-fix4/ckpt/global_step_150/actor/huggingface_bf16/model.safetensors
c4=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12c-P5 2>&1 | tail -1)
ok "[v2.5.8a] E12c-P5: from E10c-P5-fix4 s150, 100 steps" '[[ "$c4" == "DRYRUN E12c-P5 start=$T/base/root/runs/E10c-P5-fix4/ckpt/global_step_150/actor/huggingface_bf16 steps=100 "* ]]'
rm -rf $T/base/root/runs/E10c-P5-fix4; bash $T/rb/scripts/run_e12.sh 0 E12c-P5 > $T/c5 2>&1
ok "[v2.5.8a] E12c-P5 start fetched from the result repository (runs/E10c-P5-fix4)" 'grep -q "^fetch me/pub runs/E10c-P5-fix4/global_step_150/actor/huggingface $T/ck/adhoc_models/result --task E12c-P5" $CALLS'
# [v2.5.8b] E12b-P5-kl: refused before PROBE-E12B-KL passed; then E12b-P5's path with a KL loss to the start (reference = start)
S5=$T/base/root/runs/E10b-P5/ckpt/global_step_150/actor/huggingface_bf16; mkdir -p $S5; : > $S5/model.safetensors
k1=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5-kl 2>&1 | tail -1)
ok "[v2.5.8b] E12b-P5-kl refused before PROBE-E12B-KL passed" '[[ "$k1" == "PROBE-E12B-KL has not passed: E12b-P5-kl not started" ]]'
mkdir -p $T/base/root/runs/PROBE-E12B-KL; echo '{"result": "PASS"}' > $T/base/root/runs/PROBE-E12B-KL/report.json
k2=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5-kl 2>&1 | tail -1); k3=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5 2>&1 | tail -1)
ok "[v2.5.8b] E12b-P5-kl: start P5 s150, 100 steps, KL 0.01 low_var_kl, reference = the start, the plain reward" '[[ "$k2" == "DRYRUN E12b-P5-kl start=$S5 steps=100 "*"upload=100 "*"var=kl kl=[actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl +actor_rollout_ref.ref.model.path=$S5 "*"reward=reward_mathverify.py" ]]'
ok "[v2.5.8b] the config names the start as the reference; E12b-P5 unchanged (KL off, no variant)" 'grep -q "KL loss 0.01 low_var_kl to the start (reference = $S5)" $T/base/root/runs/E12b-P5-kl/config.yaml && grep -q "^variant: kl$" $T/base/root/runs/E12b-P5-kl/config.yaml && [[ "$k3" == "DRYRUN E12b-P5 start=$S5 "*"var=none kl=[actor_rollout_ref.actor.use_kl_loss=False] reward=reward_mathverify.py" ]]'
k4=$(SMOKE=1 PROBE_TASK=PROBE-E12B-KL-smoke DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5-kl 2>&1 | tail -1); rm -f $T/base/root/runs/PROBE-E12B-KL/report.json
k5=$(SMOKE=1 PROBE_TASK=PROBE-E12B-KL-smoke2 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5-kl 2>&1 | tail -1)
ok "[v2.5.8b] the probe's smoke runs without a PASS: 3 steps, no save, KL on" '[[ "$k5" == "DRYRUN PROBE-E12B-KL-smoke2 start=$S5 steps=3 save=-1 "*"var=kl kl=[actor_rollout_ref.actor.use_kl_loss=True"* ]]'
# [v2.5.8c] E12b-P5-stop: refused before PROBE-E12B-STOP passed; then E12b-P5's path with the stop reward (no KL)
t1=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5-stop 2>&1 | tail -1)
ok "[v2.5.8c] E12b-P5-stop refused before PROBE-E12B-STOP passed" '[[ "$t1" == "PROBE-E12B-STOP has not passed: E12b-P5-stop not started" ]]'
mkdir -p $T/base/root/runs/PROBE-E12B-STOP; echo '{"result": "PASS"}' > $T/base/root/runs/PROBE-E12B-STOP/report.json
t2=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5-stop 2>&1 | tail -1)
ok "[v2.5.8c] E12b-P5-stop: start P5 s150, 100 steps, KL off, the stop reward; config names it" '[[ "$t2" == "DRYRUN E12b-P5-stop start=$S5 steps=100 "*"upload=100 "*"var=stop kl=[actor_rollout_ref.actor.use_kl_loss=False] reward=reward_stop.py" ]] && grep -q "^reward: reward_stop.py " $T/base/root/runs/E12b-P5-stop/config.yaml && grep -q "^reward: reward_mathverify.py" $T/base/root/runs/E12b-P5-kl/config.yaml && grep -q "^venv: venv$" $T/base/root/runs/E12b-P5-stop/config.yaml'
rm -rf $T
[ $fails = 0 ] && echo "DR_TEST PASS ($n checks)" || echo "DR_TEST FAIL ($fails of $n)"
