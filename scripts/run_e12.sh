#!/usr/bin/env bash
# E12b: short GRPO from one 1.7B preparation checkpoint, on ONE GPU (v2.5.7o: E12d, the same from a 4B one). Same recipe as the lab's 0.6B E12: plain GRPO
# (no ref model, no KL), 64 prompts x 8 responses, 4K, T=1.0, lr 1e-6, clip 0.2/0.28, mini-batch 16, token-level IS
# correction (threshold 5), data.seed 4242; validation every 25 steps on MATH500 first 100 (avg@4, T=0.6, 4K);
# checkpoints every 20 steps with the resume state (resume_mode=auto: re-running the same command continues).
# Tier A (P0, P1, P2, P3, P5) runs 100 steps, the other starts 60. Starts: P0 = Qwen3-1.7B, P1 = SFT-P1's last
# checkpoint, others = E10b-<arm> step 150. Rollouts of every step are dumped for the group statistics.
# Status: syntax-checked; the lab runs the same configuration for 0.6B on H100/H200; not yet run on Blackwell.
# v2.5.7o (reviewer 06:0x UTC): E12d-P0 / E12d-P5 = the same GRPO from the 4B student (P0 = base Qwen3-4B, P5 = E10d-P5 step
# 150), 100 steps, flash-attn (a new ladder: the default for new arms), gradient checkpointing on, vLLM at 0.5 of the GPU;
# E12b-P2-s43 (P2's second seed, from E10b-P2-s43) on flash-attn too.
# [v2.5.8a] (Q-022 / Q-023, reviewer A1, Runming 10-20): E12b-P5self = the same GRPO from E10b-P5self step 50 (the
# self-distilled start: no outside teacher anywhere in the chain); E12c-P0 / E12c-P5 = the same GRPO for the 0.6B student
# (P0 = base Qwen3-0.6B from its local copy, P5 = E10c-P5-fix4 step 150), gated by PROBE-E12C (a 3-step smoke of E12c-P0)
# Usage: bash run_e12.sh <gpu> <task id>      e.g. E12b-P2, E12d-P5, E12c-P0
set -euo pipefail
# v2.5.7c: SMOKE=1 (FA-SPEED-E12) runs SMOKE_STEPS (3) steps in runs/$PROBE_TASK: no checkpoint, upload, DR, validation
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/dr.sh"; G=$1; T=$2; LADDER=${T%%-*}; A=${T#*-}
# [v2.5.8b] (Q-020, reviewer 22:0x UTC 10-19): E12b-P5-<variant> = E12b-P5 with one change: kl = a KL loss to the start (coef
# 0.01, low_var_kl, the reference policy = the start P5 s150), stop = the stop reward ([v2.5.8c]). Everything else is E12b-P5's.
# [v2.5.9f] (T1, reviewer 07:4x UTC 10-21): stop2 = the continuous stop reward (code/train/reward_stop2.py), 100 steps as stop
VAR=""; case $A in P5-kl|P5-stop|P5-stop2|P5-8k|P5-16k) VAR=${A#P5-}; A=P5;; esac
# [v2.5.8h] (Runming 10-20 17:5x UTC via the reviewer, Q-028: the 4K limit costs too much): E12b-P5-8k / -16k = E12b-P5 with
# max_response_length 8192 / 16384 and what the length forces: flash-attn with remove-padding (an SDPA micro-batch is
# padded to its longest sequence), a token budget of at least prompt + response, vLLM batching above the model length
MAXR=4096; case $VAR in 8k) MAXR=8192;; 16k) MAXR=16384;; esac
# [v2.5.8b] the variant starts only after its gate PROBE-E12B-<VAR> passed (a 3-step smoke that checks the reference; the smoke itself is exempt)
if [ -n "$VAR" ] && [ "${SMOKE:-0}" != 1 ]; then
  P=PROBE-E12B-$(echo $VAR | tr a-z A-Z); grep -q '"result": "PASS"' $OPD_ROOT/runs/$P/report.json 2> /dev/null || { echo "$P has not passed: $T not started"; exit 3; }
fi
SRC=E10b; [ $LADDER = E12d ] && SRC=E10d   # v2.5.7o: the preparation run a start comes from
SRUN=$SRC-$A; [ $LADDER = E12c ] && SRUN=E10c-$A-fix4   # [v2.5.8a] the 0.6B starts: E10c-<arm>-fix4, this node's 0.6B batching
SS=150; case $A in P5self|P5self-s43) SS=50;; esac       # [v2.5.8a] E10b-P5self ran 50 steps; [v2.5.8e] its seed 43 too
# [v2.5.8a] an E12c task starts only after PROBE-E12C passed (the smoke itself is exempt)
if [ $LADDER = E12c ] && [ "${SMOKE:-0}" != 1 ]; then
  grep -q '"result": "PASS"' $OPD_ROOT/runs/PROBE-E12C/report.json 2> /dev/null || { echo "PROBE-E12C has not passed: $T not started"; exit 3; }
fi
SM=${SMOKE:-0}; NAME=$T; [ $SM = 1 ] && NAME=${PROBE_TASK:-smoke-$T}
OUT=$OPD_ROOT/runs/$NAME; mkdir -p $OUT/rollouts
[ -f $OUT/.done ] && exit 0
case $A in
  P0) START=$STUDENT_17B; [ $LADDER = E12d ] && START=$STUDENT_4B; [ $LADDER = E12c ] && START=$STUDENT_06B;;   # [v2.5.8a] E12c
  P1) START=$OPD_ROOT/runs/SFT-P1/ckpt/global_step_$(cat $OPD_ROOT/runs/SFT-P1/final_step)/huggingface;;
  *)  START=$OPD_ROOT/runs/$SRUN/ckpt/global_step_$SS/actor/huggingface_bf16   # [v2.5.8a] $SRUN / $SS   # v2.4: the fp32 copy is not kept;
       has_weights $START || START=${START%_bf16};;                                 # bf16 start (deviation: weight rounding)
esac
# v2.5.7: a start on the container disk that a rebuild removed comes from the copy that survived: P1 from
# $(dr_root)/keep/SFT-P1 (dr-keep), an E10b start from this node's public result repository (its final bf16 upload)
if [ $A != P0 ] && ! has_weights $START; then
  case $A in
    P1) K=$(ls -d $(dr_root)/keep/SFT-P1/ckpt/global_step_*/huggingface 2> /dev/null | sort -V | tail -1); [ -n "$K" ] && START=$K;;
    *) R=runs/$SRUN/global_step_$SS/actor/huggingface   # [v2.5.8a]
       python $RUNBOOK/scripts/fetch_model.py "$(python $RUNBOOK/scripts/remote.py repo public)" $R $CKPT_ROOT/adhoc_models/result --task $T \
         && START=$CKPT_ROOT/adhoc_models/result/$R;;
  esac
fi
case $A in P0|P1|P2|P3|P5|P5-s43|P5-s44|P2-s43|P5self|P5self-s43) STEPS=100;; *) STEPS=60;; esac   # v2.5.6j: P5-s43 = the headline chain's second seed (queue v3 V3-c2); v2.5.7n: P5-s44 its third (from the flash-attn arm)
case $VAR in 8k|16k) STEPS=${E12B_LONG_STEPS:-50};; esac   # [v2.5.8h] reviewer 17:4x UTC: 50 steps (constant lr: extendable)
[ $A = P0 ] || has_weights $START || { echo "missing start checkpoint $START"; exit 1; }
# v2.5.7c: venv-fa (flash-attn, remove-padding) only with VENV=fa and E12B_FA=1: the E12b ladder stays on one attention
V=$OPD_ROOT/venv; [ "${VENV:-}" = fa ] && [ "${E12B_FA:-0}" = 1 ] && V=$OPD_ROOT/venv-fa
case $T in E12d-*|E12b-P2-s43|E12b-P5-8k|E12b-P5-16k) V=$OPD_ROOT/venv-fa;; esac   # v2.5.7o: new arms on flash-attn (reviewer 06:0x UTC); [v2.5.8h] the long limits
[ $LADDER = E12d ] && VLLM_UTIL=${E12D_VLLM_UTIL:-0.5}   # v2.5.7o: vLLM 0.5 for the 4B actor (a node-wide VLLM_UTIL does not reach it)
source $V/bin/activate
# [v2.5.8a] E12c-P0's start, the 0.6B base, from a local copy: $OPD_ROOT/models/Qwen3-0.6B, else the Hugging Face cache
# the E10c arms loaded it from, so the hub (blocked by the node's proxy since 10-19) is not asked
if [ $LADDER = E12c ] && [ $A = P0 ] && [ ! -d "$START" ]; then
  if has_weights $OPD_ROOT/models/Qwen3-0.6B; then START=$OPD_ROOT/models/Qwen3-0.6B
  else START=$(python -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], local_files_only=True))" $START 2> /dev/null) || START=$STUDENT_06B; fi
fi
NG=$(echo $G | tr , '\n' | grep -c .)   # [v2.5.9a] the GPUs of a #gpus task (a,b): FSDP over NG ranks, one vLLM engine each
export CUDA_VISIBLE_DEVICES=$G RAY_TMPDIR=/tmp/ray_gpu${G//,/_} PYTHONPATH=$RUNBOOK/code/patches:${PYTHONPATH:-}
unset OPD_VARIANT
if python -c "import flash_attn" 2>/dev/null; then RMPAD=True; ATTN=flash_attention_2; TOK=16384; else RMPAD=False; ATTN=sdpa; TOK=6144; fi
# [v2.5.8h] a long limit: the dynamic token budget holds one whole sequence (verl requires it), a fixed micro-batch is not used
[ $MAXR -gt 4096 ] && { TOK=$(( MAXR + 1024 > TOK ? MAXR + 1024 : TOK )); MAX_TOK_GPU=$TOK; FIXED_MICRO_BSZ=; }
VBT=$(( MAXR + 1024 > 16384 ? MAXR + 1024 : 16384 ))   # [v2.5.8h] vLLM max_num_batched_tokens >= the model length
# v2.5.4a: as run_arm.sh, FIXED_MICRO_BSZ=<n> (setenv, or set by task.sh after an out-of-memory failure) uses n sequences per
# micro-batch in the update and the log-probs instead of the token budget (without flash-attn every sequence is padded to
# the full length, so only a fixed count bounds the logits); without flash-attn the entropy is computed in chunks
if [ -n "${FIXED_MICRO_BSZ:-}" ]; then BATCH="actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$FIXED_MICRO_BSZ"; LOGP_MB=$FIXED_MICRO_BSZ
else BATCH="actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOK_GPU:-$TOK}"; LOGP_MB=8; fi
ENT=""; [ "$RMPAD" = False ] && ENT="actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
# v2.5.7: FUSED=1 / GC=0 (setenv) as in run_arm.sh (SPEED-PROBE 2026-10-18: GC=0 bit-exact, -9% step time on an E10b arm).
# E12b turns gradient checkpointing off only with E12B_GC=1 as well (setenv, after its own 3-step comparison; reviewer
# A1 07:0x UTC), so GC=0 for the E10b / E10c arms does not reach it; unset, E12b runs as before
KERN=""; [ "${FUSED:-0}" = 1 ] && KERN="actor_rollout_ref.model.use_fused_kernels=True actor_rollout_ref.model.fused_kernel_options.impl_backend=torch"
GCK=True; [ "${GC:-1}" = 0 ] && [ "${E12B_GC:-0}" = 1 ] && GCK=False
[ -f $OUT/config.yaml ] && echo "resumed: $(date -u +%FT%TZ)" >> $OUT/config.yaml || cat > $OUT/config.yaml <<CFG
task: $T
start: $START
algo: GRPO 64x8, $((MAXR / 1024))K, T=1.0, lr 1e-6, clip 0.2/0.28, $([ "$VAR" = kl ] && echo "KL loss 0.01 low_var_kl to the start (reference = $START)" || echo "KL 0"), mini-batch 16, data.seed 4242, $STEPS steps
variant: ${VAR:-none}
max_response_length: $MAXR
reward: $(case $VAR in (stop) echo "reward_stop.py (1 if correct, +0.5 if every box agrees and the response ends within 64 tokens after the first box)";; (stop2) echo "reward_stop2.py (correct x (1 + 0.5 x clip(1 - tail / 1024, 0, 1)), tail = tokens after the first box, a cut response counts 1024)";; (*) echo reward_mathverify.py;; esac)   # [v2.5.8c] [v2.5.9f]
attention: $ATTN (use_remove_padding=$RMPAD)
venv: $(basename $V)
batching: [$BATCH]
gpus: $G (n_gpus_per_node $NG)
kernels: fused=${FUSED:-0} gradient_checkpointing=$GCK
started: $(date -u +%FT%TZ)
CFG
# v2.4 save rule: dedicated node = HF weights at the last step only (queue v2 has no intermediate E12b reading; set
# SAVE=<n> when one is registered), uploaded as bf16 and not kept locally (no local E12b evaluation); SPOT=1 also saves
# the resume state every 20 steps and keeps only the newest. Checkpoints on $CKPT_ROOT (v2.4).
if [ "${SPOT:-0}" = 1 ]; then CONTENTS="['model','optimizer','extra','hf_model']"; SAVE=${SAVE:-20}
elif dr_on; then CONTENTS="['model','optimizer','extra','hf_model']"; SAVE=${SAVE:-${DR_EVERY:-25}}   # v2.5.7: resume state for dr_sync.sh
else CONTENTS="['hf_model']"; SAVE=${SAVE:-$STEPS}; fi
VALB=True; TESTF=25; [ $SM = 1 ] && { STEPS=${SMOKE_STEPS:-3}; SAVE=-1; VALB=False; TESTF=-1; }
# v2.5.7u (reviewer ruling 07:5x UTC, readings budA-p5s44rl): E12b-P5-s44's mid-RL steps 25 / 50 / 75 are registered
# readings (where does P5 -> RL lose the 32K ceiling), so they are uploaded as bf16 like the last step; until now
# upload_sync deleted every step not uploaded (E12b-P5-s43 lost s25 / s50 / s75 that way)
UW=$STEPS; case $T in E12b-P5-s44|E12d-P5) UW=25,50,75,$STEPS;; E12b-P5-8k|E12b-P5-16k) UW=25,$STEPS;; esac   # [v2.5.8h] the long limits: s25 too   # E12d-P5: the same across sizes (reviewer A1 08:0x UTC)
# [v2.5.8b] the variant's settings (KL loss and its reference; the reward function)
KL="actor_rollout_ref.actor.use_kl_loss=False"; REWARD=$RUNBOOK/code/train/reward_mathverify.py
# [v2.5.8c] stop: the stop reward, counting the student's tokens (its start's tokenizer) against the 4K response limit
[ "$VAR" = stop ] && { REWARD=$RUNBOOK/code/train/reward_stop.py; export OPD_REWARD_TOKENIZER=$START OPD_REWARD_MAXLEN=4096; }
[ "$VAR" = stop2 ] && { REWARD=$RUNBOOK/code/train/reward_stop2.py; export OPD_REWARD_TOKENIZER=$START OPD_REWARD_MAXLEN=4096; }   # [v2.5.9f]
[ "$VAR" = kl ] && KL="actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl +actor_rollout_ref.ref.model.path=$START actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGP_MB actor_rollout_ref.ref.fsdp_config.param_offload=False"
[ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN $NAME start=$START steps=$STEPS save=$SAVE contents=$CONTENTS kern=[$KERN] gc=$GCK venv=$(basename $V) ngpu=$NG rmpad=$RMPAD upload=${UPLOAD_WEIGHTS:-$UW} vllm=${VLLM_UTIL:-0.6} var=${VAR:-none} kl=[$KL] reward=$(basename $REWARD)"; exit 0; }   # tests
if [ $SM = 0 ]; then
  ckpt_link $OUT
  dr_restore $OUT || echo "DR_RESTORE_FAILED $T"; dr_start_sync $OUT   # v2.5.7
  pgrep -f "upload_syn[c].sh $OUT\$" >/dev/null || KEEP_BF16=0 UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-$UW} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OUT > $OUT/upload.log 2>&1 < /dev/null &
fi
cd $OPD_ROOT/G-OPD/verl
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=5.0 \
  algorithm.rollout_correction.rollout_rs=null algorithm.rollout_correction.bypass_mode=false \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  data.train_files=$OPD_ROOT/data/dapo_train.parquet data.val_files=$OPD_ROOT/data/math500_100.parquet \
  data.train_batch_size=64 data.max_prompt_length=1024 data.max_response_length=$MAXR \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True data.seed=4242 data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=$START +actor_rollout_ref.model.override_config.attn_implementation=$ATTN \
  actor_rollout_ref.model.use_remove_padding=$RMPAD actor_rollout_ref.model.enable_gradient_checkpointing=$GCK \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 $BATCH $ENT $KERN \
  $KL actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents=$CONTENTS \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGP_MB actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.6} \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.max_num_batched_tokens=$VBT \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 actor_rollout_ref.rollout.val_kwargs.n=4 \
  custom_reward_function.path=$REWARD custom_reward_function.name=compute_score \
  reward_model.reward_manager=naive trainer.critic_warmup=0 trainer.val_before_train=$VALB \
  trainer.logger='["console"]' trainer.project_name=opd-e10b trainer.experiment_name=$NAME \
  trainer.n_gpus_per_node=$NG trainer.nnodes=1 trainer.save_freq=$SAVE trainer.test_freq=$TESTF trainer.resume_mode=auto \
  trainer.rollout_data_dir=$OUT/rollouts trainer.total_training_steps=$STEPS trainer.total_epochs=1 \
  trainer.default_local_dir=$OUT/ckpt 2>&1 | tee -a $OUT/train.log
LAST=$(grep -oE "(^|[ )])step:[0-9]+ - " $OUT/train.log | tail -1 | grep -oE "[0-9]+")
echo "completed_steps: ${LAST:-unknown}" >> $OUT/config.yaml
if [ "${LAST:-0}" -ge $STEPS ]; then echo "finished: $(date -u +%FT%TZ)" >> $OUT/config.yaml; touch $OUT/.done
  # v2.5.7j: the E10b start (its final bf16) stays on this node after E12b (Runming's policy: final weights keep a local
  # copy after upload). Until v2.5.7i it was deleted here once E11b and E12b were done, so E10b-P2 / P3 / P4lowlr / P5 /
  # P6 / P6r / P2noEOS s150 had to be read back from the result repository (EVAL-budA17b, 2026-10-18 20:0x UTC).
else echo "stopped: $(date -u +%FT%TZ) before the last step (v2.3: no finished: line, so upload_sync keeps running for the retry)" >> $OUT/config.yaml; fi
[ $SM = 1 ] || { cd $RUNBOOK && python code/analysis/e12_rollout_stats.py $OUT --tokenizer $STUDENT_17B > /dev/null 2>&1 || true; }
