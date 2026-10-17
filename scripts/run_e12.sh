#!/usr/bin/env bash
# E12b: short GRPO from one 1.7B preparation checkpoint, on ONE GPU. Same recipe as the lab's 0.6B E12: plain GRPO
# (no ref model, no KL), 64 prompts x 8 responses, 4K, T=1.0, lr 1e-6, clip 0.2/0.28, mini-batch 16, token-level IS
# correction (threshold 5), data.seed 4242; validation every 25 steps on MATH500 first 100 (avg@4, T=0.6, 4K);
# checkpoints every 20 steps with the resume state (resume_mode=auto: re-running the same command continues).
# Tier A (P0, P1, P2, P3, P5) runs 100 steps, the other starts 60. Starts: P0 = Qwen3-1.7B, P1 = SFT-P1's last
# checkpoint, others = E10b-<arm> step 150. Rollouts of every step are dumped for the group statistics.
# Status: syntax-checked; the lab runs the same configuration for 0.6B on H100/H200; not yet run on Blackwell.
# Usage: bash run_e12.sh <gpu> <task id>      e.g. E12b-P2
set -euo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=$2; A=${T#E12b-}; OUT=$OPD_ROOT/runs/$T; mkdir -p $OUT/rollouts
[ -f $OUT/.done ] && exit 0
case $A in
  P0) START=$STUDENT_17B;;
  P1) START=$OPD_ROOT/runs/SFT-P1/ckpt/global_step_$(cat $OPD_ROOT/runs/SFT-P1/final_step)/huggingface;;
  *)  START=$OPD_ROOT/runs/E10b-$A/ckpt/global_step_150/actor/huggingface_bf16   # v2.4: the fp32 copy is not kept;
       has_weights $START || START=${START%_bf16};;                                 # bf16 start (deviation: weight rounding)
esac
case $A in P0|P1|P2|P3|P5) STEPS=100;; *) STEPS=60;; esac
[ $A = P0 ] || has_weights $START || { echo "missing start checkpoint $START"; exit 1; }
source $OPD_ROOT/venv/bin/activate
export CUDA_VISIBLE_DEVICES=$G RAY_TMPDIR=/tmp/ray_gpu$G PYTHONPATH=$RUNBOOK/code/patches:${PYTHONPATH:-}
unset OPD_VARIANT
if python -c "import flash_attn" 2>/dev/null; then RMPAD=True; ATTN=flash_attention_2; TOK=16384; else RMPAD=False; ATTN=sdpa; TOK=6144; fi
# v2.5.4a: as run_arm.sh, FIXED_MICRO_BSZ=<n> (setenv, or set by task.sh after an out-of-memory failure) uses n sequences per
# micro-batch in the update and the log-probs instead of the token budget (without flash-attn every sequence is padded to
# the full length, so only a fixed count bounds the logits); without flash-attn the entropy is computed in chunks
if [ -n "${FIXED_MICRO_BSZ:-}" ]; then BATCH="actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$FIXED_MICRO_BSZ"; LOGP_MB=$FIXED_MICRO_BSZ
else BATCH="actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOK_GPU:-$TOK}"; LOGP_MB=8; fi
ENT=""; [ "$RMPAD" = False ] && ENT="actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
[ -f $OUT/config.yaml ] && echo "resumed: $(date -u +%FT%TZ)" >> $OUT/config.yaml || cat > $OUT/config.yaml <<CFG
task: $T
start: $START
algo: GRPO 64x8, 4K, T=1.0, lr 1e-6, clip 0.2/0.28, KL 0, mini-batch 16, data.seed 4242, $STEPS steps
attention: $ATTN (use_remove_padding=$RMPAD)
started: $(date -u +%FT%TZ)
CFG
# v2.4 save rule: dedicated node = HF weights at the last step only (queue v2 has no intermediate E12b reading; set
# SAVE=<n> when one is registered), uploaded as bf16 and not kept locally (no local E12b evaluation); SPOT=1 also saves
# the resume state every 20 steps and keeps only the newest. Checkpoints on $CKPT_ROOT (v2.4).
if [ "${SPOT:-0}" = 1 ]; then CONTENTS="['model','optimizer','extra','hf_model']"; SAVE=${SAVE:-20}; else CONTENTS="['hf_model']"; SAVE=${SAVE:-$STEPS}; fi
ckpt_link $OUT
pgrep -f "upload_syn[c].sh $OUT\$" >/dev/null || KEEP_BF16=0 UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-$STEPS} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OUT > $OUT/upload.log 2>&1 < /dev/null &
cd $OPD_ROOT/G-OPD/verl
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=5.0 \
  algorithm.rollout_correction.rollout_rs=null algorithm.rollout_correction.bypass_mode=false \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  data.train_files=$OPD_ROOT/data/dapo_train.parquet data.val_files=$OPD_ROOT/data/math500_100.parquet \
  data.train_batch_size=64 data.max_prompt_length=1024 data.max_response_length=4096 \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True data.seed=4242 data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=$START +actor_rollout_ref.model.override_config.attn_implementation=$ATTN \
  actor_rollout_ref.model.use_remove_padding=$RMPAD actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 $BATCH $ENT \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents=$CONTENTS \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGP_MB actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.6} \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 actor_rollout_ref.rollout.val_kwargs.n=4 \
  custom_reward_function.path=$RUNBOOK/code/train/reward_mathverify.py custom_reward_function.name=compute_score \
  reward_model.reward_manager=naive trainer.critic_warmup=0 trainer.val_before_train=True \
  trainer.logger='["console"]' trainer.project_name=opd-e10b trainer.experiment_name=$T \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq=$SAVE trainer.test_freq=25 trainer.resume_mode=auto \
  trainer.rollout_data_dir=$OUT/rollouts trainer.total_training_steps=$STEPS trainer.total_epochs=1 \
  trainer.default_local_dir=$OUT/ckpt 2>&1 | tee -a $OUT/train.log
LAST=$(grep -oE "(^|[ )])step:[0-9]+ - " $OUT/train.log | tail -1 | grep -oE "[0-9]+")
echo "completed_steps: ${LAST:-unknown}" >> $OUT/config.yaml
if [ "${LAST:-0}" -ge $STEPS ]; then echo "finished: $(date -u +%FT%TZ)" >> $OUT/config.yaml; touch $OUT/.done
  # v2.4: the E10b start (bf16, 3.4 GB) is freed once both readers of it (E11b and this E12b) are done
  case $A in P0|P1) ;; *) [ -f $OPD_ROOT/runs/E11b-$A/.done ] && rm -rf $OPD_ROOT/runs/E10b-$A/ckpt/global_step_150/actor/huggingface_bf16;; esac
else echo "stopped: $(date -u +%FT%TZ) before the last step (v2.3: no finished: line, so upload_sync keeps running for the retry)" >> $OUT/config.yaml; fi
cd $RUNBOOK && python code/analysis/e12_rollout_stats.py $OUT --tokenizer $STUDENT_17B > /dev/null 2>&1 || true
