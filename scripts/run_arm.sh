#!/usr/bin/env bash
# One E10b arm on ONE GPU: student Qwen3-1.7B (non-thinking), teacher Qwen3-8B as verl "ref", 150 steps, 4K rollouts.
# Same recipe as the lab's 0.6B E10 arms (run_e10_arm.sh) except: the student, checkpoints every 25 steps with the
# v2.4 save rule below (dedicated node: HF weights at step 150 only, no resume state), and SDPA when flash-attn is absent.
# Idempotent: a finished task (runs/<task>/.done) is skipped. SMOKE=1 runs 3 steps without validation or checkpoints
# (the verification smoke sets STUDENT=$STUDENT_06B). Models: STUDENT_17B / TEACHER from env.sh (models.env).
# Usage: bash run_arm.sh <gpu> <task id>      task ids: E10b-{P2,P3,P4lowlr,P5,P6,P6r,P2noEOS,Pstop,P6rand,P6top5,P4EOSpen}
#        and the seed replicates E10b-P2-s43/-s44, E10b-P6r-s43/-s44
set -euo pipefail
source "$(dirname "$0")/env.sh"; GPU=$1; TASK=$2
# task id E10b-<arm>[-s<seed>]: the seed changes only the data order (data.seed); the default seed is 42
SEED=42; BASE=$TASK; [[ $TASK =~ -s(43|44)$ ]] && { SEED=${BASH_REMATCH[1]}; BASE=${TASK%-s*}; }
case $BASE in
  E10b-P2) ARM=P2; LR=3e-6;; E10b-P3) ARM=P3; LR=3e-6;; E10b-P4lowlr) ARM=P4; LR=6e-7;; E10b-P5) ARM=P5; LR=3e-6;;
  E10b-P6) ARM=P6; LR=3e-6;; E10b-P6r) ARM=P6r; LR=3e-6;; E10b-P2noEOS) ARM=P2noEOS; LR=3e-6;;
  E10b-Pstop) ARM=Pstop; LR=3e-6;; E10b-P6rand) ARM=P6rand; LR=3e-6;; E10b-P6top5) ARM=P6top5; LR=3e-6;;
  E10b-P4EOSpen) ARM=P4EOSpen; LR=3e-6;;
  *) echo "unknown task $TASK"; exit 2;;
esac
[ "${SMOKE:-0}" = 1 ] && TASK=smoke-$TASK
OUT=$OPD_ROOT/runs/$TASK; mkdir -p $OUT
[ -f $OUT/.done ] && { echo "TASK_ALREADY_DONE $TASK"; exit 0; }
STUDENT=${STUDENT:-$STUDENT_17B}   # the verification smoke uses $STUDENT_06B
source $OPD_ROOT/venv/bin/activate
export CUDA_VISIBLE_DEVICES=$GPU RAY_TMPDIR=/tmp/ray_gpu$GPU
export PYTHONPATH=$RUNBOOK/code/patches:${PYTHONPATH:-}
export OPD_VARIANT=$ARM OPD_SEED=${OPD_SEED:-1234}
export OPD_REF_THINK=$([ "$ARM" = P5 ] && echo 1 || echo 0)
if python -c "import flash_attn" 2>/dev/null; then RMPAD=True; ATTN=flash_attention_2; TOK_DEFAULT=16384
else RMPAD=False; ATTN=sdpa; TOK_DEFAULT=6144; fi
# without remove-padding every micro-batch is padded to its longest sequence, so the fp32 logits grow with the token
# budget: 16384 ran out of memory on a B200 (183 GB) in the actor update; 6144 keeps the logits under ~10 GB
STEPS=150; TEST=10; VALB=True
# save rule (v2.4, Runming 2026-10-17): no archive of resume states. A dedicated node saves only the HF weights at the
# pre-registered reading steps (E10b: step 150, the one checkpoint E11b and E12b read; SAVE=<n> for others), which
# upload_sync turns into bf16, uploads and keeps locally as bf16 only. SPOT=1 (a node that can vanish) also saves the
# resume state every 25 steps, and upload_sync keeps only the newest one.
if [ "${SPOT:-0}" = 1 ]; then SAVE=${SAVE:-25}; CONTENTS="['model','optimizer','extra','hf_model']"
else SAVE=${SAVE:-$STEPS}; CONTENTS="['hf_model']"; fi
[ "${SMOKE:-0}" = 1 ] && { STEPS=3; SAVE=-1; TEST=-1; VALB=False; }
[ -f $OUT/config.yaml ] && echo "resumed: $(date -u +%FT%TZ)" >> $OUT/config.yaml || cat > $OUT/config.yaml <<CFG
task: $TASK
arm: $ARM
lr: $LR
data_seed: $SEED
steps: $STEPS
student: $STUDENT (no_think)
teacher: $TEACHER (ref; think-template=$OPD_REF_THINK)
attention: $ATTN (use_remove_padding=$RMPAD)
gopd_commit: $GOPD_COMMIT
gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)
started: $(date -u +%FT%TZ)
CFG
[ "${SMOKE:-0}" = 1 ] || ckpt_link $OUT   # v2.4: checkpoints on $CKPT_ROOT (the large disk)
[ "${SMOKE:-0}" = 1 ] || pgrep -f "upload_syn[c].sh $OUT\$" >/dev/null || \
  UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-$STEPS} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OUT > $OUT/upload.log 2>&1 < /dev/null &
cd $OPD_ROOT/G-OPD/verl
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=5.0 \
  algorithm.rollout_correction.rollout_rs=null algorithm.rollout_correction.bypass_mode=false \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  data.train_files=$OPD_ROOT/data/dapo_train.parquet data.val_files=$OPD_ROOT/data/math500_100.parquet \
  data.train_batch_size=128 data.max_prompt_length=1024 data.max_response_length=4096 \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True data.seed=$SEED data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=$STUDENT +actor_rollout_ref.ref.model.path=$TEACHER \
  +actor_rollout_ref.model.override_config.attn_implementation=$ATTN \
  actor_rollout_ref.actor.optim.lr=$LR actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.model.use_remove_padding=$RMPAD \
  actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOK_GPU:-$TOK_DEFAULT} \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents=$CONTENTS \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.5} \
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.8 actor_rollout_ref.rollout.val_kwargs.n=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=$([ "${TEACHER_OFFLOAD:-1}" = 1 ] && echo True || echo False) \
  custom_reward_function.path=$RUNBOOK/code/train/reward_mathverify.py custom_reward_function.name=compute_score \
  reward_model.reward_manager=naive trainer.critic_warmup=0 trainer.val_before_train=$VALB \
  trainer.logger='["console"]' trainer.project_name=opd-e10b trainer.experiment_name=$TASK \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq=$SAVE trainer.test_freq=$TEST \
  trainer.resume_mode=auto trainer.total_training_steps=$STEPS trainer.total_epochs=2 \
  trainer.default_local_dir=$OUT/ckpt \
  2>&1 | tee -a $OUT/train.log
LAST=$(grep -oE "(^|[ )])step:[0-9]+ - " $OUT/train.log | tail -1 | grep -oE "[0-9]+")
echo "completed_steps: ${LAST:-unknown}" >> $OUT/config.yaml
if [ "${LAST:-0}" -ge $STEPS ]; then echo "finished: $(date -u +%FT%TZ)" >> $OUT/config.yaml; touch $OUT/.done
else echo "stopped: $(date -u +%FT%TZ) before the last step (v2.3: no finished: line, so upload_sync keeps running for the retry)" >> $OUT/config.yaml; fi
echo "TASK_FINISHED $TASK steps=${LAST:-unknown}"
