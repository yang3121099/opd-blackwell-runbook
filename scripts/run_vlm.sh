#!/usr/bin/env bash
# E15-VLM-P5 (runbook v2.5.9z; Q-051, the reviewer's pre-registration research_program §3c): P5's OPD recipe for a vision-language
# pair on ONE GPU, from venv-vl (vLLM 0.11.0, transformers 4.57): student Qwen3-VL-2B-Instruct (its own template, no think block),
# teacher Qwen3-VL-4B-Thinking ([v2.5.9z28]: was 8B-Thinking, whose vision head_dim 72 this flash-attn build refuses; a recorded
# deviation of Q-051) scoring on its thinking template. The teacher reads the student's own image-expanded prompt ids +
# '<think>\n' + the response (code/patches/opd_vlmref.py through vlmref_patch.py, OPD_VLM_REF=1): G-OPD's text re-tokenization would
# drop the image. Data: Geometry3K train (data/geo3k_train.parquet, made by code/train/prep_geo3k.py from the node's copy when
# missing), validation on its validation split every 25 steps; 128 prompts x 1 rollout a step, 4096-token responses, prompts up to
# 2048 tokens (the image's tokens included), the maths reward (the last \boxed{} under Math-Verify). 50 steps, s25 / s50 uploaded.
# Starts only after PROBE-VLM passed (the smoke itself is exempt). SMOKE=1 SMOKE_STEPS=n PROBE_TASK=<name>: n steps in runs/<name>,
# no checkpoint, upload or validation. Usage: bash run_vlm.sh <gpu> E15-VLM-P5
set -euo pipefail
source "$(dirname "$0")/env.sh"; G=$1; TASK=$2
[ "$TASK" = E15-VLM-P5 ] || { echo "unknown task $TASK"; exit 2; }
if [ "${SMOKE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
  grep -q '"result": "PASS"' $OPD_ROOT/runs/PROBE-VLM/report.json 2> /dev/null || { echo "PROBE-VLM has not passed: $TASK not started"; exit 3; }
fi
NAME=$TASK; [ "${SMOKE:-0}" = 1 ] && NAME=${PROBE_TASK:-smoke-$TASK}
OUT=$OPD_ROOT/runs/$NAME; mkdir -p $OUT
[ -f $OUT/.done ] && { echo "TASK_ALREADY_DONE $NAME"; exit 0; }
STUDENT=${VL_STUDENT:-$OPD_ROOT/models/Qwen3-VL-2B-Instruct}; TEACHER=${VL_TEACHER:-$OPD_ROOT/models/Qwen3-VL-4B-Thinking}   # [v2.5.9z28]
V=${VENV_VL:-$OPD_ROOT/venv-vl}; TRAIN=$OPD_ROOT/data/geo3k_train.parquet; VAL=$OPD_ROOT/data/geo3k_validation.parquet
NSTEPS=${VLM_STEPS:-50}; TBS=${VLM_TBS:-128}; MAXP=2048; MAXR=4096; SAVE=25; UW=25,$NSTEPS; VALB=True; TESTF=25
[ "${SMOKE:-0}" = 1 ] && { NSTEPS=${SMOKE_STEPS:-1}; SAVE=-1; VALB=False; TESTF=-1; }
# [v2.5.9z1] the teacher's suffix PROBE-VLM chose (check_b.json): think = '<think>\n' (P5), empty = '<think>\n\n</think>\n\n' (variant P5e,
# the thinking-only teacher's non-thinking reading); '<think>\n' when the check is not there yet (the probe's own smoke before b)
SUF=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('suffix') or 'think')" $OPD_ROOT/runs/PROBE-VLM/check_b.json 2> /dev/null || echo think)
case $SUF in empty) SUFTXT=$'<think>\n\n</think>\n\n'; VARIANT=P5e;; *) SUFTXT=$'<think>\n'; VARIANT=P5;; esac
export OPD_VLM_REF=1 OPD_VLM_REF_SUFFIX=${OPD_VLM_REF_SUFFIX:-$SUFTXT} OPD_VARIANT=P5 OPD_REF_THINK=0 OPD_SEED=${OPD_SEED:-1234}
[ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN $NAME student=$(basename $STUDENT) teacher=$(basename $TEACHER) venv=$(basename $V) train=$(basename $TRAIN) steps=$NSTEPS tbs=$TBS maxp=$MAXP maxr=$MAXR save=$SAVE upload=$UW vlmref=$OPD_VLM_REF variant=$VARIANT suffix=$(printf %q "$OPD_VLM_REF_SUFFIX")"; exit 0; }   # tests
for f in $STUDENT $TEACHER; do [ -f $f/.fetch_ok ] || { echo "MODEL_MISSING $(basename $f)"; exit 2; }; done
[ -x $V/bin/python ] || { echo "no venv-vl"; exit 2; }
source $V/bin/activate
if [ ! -f $TRAIN ]; then python $RUNBOOK/code/train/prep_geo3k.py $OPD_ROOT/data/hf/hiyouga__geometry3k $OPD_ROOT/data || { echo "PREP_GEO3K_FAILED"; exit 2; }; fi
[ -f $VAL ] || VAL=$OPD_ROOT/data/geo3k_test.parquet
python $RUNBOOK/code/patches/vlmref_patch.py || { echo "VLMREF_PATCH_FAILED $TASK"; exit 1; }   # idempotent
if python -c "import flash_attn" 2>/dev/null; then RMPAD=True; ATTN=flash_attention_2; else RMPAD=False; ATTN=sdpa; fi
[ -f $OUT/config.yaml ] && echo "resumed: $(date -u +%FT%TZ)" >> $OUT/config.yaml || cat > $OUT/config.yaml <<CFG
task: $TASK
student: $STUDENT (Qwen3-VL-2B-Instruct, its own template)
teacher: $TEACHER (ref; Qwen3-VL-4B-Thinking ([v2.5.9z28], was 8B) on its thinking template: the student's image-expanded prompt ids + the prefix '<think>\n' + the response, opd_vlmref.py)
variant: $VARIANT (teacher suffix $(printf %q "$OPD_VLM_REF_SUFFIX"), from PROBE-VLM's check)
algo: OPD (reverse-KL advantages, P5's recipe), ${TBS}x1, prompts <= $MAXP tokens, responses $MAXR, lr 3e-6, $NSTEPS steps
data: Geometry3K train ($(basename $TRAIN)), validation $(basename $VAL)
venv: $(basename $V); attention: $ATTN (remove-padding $RMPAD)
started: $(date -u +%FT%TZ)
CFG
[ "${SMOKE:-0}" = 1 ] || { pgrep -f "upload_syn[c].sh $OUT\$" >/dev/null || UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-$UW} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OUT > $OUT/upload.log 2>&1 < /dev/null & }
export CUDA_VISIBLE_DEVICES=$G RAY_TMPDIR=/tmp/ray_gpu$G PYTHONPATH=$RUNBOOK/code/patches:${PYTHONPATH:-}
cd $OPD_ROOT/G-OPD/verl
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=5.0 \
  algorithm.rollout_correction.rollout_rs=null algorithm.rollout_correction.bypass_mode=false \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  data.train_files=$TRAIN data.val_files=$VAL data.image_key=images \
  data.train_batch_size=$TBS data.max_prompt_length=$MAXP data.max_response_length=$MAXR \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True data.seed=42 data.return_raw_chat=True \
  actor_rollout_ref.model.path=$STUDENT +actor_rollout_ref.ref.model.path=$TEACHER \
  +actor_rollout_ref.model.override_config.attn_implementation=$ATTN actor_rollout_ref.model.use_remove_padding=$RMPAD \
  actor_rollout_ref.actor.optim.lr=3e-6 actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$TBS actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${VLM_TOK:-16384} \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.4} \
  actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  custom_reward_function.path=$RUNBOOK/code/train/reward_mathverify.py custom_reward_function.name=compute_score \
  reward_model.reward_manager=naive trainer.critic_warmup=0 trainer.val_before_train=$VALB \
  trainer.logger='["console"]' trainer.project_name=opd-e15 trainer.experiment_name=$NAME \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq=$SAVE trainer.test_freq=$TESTF trainer.resume_mode=auto \
  trainer.total_training_steps=$NSTEPS trainer.total_epochs=1 trainer.default_local_dir=$OUT/ckpt 2>&1 | tee -a $OUT/train.log
LAST=$(grep -oE "(^|[ )])step:[0-9]+ - " $OUT/train.log | tail -1 | grep -oE "[0-9]+")
echo "completed_steps: ${LAST:-unknown}" >> $OUT/config.yaml
[ "${LAST:-0}" -ge $NSTEPS ] && { echo "finished: $(date -u +%FT%TZ)" >> $OUT/config.yaml; [ "${SMOKE:-0}" = 1 ] || touch $OUT/.done; }
exit 0
