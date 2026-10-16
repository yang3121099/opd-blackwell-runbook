#!/usr/bin/env bash
# SFT-P1 (1.7B): the E12b start P1 = rejection-sampling fine-tuning of Qwen3-1.7B on the teacher's correct, terminated
# traces. Teacher traces: Qwen3-8B, non-thinking, T=0.7 / top-p 0.8 / top-k 20, 4K, one per DAPO training prompt.
#   - if $OPD_ROOT/data/p1_traces/shard*.jsonl* exist (e.g. copied from the project's archive), they are used;
#   - otherwise they are generated here (code/train/gen_teacher_traces.py, one GPU, about 16.7K prompts).
# Token budget 49.2M (same as the lab's 0.6B P1), 128 sequences per step, lr 1e-5 (verl SFT defaults: cosine, 10%
# warmup), max length 6144, an HF checkpoint at every epoch boundary and at the last step (the E12b start).
# Without flash-attn the SFT trainer uses SDPA (OPD_ATTN, see code/patches) and no remove-padding.
# Status: syntax-checked only (the lab ran the same recipe for 0.6B on H100); not yet run on Blackwell.
# Usage: bash run_sft.sh <gpu> SFT-P1
set -euo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=$2; OUT=$OPD_ROOT/runs/$T; mkdir -p $OUT $OPD_ROOT/data/p1_traces
[ -f $OUT/.done ] && exit 0
source $OPD_ROOT/venv/bin/activate; export CUDA_VISIBLE_DEVICES=$G; cd $RUNBOOK
ls $OPD_ROOT/data/p1_traces/shard*.jsonl* >/dev/null 2>&1 || \
  python code/train/gen_teacher_traces.py --teacher $TEACHER --shard 0 --nshards 1 --out $OPD_ROOT/data/p1_traces > $OUT/traces.log 2>&1
python code/train/build_sft_data.py --mode p1 --out $OPD_ROOT/data/sft_p1_17b.parquet $OPD_ROOT/data/p1_traces/shard*.jsonl* | tee $OUT/sft_data.json
read STEPS SPE EPOCHS <<< "$(python -c "import json;d=json.load(open('$OUT/sft_data.json'));print(d['steps_for_budget'],d['steps_per_epoch'],d['total_epochs_arg'])")"
if python -c "import flash_attn" 2>/dev/null; then RMPAD=True; export OPD_ATTN=flash_attention_2; else RMPAD=False; export OPD_ATTN=sdpa; fi
cat > $OUT/config.yaml <<CFG
task: $T
student: $STUDENT_17B (non-thinking rendering)
lr: 1e-5
steps: $STEPS (steps_per_epoch $SPE, token budget 49.2M)
attention: $OPD_ATTN (use_remove_padding=$RMPAD)
started: $(date -u +%FT%TZ)
CFG
echo $STEPS > $OUT/final_step
cd $OPD_ROOT/G-OPD/verl
torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.fsdp_sft_trainer \
  data.train_files=$OPD_ROOT/data/sft_p1_17b.parquet data.val_files=$OPD_ROOT/data/sft_p1_17b_val64.parquet \
  data.prompt_key=prompt data.response_key=response data.max_length=6144 data.truncation=right \
  data.train_batch_size=128 data.micro_batch_size_per_gpu=${MICRO:-2} \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  model.partial_pretrain=$STUDENT_17B model.enable_gradient_checkpointing=True use_remove_padding=$RMPAD \
  optim.lr=1e-5 trainer.default_local_dir=$OUT/ckpt trainer.project_name=opd-e10b trainer.experiment_name=$T \
  trainer.total_epochs=$EPOCHS trainer.total_training_steps=$STEPS trainer.save_freq=$SPE trainer.test_freq=-1 \
  trainer.logger='["console"]' trainer.n_gpus_per_node=1 trainer.resume_mode=disable \
  trainer.checkpoint.save_contents='["model","hf_model"]' 2>&1 | tee $OUT/train.log
has_weights $OUT/ckpt/global_step_$STEPS/huggingface || { echo "no final checkpoint"; exit 1; }
echo "finished: $(date -u +%FT%TZ)" >> $OUT/config.yaml
cd $RUNBOOK; rm -rf $OUT/final_bf16; python scripts/to_bf16.py $OUT/ckpt/global_step_$STEPS/huggingface $OUT/final_bf16
python scripts/remote.py put $OUT/final_bf16 runs/$T/global_step_$STEPS/huggingface
for f in config.yaml train.log sft_data.json; do python scripts/remote.py put $OUT/$f runs/$T/$f; done
touch $OUT/.done; python scripts/remote.py put $OUT/.done runs/$T/.done && echo "UP_OK $T"
