#!/usr/bin/env bash
# Upload a task's checkpoints while it trains: each checkpoint's HF weights as a bf16 copy to
# runs/<task>/global_step_<N>/actor/huggingface/, plus train.log / config.yaml after each one; at the end the logs, a
# per-step metrics.jsonl and the .done marker. Backend: UPLOAD_BACKEND=hf (HF_TOKEN, HF_REPO) or modelscope
# (MS_TOKEN, MS_REPO). Credentials are read from the environment only; never printed or written to disk.
# Weights are model.safetensors or shards (model-0000k-of-0000n.safetensors + model.safetensors.index.json: every
# 1.7B checkpoint is sharded). It stops only after 'finished:' AND every checkpoint is up (the last one is usually
# younger than 120 s when the run finishes). v2.1 fixed both: before, shards were skipped and the last checkpoint lost.
# KEEP_LOCAL_HF=0 deletes the local fp32 HF weights after their bf16 copy is uploaded.
# UPLOAD_DRY=1 (verify.sh): no credentials, no upload; leaves the bf16 copy in <ckpt>/.bf16 and prints UP_DRY.
# Usage: bash upload_sync.sh <task dir>
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
RUN=$1; TASK=$(basename $RUN); DRY=${UPLOAD_DRY:-0}
up() { [ "$DRY" = 1 ] && return 0; python $RUNBOOK/scripts/remote.py put "$1" "$2"; }
[ "$DRY" = 1 ] || python $RUNBOOK/scripts/remote.py check || { echo "UPLOAD_STOP: backend not usable (token or repo missing?)"; exit 1; }
while true; do
  latest=$(cat $RUN/ckpt/latest_checkpointed_iteration.txt 2>/dev/null || echo 0); new=0; pend=0
  for d in $RUN/ckpt/global_step_*; do
    h=$d/actor/huggingface; ls $h/*.safetensors >/dev/null 2>&1 || continue
    s=${d##*_}; [ "$s" -le "$latest" ] || continue
    [ -f $d/.uploaded ] && continue
    pend=1
    [ $(( $(date +%s) - $(stat -c %Y $(ls -t $h/*.safetensors | head -1)) )) -gt 120 ] || continue
    rm -rf $d/.bf16; python $RUNBOOK/scripts/to_bf16.py $h $d/.bf16 || { echo "BF16_FAIL $TASK $s"; continue; }
    if [ "$DRY" = 1 ]; then
      touch $d/.uploaded; new=1; echo "UP_DRY $TASK step $s: bf16 copy in $d/.bf16 ($(ls $d/.bf16/*.safetensors | wc -l) weight file(s))"
    elif up $d/.bf16 runs/$TASK/global_step_$s/actor/huggingface; then
      touch $d/.uploaded; rm -rf $d/.bf16; new=1; echo "UP_OK $TASK step $s $(date -u +%T)"
      [ "${KEEP_LOCAL_HF:-1}" = 0 ] && rm -f $h/*.safetensors
    else echo "UP_FAIL $TASK step $s (retried next cycle)"; fi
  done
  [ $new = 1 ] && { up $RUN/train.log runs/$TASK/train.log; up $RUN/config.yaml runs/$TASK/config.yaml; }
  if grep -q '^finished:' $RUN/config.yaml 2>/dev/null && [ $pend = 0 ]; then
    [ -f $RUN/train.log ] && python $RUNBOOK/scripts/metrics_jsonl.py $RUN/train.log > $RUN/metrics.jsonl
    for f in train.log config.yaml metrics.jsonl rollout_stats.csv; do [ -f $RUN/$f ] && up $RUN/$f runs/$TASK/$f; done
    [ -d $RUN/rollouts ] && up $RUN/rollouts runs/$TASK/rollouts
    [ -f $RUN/.done ] && up $RUN/.done runs/$TASK/.done
    echo "UPLOAD_DONE $TASK $(date -u +%T)"; break
  fi
  sleep $([ "$DRY" = 1 ] && echo 5 || echo 120)
done
