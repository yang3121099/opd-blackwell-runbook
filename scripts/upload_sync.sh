#!/usr/bin/env bash
# Upload a task's checkpoints while it trains: each checkpoint's HF weights as a bf16 copy to
# runs/<task>/global_step_<N>/actor/huggingface/, plus train.log / config.yaml after each one; at the end the logs, a
# per-step metrics.jsonl and the .done marker. Backend: UPLOAD_BACKEND=hf (HF_TOKEN, HF_REPO) or modelscope
# (MS_TOKEN, MS_REPO). Credentials are read from the environment only; never printed or written to disk.
# KEEP_LOCAL_HF=0 deletes the local fp32 HF copy after its bf16 copy is uploaded.
# Usage: bash upload_sync.sh <task dir>
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
RUN=$1; TASK=$(basename $RUN)
up() { python $RUNBOOK/scripts/remote.py put "$1" "$2"; }
python $RUNBOOK/scripts/remote.py check || { echo "UPLOAD_STOP: backend not usable (token or repo missing?)"; exit 1; }
while true; do
  latest=$(cat $RUN/ckpt/latest_checkpointed_iteration.txt 2>/dev/null || echo 0); new=0
  for d in $RUN/ckpt/global_step_*; do
    h=$d/actor/huggingface; [ -f $h/model.safetensors ] || continue
    s=${d##*_}; [ "$s" -le "$latest" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y $h/model.safetensors) )) -gt 120 ] || continue
    [ -f $d/.uploaded ] && continue
    rm -rf $d/.bf16; python $RUNBOOK/scripts/to_bf16.py $h $d/.bf16 || { echo "BF16_FAIL $TASK $s"; continue; }
    if up $d/.bf16 runs/$TASK/global_step_$s/actor/huggingface; then
      touch $d/.uploaded; rm -rf $d/.bf16; new=1; echo "UP_OK $TASK step $s $(date -u +%T)"
      [ "${KEEP_LOCAL_HF:-1}" = 0 ] && rm -f $h/model.safetensors
    else echo "UP_FAIL $TASK step $s (retried next cycle)"; fi
  done
  [ $new = 1 ] && { up $RUN/train.log runs/$TASK/train.log; up $RUN/config.yaml runs/$TASK/config.yaml; }
  if grep -q '^finished:' $RUN/config.yaml 2>/dev/null; then
    python $RUNBOOK/scripts/metrics_jsonl.py $RUN/train.log > $RUN/metrics.jsonl
    for f in train.log config.yaml metrics.jsonl; do up $RUN/$f runs/$TASK/$f; done
    [ -f $RUN/.done ] && up $RUN/.done runs/$TASK/.done
    echo "UPLOAD_DONE $TASK $(date -u +%T)"; break
  fi
  sleep 120
done
