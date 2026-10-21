#!/usr/bin/env bash
# Upload a task's checkpoints while it trains: each checkpoint's HF weights as a bf16 copy to
# runs/<task>/global_step_<N>/actor/huggingface/, plus train.log / config.yaml after each one; at the end the logs, a
# per-step metrics.jsonl and the .done marker. Backend: UPLOAD_BACKEND=hf (HF_TOKEN, HF_REPO) or modelscope
# (MS_TOKEN, MS_REPO). Credentials are read from the environment only; never printed or written to disk.
# Weights are model.safetensors or shards (model-0000k-of-0000n.safetensors + model.safetensors.index.json: every
# 1.7B checkpoint is sharded). It stops only after 'finished:' AND every checkpoint is up (the last one is usually
# younger than 120 s when the run finishes). v2.1 fixed both: before, shards were skipped and the last checkpoint lost.
# v2.4 (Runming's save rule): after the bf16 copy is uploaded the local fp32 HF weights are deleted (KEEP_LOCAL_HF=0,
# now the default) and the bf16 copy stays in <ckpt>/actor/huggingface_bf16 for an evaluation on this machine
# (KEEP_BF16=1; run_e12.sh passes 0). On every run (not only SPOT=1; [v2.5.8] comment corrected, checked 10-19) the
# resume files (*.pt) of every step older than the newest checkpoint are deleted, so one resume state stays. UPLOAD_WEIGHTS (v2.4): which steps' weights go to the result
# repository: a comma list (run_arm.sh / run_e12.sh pass the last step), or all; a step not in it gets no bf16 copy,
# its fp32 HF weights are deleted (unless KEEP_LOCAL_HF=1) and it is marked .not_uploaded.
# UPLOAD_DRY=1 (verify.sh): no credentials, no upload; leaves the bf16 copy in <ckpt>/.bf16 and prints UP_DRY.
# Usage: bash upload_sync.sh <task dir>
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
RUN=$1; TASK=$(basename $RUN); DRY=${UPLOAD_DRY:-0}
up() { [ "$DRY" = 1 ] && return 0; python $RUNBOOK/scripts/remote.py put "$1" "$2"; }
[ "$DRY" = 1 ] || python $RUNBOOK/scripts/remote.py check || { echo "UPLOAD_STOP: backend not usable (token or repo missing?)"; exit 1; }
echo "UPLOAD_SETTINGS $(date -u +%FT%TZ) $TASK UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-all} KEEP_LOCAL_HF=${KEEP_LOCAL_HF:-0} KEEP_BF16=${KEEP_BF16:-1} DRY=$DRY"   # [v2.5.8]
while true; do
  latest=$(cat $RUN/ckpt/latest_checkpointed_iteration.txt 2>/dev/null || echo 0); new=0; pend=0
  for d in $RUN/ckpt/global_step_*; do
    h=$d/actor/huggingface; ls $h/*.safetensors >/dev/null 2>&1 || continue
    s=${d##*_}; [ "$s" -le "$latest" ] || continue
    [ -f $d/.uploaded ] && continue
    pend=1
    [ $(( $(date +%s) - $(stat -c %Y $(ls -t $h/*.safetensors | head -1)) )) -gt 120 ] || continue
    case ",${UPLOAD_WEIGHTS:-all}," in *",all,"*|*",$s,"*) ;; *)
      [ "${KEEP_LOCAL_HF:-0}" = 0 ] && rm -f $h/*.safetensors; touch $d/.uploaded $d/.not_uploaded; echo "NOT_UPLOADED $TASK step $s (UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-all})"; continue;; esac
    rm -rf $d/.bf16; python $RUNBOOK/scripts/to_bf16.py $h $d/.bf16 || { echo "BF16_FAIL $TASK $s"; continue; }
    if [ "$DRY" = 1 ]; then
      touch $d/.uploaded; new=1; echo "UP_DRY $TASK step $s: bf16 copy in $d/.bf16 ($(ls $d/.bf16/*.safetensors | wc -l) weight file(s))"
    elif up $d/.bf16 runs/$TASK/global_step_$s/actor/huggingface; then
      touch $d/.uploaded; new=1; echo "UP_OK $TASK step $s $(date -u +%T)"
      # v2.5: optional bf16 archive under a candidate root (ARCHIVE_ROOT in paths.env; none by default)
      [ -n "${ARCHIVE_ROOT:-}" ] && mkdir -p $ARCHIVE_ROOT/opd-archive/$TASK && rm -rf $ARCHIVE_ROOT/opd-archive/$TASK/global_step_$s \
        && cp -r $d/.bf16 $ARCHIVE_ROOT/opd-archive/$TASK/global_step_$s && echo "ARCHIVED $TASK step $s in $ARCHIVE_ROOT/opd-archive"
      if [ "${KEEP_BF16:-1}" = 1 ]; then rm -rf $d/actor/huggingface_bf16; mv $d/.bf16 $d/actor/huggingface_bf16; else rm -rf $d/.bf16; fi
      [ "${KEEP_LOCAL_HF:-0}" = 0 ] && rm -f $h/*.safetensors
    else echo "UP_FAIL $TASK step $s (retried next cycle)"; fi
  done
  [ $new = 1 ] && { up $RUN/train.log runs/$TASK/train.log; up $RUN/config.yaml runs/$TASK/config.yaml; }
  for d in $RUN/ckpt/global_step_*; do   # one resume state only: older steps lose their *.pt once a newer one is complete
    [ -d $d ] && [ "${d##*_}" -lt "$latest" ] && ls $d/actor/*.pt >/dev/null 2>&1 && rm -f $d/actor/*.pt && echo "PRUNED resume state $TASK step ${d##*_}"
  done
  if grep -q '^finished:' $RUN/config.yaml 2>/dev/null && [ $pend = 0 ]; then
    [ -f $RUN/train.log ] && python $RUNBOOK/scripts/metrics_jsonl.py $RUN/train.log > $RUN/metrics.jsonl
    for f in train.log config.yaml metrics.jsonl rollout_stats.csv lensel.jsonl; do [ -f $RUN/$f ] && up $RUN/$f runs/$TASK/$f; done   # [v2.5.8m] lensel.jsonl: the length-selected arms' per-prompt record
    [ -d $RUN/rollouts ] && up $RUN/rollouts runs/$TASK/rollouts
    [ -f $RUN/.done ] && up $RUN/.done runs/$TASK/.done
    echo "UPLOAD_DONE $TASK $(date -u +%T)"
    # [v2.5.8] the final step and every uploaded step kept on the share (dr-keep, under DR_CAP_GB; DR=0: none)
    [ "$DRY" = 1 ] || python $RUNBOOK/scripts/control.py dr_keep_auto $TASK 2>&1 | tail -1
    break
  fi
  # [v2.5.8] a task that is paused or failed for good (3 attempts) and no longer running gets no retry: its loop ends
  # once nothing is pending (10-19: the loops of E10c-dyn16k (failed) and -dyn6144 (paused) ran on for a day); a
  # resumed or rerun task's run starts a new one
  why=""; [ -f $RUN/.paused ] && why=paused
  [ "$(sed -n 's/^attempts: //p' $RUN/.failed 2>/dev/null)" -ge 3 ] 2> /dev/null && why=failed
  if [ -n "$why" ] && [ $pend = 0 ] && ! pgrep -f "scripts/task\.sh ([0-9]+(,[0-9]+)*|cpu) $TASK( |\$)" > /dev/null; then
    for f in train.log config.yaml lensel.jsonl; do [ -f $RUN/$f ] && up $RUN/$f runs/$TASK/$f; done
    echo "UPLOAD_STOPPED $TASK $why $(date -u +%T)"; break
  fi
  sleep $([ "$DRY" = 1 ] && echo 5 || echo 120)
done
