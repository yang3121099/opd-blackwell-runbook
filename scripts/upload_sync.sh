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
# [v2.5.9z52] (ops-lead / Fable 10-22: the HF account answered 429) the files that go up while a task trains (train.log and
# config.yaml after a checkpoint, wall.json, len_by_step.csv, rollout_stats.csv) wait in $RUN/.up_due and go up together, in
# one commit, at most every RUNS_PUT_S (600) s per task; the final files go in one commit too. Weights are not held back.
UPT=$RUN/.last_commit
runs_put_s() { local v; v=$(sed -n 's/^\(export \)\{0,1\}RUNS_PUT_S=//p' $OPD_ROOT/control.env 2> /dev/null | tail -1 | tr -cd 0-9); echo ${v:-${RUNS_PUT_S:-600}}; }
due() { [ "$DRY" = 1 ] || [ $(( $(date +%s) - $(stat -c %Y $UPT 2> /dev/null || stat -f %m $UPT 2> /dev/null || echo 0) )) -ge $(runs_put_s) ]; }
upn() {   # files of $RUN, one commit under runs/$TASK/
  local b=$RUN/.up_batch f n=0; rm -rf $b; mkdir -p $b
  for f in "$@"; do [ -f $RUN/$f ] && cp -p $RUN/$f $b/ && n=$((n + 1)); done
  [ $n = 0 ] && return 0
  up $b runs/$TASK && touch $UPT; }
flush_due() { [ -s $RUN/.up_due ] && due || return 0; local fl; fl=$(sort -u $RUN/.up_due | tr '\n' ' ')
  upn $fl > /dev/null 2>&1 && { rm -f $RUN/.up_due; echo "UP_BATCH $TASK ${fl}$(date -u +%T)"; }; }
stats() {   # [v2.5.9g] $1 = settle seconds: <run>/rollout_stats.csv from the run's rollout dumps (an E12 run; per-step cache)
  ls $RUN/rollouts/*.jsonl > /dev/null 2>&1 || return 1
  local out; out=$(cd $RUNBOOK && timeout 900 python code/analysis/e12_rollout_stats.py $RUN --tokenizer $STUDENT_17B --settle $1 2>&1) \
    || { echo "STATS_ERR $TASK $(echo "$out" | tail -1 | cut -c1-200)"; return 1; }; }
lenwall() {   # [v2.5.9z38] (Q-060) an OPD run's per-step length / clip and its 4K-wall step (len_wall.py); the CSV every LEN_EVERY (5) steps
  local w n u; w=$(timeout 120 python $RUNBOOK/scripts/len_wall.py $RUN $TASK 2> /dev/null | grep '^WALL ')
  if [ -n "$w" ]; then echo "$w"; [ "$DRY" = 1 ] || python $RUNBOOK/scripts/control.py event $TASK wall_cross "${w#WALL $TASK }" > /dev/null 2>&1
    echo wall.json >> $RUN/.up_due; fi   # [v2.5.9z52]
  n=$(awk 'NR > 1' $RUN/len_by_step.csv 2> /dev/null | grep -c .); u=$(cat $RUN/.len_uploaded 2> /dev/null || echo 0)
  [ $n -ge $(( u + ${LEN_EVERY:-5} )) ] && echo len_by_step.csv >> $RUN/.up_due && { echo $n > $RUN/.len_uploaded; echo "LEN_UP $TASK $n steps $(date -u +%T) (queued)"; }
  return 0; }
x5alert() {   # [v2.5.9z59] (Fable 10-22) an E16 step whose responses carry special ids in more than X5_SPECIAL_ALERT (2%) of its rows: one event
  [ "$DRY" = 1 ] && return 0   # [v2.5.9z61] (ops-lead's follow-up) as lenwall
  [ -f $RUN/x5.jsonl ] || return 0
  python3 - $RUN/x5.jsonl $RUN/.x5_seen $TASK "${X5_SPECIAL_ALERT:-0.02}" $RUNBOOK/scripts/control.py <<'PY'
import json, subprocess, sys
log, seen_f, task, lim, ctl = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
raw = open(log).read()
lines = raw.split("\n")[:-1]   # [v2.5.9z61] (ops-lead's follow-up) whole lines only: a line still being written waits for the next round
try:
    seen = int(open(seen_f).read().strip() or 0)
except (OSError, ValueError):
    seen = 0
for l in lines[seen:]:
    try:
        c = json.loads(l)
    except ValueError:
        continue
    n = c.get("rows_special", sum((c.get("masked") or {}).values())); r = c.get("rows") or 0
    if r and n / r > lim:
        subprocess.run([sys.executable, ctl, "event", task, "x5_special_rows",
                        f"call {c.get('call')}: {n} of {r} rows ({n / r:.1%} > {lim:.0%}) {json.dumps(c.get('masked'))}"])
open(seen_f, "w").write(str(len(lines)))
PY
}
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
  [ $new = 1 ] && printf 'train.log\nconfig.yaml\n' >> $RUN/.up_due   # [v2.5.9z52]
  for d in $RUN/ckpt/global_step_*; do   # one resume state only: older steps lose their *.pt once a newer one is complete
    [ -d $d ] && [ "${d##*_}" -lt "$latest" ] && ls $d/actor/*.pt >/dev/null 2>&1 && rm -f $d/actor/*.pt && echo "PRUNED resume state $TASK step ${d##*_}"
  done
  # [v2.5.9g] Q-034 (E12b-P5-stop2 is paused if its zero-variance groups average > 80% over s1-s10): while an E12 run trains,
  # its rollout_stats.csv every round from the settled dumps (older than 60 s; only new steps are read), uploaded to
  # runs/<task>/ each time it has STATS_EVERY (10) more rows than at the last upload: s10, s20, ...
  case $TASK in E10[bcde]-*) grep -q '^finished:' $RUN/config.yaml 2>/dev/null || lenwall;; esac   # [v2.5.9z38]; [v2.5.9z68] E10e
  case $TASK in E16-*) x5alert;; esac   # [v2.5.9z59]
  case $TASK in E12[bcde]-*) if ! grep -q '^finished:' $RUN/config.yaml 2>/dev/null && stats 60; then
    n=$(awk 'NR > 1' $RUN/rollout_stats.csv 2> /dev/null | grep -c .); u=$(cat $RUN/.stats_uploaded 2> /dev/null || echo 0)
    [ $n -ge $(( u + ${STATS_EVERY:-10} )) ] && echo rollout_stats.csv >> $RUN/.up_due \
      && { echo $n > $RUN/.stats_uploaded; echo "STATS_UP $TASK $n steps $(date -u +%T) (queued)"; }
  fi;; esac
  grep -q '^finished:' $RUN/config.yaml 2> /dev/null || flush_due   # [v2.5.9z52]
  if grep -q '^finished:' $RUN/config.yaml 2>/dev/null && [ $pend = 0 ]; then
    [ -f $RUN/train.log ] && python $RUNBOOK/scripts/metrics_jsonl.py $RUN/train.log > $RUN/metrics.jsonl
    case $TASK in E12[bcde]-*) stats 0;; esac   # [v2.5.9g] complete, whether or not run_e12.sh's own pass (after finished:) is done
    case $TASK in E10[bcde]-*) lenwall;; esac   # [v2.5.9z38] the last steps' rows and a late wall; [v2.5.9z68] E10e
    upn train.log config.yaml metrics.jsonl rollout_stats.csv lensel.jsonl stopmerge.jsonl tailswap.jsonl len_by_step.csv wall.json && rm -f $RUN/.up_due   # [v2.5.8m] lensel.jsonl: the length-selected arms' per-prompt record; [v2.5.9z52] one commit
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
  # [v2.5.9z88c] a CORPUS-RECORD sandbox's task.sh <90+> $TASK --dryrun-sandbox is not this task running
  if [ -n "$why" ] && [ $pend = 0 ] && [ -z "$(pgrep -af "scripts/task\.sh ([0-9]+(,[0-9]+)*|cpu) $TASK( |\$)" | grep -v -- --dryrun-sandbox)" ]; then
    upn train.log config.yaml lensel.jsonl stopmerge.jsonl tailswap.jsonl $(sort -u $RUN/.up_due 2> /dev/null) && rm -f $RUN/.up_due   # [v2.5.9z52] one commit
    echo "UPLOAD_STOPPED $TASK $why $(date -u +%T)"; break
  fi
  sleep $([ "$DRY" = 1 ] && echo 5 || echo 120)
done
