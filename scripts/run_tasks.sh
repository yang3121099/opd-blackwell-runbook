#!/usr/bin/env bash
# Remote-node loop (queue version 2): every 5 minutes, start the next runnable tasks of the open stages, strictly in
# queue order, one task per free GPU (no compute process and <= 2 GB used), 90 s apart. `git pull` every 30 minutes.
#  - an "auto" stage opens once every task of every earlier stage is finished (.done locally or in the result
#    repository, or failed 3 times); a "locked" stage never opens and holds back everything after it
#  - a task starts only when its deps are .done; a task whose dep failed for good gets runs/<task>/.skipped and counts
#    as finished, so later stages still open (v2.3: before, it held its stage open forever and stalled the ladder)
#  - fail-fast: a task that exits without .done gets runs/<task>/.failed (attempt count + the log tail, reported by the
#    heartbeat) and is retried at most twice (3 attempts in all)
# DRY=1 prints what would start, without starting anything.
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
Q=$RUNBOOK/tasks/queue.txt; last_pull=0
# done locally, or in the result repository; a remote hit is cached as a local .done so it is asked only once (v2.3)
done_() { [ -f $OPD_ROOT/runs/$1/.done ] && return 0
  python $RUNBOOK/scripts/remote.py done $1 2>/dev/null && { mkdir -p $OPD_ROOT/runs/$1; touch $OPD_ROOT/runs/$1/.done; }; }
skipped() { [ -f $OPD_ROOT/runs/$1/.skipped ]; }
attempts() { sed -n 's/^attempts: //p' $OPD_ROOT/runs/$1/.failed 2>/dev/null || echo 0; }
dead() { [ "$(attempts $1)" -ge 3 ]; }
running() { pgrep -f "task.sh [0-9]+ $1\$" >/dev/null; }
free_gpus() { for i in $(seq 0 $((NGPU-1))); do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $i)" ] || continue
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i | tr -d ' ')" -le 2048 ] && echo $i; done; }
launch() { # gpu task
  mkdir -p $OPD_ROOT/runs/$2
  setsid nohup bash -c "bash $RUNBOOK/scripts/task.sh $1 $2 > $OPD_ROOT/logs/$2.log 2>&1
    if [ ! -f $OPD_ROOT/runs/$2/.done ]; then a=\$(( \$(sed -n 's/^attempts: //p' $OPD_ROOT/runs/$2/.failed 2>/dev/null || echo 0) + 1 ))
      { echo \"attempts: \$a\"; echo \"last: \$(date -u +%FT%TZ)\"; tail -20 $OPD_ROOT/logs/$2.log; } > $OPD_ROOT/runs/$2/.failed; fi" < /dev/null > /dev/null 2>&1 &
}
while true; do
  if [ "${PULL:-1}" = 1 ] && [ $(( $(date +%s) - last_pull )) -ge 1800 ]; then git -C "$RUNBOOK" pull -q --ff-only 2>/dev/null; last_pull=$(date +%s); fi
  FREE=($(free_gpus)); earlier_finished=1
  while read -r stage status rest; do
    case $stage in ''|\#*|version:) continue;; esac
    opened=0; [ "$status" = open ] && opened=1; [ "$status" = auto ] && [ $earlier_finished = 1 ] && opened=1
    stage_finished=1
    for spec in $rest; do
      t=${spec%%:*}; deps=""; [ "$spec" != "$t" ] && deps=$(echo ${spec#*:} | tr '+' ' ')
      if done_ $t || dead $t || skipped $t; then continue; fi
      bad=""; for d in $deps; do dead $d || skipped $d && bad="$bad $d"; done
      if [ -n "$bad" ]; then   # a dependency failed for good: this task can never run; it no longer blocks its stage
        mkdir -p $OPD_ROOT/runs/$t; echo "skipped: dependency failed for good:$bad ($(date -u +%FT%TZ))" > $OPD_ROOT/runs/$t/.skipped
        echo "SKIP $t: dependency$bad failed for good"; continue
      fi
      stage_finished=0
      [ $opened = 1 ] || continue
      running $t && continue
      ok=1; for d in $deps; do done_ $d || ok=0; done
      [ $ok = 1 ] || continue
      [ ${#FREE[@]} -gt 0 ] || continue
      g=${FREE[0]}; FREE=("${FREE[@]:1}")
      if [ "${DRY:-0}" = 1 ]; then echo "WOULD_START gpu=$g $t (stage $stage, attempt $(( $(attempts $t) + 1 )))"; continue; fi
      echo "START $(date -u +%T) gpu=$g $t (stage $stage, queue v$(sed -n 's/^version:[[:space:]]*//p' "$Q"), attempt $(( $(attempts $t) + 1 )))"
      launch $g $t; sleep 90
    done
    [ "$status" = locked ] && stage_finished=0
    [ $stage_finished = 1 ] || earlier_finished=0
  done < "$Q"
  [ "${DRY:-0}" = 1 ] && break
  sleep 300
done
