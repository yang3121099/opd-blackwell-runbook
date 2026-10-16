#!/usr/bin/env bash
# Remote-node loop: every 30 minutes `git pull` this repository, read tasks/queue.txt, and start the open stage's tasks
# on free GPUs (no compute process, <= 2 GB used), one task per GPU, 90 s apart. Only whitelisted task IDs run.
# Idempotent: a task with runs/<task>/.done is skipped; a task already running is left alone; an interrupted task
# resumes from its latest checkpoint when started again.
set -uo pipefail
source "$(dirname "$0")/env.sh"
WHITELIST="E10b-P2 E10b-P3 E10b-P4lowlr E10b-P5 E10b-P6 E10b-P6r E10b-P2noEOS"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
while true; do
  [ "${PULL:-1}" = 1 ] && git -C $RUNBOOK pull -q --ff-only 2>/dev/null
  Q=$RUNBOOK/tasks/queue.txt; open=$(sed -n 's/^open:[[:space:]]*\([^[:space:]]*\).*/\1/p' $Q)
  tasks=$(awk -v o="$open" '$1==o && $2=="open" {for (i=3;i<=NF;i++) print $i}' $Q)
  for t in $tasks; do
    case " $WHITELIST " in *" $t "*) ;; *) echo "IGNORED $t (not whitelisted)"; continue;; esac
    [ -f $OPD_ROOT/runs/$t/.done ] && continue
    pgrep -f "run_arm.sh [0-9]+ $t\$" >/dev/null && continue
    g=""; for i in $(seq 0 $((NGPU-1))); do
      [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $i)" ] || continue
      [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i | tr -d ' ')" -le 2048 ] && { g=$i; break; }
    done
    [ -z "$g" ] && break
    echo "START $(date -u +%T) gpu=$g $t (queue version $(sed -n 's/^version:[[:space:]]*//p' $Q))"
    setsid nohup bash $RUNBOOK/scripts/run_arm.sh $g $t > $OPD_ROOT/logs/$t.log 2>&1 < /dev/null &
    sleep 90
  done
  sleep 1800
done
