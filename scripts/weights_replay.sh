#!/usr/bin/env bash
# weights_replay.sh (runbook v2.5.9z62; ops-lead A1 10-23): the held weights to the public repository, one item at a time
# (weights_replay.py once: order, size-bounded timeout, 429 backoff, the control plane first, verified before it is struck; the
# held copies are never deleted). Again at once after a success, else every WEIGHTS_REPLAY_POLL (120) s. One copy (flock on
# control/weights_replay.lock); run_tasks.sh starts it and restarts it when this file changes (ensure_loop). setenv
# WEIGHTS_REPLAY=0 pauses it. Usage: setsid nohup bash scripts/weights_replay.sh >> $OPD_ROOT/logs/weights_replay.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
CDIR=$OPD_ROOT/control; mkdir -p $CDIR $OPD_ROOT/logs
if command -v flock > /dev/null; then
  exec 9> $CDIR/weights_replay.lock; flock -n 9 || { echo "WEIGHTS_REPLAY already running"; exit 0; }
fi
echo "WEIGHTS_REPLAY_START $(date -u +%T)"
while true; do
  date -u +%s > $CDIR/loop_weights_replay
  out=$(python $RUNBOOK/scripts/weights_replay.py once 2>> $OPD_ROOT/logs/weights_replay.err 9>&-)
  case "$out" in REPLAY_IDLE|REPLAY_OFF) ;; *) echo "$(date -u +%T) ${out:-round failed}";; esac
  [ "${WEIGHTS_REPLAY_ONCE:-0}" = 1 ] && break
  case "$out" in REPLAY_OK*) sleep 5;; *) sleep ${WEIGHTS_REPLAY_POLL:-120};; esac
done
