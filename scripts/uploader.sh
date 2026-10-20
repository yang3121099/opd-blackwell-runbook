#!/usr/bin/env bash
# uploader.sh (runbook v2.5.8; design 4b.2 of lab/reviews/notes/v258_backlog.md, reviewer A1 2026-10-19 13:3x UTC with the
# additions a-d): ships the control plane's outbox ($OPD_ROOT/control/outbox/, control.py enqueue / queue_put) with the
# bounded remote_put, one pass after another (`control.py ship_outbox`: time order, the newest copy of a coalesced dst,
# a failed upload ends the pass). This loop has no other duty, so a hung upload stops only it: dispatching (run_tasks)
# and the heartbeat only write local files. control/loop_uploader is its liveness file (every pass and every item);
# the watchdog restarts the loop when it is older than WD_STALE_MIN, and control.py uploads directly while it is stale.
# One uploader only (flock on control/uploader.lock; without flock, control/uploader.pid). run_tasks and the watchdog start it.
# Usage: setsid nohup bash scripts/uploader.sh >> $OPD_ROOT/logs/uploader.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
CDIR=$OPD_ROOT/control; mkdir -p $CDIR/outbox $OPD_ROOT/logs
if command -v flock > /dev/null; then
  exec 9> $CDIR/uploader.lock; flock -n 9 || { echo "UPLOADER already running"; exit 0; }
else   # [v2.5.8] a pid file: the pgrep count also saw this script's own $(...) subshells and refused every start
  p=$(cat $CDIR/uploader.pid 2> /dev/null)
  [ -n "$p" ] && [ "$p" != $$ ] && ps -o args= -p $p 2> /dev/null | grep -q "scripts/uploader.s[h]" && { echo "UPLOADER already running"; exit 0; }
  echo $$ > $CDIR/uploader.pid
fi
TO=""; command -v timeout > /dev/null && TO="timeout -k 30 ${UPLOADER_PASS_TIMEOUT:-900}"   # the pass itself is bounded too
echo "UPLOADER_START $(date -u +%T) poll ${UPLOADER_POLL:-5}s"
while true; do
  date -u +%s > $CDIR/loop_uploader
  out=$($TO python $RUNBOOK/scripts/control.py ship_outbox 2>> $OPD_ROOT/logs/uploader.err 9>&-); rc=$?   # the lock stays with this loop only
  date -u +%s > $CDIR/loop_uploader
  case "$out" in
    "SHIPPED 0 DROPPED 0 FAILED 0 "*) sleep ${UPLOADER_POLL:-5};;
    *"FAILED 0 "*) echo "$(date -u +%T) $out";;                                  # more may be waiting: next pass at once
    *) echo "$(date -u +%T) ${out:-pass rc=$rc}"; sleep ${UPLOADER_BACKOFF:-30};;  # an upload failed or timed out
  esac
done
