#!/usr/bin/env bash
# [z168] (Fable board v22) the registered-step keeper: every KEEP_SEC (20) s, python keep_steps.py hard-links each task's registered steps
# (its UPLOAD_WEIGHTS + KEEP_STEPS) out of the local checkpoint rotation and publishes them to the shared disk's adhoc_models/result (see
# keep_steps.py). Started by run_tasks (ensure_loop keep_steps) only when CKPT_LOCAL is set and KEEP_STEPS_OFF is not 1; stopped by
# loops.sh stop. Liveness: control/loop_keep_steps (scripts/preflight.sh reads it). Log: logs/keep_steps.log.
set -uo pipefail
source "$(dirname "$0")/env.sh"
[ -n "${CKPT_LOCAL:-}" ] || { echo "KEEP_STEPS_OFF $(date -u +%T): CKPT_LOCAL unset (no local rotation)"; exit 0; }
mkdir -p $OPD_ROOT/control $OPD_ROOT/logs
if command -v flock > /dev/null; then exec 9> $OPD_ROOT/control/keep_steps.lock; flock -n 9 || exit 0; fi   # one copy
echo "KEEP_STEPS_START $(date -u +%T) local $CKPT_LOCAL"
while true; do
  date -u +%s > $OPD_ROOT/control/loop_keep_steps
  nice -n 10 python3 $RUNBOOK/scripts/keep_steps.py
  [ "${KEEP_ONCE:-0}" = 1 ] && break   # the tests
  sleep ${KEEP_SEC:-20}
done
