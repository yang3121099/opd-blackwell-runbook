#!/usr/bin/env bash
# [z155] (lab WO-011; Fable's A1 design) the checkpoint mirror: every MIRROR_SEC (60) s, python ckpt_lib.py mirror copies each task's newest
# LOCAL checkpoint ($CKPT_LOCAL/<task>/ckpt) to the shared disk ($CKPT_ROOT/<task>/ckpt) - .part, manifest check, rename, then the shared
# marker; a copy in flight is finished first; MIRROR_LAG lines while behind; keep 2 verified on each side. Never inside the training: its own
# process, nice 19 / idle I/O class. Started by run_tasks (ensure_loop ckpt_mirror) only when CKPT_LOCAL is set; stopped by loops.sh stop.
# Liveness: control/loop_ckpt_mirror. Log: logs/ckpt_mirror.log.
set -uo pipefail
source "$(dirname "$0")/env.sh"
[ -n "${CKPT_LOCAL:-}" ] || { echo "CKPT_MIRROR_OFF $(date -u +%T): CKPT_LOCAL unset (checkpoints go to the shared disk directly)"; exit 0; }
mkdir -p $OPD_ROOT/control $OPD_ROOT/logs
if command -v flock > /dev/null; then exec 9> $OPD_ROOT/control/ckpt_mirror.lock; flock -n 9 || exit 0; fi   # one copy
io=""; command -v ionice > /dev/null && io="ionice -c3"
echo "CKPT_MIRROR_START $(date -u +%T) local $CKPT_LOCAL"
while true; do
  date -u +%s > $OPD_ROOT/control/loop_ckpt_mirror
  nice -n 19 $io python3 $RUNBOOK/scripts/ckpt_lib.py mirror 2>&1 | sed "s/^/$(date -u +%T) /"
  [ "${MIRROR_ONCE:-0}" = 1 ] && break   # the tests
  sleep ${MIRROR_SEC:-60}
done
