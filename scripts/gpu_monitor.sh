#!/usr/bin/env bash
# [v3.1] gpu_monitor.sh: one copy of scripts/gpu_monitor.py (this node's own record of GPU utilisation against the machine's 85 % rule).
# run_tasks.sh starts it and starts it again when this file changes (ensure_loop: it finds and ends this script; the trap ends the
# python below it). One copy at a time (flock). Reads only nvidia-smi and the process list; writes control/gpu_monitor/state.json
# and logs/gpu_util.csv.
set -uo pipefail
source "$(dirname "$0")/env.sh"
[ -f $OPD_ROOT/node.env ] && { set -a; source $OPD_ROOT/node.env; set +a; }
mkdir -p $OPD_ROOT/control/gpu_monitor
exec 9> $OPD_ROOT/control/gpu_monitor/lock
if command -v flock > /dev/null; then flock -n 9 || { echo "GPU_MONITOR $(date -u +%T) another copy holds the lock"; exit 0; }; fi
python3 $RUNBOOK/scripts/gpu_monitor.py run 9>&- &
C=$!; trap 'kill $C 2> /dev/null' TERM INT EXIT
wait $C
