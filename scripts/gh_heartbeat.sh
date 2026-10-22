#!/usr/bin/env bash
# gh_heartbeat.sh (runbook v2.5.9z54; Runming 10-22 21:5x Beijing via Fable, ops-lead A1): the GitHub fallback heartbeat. Every
# GH_HB_POLL (10) s one `control.py gh_round`: every GH_HB_SEC (600) s, and at once after an urgent event, hb/<node>.json (the
# pulse, the latest receipts and events, the control plane's state) is committed and pushed to a private GitHub repository over
# HTTPS (GH_HB_URL; git's credential store reads the token file Runming wrote, never this code); a failed push backs off 60 s
# doubling to 900 s. Independent of the HF path: nothing here
# waits on an HF call, and the uploader / heartbeat never wait on this loop. setenv GH_HB=0 turns it off. One copy (flock on
# control/gh_heartbeat.lock); run_tasks.sh starts it and restarts it when this file changes (ensure_loop).
# Usage: setsid nohup bash scripts/gh_heartbeat.sh >> $OPD_ROOT/logs/gh_heartbeat.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
CDIR=$OPD_ROOT/control; mkdir -p $CDIR $OPD_ROOT/logs
if command -v flock > /dev/null; then
  exec 9> $CDIR/gh_heartbeat.lock; flock -n 9 || { echo "GH_HEARTBEAT already running"; exit 0; }
fi
TO=""; command -v timeout > /dev/null && TO="timeout -k 10 ${GH_HB_ROUND_TIMEOUT:-400}"   # a round is bounded too
echo "GH_HEARTBEAT_START $(date -u +%T)"
while true; do
  date -u +%s > $CDIR/loop_gh_heartbeat
  out=$($TO python3 $RUNBOOK/scripts/control.py gh_round 2>> $OPD_ROOT/logs/gh_heartbeat.err 9>&-)
  case "$out" in idle|off|backoff) ;; *) echo "$(date -u +%T) ${out:-round failed}";; esac
  [ "${GH_HB_ONCE:-0}" = 1 ] && break
  sleep ${GH_HB_POLL:-10}
done
