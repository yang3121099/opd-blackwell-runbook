#!/usr/bin/env bash
# watchdog.sh (runbook v2.5.7zb; Runming / reviewer 2026-10-19 14:0x UTC, after the control-plane hang of 12:16-13:50 UTC):
# restarts run_tasks.sh or heartbeat.sh when its liveness file (control/loop_run_tasks: every round of run_tasks;
# control/loop_heartbeat: every 10 s and after every bounded upload of the heartbeat; both hold epoch seconds) is older
# than WD_STALE_MIN (10) minutes. It makes no network call of its own (its events are bounded, in the background).
# A restart TERMs the loop's process group, KILLs it after 30 s, checks it is gone, and starts the loop the way
# overnight.sh / run_tasks do. Both loops are started with setsid, so their groups hold only the loop and its direct
# children. Task sessions, upload_sync, prefetches and this watchdog run in sessions of their own and are never
# signalled. At most WD_MAX (3) restarts of one loop in WD_WINDOW_MIN (30) minutes: beyond that, one event
# watchdog_gave_up and no automatic restart of that loop until the file control/watchdog_<loop>.gave_up is removed.
# One watchdog only (flock on control/watchdog.lock; without flock, pgrep). run_tasks starts it when it is missing.
# Usage: setsid nohup bash scripts/watchdog.sh >> $OPD_ROOT/logs/watchdog.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
CDIR=$OPD_ROOT/control; mkdir -p $CDIR $OPD_ROOT/logs
if command -v flock > /dev/null; then
  exec 9> $CDIR/watchdog.lock; flock -n 9 || { echo "WATCHDOG already running"; exit 0; }
elif [ "$(pgrep -f 'scripts/watchdog.s[h]' | grep -vc "^$$\$")" -gt 0 ]; then echo "WATCHDOG already running"; exit 0; fi
STALE=${WD_STALE_SEC:-$(( ${WD_STALE_MIN:-10} * 60 ))}; WIN=$(( ${WD_WINDOW_MIN:-30} * 60 )); MAX=${WD_MAX:-3}
echo "WATCHDOG_START $(date -u +%T) stale ${STALE}s, at most $MAX restarts per loop in ${WIN}s"
ev() { command -v timeout > /dev/null && (timeout 60 python $RUNBOOK/scripts/control.py event node "$1" "$2" > /dev/null 2>&1 &); }
pids_of() { pgrep -f "scripts/$1.s[h]"; }
start_loop() {
  case $1 in
    run_tasks) (cd $RUNBOOK && setsid nohup bash scripts/run_tasks.sh >> $OPD_ROOT/logs/run_tasks.log 2>&1 9>&- < /dev/null &);;
    heartbeat) (setsid nohup bash $RUNBOOK/scripts/heartbeat.sh >> $OPD_ROOT/logs/heartbeat.log 2>&1 9>&- < /dev/null &);;
  esac; }
sig_loop() {   # $1 loop, $2 signal: the loop's process group when it leads one, else the loop and its children
  local p g
  for p in $(pids_of $1); do
    g=$(ps -o pgid= -p $p 2>/dev/null | tr -d ' ')
    if [ "$g" = "$p" ]; then kill -$2 -- -$g 2>/dev/null; else pkill -$2 -P $p 2>/dev/null; kill -$2 $p 2>/dev/null; fi
  done; }
stop_loop() {
  sig_loop $1 TERM
  for _ in $(seq ${WD_TERM_WAIT:-30}); do [ -z "$(pids_of $1)" ] && return 0; sleep 1; done
  sig_loop $1 KILL; sleep 2; [ -z "$(pids_of $1)" ]; }
while true; do
  now=$(date +%s)
  for L in run_tasks heartbeat; do
    f=$CDIR/loop_$L; [ -f $f ] || continue   # never seen alive (not started yet): not the watchdog's business
    t=$(cat $f 2>/dev/null | tr -cd 0-9); age=$(( now - ${t:-$now} ))
    [ $age -le $STALE ] && continue
    [ -f $CDIR/watchdog_$L.gave_up ] && continue
    recent=$(awk -v n=$now -v w=$WIN '$1 > n - w' $CDIR/watchdog_$L.times 2>/dev/null | grep -c .)
    if [ "${recent:-0}" -ge $MAX ]; then
      date -u +%FT%TZ > $CDIR/watchdog_$L.gave_up
      echo "WATCHDOG $(date -u +%T) $L stale ${age}s: $recent restarts in ${WIN}s, giving up"; ev watchdog_gave_up "$L stale ${age}s"; continue
    fi
    echo "WATCHDOG $(date -u +%T) $L stale ${age}s: restart"
    if ! stop_loop $L; then echo "WATCHDOG $(date -u +%T) $L still running after KILL"; ev watchdog_stuck "$L"; continue; fi
    start_loop $L; echo $now >> $CDIR/watchdog_$L.times; date -u +%s > $f   # the new loop gets a full window
    ev watchdog_restart "$L stale ${age}s"
  done
  sleep ${WD_POLL:-60}
done
