#!/usr/bin/env bash
# watchdog.sh (runbook v2.5.7zb; Runming / reviewer 2026-10-19 14:0x UTC, after the control-plane hang of 12:16-13:50 UTC):
# restarts run_tasks.sh, heartbeat.sh or ([v2.5.8]) uploader.sh (control/loop_uploader) when its liveness file (control/loop_run_tasks: every round of run_tasks;
# control/loop_heartbeat: every 10 s and after every bounded upload of the heartbeat; both hold epoch seconds) is older
# than WD_STALE_MIN (10) minutes. It makes no network call of its own (its events are bounded, in the background).
# A restart TERMs the loop's tree ([v2.5.8]: see tree_of; was its process group), KILLs it after 30 s, checks it is
# gone, and starts the loop the way overnight.sh / run_tasks do. Task sessions, upload_sync, prefetches and this
# watchdog run in sessions of their own and are never signalled. At most WD_MAX (3) restarts of one loop in WD_WINDOW_MIN (30) minutes: beyond that, one event
# watchdog_gave_up and no automatic restart of that loop until the file control/watchdog_<loop>.gave_up is removed.
# One watchdog only (flock on control/watchdog.lock; without flock, [v2.5.8] control/watchdog.pid). run_tasks starts it when it is missing.
# Usage: setsid nohup bash scripts/watchdog.sh >> $OPD_ROOT/logs/watchdog.log 2>&1 &
set -uo pipefail
source "$(dirname "$0")/env.sh"
CDIR=$OPD_ROOT/control; mkdir -p $CDIR $OPD_ROOT/logs
if command -v flock > /dev/null; then
  exec 9> $CDIR/watchdog.lock; flock -n 9 || { echo "WATCHDOG already running"; exit 0; }
else   # [v2.5.8] a pid file: the pgrep count also saw this script's own $(...) subshells and refused every start
  p=$(cat $CDIR/watchdog.pid 2> /dev/null)
  [ -n "$p" ] && [ "$p" != $$ ] && ps -o args= -p $p 2> /dev/null | grep -q "scripts/watchdog.s[h]" && { echo "WATCHDOG already running"; exit 0; }
  echo $$ > $CDIR/watchdog.pid
fi
STALE=${WD_STALE_SEC:-$(( ${WD_STALE_MIN:-10} * 60 ))}; WIN=$(( ${WD_WINDOW_MIN:-30} * 60 )); MAX=${WD_MAX:-3}
echo "WATCHDOG_START $(date -u +%T) stale ${STALE}s, at most $MAX restarts per loop in ${WIN}s"
ev() { command -v timeout > /dev/null && (timeout -k 10 60 python $RUNBOOK/scripts/control.py event node "$1" "$2" > /dev/null 2>&1 &); }
pids_of() { pgrep -f "scripts/$1.s[h]"; }
start_loop() {
  case $1 in
    run_tasks) (cd $RUNBOOK && setsid nohup bash scripts/run_tasks.sh >> $OPD_ROOT/logs/run_tasks.log 2>&1 9>&- < /dev/null &);;
    heartbeat) (setsid nohup bash $RUNBOOK/scripts/heartbeat.sh >> $OPD_ROOT/logs/heartbeat.log 2>&1 9>&- < /dev/null &);;
    uploader) (cd $RUNBOOK && setsid nohup bash scripts/uploader.sh >> $OPD_ROOT/logs/uploader.log 2>&1 9>&- < /dev/null &);;   # [v2.5.8]
  esac; }
# [v2.5.8] (reviewer 15:1x UTC, after drill B) a restart signals the loop's whole tree, captured before the first
# signal: the loop and every descendant in the loop's own session. That includes a `timeout` and its upload, which
# sit in a process group of their own (GNU timeout calls setpgid), so a group kill missed them and left a hung upload
# behind. Never included: other sessions (task sessions, prefetches, upload_sync), and any process whose command is a
# task, a prefetch, a fetch or another loop (TREE_SKIP), with everything below it. (Linux: session by ps -o sid;
# where ps has no sid, as on macOS in the tests, only TREE_SKIP applies.)
TREE_SKIP='scripts/(task|run_adhoc_eval|upload_sync|run_queue_eval|heartbeat|watchdog|run_tasks|uploader|dr|dr_sync)[.]sh|fetch_model[.]py'
tree_of() {   # $1 = a loop's pid -> that pid and its descendants as above, one per line
  local hassid=1; ps -o sid= -p $$ > /dev/null 2>&1 || hassid=0
  { if [ $hassid = 1 ]; then ps -A -o pid=,ppid=,sid=,args=; else ps -A -o pid=,ppid=,args= | awk '{$2 = $2 " -"; print}'; fi; } |
  awk -v r=$1 -v skip="$TREE_SKIP" '{pid = $1; par[pid] = $2; s[pid] = $3; $1 = $2 = $3 = ""; a[pid] = $0}
    END {inn[r] = 1; ch = 1
         while (ch) { ch = 0; for (p in par) if (!(p in inn) && (par[p] in inn) && s[p] == s[r] && a[p] !~ skip) { inn[p] = 1; ch = 1 } }
         for (p in inn) print p}'; }
alive_of() { local p; for p in "$@"; do kill -0 $p 2> /dev/null && echo $p; done; }
stop_loop() {   # TERM the trees of loop $1, KILL what is left after WD_TERM_WAIT s; 0 when all of it is gone
  local p members="" left
  for p in $(pids_of $1); do members="$members $(tree_of $p | tr '\n' ' ')"; done
  [ -z "${members// /}" ] && return 0
  echo "WATCHDOG $(date -u +%T) $1 tree:$members"
  kill -TERM $members 2> /dev/null
  for _ in $(seq ${WD_TERM_WAIT:-30}); do [ -z "$(alive_of $members)" ] && return 0; sleep 1; done
  left=$(alive_of $members | tr '\n' ' '); echo "WATCHDOG $(date -u +%T) $1 KILL $left"
  kill -KILL $left 2> /dev/null; sleep 2
  [ -z "$(alive_of $members)" ] && [ -z "$(pids_of $1)" ]; }
while true; do
  now=$(date +%s)
  for L in run_tasks heartbeat uploader; do   # [v2.5.8] uploader
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
