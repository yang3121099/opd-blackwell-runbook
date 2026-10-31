#!/usr/bin/env bash
# [v2.5.9z106] loops.sh on fake loops (ops-lead's A1 11-01 C2 / C6): one heartbeat and one run_tasks at most; a loop started
# elsewhere (h800_loops.sh's way: cd runbook && bash scripts/heartbeat.sh) is adopted, never doubled; duplicates refused, then
# cured by restart; a task session and another runbook's loop are never signalled; ensure restarts a changed heartbeat.
# Run: bash scripts/test_loops.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]: ${out:0:400}"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); RB=$T/rb; O=$T/opd; mkdir -p $RB/scripts $O/control $O/logs $T/bin $T/other/scripts
cp $SRC/scripts/loops.sh $RB/scripts/
printf 'export OPD_ROOT=%s RUNBOOK=%s\n' $O $RB > $RB/scripts/env.sh
printf 'import sys, os; open(os.path.join(os.environ["OPD_ROOT"], "events"), "a").write(" ".join(sys.argv[1:]) + "\\n")\n' > $RB/scripts/control.py
for L in heartbeat run_tasks; do printf 'while true; do x=$(sleep 1; echo); done\n' > $RB/scripts/$L.sh; done
cp $RB/scripts/heartbeat.sh $T/other/scripts/heartbeat.sh
printf 'sleep 300\n' > $RB/scripts/task.sh
command -v setsid > /dev/null || { printf '#!/bin/bash\nexec "$@"\n' > $T/bin/setsid; chmod +x $T/bin/setsid; }
command -v timeout > /dev/null || { printf '#!/bin/bash\nwhile [[ $1 == -* ]]; do shift 2; done; shift; exec "$@"\n' > $T/bin/timeout; chmod +x $T/bin/timeout; }   # GNU timeout is not on the Mac
export PATH=$T/bin:$PATH LOOPS_TEST_CWD=$RB LOOPS_TERM_WAIT=3 LOOPS_LOCK_WAIT=5 LOOPS_SYNC_EV=1
LP() { bash $RB/scripts/loops.sh "$@" 2>&1; }
cnt() { bash $RB/scripts/loops.sh status | awk -v l=$1 '$2 == l {print $3}'; }
pids_all() { pgrep -f "$1/scripts/[a-z_]*[.]sh|scripts/[a-z_]*[.]sh" 2> /dev/null; }

out=$(LP status); ok "status: none running" '[ "$(cnt heartbeat)" = 0 ] && [ "$(cnt run_tasks)" = 0 ]'
out=$(LP start); rc=$?
ok "start: one heartbeat, one run_tasks, pid files" '[ $rc = 0 ] && [ "$(cnt heartbeat)" = 1 ] && [ "$(cnt run_tasks)" = 1 ] && [ -s $O/control/loops_heartbeat.pid ] && [ -s $O/control/loops_run_tasks.pid ]'
h1=$(cat $O/control/loops_heartbeat.pid)
out=$(LP start); rc=$?
ok "start again: no second loop (C2)" '[ $rc = 0 ] && [ "$(cnt heartbeat)" = 1 ] && [ "$(cat $O/control/loops_heartbeat.pid)" = "$h1" ]'
out=$(LP stop heartbeat); rm -f $O/control/loops_heartbeat.pid
(cd $RB; setsid nohup bash scripts/heartbeat.sh > /dev/null 2>&1 < /dev/null &) > /dev/null 2>&1; sleep 1   # h800_loops.sh's way, no pid file
out=$(LP start heartbeat); rc=$?; sleep 1
ok "a loop started elsewhere is adopted, not doubled (C2)" '[ $rc = 0 ] && [ "$(cnt heartbeat)" = 1 ] && [[ "$out" == *"adopted heartbeat"* ]] && grep -q "loops_adopted heartbeat" $O/events'
(cd $RB; setsid nohup bash scripts/heartbeat.sh > /dev/null 2>&1 < /dev/null &) > /dev/null 2>&1; sleep 1   # a second one: duplicate
out=$(LP start heartbeat); rc=$?; sleep 1   # the event is written in the background
ok "two heartbeats: start refuses (exit 3, event)" '[ $rc = 3 ] && [[ "$out" == *"DUPLICATE heartbeat"* ]] && grep -q "loops_duplicate heartbeat" $O/events'
(cd $RB; setsid nohup bash scripts/task.sh 0 X > /dev/null 2>&1 < /dev/null &) > /dev/null 2>&1   # a task session
(setsid nohup bash $T/other/scripts/heartbeat.sh > /dev/null 2>&1 < /dev/null &); sleep 1   # another runbook's loop
task=$(pgrep -f "scripts/task[.]sh 0 X"); other=$(pgrep -f "$T/other/scripts/heartbeat[.]sh" | head -1)
out=$(LP restart heartbeat); rc=$?
ok "restart cures the duplicate: exactly one heartbeat" '[ $rc = 0 ] && [ "$(cnt heartbeat)" = 1 ]'
ok "restart never signals a task session (C6)" 'kill -0 $task 2> /dev/null'
ok "another runbook's heartbeat is neither counted nor signalled" 'kill -0 $other 2> /dev/null'
h2=$(cat $O/control/loops_heartbeat.pid); echo $(sha1sum $RB/scripts/heartbeat.sh 2> /dev/null | cut -c1-12 || shasum $RB/scripts/heartbeat.sh | cut -c1-12) > $O/control/heartbeat.sha
echo "# changed" >> $RB/scripts/heartbeat.sh
out=$(LP ensure heartbeat); rc=$?
ok "ensure: a changed heartbeat is restarted (one, new pid)" '[ $rc = 0 ] && [ "$(cnt heartbeat)" = 1 ] && [ "$(cat $O/control/loops_heartbeat.pid)" != "$h2" ]'
js() { python3 -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if len(d[\"heartbeat\"]) == 1 and len(d[\"run_tasks\"]) == 1 else 1)" $O/control/loops.json; }
out=$(LP status); js || { sleep 1; out=$(LP status); }   # macOS has no session ids: a subshell caught mid-fork counts once; again
ok "status writes control/loops.json" 'python3 -c "import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if len(d[\"heartbeat\"]) == 1 and len(d[\"run_tasks\"]) == 1 else 1)" $O/control/loops.json'
out=$(LP bogus); rc=$?; ok "bad verb: usage, exit 2" '[ $rc = 2 ]'
LP stop > /dev/null; kill $task $other 2> /dev/null; sleep 1
ok "stop: none left" '[ "$(cnt heartbeat)" = 0 ] && [ "$(cnt run_tasks)" = 0 ]'
pkill -f "$T/" 2> /dev/null; [ -n "${KEEP_T:-}" ] && echo "kept $T" || rm -rf $T
[ $fails = 0 ] && echo "LOOPS_TEST PASS ($n checks)" || { echo "LOOPS_TEST FAIL ($fails of $n)"; exit 1; }
