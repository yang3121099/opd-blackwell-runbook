#!/usr/bin/env bash
# Stub tests of v2.5.7zb watchdog.sh (no GPU, network or HF). Usage: bash scripts/test_watchdog.sh -> WATCHDOG_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/control $T/root/logs $T/bin
  cp $SRC/scripts/watchdog.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  for L in run_tasks heartbeat; do printf 'echo $$ >> %s/root/%s.starts\nwhile true; do sleep 1; done\n' $T $L > $T/rb/scripts/$L.sh; done
  # setsid is Linux-only: a stub that really starts a new session (python os.setsid) keeps the process-group logic real
  printf '#!/usr/bin/env bash\nexec python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" "$@"\n' > $T/bin/setsid; chmod +x $T/bin/setsid
  export PATH=$T/bin:$PATH
  (cd $T/rb && setsid bash scripts/run_tasks.sh > /dev/null 2>&1 < /dev/null &)
  (setsid bash $T/rb/scripts/heartbeat.sh > /dev/null 2>&1 < /dev/null &)
  sleep 2
}
rt_pids() { pgrep -f "bash scripts/run_tasks.s[h]"; }
cleanup() { pkill -f "$T/rb/scripts/watchdog.s[h]" 2>/dev/null; pkill -f "$T/rb/scripts/heartbeat.s[h]" 2>/dev/null; pkill -f "bash scripts/run_tasks.s[h]" 2>/dev/null; sleep 1; rm -rf $T; }
run_wd() { (WD_STALE_SEC=100 WD_POLL=1 WD_TERM_WAIT=3 bash $T/rb/scripts/watchdog.sh >> $T/root/logs/watchdog.log 2>&1 < /dev/null &); sleep ${1:-6}; pkill -f "$T/rb/scripts/watchdog.s[h]" 2>/dev/null; sleep 1; }
# 1. a stale run_tasks is restarted once (its old process gone); a fresh heartbeat is left alone
setup
now=$(date +%s); echo $((now - 1000)) > $T/root/control/loop_run_tasks; echo $now > $T/root/control/loop_heartbeat
old=$(rt_pids | head -1)
run_wd 6
ok "stale run_tasks restarted once, the old process gone" '[ -n "$old" ] && ! kill -0 $old 2>/dev/null && [ "$(grep -c . $T/root/run_tasks.starts)" = 2 ] && [ "$(rt_pids | grep -c .)" = 1 ] && [ "$(grep -c . $T/root/control/watchdog_run_tasks.times)" = 1 ] && grep -q "run_tasks stale .*restart" $T/root/logs/watchdog.log'
ok "fresh heartbeat untouched" '[ "$(grep -c . $T/root/heartbeat.starts)" = 1 ] && ! grep -q "heartbeat stale" $T/root/logs/watchdog.log'
cleanup
# 2. rate limit: 3 restarts within the window -> gave_up, no restart
setup
now=$(date +%s); echo $((now - 1000)) > $T/root/control/loop_run_tasks; echo $now > $T/root/control/loop_heartbeat
printf '%s\n' $((now - 60)) $((now - 120)) $((now - 180)) > $T/root/control/watchdog_run_tasks.times
run_wd 4
ok "rate limit: gave up after 3 restarts in the window, no restart" '[ -f $T/root/control/watchdog_run_tasks.gave_up ] && [ "$(grep -c . $T/root/run_tasks.starts)" = 1 ] && grep -q "giving up" $T/root/logs/watchdog.log'
cleanup
# 3. one watchdog only: the second exits at once (both in the background, so a failure cannot hang the test)
setup
(WD_POLL=1 bash $T/rb/scripts/watchdog.sh >> $T/root/logs/w1.log 2>&1 < /dev/null &); sleep 2
(WD_POLL=1 bash $T/rb/scripts/watchdog.sh > $T/root/logs/w2.log 2>&1 < /dev/null &); sleep 2
ok "a second watchdog exits at once" 'grep -q "WATCHDOG_START" $T/root/logs/w1.log && grep -q "already running" $T/root/logs/w2.log && [ "$(pgrep -f "$T/rb/scripts/watchdog.s[h]" | grep -c .)" = 1 ]'
cleanup
[ $fails = 0 ] && echo "WATCHDOG_TEST PASS ($n checks)" || echo "WATCHDOG_TEST FAIL ($fails of $n)"
