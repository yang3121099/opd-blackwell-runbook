#!/usr/bin/env bash
# [v2.5.9z71] after the switch of control/ and logs/ the loops' log fds point at the new paths (ops-lead's test ③): run_tasks.sh's
# ctrl_reopen reopens its own log there; ensure_loop restarts a loop whose .sha the switch removed, and the new copy writes on
# the checkpoint disk while the old file stops growing; the weights replay restarts only between two uploads. No GPU, network
# or HF. Usage: bash scripts/test_ctrl_switch.sh -> CTRL_SWITCH_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); export OPD_ROOT=$T/root CKPT_ROOT=$T/ckpt RUNBOOK=$T/rb PATH=$T/bin:$PATH
mkdir -p $OPD_ROOT/control $OPD_ROOT/logs $CKPT_ROOT $RUNBOOK/scripts $T/bin $T/x/scripts
printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; chmod +x $T/bin/setsid
command -v sha1sum > /dev/null || { printf '#!/usr/bin/env bash\nshasum -a 1 "$@"\n' > $T/bin/sha1sum; chmod +x $T/bin/sha1sum; }
printf 'import sys\n' > $RUNBOOK/scripts/control.py   # ensure_loop's events: nothing
printf 'while true; do echo "tick $$"; sleep 0.2; done\n' > $RUNBOOK/scripts/watchdog.sh
CDIR=$OPD_ROOT/control
FN=$T/fns.sh; sed -n '/^ensure_loop() {/,/; }$/p;/^ctrl_reopen() {/,/^}$/p' $SRC/scripts/run_tasks.sh > $FN; echo "CDIR=$CDIR" >> $FN
ok "the functions extracted" 'grep -q "^ensure_loop() {" $FN && grep -q "^ctrl_reopen() {" $FN'
source $FN
switch() { python3 -c "import sys; sys.path.insert(0, '$SRC/scripts'); import control as C; C.event = lambda *a, **k: None; print(C.disk_fallback(check_dev=False, free_fn=lambda p: 5000, stamp='$1'))"; }
# 1. before: the watchdog loop and a run_tasks-like loop write to logs/ on the work disk
ensure_loop watchdog; sleep 0.6
bash -c "source $FN; exec >> $OPD_ROOT/logs/run_tasks.log 2>&1; echo before; while [ ! -f $T/go ]; do sleep 0.1; done; ctrl_reopen; echo after; touch $T/reopened" &
sleep 0.4
ok "before the switch the loop writes on the work disk" 'grep -q tick $OPD_ROOT/logs/watchdog.log && [ ! -L $OPD_ROOT/logs ]'
# 2. the switch
r=$(switch S1)
ok "switched_on" '[ "$r" = switched_on ] && [ -L $OPD_ROOT/control ] && [ -L $OPD_ROOT/logs ]'
NEW=$(cd $OPD_ROOT/logs && pwd -P); OLD=$OPD_ROOT/logs.pre_fallback_S1
ok "the new logs/ is on the checkpoint disk" '[[ "$NEW" == "$(cd $CKPT_ROOT && pwd -P)"/opd_fallback/logs.S1 ]]'
ok "the switch asked for the reopen and removed the watchdog's sha" '[ -f $CDIR/ctrl_reopen ] && [ ! -f $CDIR/watchdog.sha ]'
# 3. run_tasks.sh's own log reopened at the new path
touch $T/go; for i in $(seq 50); do [ -f $T/reopened ] && break; sleep 0.1; done
ok "run_tasks reopened: 'after' on the new path, not the old" 'grep -q "^after$" $NEW/run_tasks.log && ! grep -q "^after$" $OLD/run_tasks.log && grep -q "^before$" $OLD/run_tasks.log'
ok "its reopen line names the new control/" 'grep -q "^CTRL_REOPEN .* control -> .*opd_fallback/control.S1" $NEW/run_tasks.log'
ok "ctrl_reopen removed its marker" '[ ! -f $CDIR/ctrl_reopen ]'
# 4. ensure_loop restarts the watchdog: the new copy writes on the new path, the old file stops
old_pid=$(pgrep -f "scripts/watchdog.s[h]" | head -1)
ensure_loop watchdog; sleep 0.8
new_pid=$(pgrep -f "scripts/watchdog.s[h]" | head -1)
ok "a new watchdog process" '[ -n "$new_pid" ] && [ "$new_pid" != "$old_pid" ] && ! kill -0 $old_pid 2> /dev/null'
ok "the new one writes on the checkpoint disk" 'grep -q "tick $new_pid" $NEW/watchdog.log && ! grep -q "tick $new_pid" $OLD/watchdog.log'
s1=$(wc -c < $OLD/watchdog.log); m1=$(wc -c < $NEW/watchdog.log); sleep 0.6; s2=$(wc -c < $OLD/watchdog.log); m2=$(wc -c < $NEW/watchdog.log)
ok "the old file stopped growing, the new one grows" '[ $s1 = $s2 ] && [ $m2 -gt $m1 ]'
ok "the sha written again" '[ -f $CDIR/watchdog.sha ]'
# 5. the weights replay: restarted only between two uploads
echo abc > $CDIR/weights_replay.sha; echo S1 > $CDIR/ctrl_reopen_wr   # (the reopen in 3 found no upload and took its marker)
printf 'import time\ntime.sleep(2)\n' > $T/x/scripts/weights_replay.py; python3 $T/x/scripts/weights_replay.py & wr=$!
sleep 0.3; ctrl_reopen
ok "an upload running: the replay's sha kept" '[ -f $CDIR/weights_replay.sha ] && [ -f $CDIR/ctrl_reopen_wr ]'
wait $wr; ctrl_reopen
ok "between two uploads: its sha removed (ensure_loop restarts it)" '[ ! -f $CDIR/weights_replay.sha ] && [ ! -f $CDIR/ctrl_reopen_wr ]'
# 6. the rollback: setenv CTRL_ON_CKPT=0 brings them back; the loops reopen again
echo "export CTRL_ON_CKPT=0" > $OPD_ROOT/control.env
r=$(switch S2)
ok "rollback: switched_off, real directories again" '[ "$r" = switched_off ] && [ ! -L $OPD_ROOT/control ] && [ -d $OPD_ROOT/control ] && [ ! -L $OPD_ROOT/logs ] && [ -f $CDIR/ctrl_reopen ]'
ensure_loop watchdog; sleep 0.6; p3=$(pgrep -f "scripts/watchdog.s[h]" | head -1)
ok "after the rollback the loop writes on the work disk again" 'grep -q "tick $p3" $OPD_ROOT/logs/watchdog.log && [ "$(cd $OPD_ROOT/logs && pwd -P)" = "$(cd $OPD_ROOT && pwd -P)/logs" ]'
for p in $(pgrep -f "scripts/watchdog.s[h]"); do kill $p 2> /dev/null; done
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "CTRL_SWITCH_TEST PASS ($n checks)" || echo "CTRL_SWITCH_TEST FAIL ($fails of $n)"
