#!/usr/bin/env bash
# [v3.1 / v3.2] gpu_fill.sh on stubs (no GPU: nvidia-smi answers from files; the "owner's filler" is a sleep): off without the command
# or without GPU_FILL=1; a GPU empty at two looks in a row gets one filler in its own session with CUDA_VISIBLE_DEVICES and the stored
# PATH; fill_stop ends it and waits for the card; a filler that was on a card before us makes the card free and is cleared too, any
# other process is never touched; the loop fills without run_tasks and with a logs directory it cannot write; a running setup clears
# no card and a card on hold is not filled; switching off ends our fillers; run_tasks.sh's free_gpus counts a filler-only card as free.
# Run: bash scripts/test_gpu_fill.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); export OPD_ROOT=$T/root; mkdir -p $OPD_ROOT/control $OPD_ROOT/logs $T/bin $T/mem
# nvidia-smi stub: memory.used of GPU <i> from $T/mem/<i> (default 0); compute-apps from $T/apps/<i> (default none)
cat > $T/bin/nvidia-smi <<'STUB'
#!/bin/bash
i=""; q=""; while [ $# -gt 0 ]; do case $1 in -i) i=$2; shift 2;; --query-gpu=*) q=${1#--query-gpu=}; shift;; --query-compute-apps=*) q=apps; shift;; *) shift;; esac; done
case $q in memory.used) cat $TROOT/mem/$i 2> /dev/null || echo 0;; apps) cat $TROOT/apps/$i 2> /dev/null;; index) printf '0\n1\n2\n3\n';; esac
STUB
command -v setsid > /dev/null || printf '#!/bin/bash\nexec python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" "$@"\n' > $T/bin/setsid   # macOS has no setsid
chmod +x $T/bin/*; export PATH=$T/bin:$PATH TROOT=$T FILL_TERM_S=3 FILL_FREE_S=3 FILL_DIR=$T/shm
source $SRC/scripts/gpu_fill.sh
SL=$((30000 + RANDOM))   # this run's own filler: `sleep $SL` (two runs of this test at once do not see each other's)
live() { ls $FILL_DIR/*.pid 2> /dev/null | while read -r f; do fill_on $(basename $f .pid) && echo $(basename $f .pid); done | sort | paste -sd, -; }

fill_round "0 1 2 3" > /dev/null; fill_round "0 1 2 3" > /dev/null
ok "no command, no GPU_FILL: nothing starts" '[ -z "$(live)" ]'
printf 'echo "cvd=$CUDA_VISIBLE_DEVICES path=$PATH"; exec sleep %s\n' $SL > $OPD_ROOT/control/gpu_fill_secret.cmd
fill_round "0 1 2 3" > /dev/null; fill_round "0 1 2 3" > /dev/null
ok "the command without GPU_FILL=1: nothing starts" '[ -z "$(live)" ]'
echo 'export GPU_FILL=1' > $OPD_ROOT/control.env; echo "export PATH=/fillpath:$PATH" > $OPD_ROOT/control/gpu_fill_secret.env
fill_round "0 1 2 3" > /dev/null
ok "first round: marked empty, not filled yet" '[ -z "$(live)" ] && [ -f $FILL_DIR/2.empty ]'
fill_round "0 1 3" > /dev/null; sleep 1
ok "second round: the GPUs empty twice are filled, the one that was taken meanwhile is not" '[ "$(live)" = 0,1,3 ] && [ ! -f $FILL_DIR/2.empty ]'
ok "its own session, CUDA_VISIBLE_DEVICES and the stored PATH" 'grep -q "^cvd=1 path=/fillpath:" $OPD_ROOT/logs/gpu_fill_1.log && [ "$(ps -o pgid= -p $(cat $FILL_DIR/1.pid) | tr -d " ")" = "$(cat $FILL_DIR/1.pid)" ]'
fill_round "" > /dev/null; fill_round "" > /dev/null
ok "a filled GPU is not filled twice" '[ "$(pgrep -f "sleep $SL" | wc -l | tr -d " ")" = 3 ]'
P1=$(cat $FILL_DIR/1.pid); out=$(fill_stop 1,3)
ok "fill_stop ends those fillers (their whole session) and reports the cards empty" '[ "$(live)" = 0 ] && ! kill -0 -- -$P1 2> /dev/null && [[ "$out" == *"cards empty"* ]]'
fill_round "1 3" > /dev/null; fill_round "1 3" > /dev/null; echo 5000 > $T/mem/3; out=$(fill_stop 3); rc=$?
ok "a card whose memory does not come back: fill_stop says so and returns 1" '[ $rc = 1 ] && [[ "$out" == *"not back under 2 GB"* ]]'
echo 0 > $T/mem/3
# run_tasks.sh's free_gpus and the loop's own list: a filler-only card is free for the queue, and it is not empty
NGPU=4; gpu_held() { return 1; }
eval "$(sed -n '/^free_gpus() {/,/^    \[ .*-le 2048 \] && echo \$i; done; }/p' $SRC/scripts/run_tasks.sh)"
mkdir -p $T/apps; echo 4242 > $T/apps/0; echo 9000 > $T/mem/0; echo 4243 > $T/apps/1; echo 9000 > $T/mem/1; echo 777 > $T/apps/2; echo 30000 > $T/mem/2   # 0 and 1: our fillers; 2: someone's process
ok "free_gpus: filler-only cards and empty cards; fill_empty_gpus: empty cards only" '[ "$(free_gpus | paste -sd, -)" = 0,1,3 ] && [ "$(fill_empty_gpus | paste -sd, -)" = 3 ]'
# [v3.2] a filler that was on the card before us (the owner started it): known by the script's name on its command line
fill_stop 0,1 > /dev/null; rm -f $T/apps/*; for g in 0 1 2 3; do echo 0 > $T/mem/$g; done
printf 'python /own/dir/filler_train.py --duration_hours 888\n' > $OPD_ROOT/control/gpu_fill_secret.cmd
mkdir -p $T/own/dir; printf 'import time\ntime.sleep(300)\n' > $T/own/dir/filler_train.py; printf 'import time\ntime.sleep(300)\n' > $T/own/dir/real_job.py
python3 $T/own/dir/filler_train.py --duration_hours 888 & OWN=$!; python3 $T/own/dir/real_job.py & REAL=$!; sleep 0.5
echo $OWN > $T/apps/1; echo 20000 > $T/mem/1; echo $REAL > $T/apps/2; echo 20000 > $T/mem/2; printf '%s\n%s\n' $OWN $REAL > $T/apps/3; echo 20000 > $T/mem/3
ok "the owner's own filler on a card: the card is free; another process (alone or beside it): not free" 'fill_free 1 && ! fill_free 2 && ! fill_free 3 && [ "$(free_gpus | paste -sd, -)" = 0,1 ]'
( sleep 1; echo 0 > $T/mem/1 ) & out=$(fill_stop 1,2)
ok "fill_stop clears the owner's filler from its cards and never touches the other process" '! kill -0 $OWN 2> /dev/null && kill -0 $REAL 2> /dev/null'
# [v3.4] its exact command was learned before it was cleared, and that is what an empty card is filled with (the stored line says
# `python /own/dir/...`, which does not exist here: without the learned command the filler would end at once, as on 11-17)
ok "the owner's exact command is learned from his process (node-local, 0600)" 'grep -q "$T/own/dir/filler_train.py --duration_hours 888" $OPD_ROOT/control/gpu_fill_secret.seen && [ "$(stat -c %a $OPD_ROOT/control/gpu_fill_secret.seen 2> /dev/null || stat -f %Lp $OPD_ROOT/control/gpu_fill_secret.seen)" = 600 ]'
rm -f $T/apps/*; for g in 0 1 2 3; do echo 0 > $T/mem/$g; done
fill_round "3" > /dev/null; fill_round "3" > /dev/null; sleep 1
ok "an empty card is filled with the learned command, on that card" 'fill_on 3 && ps -o args= -g $(cat $FILL_DIR/3.pid) 2> /dev/null | grep -q "$T/own/dir/filler_train.py --duration_hours 888" || pgrep -f "$T/own/dir/filler_train.py --duration_hours 888" > /dev/null'
fill_stop 3 > /dev/null; pkill -f "$T/own/dir/filler_train.py" 2> /dev/null
# [v3.4] the owner replaces his filler: only its file name is told (setenv GPU_FILL_NAME), the command is learned from his process
printf 'import time\ntime.sleep(300)\n' > $T/own/dir/filler2.py; python3 $T/own/dir/filler2.py --duration_hours 888 & OWN2=$!; sleep 0.5
echo $OWN2 > $T/apps/2; echo 60000 > $T/mem/2
ok "a filler under a new name is not known as one yet" '! fill_free 2'
printf 'export GPU_FILL=1\nexport GPU_FILL_NAME=filler2.py\n' > $OPD_ROOT/control.env
ok "GPU_FILL_NAME: the card it is on is free, the old learned command no longer counts, no command is stored for it" 'fill_free 2 && ! fill_seen_ok && [ -z "$(fill_cmd_line)" ]'
out=$(fill_round "3"; fill_round "3")
ok "the loop learns the new command from his process and does not touch it; nothing is started before that" 'fill_seen_ok && kill -0 $OWN2 2> /dev/null && grep -q "$T/own/dir/filler2.py --duration_hours 888" $OPD_ROOT/control/gpu_fill_secret.seen'
sleep 1; ok "the empty card then runs the learned new command" 'fill_on 3 && pgrep -f "$T/own/dir/filler2.py --duration_hours 888" | grep -qv "^$OWN2\$"'
fill_stop 3 > /dev/null; kill $OWN2 2> /dev/null; pkill -f "$T/own/dir/filler2.py" 2> /dev/null; rm -f $T/apps/*; echo 0 > $T/mem/2
rm -f $OPD_ROOT/control/gpu_fill_secret.seen; echo 'export GPU_FILL=1' > $OPD_ROOT/control.env
kill $REAL 2> /dev/null; rm -f $T/apps/*; for g in 0 1 2 3; do echo 0 > $T/mem/$g; done
printf 'echo "cvd=$CUDA_VISIBLE_DEVICES path=$PATH"; exec sleep %s\n' $SL > $OPD_ROOT/control/gpu_fill_secret.cmd
# [v3.2] the loop on its own: no run_tasks, and a logs directory that cannot be written (a full disk)
rm -rf $OPD_ROOT/logs; : > $OPD_ROOT/logs   # a file where the directory should be: every log write fails
EXP=$(for g in 0 1 2 3; do fill_held $g || echo $g; done | paste -sd, -)   # (a stray task.sh of another test on this machine would hold its GPU number)
( cd $T && RUNBOOK=$SRC CKPT_ROOT=$T/ck FILL_POLL_S=1 exec bash $SRC/scripts/gpu_fill.sh loop > /dev/null 2>&1 ) & echo $! > $T/loop.pid; sleep 6
ok "the loop fills the empty cards by itself, with logs it cannot write" '[ -n "$EXP" ] && [ "$(live)" = "$EXP" ]'
kill $(cat $T/loop.pid) 2> /dev/null; sleep 0.5
ok "the loop is gone and the fillers outlive it" '! kill -0 $(cat $T/loop.pid) 2> /dev/null && [ "$(live)" = "$EXP" ]'
rm -f $OPD_ROOT/logs; mkdir -p $OPD_ROOT/logs; fill_stop 0,1,2,3 > /dev/null
fill_round "1 2 3" > /dev/null; fill_round "1 2 3" > /dev/null; sleep 0.5; echo 4242 > $T/apps/0
# setup running: its card (VERIFY_GPU, default 0) is left empty, a filler on it is ended
rm -f $T/apps/* ; for g in 0 1 2 3; do echo 0 > $T/mem/$g; done; fill_round "0" > /dev/null; fill_round "0" > /dev/null; sleep 0.5
# [v3.3] a running setup clears nothing (11-17: it ended the owner's filler for the whole install); a card is kept empty only on request
mkdir -p $T/scripts; printf '#!/bin/bash\nsleep 30\n' > $T/scripts/verify.sh; bash $T/scripts/verify.sh & VP=$!; sleep 0.5
fill_round "" > /dev/null; fill_round "" > /dev/null; sleep 0.5
ok "a running setup clears no card: every filler stays" '[ "$(live)" = 0,1,2,3 ]'
kill $VP 2> /dev/null
fill_hold 0 60; fill_stop 0 > /dev/null; fill_round "0" > /dev/null; fill_round "0" > /dev/null; sleep 0.5
ok "a card on hold (a check that needs it empty) is not filled, and is not in the loop's list" '[ "$(live)" = 1,2,3 ] && [ -z "$(fill_empty_gpus)" ]'
fill_unhold 0; fill_round "0" > /dev/null; fill_round "0" > /dev/null; sleep 0.5
ok "the hold lifted: the card is filled again" '[ "$(live)" = 0,1,2,3 ]'
echo 'export GPU_FILL=0' > $OPD_ROOT/control.env; fill_round "" > /dev/null
sleep 1
ok "GPU_FILL=0: every filler ends" '[ -z "$(live)" ] && [ "$(pgrep -f "sleep $SL" | wc -l | tr -d " ")" = 0 ]'
ok "the command file's name is one snapshot / upload refuse (SECRET_NAME)" 'python3 -c "import re, sys; sys.exit(0 if re.search(r\"token|secret|credential\", \"gpu_fill_secret.cmd\", re.I) else 1)"'
pkill -f "sleep $SL" 2> /dev/null; rm -rf $T
[ $fails = 0 ] && echo "GPU_FILL_TEST PASS ($n checks)" || { echo "GPU_FILL_TEST FAIL ($fails of $n)"; exit 1; }
