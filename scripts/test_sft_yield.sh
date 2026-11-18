#!/usr/bin/env bash
# test_sft_yield.sh ([v3.12]): SFT-WARM gives way by itself - run_tasks.sh's free_gpus counts a card with only its worker (and the
# owner's filler) as free, sft_yield ends the workers on the cards a task takes and leaves the others, and sft_warm.sh's own
# bookkeeping takes a worker that gave way for one that ended well.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
export OPD_ROOT=$T/root; mkdir -p $OPD_ROOT/control $OPD_ROOT/logs $OPD_ROOT/runs/SFT-WARM $T/bin $T/mem $T/apps $T/own
cat > $T/bin/nvidia-smi <<'STUB'
#!/bin/bash
i=""; q=""; while [ $# -gt 0 ]; do case $1 in -i) i=$2; shift 2;; --query-gpu=*) q=${1#--query-gpu=}; shift;; --query-compute-apps=*) q=apps; shift;; *) shift;; esac; done
case $q in memory.used) cat $TROOT/mem/$i 2> /dev/null || echo 0;; apps) cat $TROOT/apps/$i 2> /dev/null;; index) printf '0\n1\n2\n3\n';; esac
STUB
chmod +x $T/bin/*; export PATH=$T/bin:$PATH TROOT=$T FILL_DIR=$T/shm
printf 'import time\ntime.sleep(300)\n' > $T/own/hold_cards.py; printf 'import time\ntime.sleep(300)\n' > $T/own/other_job.py
( cd $T/own && exec python3 hold_cards.py ) & OWN=$!; python3 $T/own/other_job.py & OTHER=$!
W=""; for g in 1 2 3; do sleep 300 & echo $! > $OPD_ROOT/runs/SFT-WARM/gpu$g.pid; W="$W $!"; done; sleep 0.3
for g in 0 1 2 3; do echo $OWN > $T/apps/$g; echo 1400 > $T/mem/$g; done
for g in 1 2; do cat $OPD_ROOT/runs/SFT-WARM/gpu$g.pid >> $T/apps/$g; done
{ cat $OPD_ROOT/runs/SFT-WARM/gpu3.pid; echo $OTHER; } >> $T/apps/3            # card 3: the worker, the filler and someone's job
printf 'export GPU_FILL=1\nexport GPU_FILL_NAME=hold_cards.py\n' > $OPD_ROOT/control.env
source $SRC/scripts/gpu_fill.sh
NGPU=4; gpu_held() { return 1; }
eval "$(sed -n '/^sft_pid() {/,/^free_gpus() {/p' $SRC/scripts/run_tasks.sh | sed '$d')"
eval "$(sed -n '/^free_gpus() {/,/^    \[ .*-le 2048 \] && echo \$i; done; }/p' $SRC/scripts/run_tasks.sh)"
ok "filler-only and worker+filler cards are free; a card with someone else's job is not" '[ "$(free_gpus | paste -sd, -)" = 0,1,2 ]'
printf 'export GPU_FILL=1\n' > $OPD_ROOT/control.env
ok "a filler the node does not know by name keeps the worker's card busy (nothing is assumed about an unknown process)" '[ -z "$(free_gpus | paste -sd, -)" ]'
printf 'export GPU_FILL=1\nexport GPU_FILL_NAME=hold_cards.py\n' > $OPD_ROOT/control.env
P1=$(cat $OPD_ROOT/runs/SFT-WARM/gpu1.pid); P2=$(cat $OPD_ROOT/runs/SFT-WARM/gpu2.pid); P3=$(cat $OPD_ROOT/runs/SFT-WARM/gpu3.pid)
out=$(sft_yield 0,1)
ok "sft_yield ends the worker on the task's cards only, and says so" '! kill -0 $P1 2> /dev/null && kill -0 $P2 2> /dev/null && kill -0 $P3 2> /dev/null && kill -0 $OWN 2> /dev/null && [ -f $OPD_ROOT/runs/SFT-WARM/yield_gpu1 ] && [ ! -e $OPD_ROOT/runs/SFT-WARM/yield_gpu0 ] && [[ "$out" == *"SFT_YIELD"*"gpu=0,1"* ]]'
ok "a card without a worker: nothing to do, nothing printed" '[ -z "$(sft_yield 0)" ]'
# sft_warm.sh's end: a worker that gave way counts as ended well, one that died does not
R=$OPD_ROOT/runs/SFT-WARM; GP=1,2
wait_rc() { local rc=0 g; for g in $(echo $GP | tr , ' '); do wait $(cat $R/gpu$g.pid) 2> /dev/null || { [ -f $R/yield_gpu$g ] || rc=1; }; done; echo $rc; }
( sleep 0.5; : > $R/yield_gpu2; kill $P2 ) & wait_rc > $T/rc.out 2> /dev/null
ok "every worker gave way: the task ends well" '[ "$(cat $T/rc.out)" = 0 ]'
rm -f $R/yield_gpu*; sleep 300 & P4=$!; echo $P4 > $R/gpu1.pid; ( sleep 0.3; kill $P4 ) & GP=1; wait_rc > $T/rc.out 2> /dev/null
ok "a worker that died by itself: the task has failed" '[ "$(cat $T/rc.out)" = 1 ]'
kill $P4 2> /dev/null
kill $OWN $OTHER $P3 2> /dev/null; rm -rf $T
[ $fails = 0 ] && echo "SFT_YIELD_TEST PASS ($n checks)" || { echo "SFT_YIELD_TEST FAIL ($fails of $n)"; exit 1; }
