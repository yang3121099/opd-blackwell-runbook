#!/usr/bin/env bash
# test_gpu_fill_shared.sh ([v3.10b], the 11-18 machine replayed): the owner's filler is ONE process on every card, started from its
# own directory as `python <name>` under his own interpreter. With GPU_FILL_NAME: known by the bare file name, its command learned
# from the process (FILL_PROC stands in for /proc: real interpreter path, its PATH / CONDA_PREFIX, its directory), the restart
# self-test passes, fill_stop ends it for one card and the other card gets its own instance at once.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); fails=0; ok() { if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
 export OPD_ROOT=$T/root; mkdir -p $OPD_ROOT/control $OPD_ROOT/logs $T/own/dir $T/bin $T/mem $T/apps
cat > $T/bin/nvidia-smi <<'STUB'
#!/bin/bash
i=""; q=""; while [ $# -gt 0 ]; do case $1 in -i) i=$2; shift 2;; --query-gpu=*) q=${1#--query-gpu=}; shift;; --query-compute-apps=*) q=apps; shift;; *) shift;; esac; done
case $q in memory.used) cat $TROOT/mem/$i 2> /dev/null || echo 0;; apps) cat $TROOT/apps/$i 2> /dev/null;; index) printf '0\n1\n';; esac
STUB
printf '#!/bin/bash\nexec python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" "$@"\n' > $T/bin/setsid
cat > $T/bin/timeout <<'STUB'
#!/usr/bin/env python3
import subprocess, sys
a = sys.argv[1:]
if a[0] == "-k": a = a[2:]
try: sys.exit(subprocess.run(a[1:], timeout=float(a[0])).returncode)
except subprocess.TimeoutExpired: sys.exit(124)
STUB
chmod +x $T/bin/*; export PATH=$T/bin:$PATH TROOT=$T FILL_DIR=$T/shm FILL_SELFTEST_S=2 FILL_PROC=$T/proc FILL_TERM_S=3 FILL_FREE_S=3
printf 'import time, os\nopen(os.environ["TROOT"] + "/ran_" + os.environ.get("CUDA_VISIBLE_DEVICES", "x"), "w").write(os.environ.get("CONDA_PREFIX", "") + " " + os.getcwd())\ntime.sleep(300)\n' > $T/own/dir/gpu_consum.py
# the owner starts his filler from its own directory with a relative script name, under his own interpreter: one process, both cards
( cd $T/own/dir && exec python3 gpu_consum.py --hours 888 ) & OWN=$!; sleep 0.5
echo $OWN > $T/apps/0; echo $OWN > $T/apps/1; echo 1400 > $T/mem/0; echo 1400 > $T/mem/1
mkdir -p $T/proc/$OWN; printf 'python\0gpu_consum.py\0--hours\0888\0' > $T/proc/$OWN/cmdline; ln -s $T/own/dir $T/proc/$OWN/cwd; ln -s $(command -v python3) $T/proc/$OWN/exe
printf 'PATH=%s\0FOO=1\0CONDA_PREFIX=/opt/conda/envs/x\0' "$PATH" > $T/proc/$OWN/environ
printf 'export GPU_FILL=1\nexport GPU_FILL_NAME=gpu_consum.py\n' > $OPD_ROOT/control.env
printf 'python /somewhere/train.py --duration_hours 888\n' > $OPD_ROOT/control/gpu_fill_secret.cmd     # the stored line names another script
source $SRC/scripts/gpu_fill.sh
ok "known by the bare file name; the card is free; the stored line (another script) is not taken for its command" '[ "$(fill_sig)" = gpu_consum.py ] && [ "$(fill_app_pids 0)" = "$OWN" ] && fill_free 0 && [ -z "$(fill_cmd_line)" ]'
out=$(fill_round "")
ok "the loop learns the command (interpreter by its real path, the filler's own environment and directory)" 'fill_seen_ok && grep -q "export CONDA_PREFIX=/opt/conda/envs/x; " $OPD_ROOT/control/gpu_fill_secret.seen && grep -q "cd .*own/dir 2> /dev/null; /.*python[0-9.]* gpu_consum.py --hours 888" $OPD_ROOT/control/gpu_fill_secret.seen'
ok "the restart self-test ran it beside the filler and it stayed alive" '[[ "$out" == *"FILL_SELFTEST"*"ok"* ]] && fill_selftest_ok && grep -q "^/opt/conda/envs/x .*own/dir$" $T/ran_0'
( sleep 2.5; rm -f $T/apps/0 $T/apps/1; echo 0 > $T/mem/0; echo 0 > $T/mem/1 ) &
if pgrep -f "task\.sh ([0-9]+,)*1(,[0-9]+)* " > /dev/null; then echo "  SKIP the refill check: a task.sh of another test holds GPU 1 on this machine"; fill_stop 0 > /dev/null
else out=$(fill_stop 0); sleep 1
  ok "fill_stop for card 0 ends the one-process filler; card 1 gets its own instance at once, in his directory with his environment" '! kill -0 $OWN 2> /dev/null && [[ "$out" == *"FILL_START"*"gpu=1"* ]] && fill_on 1 && grep -q "^/opt/conda/envs/x .*own/dir$" $T/ran_1'; fi
fill_stop 1 > /dev/null 2>&1; kill $OWN 2> /dev/null; pkill -f "$T/own/dir" 2> /dev/null; rm -rf $T
[ $fails = 0 ] && echo "GPU_FILL_SHARED_TEST PASS" || { echo "GPU_FILL_SHARED_TEST FAIL ($fails)"; exit 1; }
