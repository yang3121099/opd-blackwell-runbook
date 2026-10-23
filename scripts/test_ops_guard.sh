#!/usr/bin/env bash
# [v2.5.9z12] Stub tests of run_tasks.sh's disk guard and reserve line (no GPU, network or HF; one DRY=1 dispatch pass each):
#  - while control.py disk_low names a disk, trainings (E10b/c/d, E12b/c/d, E15, SFT) are held (DISK_GUARD, RESERVE held=[...]);
#    readings (EVAL), probes and the CPU lane still start; with no disk low the training starts;
#  - RESERVE ready=<n> free=<m>: GPU tasks that passed every check but found no GPU, against the GPUs left free;
#  - the loop calls `control.py auto_reruns <queue>` (not in DRY) before it dispatches, and reserve_note after.
# Two GPUs (the nvidia-smi stub lists 0 and 1).
# Usage: bash scripts/test_ops_guard.sh -> OPS_GUARD_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {   # $1 = the queue's stage line; NGPU=2
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/tasks $T/root/runs $T/root/logs $T/root/control $T/bin
  cp $SRC/scripts/run_tasks.sh $T/rb/scripts/
  printf 'export OPD_ROOT=%s/root RUNBOOK=%s/rb CDIR=%s/root/control NGPU=2\n' $T $T $T > $T/rb/scripts/env.sh
  printf 'import sys, os\nopen(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\nif sys.argv[1:2] == ["disk_low"]: print(os.environ.get("FAKE_DISK_LOW", ""))\nif sys.argv[1:2] == ["readings_disk_low"]: print(os.environ.get("FAKE_READ_LOW", ""))\n' > $T/rb/scripts/control.py
  printf 'import sys; sys.exit(1)\n' > $T/rb/scripts/remote.py
  printf '#!/usr/bin/env bash\ncase "$*" in *compute-apps*) ;; *query-gpu=index*) printf "0\\n1\\n";; *) echo 0;; esac\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  mkdir -p $T/root/venv/bin; : > $T/root/venv/bin/activate
  printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; chmod +x $T/bin/setsid
  printf '#!/usr/bin/env bash\n[ "$1" = -c ] && { python3 -c "import os, sys; print(int(os.path.getmtime(sys.argv[1])))" "$3"; exit; }\nexec /usr/bin/stat "$@"\n' > $T/bin/stat; chmod +x $T/bin/stat
  printf 'version: 2\n%s\n#eval EVAL-a a local:models/Qwen3-1.7B math500 4 0.6\n#eval EVAL-b b local:models/Qwen3-1.7B math500 4 0.6\n' "$1" > $T/rb/tasks/queue.txt
  export CALLS=$T/calls PATH=$T/bin:$PATH; : > $CALLS
}
dry() { ( cd $T && DRY=1 QUEUE_FILE=$T/rb/tasks/queue.txt bash $T/rb/scripts/run_tasks.sh 2>&1 ) > $T/dry.out; }
# 1. the work disk low: the training is held, the reading and the probe start
setup "g1       open   E10b-P5 EVAL-a PROBE-X"; FAKE_DISK_LOW=work dry
ok "disk low: DISK_GUARD work; E10b-P5 held (not started, in held=[]); EVAL-a and PROBE-X start" 'grep -q "^DISK_GUARD work$" $T/dry.out && ! grep -q "WOULD_START .* E10b-P5 " $T/dry.out && grep -q "WOULD_START .* EVAL-a " $T/dry.out && grep -q "WOULD_START .* PROBE-X " $T/dry.out && grep -q "^RESERVE ready=0 free=0 tasks=\[\] held=\[E10b-P5\]$" $T/dry.out'
ok "DRY: no disk_guard state, no event, no auto_reruns, no reserve_note" '[ ! -f $T/root/control/disk_guard ] && ! grep -q "^control event node disk_guard" $CALLS && ! grep -q "^control auto_reruns" $CALLS && ! grep -q "^control reserve_note" $CALLS'
rm -rf $T
# 2. no disk low: the training starts
setup "g1       open   E10b-P5 EVAL-a"; FAKE_DISK_LOW= dry
ok "disks fine: no DISK_GUARD; E10b-P5 and EVAL-a start; reserve 0 / 0" '! grep -q "^DISK_GUARD" $T/dry.out && grep -q "WOULD_START .* E10b-P5 " $T/dry.out && grep -q "WOULD_START .* EVAL-a " $T/dry.out && grep -q "^RESERVE ready=0 free=0 tasks=\[\] held=\[\]$" $T/dry.out'
rm -rf $T
# 3. the reserve: two GPUs, three ready tasks -> one ready in reserve; a lone task -> one GPU left free
setup "g1       open   PROBE-X EVAL-a EVAL-b"; dry
ok "3 ready on 2 GPUs: RESERVE ready=1 free=0 tasks=[EVAL-b]" 'grep -q "^RESERVE ready=1 free=0 tasks=\[EVAL-b\] held=\[\]$" $T/dry.out'
rm -rf $T
setup "g1       open   EVAL-a"; dry
ok "1 ready on 2 GPUs: RESERVE ready=0 free=1" 'grep -q "^RESERVE ready=0 free=1 tasks=\[\] held=\[\]$" $T/dry.out'
rm -rf $T
# 3b. [v2.5.9z23] an EVAL task with a gpqa row starts only once GPQA's #fetch copy is in (mc_data_ok)
setup "g1       open   EVAL-g"; printf '#eval EVAL-g g_mm local:models/Qwen3-1.7B mmlupro1k 1 0 4096\n#eval EVAL-g g_gp local:models/Qwen3-1.7B gpqa 4 0.6 16384\n' >> $T/rb/tasks/queue.txt
mkdir -p $T/root/data/hf/TIGER-Lab__MMLU-Pro; : > $T/root/data/hf/TIGER-Lab__MMLU-Pro/.fetch_ok; dry
ok "[v2.5.9z23] gpqa data missing: EVAL-g not started (nor counted ready)" '! grep -q "WOULD_START .* EVAL-g " $T/dry.out && grep -q "^RESERVE ready=0 free=2 " $T/dry.out'
mkdir -p $T/root/data/hf/Idavidrein__gpqa; : > $T/root/data/hf/Idavidrein__gpqa/.fetch_ok; dry
ok "[v2.5.9z23] both datasets in: EVAL-g starts" 'grep -q "WOULD_START .* EVAL-g " $T/dry.out'
rm -rf $T
# 4. the loop's own calls (outside DRY): auto_reruns before the dispatch, reserve_note after it
ok "run_tasks.sh calls auto_reruns before the dispatch and reserve_note after it" 'a=$(grep -n "control.py auto_reruns \"\$Q\"" $SRC/scripts/run_tasks.sh | cut -d: -f1); b=$(grep -n "^  FREE=(\$(free_gpus))" $SRC/scripts/run_tasks.sh | cut -d: -f1); c=$(grep -n "control.py reserve_note \$READY" $SRC/scripts/run_tasks.sh | cut -d: -f1); [ -n "$a" ] && [ -n "$b" ] && [ -n "$c" ] && [ $a -lt $b ] && [ $b -lt $c ]'
# [v2.5.9z69] the readings' disk guard: below READINGS_DISK_MIN_GB no new EVAL starts (READINGS_GUARD, READINGS_HELD); trainings and
# probes are not held by it; with the disk fine the reading starts
setup "g1       open   E10b-P5 EVAL-a PROBE-X"; FAKE_READ_LOW=work dry
ok "[v2.5.9z69] readings guard: EVAL-a held (READINGS_HELD [EVAL-a]); E10b-P5 and PROBE-X not held by it" 'grep -q "^READINGS_GUARD work$" $T/dry.out && grep -q "^READINGS_HELD \[EVAL-a\]$" $T/dry.out && ! grep -q "WOULD_START .* EVAL-a " $T/dry.out && ! grep -q "^DISK_GUARD" $T/dry.out'
rm -rf $T
setup "g1       open   EVAL-a"; FAKE_READ_LOW= dry
ok "[v2.5.9z69] readings guard off: no READINGS_GUARD / READINGS_HELD lines, nothing held by it" '! grep -q "^READINGS_GUARD" $T/dry.out && ! grep -q "^READINGS_HELD" $T/dry.out'
rm -rf $T
[ $fails = 0 ] && echo "OPS_GUARD_TEST PASS ($n checks)" || echo "OPS_GUARD_TEST FAIL ($fails of $n)"
