#!/usr/bin/env bash
# SPEED-PROBE (runbook v2.5.6o, rules 2.5.12; Runming 2026-10-18 via the reviewer: speed items first, a GPU for them is
# worth it). Two items of the B300 speed list (v2.5.4 decisions b and e), on ONE GPU, ~1 h, no checkpoint or upload of
# weights, nothing of a running task touched:
#   1. use_fused_kernels (verl, torch backend; type B: same maths, other accumulation): per-token log-probs of 16 rollout
#      sequences (the newest E12b rollouts on this node) through the plain and the fused path, for the student
#      ($STUDENT_17B) and the teacher ($TEACHER); pass = max |diff| < 1e-2 and mean |diff| < 1e-3
#   2. four 3-step smokes of the E10b-P2 configuration (run_arm.sh SMOKE=1, this node's current batching settings and
#      VENV): base, fused (FUSED=1), gc0 (GC=0: no gradient checkpointing in the actor), fused_gc0; nvidia-smi samples
#      this GPU every 5 s (peak); timing segments from train.log (steps 2-3, step 1 is warm-up)
#   3. report.json (speed_probe.py report) + logs to logs/<node>/speed_probe/ in the private repository, an event, .done
# A failed item is a result: the task ends with .done either way, and the event says which items passed.
# Usage: bash speed_probe.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=SPEED-PROBE; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "SPEED_PROBE $(date -u +%T) $*" | tee -a $LOG; }
finish() {
  python3 $RUNBOOK/scripts/speed_probe.py report $D > $D/report.out 2>&1
  S=$(python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print('fused', 'PASS' if r['fused_pass'] else 'FAIL', 'gc0', 'PASS' if r['gc0_pass'] else 'FAIL', 'fused_gc0', 'PASS' if r['fused_gc0_pass'] else 'FAIL')" $D/report.json 2> /dev/null || echo "report missing")
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/lp_*.json $D/*.rc $D/up/ 2> /dev/null
  for v in base fused gc0 fused_gc0; do [ -f $D/$v/train.log ] && mkdir -p $D/up/$v && tail -c 200000 $D/$v/train.log > $D/up/$v/train.log; done
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/speed_probe >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T speed_probe_done "${1:-} $S" > /dev/null 2>&1
  step "DONE ${1:-} $S"; touch $D/.done; exit 0
}
for _w in $(seq 19); do   # the previous task's vLLM can take tens of seconds to release its memory
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish "gpu $G not empty after 180 s"; sleep ${SP_WAIT_STEP:-10}
done
case ${VENV:-} in fa) PY=$OPD_ROOT/venv-fa/bin/python;; *) PY=$OPD_ROOT/venv/bin/python;; esac
step "start gpu=$G python=$PY FIXED_MICRO_BSZ=${FIXED_MICRO_BSZ:-} VENV=${VENV:-}"

# 1. per-token log-probs, plain vs fused
R=$(ls -t $OPD_ROOT/runs/E12b-*/rollouts/*.jsonl 2> /dev/null | head -1)
if [ -z "$R" ]; then step "no E12b rollouts on this node: log-prob check skipped"
else
  for who in student teacher; do
    M=$([ $who = student ] && echo $STUDENT_17B || echo $TEACHER)
    ( cd $OPD_ROOT/G-OPD/verl && PYTHONPATH=$OPD_ROOT/G-OPD/verl:${PYTHONPATH:-} CUDA_VISIBLE_DEVICES=$G $PY $RUNBOOK/scripts/speed_probe.py lp $M $R ${SP_SEQS:-16} ) > $D/lp_$who.json 2> $D/lp_$who.err
    step "log-prob $who: $(cat $D/lp_$who.json 2> /dev/null | tail -1) $( [ -s $D/lp_$who.json ] || tail -c 300 $D/lp_$who.err | tr '\n' ' ')"
  done
fi

# 2. four 3-step smokes
for V in base fused gc0 fused_gc0; do
  case $V in base) F=0; GCV=1;; fused) F=1; GCV=1;; gc0) F=0; GCV=0;; fused_gc0) F=1; GCV=0;; esac
  ( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/$V.mem.csv; sleep 5; done ) &
  MP=$!
  rm -rf $OPD_ROOT/runs/$T-$V $D/$V
  FUSED=$F GC=$GCV SMOKE=1 PROBE_TASK=$T-$V bash $RUNBOOK/scripts/run_arm.sh $G E10b-P2 > $D/$V.out 2>&1; echo $? > $D/$V.rc
  kill $MP 2> /dev/null
  mkdir -p $D/$V; cp $OPD_ROOT/runs/$T-$V/train.log $OPD_ROOT/runs/$T-$V/config.yaml $D/$V/ 2> /dev/null
  step "smoke $V exit $(cat $D/$V.rc), peak $(cut -d, -f2 $D/$V.mem.csv 2> /dev/null | sort -n | tail -1) MiB"
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true   # release this smoke's engines before the next
done
finish
