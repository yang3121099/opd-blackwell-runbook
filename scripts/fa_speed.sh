#!/usr/bin/env bash
# FA-SPEED probes (runbook v2.5.7c; Runming 2026-10-18 08:5x UTC: "flash-attn 加速优先级高，多测试一些，尽快推上线").
# flash-attn 2.8.3.post2 is in venv-fa (fa-build seq 103 PASS, FA-PROBE varlen PASS). One probe per GPU, each ~1-1.5 h,
# no checkpoint or weight upload, nothing of a running task touched:
#   FA-SPEED-17   per-token log-probs, SDPA vs flash_attention_2 (1.7B student, 8B teacher; fa_speed.py lp), then three
#                 3-step smokes of E10b-P5 (the longest answers): sdpa_fix4 (today's ladder: venv, 4 per micro-batch,
#                 GC=0), fa_tok (venv-fa, remove-padding, token budget 16384, GC=0), fa_fix4 (venv-fa, 4 per micro-batch)
#   FA-SPEED-4B   two 3-step smokes of E10d-P5 (4B student, GC on): sdpa_fix4, fa_tok
#   FA-SPEED-E12  three 3-step smokes of E12b-P0 (GRPO from the base 1.7B): sdpa_gc1 (today's E12b), sdpa_gc0 (GC off:
#                 E12b's own GC check), fa_gc0 (venv-fa, remove-padding, token budget, GC off)
# Per smoke: exit code, out of memory, peak memory (nvidia-smi every 5 s), steps 2+ timing segments, speed-up vs the
# first variant; report.json (fa_speed.py report) and logs to logs/<node>/fa_speed/<task>/ (private), an event, .done.
# Usage: bash fa_speed.sh <gpu> <FA-SPEED-17|FA-SPEED-4B|FA-SPEED-E12>
set -uo pipefail
G=$1; T=$2; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "FA_SPEED $(date -u +%T) $*" | tee -a $LOG; }
case $T in
  FA-SPEED-17)  VARS="sdpa_fix4 fa_tok fa_fix4";;
  FA-SPEED-4B)  VARS="sdpa_fix4 fa_tok";;
  FA-SPEED-E12) VARS="sdpa_gc1 sdpa_gc0 fa_gc0";;
  *) echo "unknown probe $T"; exit 2;;
esac
finish() {
  python3 $RUNBOOK/scripts/fa_speed.py report $D $VARS > $D/report.out 2>&1
  S=$(python3 -c "import json, sys; r = json.load(open(sys.argv[1])); v = r['variants']; b = list(v)[0]
print('lp', 'PASS' if r['lp_pass'] else ('n/a' if not any(r['lp'].values()) else 'FAIL'), 'smokes', 'ok' if r['smokes_ok'] else 'FAIL',
      ' '.join(f\"{k}={x['mean_s2plus'].get('step')}s/{x['mem_peak_mib']}MiB\" for k, x in v.items()))" $D/report.json 2> /dev/null || echo "report missing")
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/lp_*.json $D/*.rc $D/up/ 2> /dev/null
  for v in $VARS; do [ -f $D/$v/train.log ] && mkdir -p $D/up/$v && tail -c 200000 $D/$v/train.log > $D/up/$v/train.log; done
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/fa_speed/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T fa_speed_done "${1:-} $S" > /dev/null 2>&1
  step "DONE ${1:-} $S"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish "gpu $G not empty after 180 s"; sleep ${FS_WAIT_STEP:-10}
done
[ -f $OPD_ROOT/venv-fa/bin/python ] && $OPD_ROOT/venv-fa/bin/python -c "import flash_attn" 2> /dev/null || finish "venv-fa has no flash_attn"
step "start $T gpu=$G variants=$VARS"
# 1. log-probs SDPA vs flash-attn (FA-SPEED-17 only)
if [ $T = FA-SPEED-17 ]; then
  R=$(ls -t $OPD_ROOT/runs/E12b-*/rollouts/*.jsonl 2> /dev/null | head -1)
  for who in student teacher; do
    M=$([ $who = student ] && echo $STUDENT_17B || echo $TEACHER)
    [ -n "$R" ] && ( cd / && CUDA_VISIBLE_DEVICES=$G $OPD_ROOT/venv-fa/bin/python $RUNBOOK/scripts/fa_speed.py lp $M $R ${FS_SEQS:-16} ) > $D/lp_$who.json 2> $D/lp_$who.err
    step "log-prob $who sdpa vs flash: $(tail -1 $D/lp_$who.json 2> /dev/null) $( [ -s $D/lp_$who.json ] || tail -c 300 $D/lp_$who.err 2> /dev/null | tr '\n' ' ')"
  done
fi
# the 4B student on the share (as PROBE-4B)
if [ $T = FA-SPEED-4B ] && [ ! -f $OPD_ROOT/models/Qwen3-4B/.fetch_ok ]; then
  python3 $RUNBOOK/scripts/fetch_model.py Qwen/Qwen3-4B "" $OPD_ROOT/models/Qwen3-4B --task $T >> $LOG 2>&1 && touch $OPD_ROOT/models/Qwen3-4B/.fetch_ok || finish "Qwen/Qwen3-4B could not be fetched"
  export STUDENT_4B=$OPD_ROOT/models/Qwen3-4B
fi
# 2. the smokes
for V in $VARS; do
  case $V in
    sdpa_fix4) E="VENV= FIXED_MICRO_BSZ=4 GC=0";;   fa_tok) E="VENV=fa E10D_FA=1 FIXED_MICRO_BSZ= GC=0";;  fa_fix4) E="VENV=fa E10D_FA=1 FIXED_MICRO_BSZ=4 GC=0";;
    sdpa_gc1) E="VENV= E12B_FA=0 FIXED_MICRO_BSZ=4 GC=1 E12B_GC=0";;   sdpa_gc0) E="VENV= E12B_FA=0 FIXED_MICRO_BSZ=4 GC=0 E12B_GC=1";;
    fa_gc0) E="VENV=fa E12B_FA=1 FIXED_MICRO_BSZ= GC=0 E12B_GC=1";;
  esac
  case $T in FA-SPEED-17) S="run_arm.sh $G E10b-P5";; FA-SPEED-4B) S="run_arm.sh $G E10d-P5";; FA-SPEED-E12) S="run_e12.sh $G E12b-P0";; esac
  ( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/$V.mem.csv; sleep 5; done ) &
  MP=$!
  rm -rf $OPD_ROOT/runs/$T-$V $D/$V
  env $E SMOKE=1 SMOKE_STEPS=${FS_STEPS:-3} PROBE_TASK=$T-$V bash $RUNBOOK/scripts/$S > $D/$V.out 2>&1; echo $? > $D/$V.rc
  kill $MP 2> /dev/null
  mkdir -p $D/$V; cp $OPD_ROOT/runs/$T-$V/train.log $OPD_ROOT/runs/$T-$V/config.yaml $D/$V/ 2> /dev/null
  step "smoke $V ($E) exit $(cat $D/$V.rc), peak $(cut -d, -f2 $D/$V.mem.csv 2> /dev/null | sort -n | tail -1) MiB"
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
done
finish
