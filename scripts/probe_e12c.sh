#!/usr/bin/env bash
# PROBE-E12C (runbook v2.5.8a, Q-023; reviewer 02:5x UTC 10-20: "先 3 步 smoke"): the gate of the 0.6B GRPO ladder E12c.
# On ONE GPU a 3-step smoke of E12c-P0 with the node's settings (no checkpoint, upload, validation; nvidia-smi every 5 s);
# report.json: exit code, out of memory or not, peak memory, the last step, the start it loaded. PASS = exit 0, no OOM,
# peak < 260 GB and step 3 reached. run_e12.sh starts E12c-P0 / E12c-P5 only after a PASS. Usage: bash probe_e12c.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-E12C; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_E12C $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" "${last:-}" "${start:-}" <<'PY'
import json, os, sys
d, res, why, last, start = sys.argv[1:6]
r = {"result": res, "reason": why, "last_step": int(last) if last.isdigit() else None, "start": start or None}
try:
    vals = [float(x.split(",")[1]) for x in open(os.path.join(d, "mem.csv")) if "," in x]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_e12c >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_e12c_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PEC_WAIT_STEP:-10}
done
step "start gpu=$G FIXED_MICRO_BSZ=${FIXED_MICRO_BSZ:-} VENV=${VENV:-}"
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PEC_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_e12.sh $G E12c-P0 > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
last=$(grep -oE "(^|[ )])step:[0-9]+ - " $D/smoke/train.log 2> /dev/null | tail -1 | grep -oE "[0-9]+")
start=$(sed -n 's/^start: //p' $D/smoke/config.yaml 2> /dev/null | head -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PEC_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
[ "${last:-0}" -ge ${PEC_STEPS:-3} ] || finish FAIL "the smoke stopped at step ${last:-none} of ${PEC_STEPS:-3}"
finish PASS "${PEC_STEPS:-3}-step smoke of E12c-P0 ok, peak ${peak} MiB, start $start"
