#!/usr/bin/env bash
# PROBE-P5SELF (runbook v2.5.7v, Q-017; reviewer 08:2x UTC): the gate of E10b-P5self (P5 with the student's own base as
# the teacher). On ONE GPU a 3-step smoke of E10b-P5self with the node's settings (no checkpoint, upload, validation;
# nvidia-smi every 5 s); report.json: exit code, out of memory or not, peak memory, steps 2-3 mean per timing segment,
# the projected hours for 150 steps; PASS = exit 0, no OOM, peak < 260 GB; to logs/<node>/probe_p5self/ (private), an
# event, .done. run_arm.sh starts E10b-P5self only after a PASS. Usage: bash probe_p5self.sh <gpu> [4b]
# [v2.5.8e] (reviewer 15:5x UTC 10-20) with 4b: PROBE-P5SELF-4B, the gate of E10d-P5self (the 4B student, its own base as
# the teacher), report to logs/<node>/probe_p5self_4b/. Both: FAIL unless the smoke's teacher is the student itself.
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-P5SELF; ARMT=E10b-P5self; UPD=probe_p5self
[ "${2:-}" = 4b ] && { T=PROBE-P5SELF-4B; ARMT=E10d-P5self; UPD=probe_p5self_4b; }   # [v2.5.8e]
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_P5SELF $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["RUNBOOK"], "scripts"))
from speed_probe import steps_of
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    vals = [float(x.split(",")[1]) for x in open(os.path.join(d, "mem.csv")) if "," in x]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
try:
    text = open(os.path.join(d, "smoke", "train.log"), errors="ignore").read()
except OSError:
    text = ""
st = steps_of(text); r["steps"] = len(st); r["oom"] = "CUDA out of memory" in text or "OutOfMemoryError" in text
late = [s for s in st if s["i"] >= 2]
r["mean_s2plus"] = {k: round(sum(s[k] for s in late if s.get(k) is not None) / max(1, len([s for s in late if s.get(k) is not None])), 1)
                    for k in ("gen", "old_log_prob", "ref", "update_actor", "step")} if late else {}
r["projected_hours_150"] = round(150 * r["mean_s2plus"]["step"] / 3600, 1) if r["mean_s2plus"].get("step") else None
r["response_length"] = [s.get("response_length/mean") for s in st]
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/$UPD >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_p5self_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PSE_WAIT_STEP:-10}
done
step "start gpu=$G FIXED_MICRO_BSZ=${FIXED_MICRO_BSZ:-} VENV=${VENV:-}"
# 2. the 3-step smoke
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PSE_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G $ARMT > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PSE_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
S=$(sed -n 's/^student: \(.*\) (no_think)$/\1/p' $D/smoke/config.yaml); TT=$(sed -n 's/^teacher: \(.*\) (ref;.*$/\1/p' $D/smoke/config.yaml)   # [v2.5.8e]
[ -n "$S" ] && [ "$S" = "$TT" ] || finish FAIL "the smoke's teacher ($TT) is not the student itself ($S)"
finish PASS "${PSE_STEPS:-3}-step smoke ok, peak ${peak} MiB"
