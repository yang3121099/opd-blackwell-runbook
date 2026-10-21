#!/usr/bin/env bash
# PROBE-T14 (runbook v2.5.7p, Q-015; reviewer A1 07:1x UTC): the gate of the 14B-teacher arms E10b-P5-t14 / E10b-P2-t14.
# On ONE GPU:
#   1. Qwen/Qwen3-14B into $OPD_ROOT/models/Qwen3-14B on the share (fetch_model.py: complete by size / sha256; ~29.5 GB),
#      so the arms and a rebuilt container find it (env.sh then sets TEACHER_14B to it)
#   2. a 3-step smoke of E10b-P5-t14 (1.7B student, the 14B teacher with the thinking template, not offloaded: the
#      longest answers, so the highest memory of the two arms) with the node's settings: no checkpoint, no upload, no
#      validation; nvidia-smi samples this GPU every 5 s
#   3. report.json: exit code, out of memory or not, peak memory, steps 2-3 mean per timing segment (the ref segment is
#      the 14B forward), the projected hours for 150 steps; PASS = exit 0, no OOM, peak < 260 GB; to
#      logs/<node>/probe_t14/ (private), an event, .done
# run_arm.sh starts the -t14 arms only after a PASS. Usage: bash probe_t14.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-T14; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv   # [v2.5.9z11] this run's memory samples only (a rerun keeps the directory: PROBE-STOPMERGE's 4th peak was the 2nd run's)
step() { echo "PROBE_T14 $(date -u +%T) $*" | tee -a $LOG; }
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
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_t14 >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_t14_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PT14_WAIT_STEP:-10}
done
step "start gpu=$G FIXED_MICRO_BSZ=${FIXED_MICRO_BSZ:-} VENV=${VENV:-}"
# 1. the 14B teacher on the share
M=$OPD_ROOT/models/Qwen3-14B
python3 $RUNBOOK/scripts/fetch_model.py Qwen/Qwen3-14B "" $M --task $T >> $LOG 2>&1 || finish FAIL "Qwen/Qwen3-14B could not be fetched"
[ -f $M/config.json ] || finish FAIL "Qwen3-14B fetched without config.json"
touch $M/.fetch_ok; export TEACHER_14B=$M; step "14B teacher at \$OPD_ROOT/models/Qwen3-14B"
# 2. the 3-step smoke
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PT14_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10b-P5-t14 > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PT14_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
finish PASS "${PT14_STEPS:-3}-step smoke ok, peak ${peak} MiB"
