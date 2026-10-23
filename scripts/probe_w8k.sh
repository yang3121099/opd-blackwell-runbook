#!/usr/bin/env bash
# PROBE-P5-8K (runbook [v2.5.9z66], Q-059; Fable 10-23 04:2x UTC): the gate of E10b-P5-8k (P5's recipe with 8K rollouts, no merge).
# On ONE GPU:
#   1. a 3-step smoke of E10b-P5-8k with the node's settings (SMOKE=1: no checkpoint, upload, validation; nvidia-smi every 5 s)
#   2. PASS = exit 0, no OOM, peak < 260 GB, every step in train.log with critic/score/mean in [0, 1], and the smoke's config
#      naming max_response_length 8192 on flash-attn (use_remove_padding=True)
# report.json: the checks, peak memory, steps 2-3 mean per timing segment, the projected hours for 50 steps, and per step
# response_length/mean and clip_ratio (the 4K arm's late steps clip about half; 8K should not), to logs/<node>/probe_w8k/
# (private), an event, .done. ~30 min on one GPU. run_arm.sh starts E10b-P5-8k only after a PASS.
# Usage: bash probe_w8k.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-P5-8K; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv   # this run's memory samples only
step() { echo "PROBE_P5_8K $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" "${PW8_STEPS:-3}" <<'PY'
import json, os, re, sys
sys.path.insert(0, os.path.join(os.environ["RUNBOOK"], "scripts"))
from speed_probe import steps_of
d, res, why, nsteps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
r = {"result": res, "reason": why}
try:
    vals = [float(x.split(",")[1]) for x in open(os.path.join(d, "mem.csv")) if "," in x]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
text = rd(os.path.join(d, "smoke", "train.log"))
st = steps_of(text); r["steps"] = len(st); r["oom"] = "CUDA out of memory" in text or "OutOfMemoryError" in text
late = [s for s in st if s["i"] >= 2]
r["mean_s2plus"] = {k: round(sum(s[k] for s in late if s.get(k) is not None) / max(1, len([s for s in late if s.get(k) is not None])), 1)
                    for k in ("gen", "old_log_prob", "ref", "update_actor", "step")} if late else {}
r["projected_hours_50"] = round(50 * r["mean_s2plus"]["step"] / 3600, 1) if r["mean_s2plus"].get("step") else None
r["response_length"] = [s.get("response_length/mean") for s in st]
r["response_clip_ratio"] = [s.get("response_length/clip_ratio") for s in st]
r["score_by_step"] = {int(s): float(v) for s, v in re.findall(r"(?:^|[ )\]])step:(\d+) - .*?critic/score/mean:([-0-9.e]+)", text)}
cfg = rd(os.path.join(d, "smoke", "config.yaml"))
r["config"] = {"max_response_length_8192": "max_response_length 8192" in cfg, "remove_padding": "use_remove_padding=True" in cfg}
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  cp $D/smoke/config.yaml $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_w8k >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_w8k_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PW8_WAIT_STEP:-10}
done
step "start gpu=$G FIXED_MICRO_BSZ=${FIXED_MICRO_BSZ:-} GC=${GC:-}"
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PW8_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10b-P5-8k > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PW8_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
why=$(python3 - "$D" "${PW8_STEPS:-3}" <<'PY'
import os, re, sys
d, nsteps = sys.argv[1], int(sys.argv[2])
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
why = []
cfg = rd(os.path.join(d, "smoke", "config.yaml"))
if "max_response_length 8192" not in cfg:
    why.append("the smoke's config does not name max_response_length 8192")
if "use_remove_padding=True" not in cfg:
    why.append("the smoke did not run with remove-padding (flash-attn, venv-fa)")
scores = {int(s): float(v) for s, v in re.findall(r"(?:^|[ )\]])step:(\d+) - .*?critic/score/mean:([-0-9.e]+)", rd(os.path.join(d, "smoke", "train.log")))}
if sorted(scores) != list(range(1, nsteps + 1)):
    why.append(f"train.log steps with a score {sorted(scores)}, expected 1..{nsteps}")
if any(not 0.0 <= v <= 1.0 for v in scores.values()):
    why.append(f"a score outside [0, 1]: {scores}")
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "${PW8_STEPS:-3}-step smoke ok, 8K window on flash-attn, peak ${peak} MiB"
