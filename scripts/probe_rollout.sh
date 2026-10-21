#!/usr/bin/env bash
# PROBE-ROLLOUT-<U60|U75|U85> (runbook [v2.5.9i]; Runming 10-21 18:5x Beijing: GPU2 / GPU7 at 81% under the 16K arm, "squeeze out
# all the performance"; sched-preempt's utilization_20261021.md): does more vLLM memory make an E12b step faster? The long
# arms spend most of a step generating (8K: timing_s/gen 504 of 691 s at s25) with vLLM at gpu_memory_utilization 0.6, and
# at 8K / 16K the KV cache holds fewer sequences than a rank generates at once. (enforce_eager is already False on the
# node: CUDA graphs are on, so it is not a variant.) On ONE GPU a 3-step smoke of E12b-P5-8k with the arm's own settings
# except VLLM_UTIL (through PROBE_ENV, which wins over control.env): U60 = 0.6 (the arm as it runs), U75 = 0.75, U85 = 0.85;
# nvidia-smi every 5 s. Each variant is its own task (one GPU, no waiting for three at once).
# report.json: exit code, out of memory or not, peak memory, per step timing_s/step / gen / update_actor, response_length
# mean / clip_ratio, critic/score/mean, actor/pg_loss, perf/max_memory_reserved_gb, perf/throughput; the means over the steps
# after the first (warm-up); and against U60's report when it exists: the step and generation time ratios and the length /
# score differences, with faster = step <= 0.85 x U60 and comparable = |length| <= 15%, |score| <= 0.10 (readings: taking a
# setting into an arm is decided on the review). PASS = exit 0, no OOM, peak < 260 GB, every step logged. Private:
# logs/<node>/probe_rollout/<task>; an event; .done. ~45 min.
# Usage: bash probe_rollout.sh <gpu> <U60|U75|U85>
set -uo pipefail
G=$1; V=$2; source "$(dirname "$0")/env.sh"
case $V in U60) U=0.6;; U75) U=0.75;; U85) U=0.85;; *) echo "unknown variant $V"; exit 2;; esac
T=PROBE-ROLLOUT-$V; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_ROLLOUT $(date -u +%T) $*" | tee -a $LOG; }
STEPS=${PR_STEPS:-3}; ARM=${PR_ARM:-E12b-P5-8k}
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" "$V" "$U" "$OPD_ROOT/runs/PROBE-ROLLOUT-U60/report.json" <<'PY'
import json, os, re, sys
d, res, why, v, u, ref = sys.argv[1:7]
r = {"result": res, "reason": why, "variant": v, "vllm_util": float(u)}
try:
    vals = [float(x.split(",")[1]) for x in open(os.path.join(d, "mem.csv")) if "," in x]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
KEYS = ("timing_s/step", "timing_s/gen", "timing_s/update_actor", "response_length/mean", "response_length/clip_ratio",
        "critic/score/mean", "actor/pg_loss", "perf/max_memory_reserved_gb", "perf/throughput")
steps = {}
p = os.path.join(d, "smoke", "train.log")
for line in (open(p, errors="ignore") if os.path.exists(p) else []):
    m = re.search(r"(?:^|[ )\]])step:(\d+) - ", line)
    if m:
        steps[int(m.group(1))] = {k: float(x.group(1)) for k in KEYS if (x := re.search(re.escape(k) + r":([-+0-9.eE]+)", line))}
r["by_step"] = {str(s): steps[s] for s in sorted(steps)}
later = [steps[s] for s in sorted(steps) if s > 1] or [steps[s] for s in sorted(steps)]
r["mean_after_step1"] = {k: round(sum(x[k] for x in later) / len(later), 4) for k in KEYS if later and all(k in x for x in later)}
if v != "U60" and os.path.exists(ref):
    b = json.load(open(ref)).get("mean_after_step1", {}); m = r["mean_after_step1"]
    if b.get("timing_s/step") and m.get("timing_s/step"):
        c = {"step_ratio": round(m["timing_s/step"] / b["timing_s/step"], 3),
             "gen_ratio": round(m["timing_s/gen"] / b["timing_s/gen"], 3) if b.get("timing_s/gen") and m.get("timing_s/gen") else None,
             "length_rel_diff": round(m["response_length/mean"] / b["response_length/mean"] - 1, 3) if b.get("response_length/mean") else None,
             "score_diff": round(m.get("critic/score/mean", 0) - b.get("critic/score/mean", 0), 3)}
        c["faster"] = c["step_ratio"] <= 0.85
        c["comparable"] = (c["length_rel_diff"] is not None and abs(c["length_rel_diff"]) <= 0.15) and abs(c["score_diff"]) <= 0.10
        r["vs_U60"] = c
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_rollout/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_rollout_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PR_WAIT_STEP:-10}
done
step "start gpu=$G variant=$V VLLM_UTIL=$U arm=$ARM steps=$STEPS"
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
PROBE_ENV="${PROBE_ENV:-} VLLM_UTIL=$U" SMOKE=1 SMOKE_STEPS=$STEPS PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_e12.sh $G $ARM > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PR_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
grep -Eq "gpu_memory_utilization.{0,4}${U}([^0-9]|$)" $D/smoke/train.log 2> /dev/null || finish FAIL "the smoke's config does not show gpu_memory_utilization $U"
ns=$(grep -c "timing_s/step:" $D/smoke/train.log 2> /dev/null); [ "${ns:-0}" -ge $STEPS ] || finish FAIL "train.log has ${ns:-0} steps with timing, expected $STEPS"
finish PASS "$STEPS-step smoke of $ARM at VLLM_UTIL $U ok, peak ${peak} MiB"
