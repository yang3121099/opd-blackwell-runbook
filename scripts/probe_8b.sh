#!/usr/bin/env bash
# PROBE-8B (runbook [v2.5.9z68], Q-068; Fable 10-23 15:1x UTC): the gate of E10e-P5self (the 8B student taught by its own weights on the
# thinking template, four GPUs). On the task's FOUR GPUs (a,b,c,d):
#   1. a 3-step smoke of E10e-P5self (SMOKE=1: no checkpoint, upload, validation; nvidia-smi every 5 s on the four) with opd_route.py's
#      count mode (the teacher template unchanged; route.jsonl: the teacher's input rows per call)
#   2. PASS = exit 0, no out of memory, peak < 260 GB on every GPU; every step in train.log with critic/score/mean in [0, 1];
#      route.jsonl one line per step, each with rows = batch = 128 all on the thinking template (the teacher's forward count equals
#      the samples); the smoke's config naming the 8B as both student and teacher, think-template=1, param_offload=1 and four GPUs
#   3. report.json: the checks, the peak per GPU, steps 2-3 mean per timing segment, the projected hours for 50 steps, per step
#      response length and clip ratio; to logs/<node>/probe_8b/ (private), an event, .done. run_arm.sh starts E10e-P5self after a PASS.
# Usage: bash probe_8b.sh <gpu,gpu,gpu,gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-8B; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv   # this run's memory samples only
step() { echo "PROBE_8B $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, re, sys
sys.path.insert(0, os.path.join(os.environ["RUNBOOK"], "scripts"))
from speed_probe import steps_of
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
peak = {}
try:
    for x in open(os.path.join(d, "mem.csv")):
        f = x.strip().split(",")
        if len(f) == 3:
            peak[f[1]] = max(peak.get(f[1], 0.0), float(f[2]))
except (OSError, ValueError):
    pass
r["mem_peak_mib"] = peak or None
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
text = rd(os.path.join(d, "smoke", "train.log"))
st = steps_of(text); r["steps"] = len(st); r["oom"] = "CUDA out of memory" in text or "OutOfMemoryError" in text
late = [s for s in st if s["i"] >= 2]
r["mean_s2plus"] = {k: round(sum(s[k] for s in late if s.get(k) is not None) / max(1, len([s for s in late if s.get(k) is not None])), 1)
                    for k in ("gen", "old_log_prob", "ref", "update_actor", "step")} if late else {}
r["projected_hours_50"] = round(50 * r["mean_s2plus"]["step"] / 3600, 1) if r["mean_s2plus"].get("step") else None
r["response_length"] = [s.get("response_length/mean") for s in st]
r["response_clip_ratio"] = [s.get("response_length/clip_ratio") for s in st]
p = os.path.join(d, "checks.json")
if os.path.exists(p):
    r["checks"] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  cp $D/smoke/config.yaml $D/smoke/route.jsonl $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_8b >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_8b_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
[ "$(echo $G | tr , '\n' | grep -c .)" = 4 ] || finish FAIL "PROBE-8B needs four GPUs (got $G): the queue's #gpus PROBE-8B 4"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ' | sort -n | tail -1)
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpus $G not empty after 180 s"; sleep ${P8B_WAIT_STEP:-10}
done
step "start gpus=$G GC=${GC:-}"
( while true; do t=$(date -u +%s); nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i $G | tr -d ' ' | sed "s/^/$t,/" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${P8B_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10e-P5self > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $OPD_ROOT/runs/$T-smoke/route.jsonl $D/smoke/ 2> /dev/null
for g in $(echo $G | tr , ' '); do python3 $RUNBOOK/scripts/control.py reap $g $T > /dev/null 2>&1 || true; done
peak=$(cut -d, -f3 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${P8B_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB on a GPU"
why=$(python3 - "$D" "${P8B_STEPS:-3}" "${P8B_ROWS:-128}" <<'PY'
import json, os, re, sys
d, nsteps, nrows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sm = os.path.join(d, "smoke"); why = []
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
cfg = rd(os.path.join(sm, "config.yaml"))
st = re.search(r"^student: (\S+)", cfg, re.M); te = re.search(r"^teacher: (\S+) \((.*)\)", cfg, re.M)
if not st or not te or st.group(1) != te.group(1) or "8B" not in st.group(1):
    why.append(f"the smoke's config does not name the 8B as both student and teacher: {st.group(1) if st else None} / {te.group(1) if te else None}")
if not te or "think-template=1" not in te.group(2) or "param_offload=1" not in te.group(2):
    why.append("the teacher is not on its thinking template, offloaded")
if not re.search(r"^gpu: .* x4 ", cfg, re.M):
    why.append("the smoke did not run on four GPUs")
scores = {int(s): float(v) for s, v in re.findall(r"(?:^|[ )\]])step:(\d+) - .*?critic/score/mean:([-0-9.e]+)", rd(os.path.join(sm, "train.log")))}
if sorted(scores) != list(range(1, nsteps + 1)):
    why.append(f"train.log steps with a score {sorted(scores)}, expected 1..{nsteps}")
if any(not 0.0 <= v <= 1.0 for v in scores.values()):
    why.append(f"a score outside [0, 1]: {scores}")
calls = [json.loads(x) for x in rd(os.path.join(sm, "route.jsonl")).splitlines() if x.strip()]
if len(calls) != nsteps:
    why.append(f"route.jsonl has {len(calls)} reference calls, expected one per step ({nsteps})")
bad = [(c.get("call"), c.get("batch"), c.get("rows"), c.get("think")) for c in calls
       if c.get("mode") != "count" or not (c.get("rows") == c.get("batch") == nrows == c.get("think"))]
if bad:
    why.append(f"reference calls whose teacher rows are not {nrows} samples on the thinking template (call, batch, rows, think): {bad}")
json.dump({"score_by_step": scores, "teacher_calls": [{k: c.get(k) for k in ("call", "batch", "rows", "think", "plain")} for c in calls]},
          open(os.path.join(d, "checks.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "${P8B_STEPS:-3}-step smoke on four GPUs ok, peak ${peak} MiB; ${P8B_ROWS:-128} teacher rows per step on the thinking template"
