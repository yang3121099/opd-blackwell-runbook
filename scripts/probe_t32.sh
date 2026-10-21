#!/usr/bin/env bash
# PROBE-T32 (runbook [v2.5.9z9], Q-053; the reviewer's pre-registration research_program §3c, 10-22): the gate of E10d-P5-t32 (the 4B
# student, Qwen3-32B as the teacher on its thinking template, offloaded). On ONE GPU:
#   0. the 32B on the share ($TEACHER_32B, default $OPD_ROOT/models/Qwen3-32B, with .fetch_ok and config.json); while it is not
#      there the probe says WAIT_INPUTS (waiting, no attempt, no .done)
#   1. a 3-step smoke of E10d-P5-t32 (SMOKE=1: no checkpoint, upload, validation; nvidia-smi every 5 s) with opd_route.py's count
#      mode (the teacher template unchanged; route.jsonl: the teacher's input rows per call)
#   2. PASS = exit 0, no out of memory, peak < 260 GB; every step in train.log with critic/score/mean in [0, 1]; route.jsonl one
#      line per step (one reference call a step), each with rows = batch = 128 (one teacher row per sample: the teacher's forward
#      count equals the samples) and all on the thinking template; the smoke's config naming the 32B teacher, its thinking template
#      (think-template=1) and param_offload=1
#   3. report.json: the checks, steps 2-3 mean per timing segment (ref = the 32B forward), projected hours for 50 steps, peak; to
#      logs/<node>/probe_t32/ (private), an event, .done
# run_arm.sh starts E10d-P5-t32 only after a PASS. Usage: bash probe_t32.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-T32; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv   # [v2.5.9z11] this run's memory samples only (a rerun keeps the directory: PROBE-STOPMERGE's 4th peak was the 2nd run's)
step() { echo "PROBE_T32 $(date -u +%T) $*" | tee -a $LOG; }
M=${TEACHER_32B:-$OPD_ROOT/models/Qwen3-32B}
[ -f $M/.fetch_ok ] && [ -f $M/config.json ] || { echo "WAIT_INPUTS $T: Qwen3-32B not in (no .fetch_ok or config.json)"; exit 1; }
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
st = steps_of(text); r["steps"] = len(st)
late = [s for s in st if s["i"] >= 2]
r["mean_s2plus"] = {k: round(sum(s[k] for s in late if s.get(k) is not None) / max(1, len([s for s in late if s.get(k) is not None])), 1)
                    for k in ("gen", "old_log_prob", "ref", "update_actor", "step")} if late else {}
r["projected_hours_50"] = round(50 * r["mean_s2plus"]["step"] / 3600, 1) if r["mean_s2plus"].get("step") else None
r["response_length"] = [s.get("response_length/mean") for s in st]
p = os.path.join(d, "checks.json")
if os.path.exists(p):
    r["checks"] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  [ -f $D/smoke/config.yaml ] && cp $D/smoke/config.yaml $D/up/smoke_config.yaml
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_t32 >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_t32_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PT32_WAIT_STEP:-10}
done
step "start gpu=$G teacher \$OPD_ROOT/models/$(basename $M)"
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
rm -rf $OPD_ROOT/runs/$T-smoke
TEACHER_32B=$M SMOKE=1 SMOKE_STEPS=${PT32_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10d-P5-t32 > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $OPD_ROOT/runs/$T-smoke/route.jsonl $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PT32_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
why=$(python3 - "$D" "${PT32_STEPS:-3}" "${PT32_ROWS:-128}" <<'PY'
import json, os, re, sys
d, nsteps, nrows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sm = os.path.join(d, "smoke"); why = []
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
cfg = rd(os.path.join(sm, "config.yaml"))
tl = [x for x in cfg.splitlines() if x.startswith("teacher:")]
tpath = tl[0].split()[1] if tl and len(tl[0].split()) > 1 else ""   # the teacher's path itself, not the line's label
if "Qwen3-32B" not in tpath or "think-template=1" not in tl[0] or "param_offload=1" not in tl[0]:
    why.append(f"the smoke's config does not name the 32B teacher on its thinking template, offloaded: {tl[0] if tl else 'no teacher line'}")
if not any(x.startswith("student:") and "Qwen3-4B" in x for x in cfg.splitlines()):
    why.append("the smoke's config does not name the 4B student")
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
json.dump({"score_by_step": scores, "teacher_calls": [{k: c.get(k) for k in ("call", "batch", "rows", "think", "plain")} for c in calls],
           "teacher_line": tl[0] if tl else None}, open(os.path.join(d, "checks.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "${PT32_STEPS:-3}-step smoke ok, peak ${peak} MiB; ${PT32_ROWS:-128} teacher rows per step on the thinking template (one reference call a step)"
