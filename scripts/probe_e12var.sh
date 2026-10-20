#!/usr/bin/env bash
# PROBE-E12B-<VARIANT> (runbook [v2.5.8b], Q-020; reviewer 22:0x UTC 10-19): the gate of E12b-P5-<variant>. On ONE GPU a 3-step
# smoke of the variant with the node's settings (no checkpoint, upload, validation; nvidia-smi every 5 s); report.json:
# exit code, out of memory or not, peak memory, steps, the variant's own checks; PASS = exit 0, no OOM, peak < 260 GB and
# the variant's checks. run_e12.sh starts E12b-P5-<variant> only after a PASS. Usage: bash probe_e12var.sh <gpu> <variant>
#   kl: the reference must be the start (the reviewer's main risk: G-OPD's ref slot is the OPD teacher's). Checked two ways:
#       the smoke's config names the start as the reference, and the KL loss of step 1 is near 0 (actor = reference at the
#       start; a teacher or the bare base would give a large one), with a positive KL loss at a later step (the loss works).
#       [v2.5.8b] And the reward is the verifier's alone: critic/rewards/mean = critic/score/mean at every step (G-OPD's
#       reference slot must not add a teacher term to the reward; use_kl_in_reward stays off).
#   [v2.5.8c] stop: the stop reward is the one used and behaves on the node. Fixed cases scored with the node's own
#       grader and the start's tokenizer (the training's venv) give 1.5 / 1.0 / 1.0 / 0.0 (one box ending at once,
#       a changed box, a long tail, a wrong box); the smoke's config names reward_stop.py; every step's score lies in
#       [0, 1.5]; and critic/rewards/mean = critic/score/mean (nothing else added to the reward).
set -uo pipefail
G=$1; VAR=$2; source "$(dirname "$0")/env.sh"
case $VAR in kl|stop) ;; *) echo "unknown variant $VAR"; exit 2;; esac   # [v2.5.8c] stop
T=PROBE-E12B-$(echo $VAR | tr a-z A-Z); D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_E12VAR $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" "$VAR" <<'PY'
import json, os, re, sys
d, res, why, var = sys.argv[1:5]
r = {"result": res, "reason": why, "variant": var}
try:
    vals = [float(x.split(",")[1]) for x in open(os.path.join(d, "mem.csv")) if "," in x]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
for k in ("kl_loss_by_step", "ref_in_config", "reward_vs_score", "stop_cases", "score_by_step"):   # [v2.5.8b] reward_vs_score; [v2.5.8c] stop
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_e12var/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_e12var_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PEV_WAIT_STEP:-10}
done
step "start gpu=$G variant=$VAR FIXED_MICRO_BSZ=${FIXED_MICRO_BSZ:-} VENV=${VENV:-}"
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PEV_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_e12.sh $G E12b-P5-$VAR > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PEV_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
if [ $VAR = kl ]; then
  START=$(sed -n 's/^start: //p' $D/smoke/config.yaml | head -1)
  python3 - "$D" "$START" "${PEV_KL_STEP1_MAX:-0.005}" <<'PY'
import json, os, re, sys
d, start, step1_max = sys.argv[1], sys.argv[2], float(sys.argv[3])
log = open(os.path.join(d, "smoke", "train.log"), errors="ignore").read()
kl, rw = {}, {}
for line in log.splitlines():
    m = re.search(r"(?:^|[ )\]])step:(\d+) - ", line); k = re.search(r"actor/kl_loss:([-+0-9.eE]+)", line)
    if m and k:
        kl[int(m.group(1))] = float(k.group(1))
    sc, rv = re.search(r"critic/score/mean:([-+0-9.eE]+)", line), re.search(r"critic/rewards/mean:([-+0-9.eE]+)", line)
    if m and sc and rv:   # [v2.5.8b]
        rw[int(m.group(1))] = (float(sc.group(1)), float(rv.group(1)))
cfg = open(os.path.join(d, "smoke", "config.yaml"), errors="ignore").read()
ref = {"start": start, "config_yaml_reference": f"reference = {start}" in cfg, "start_in_log": log.count(start)}
json.dump(kl, open(os.path.join(d, "kl_loss_by_step.json"), "w")); json.dump(ref, open(os.path.join(d, "ref_in_config.json"), "w"))
why = []
if not start:
    why.append("no start in the smoke's config.yaml")
if not ref["config_yaml_reference"]:
    why.append("config.yaml does not name the start as the reference")
if ref["start_in_log"] < 2:
    why.append(f"the start path appears {ref['start_in_log']}x in the training log's config (actor + reference expected)")
if 1 not in kl:
    why.append("no actor/kl_loss at step 1")
elif abs(kl[1]) > step1_max:
    why.append(f"step-1 KL loss {kl[1]} > {step1_max}: the reference is not the start")
if not any(v > 0 for s, v in kl.items() if s >= 2):
    why.append("no positive KL loss after step 1")
if not rw:   # [v2.5.8b] the reward is the verifier's alone
    why.append("no critic/score/mean and critic/rewards/mean in the log")
elif any(abs(a - b) > 1e-6 for a, b in rw.values()):
    why.append("reward differs from the verifier score at steps " + ",".join(str(s) for s, (a, b) in sorted(rw.items()) if abs(a - b) > 1e-6))
json.dump({str(s): v for s, v in rw.items()}, open(os.path.join(d, "reward_vs_score.json"), "w"))
open(os.path.join(d, "kl_check.txt"), "w").write("; ".join(why))
PY
  why=$(cat $D/kl_check.txt)
  [ -z "$why" ] || finish FAIL "kl: $why"
  finish PASS "${PEV_STEPS:-3}-step smoke ok, peak ${peak} MiB, the reference is the start (step-1 KL $(python3 -c "import json; print(json.load(open('$D/kl_loss_by_step.json')).get('1'))"))"
fi
if [ $VAR = stop ]; then   # [v2.5.8c]
  START=$(sed -n 's/^start: //p' $D/smoke/config.yaml | head -1)
  SV=$(sed -n 's/^venv: //p' $D/smoke/config.yaml | head -1)   # the training's own venv (reviewer A1 note 07:1x UTC)
  cases=$(OPD_REWARD_TOKENIZER=$START OPD_REWARD_MAXLEN=4096 ${PEV_PY:-$OPD_ROOT/${SV:-venv}/bin/python} - "$RUNBOOK/code/train" 2>> $LOG <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
import reward_stop as r
cases = [("So the answer is \\boxed{2}.", "2"), ("First \\boxed{3}. Wait, let me recheck. So \\boxed{2}.", "2"),
         ("So \\boxed{2}. " + "Let me double check this result once more. " * 20, "2"), ("So \\boxed{3}.", "2")]
print(json.dumps([r.compute_score("math", s, g) for s, g in cases]))
PY
)
  python3 - "$D" "$cases" <<'PY'
import json, os, re, sys
d, cases = sys.argv[1], sys.argv[2]
log = open(os.path.join(d, "smoke", "train.log"), errors="ignore").read()
cfg = open(os.path.join(d, "smoke", "config.yaml"), errors="ignore").read()
sc = {}
for line in log.splitlines():
    m = re.search(r"(?:^|[ )\]])step:(\d+) - ", line)
    v = {k: re.search(rf"critic/{k}:([-+0-9.eE]+)", line) for k in ("score/min", "score/max", "score/mean", "rewards/mean")}
    if m and all(v.values()):
        sc[int(m.group(1))] = {k: float(x.group(1)) for k, x in v.items()}
why = []
try:
    got = json.loads(cases.strip().splitlines()[-1])
except (ValueError, IndexError):
    got = None
json.dump(got, open(os.path.join(d, "stop_cases.json"), "w")); json.dump(sc, open(os.path.join(d, "score_by_step.json"), "w"))
if got != [1.5, 1.0, 1.0, 0.0]:
    why.append(f"the reward on the fixed cases is {got}, expected [1.5, 1.0, 1.0, 0.0]")
if "reward: reward_stop.py" not in cfg:
    why.append("the smoke's config does not name reward_stop.py")
if not sc:
    why.append("no critic/score and critic/rewards in the log")
for s, v in sorted(sc.items()):
    if v["score/min"] < 0 or v["score/max"] > 1.5 + 1e-6:
        why.append(f"step {s}: score outside [0, 1.5]")
    if abs(v["rewards/mean"] - v["score/mean"]) > 1e-6:
        why.append(f"step {s}: reward differs from the score")
open(os.path.join(d, "stop_check.txt"), "w").write("; ".join(why))
PY
  why=$(cat $D/stop_check.txt)
  [ -z "$why" ] || finish FAIL "stop: $why"
  finish PASS "${PEV_STEPS:-3}-step smoke ok, peak ${peak} MiB, the stop reward scores the fixed cases 1.5 / 1.0 / 1.0 / 0.0 on the node"
fi
finish PASS "${PEV_STEPS:-3}-step smoke ok, peak ${peak} MiB"
