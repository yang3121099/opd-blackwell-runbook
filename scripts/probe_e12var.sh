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
#   [v2.5.9f] stop2: the continuous stop reward on the node: fixed cases (the start's tokenizer, the node's grader): an
#       answer that stops at once in (1.45, 1.5], a ~180-token tail in (1.0, 1.45), a wrong box 0, a tail past 1024 tokens
#       1.0; the smoke's config names reward_stop2.py; every step's score in [0, 1.5]; rewards = score; and the smoke's
#       rollout dump carries the reward's parts (tail), with the share of zero-variance groups reported.
#   [v2.5.8h] 8k / 16k: the long response limits fit (the smoke's config names the limit; its response lengths and the
#       peak are reported; the peak check above is the main one).
#   [v2.5.9w] r3 (Q-039): first the unit tests of code/patches/test_opd_vardrop.py with the training's python and
#       GOPD_RAY_TRAINER = the installed ray_trainer.py (both hook anchors found once, the patched file compiles); then, at
#       every smoke step: 128 groups scored, 0 < kept groups <= 64, kept rows = update rows = 8 x kept groups, update
#       prompts = kept groups, the PPO mini-batch count = ceil(rows / 128), the advantages' mean ~ 0 and variance ~ 1 over
#       the kept tokens (the batch whitening ran on the kept rows only), the rollout dump holds the kept rows only, and
#       vardrop.jsonl has every group (kept ones with a spread); the config names reinforce_plus_plus_baseline 128x8.
set -uo pipefail
G=$1; VAR=$2; source "$(dirname "$0")/env.sh"
case $VAR in kl|stop|stop2|8k|16k|r3) ;; *) echo "unknown variant $VAR"; exit 2;; esac   # [v2.5.8c] stop; [v2.5.8h] 8k / 16k; [v2.5.9f] stop2
# [v2.5.9z85] (Q-077) PEV_LADDER=E12c: PROBE-E12C-8K / -16K, the same smoke of the 0.6B window arm E12c-P5-<8k|16k> (the 1.7B one by default)
L=${PEV_LADDER:-E12b}; case $L:$VAR in E12b:*|E12c:8k|E12c:16k|E12e:8k) ;; *) echo "unknown ladder / variant $L $VAR"; exit 2;; esac
T=PROBE-$(echo $L | tr a-z A-Z)-$(echo $VAR | tr a-z A-Z); D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv   # [v2.5.9z11] this run's memory samples only (a rerun keeps the directory: PROBE-STOPMERGE's 4th peak was the 2nd run's)
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
for k in ("kl_loss_by_step", "ref_in_config", "reward_vs_score", "stop_cases", "score_by_step", "r3_by_step"):   # [v2.5.8b] reward_vs_score; [v2.5.8c] stop
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; cp $D/unit_tests.txt $D/smoke/vardrop.jsonl $D/up/ 2> /dev/null   # [v2.5.9w] r3's
  [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_e12var/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_e12var_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
if [ $VAR = r3 ]; then   # [v2.5.9w] the unit tests with the training's python, before the GPU is taken
  PY=${PEV_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
  RT=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ray_trainer').origin)" 2> /dev/null)
  [ -n "$RT" ] && [ -f "$RT" ] || finish FAIL "r3: verl's ray_trainer.py not found by $PY"
  ( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_RAY_TRAINER=$RT $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_opd_vardrop")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_opd_vardrop":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
  ); rc=$?
  step "r3 unit tests: $(tail -1 $D/unit_tests.txt)"
  [ $rc = 0 ] || finish FAIL "r3 unit tests: $(tail -1 $D/unit_tests.txt)"
fi
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ' | sort -n | tail -1)   # [v2.5.9z91] the fullest of a #gpus list
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PEV_WAIT_STEP:-10}
done
step "start gpu=$G variant=$VAR FIXED_MICRO_BSZ=${FIXED_MICRO_BSZ:-} VENV=${VENV:-}"
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ' | sort -n | tail -1)" >> $D/mem.csv; sleep 5; done ) &   # [v2.5.9z91] the fullest GPU
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
ARM=$L-P5-$VAR; [ $L = E12e ] && ARM=$L-P5self-$VAR   # [v2.5.9z91] (Q-081) PEV_LADDER=E12e: PROBE-E12E-8K smokes E12e-P5self-8k on its four GPUs (#gpus 4)
SMOKE=1 SMOKE_STEPS=${PEV_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_e12.sh $G $ARM > $D/smoke.out 2>&1; rc=$?   # [v2.5.9z85] $L
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
for g in $(echo $G | tr , ' '); do python3 $RUNBOOK/scripts/control.py reap $g $T > /dev/null 2>&1 || true; done   # [v2.5.9z91] each GPU of a #gpus list
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
if [ $VAR = stop2 ]; then   # [v2.5.9f]
  START=$(sed -n 's/^start: //p' $D/smoke/config.yaml | head -1)
  SV=$(sed -n 's/^venv: //p' $D/smoke/config.yaml | head -1)
  cases=$(OPD_REWARD_TOKENIZER=$START OPD_REWARD_MAXLEN=4096 ${PEV_PY:-$OPD_ROOT/${SV:-venv}/bin/python} - "$RUNBOOK/code/train" 2>> $LOG <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
import reward_stop2 as r
tail = "Let me double check this result once more. "
cases = [("So the answer is \\boxed{2}.", "2"), ("So \\boxed{2}. " + tail * 20, "2"), ("So \\boxed{3}.", "2"), ("So \\boxed{2}. " + tail * 150, "2")]
print(json.dumps([r.compute_score("math", s, g)["score"] for s, g in cases]))
PY
)
  cp $OPD_ROOT/runs/$T-smoke/rollouts/1.jsonl $D/smoke/rollouts_1.jsonl 2> /dev/null
  python3 - "$D" "$cases" "$RUNBOOK" <<'PY'
import json, os, re, sys
d, cases, rb = sys.argv[1], sys.argv[2], sys.argv[3]
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
if not (got and 1.45 < got[0] <= 1.5 and 1.0 < got[1] < 1.45 and got[2] == 0.0 and got[3] == 1.0):
    why.append(f"the reward on the fixed cases is {got}, expected (1.45, 1.5] / (1.0, 1.45) / 0 / 1.0")
if "reward: reward_stop2.py" not in cfg:
    why.append("the smoke's config does not name reward_stop2.py")
if not sc:
    why.append("no critic/score and critic/rewards in the log")
for s, v in sorted(sc.items()):
    if v["score/min"] < 0 or v["score/max"] > 1.5 + 1e-6:
        why.append(f"step {s}: score outside [0, 1.5]")
    if abs(v["rewards/mean"] - v["score/mean"]) > 1e-6:
        why.append(f"step {s}: reward differs from the score")
rp = os.path.join(d, "smoke", "rollouts_1.jsonl")
recs = [json.loads(x) for x in open(rp)] if os.path.exists(rp) else []
if not recs or "tail" not in recs[0]:
    why.append("the smoke's rollout dump does not carry the reward's parts (tail)")
else:
    try:
        sys.path.insert(0, os.path.join(rb, "code", "analysis")); import e12_rollout_stats as E
        json.dump(E.step_stats(1, recs, lambda t: len(t.split())), open(os.path.join(d, "stop2_step1_stats.json"), "w"))
    except Exception as e:
        why.append(f"the rollout statistics failed ({type(e).__name__})")
open(os.path.join(d, "stop_check.txt"), "w").write("; ".join(why))
PY
  [ -f $D/stop_check.txt ] || finish FAIL "stop2: the checks did not run"
  why=$(cat $D/stop_check.txt)
  [ -z "$why" ] || finish FAIL "stop2: $why"
  finish PASS "${PEV_STEPS:-3}-step smoke ok, peak ${peak} MiB, the continuous stop reward scores the fixed cases $(python3 -c "import json; print(json.load(open('$D/stop_cases.json')))" ) on the node; step-1 zero-variance groups $(python3 -c "import json; print(json.load(open('$D/stop2_step1_stats.json')).get('zero_var_groups'))")"
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
if [ $VAR = r3 ]; then   # [v2.5.9w]
  SD=$OPD_ROOT/runs/$T-smoke; cp $SD/vardrop.jsonl $D/smoke/ 2> /dev/null
  for f in $SD/rollouts/*.jsonl; do [ -f "$f" ] && cp "$f" $D/smoke/rollouts_$(basename "$f"); done
  grep -q "VARDROP_HOOK_PATCHED\|VARDROP_HOOK_OK" $D/smoke.out || finish FAIL "r3: vardrop_patch.py did not run before the smoke"
  python3 - "$D" "${PEV_R3_K:-64}" "${PEV_R3_GROUPS:-128}" <<'PY'
import json, math, os, re, sys
d, k, ng = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
log = open(os.path.join(d, "smoke", "train.log"), errors="ignore").read()
cfg = open(os.path.join(d, "smoke", "config.yaml"), errors="ignore").read()
keys = ("groups", "nonzero_groups", "kept_groups", "kept_rows", "update_rows", "update_prompts", "ppo_mini_batches",
        "adv_mean", "adv_var", "empty", "gen_tokens", "kept_tokens", "zero_var_frac")
st = {}
for line in log.splitlines():
    m = re.search(r"(?:^|[ )\]])step:(\d+) - ", line)
    v = {x: re.search(rf"r3/{x}:([-+0-9.eE]+)", line) for x in keys}
    if m and all(v.values()):
        st[int(m.group(1))] = {x: float(y.group(1)) for x, y in v.items()}
groups = {}
try:
    for line in open(os.path.join(d, "smoke", "vardrop.jsonl")):
        r = json.loads(line); groups.setdefault(r["step"], []).append(r)
except (OSError, ValueError):
    pass
why, dump = [], {}
if "algo: reinforce_plus_plus_baseline" not in cfg or f"{ng}x8" not in cfg:
    why.append(f"the smoke's config does not name reinforce_plus_plus_baseline {ng}x8")
if not st:
    why.append("no r3/* metrics in the log")
for s, v in sorted(st.items()):
    kg, kr = v["kept_groups"], v["kept_rows"]
    if v["groups"] != ng:
        why.append(f"step {s}: {v['groups']:.0f} groups scored, expected {ng}")
    if not 0 < kg <= k:
        why.append(f"step {s}: {kg:.0f} kept groups (1..{k})")
    if kr != 8 * kg or v["update_rows"] != kr or v["update_prompts"] != kg:
        why.append(f"step {s}: kept rows {kr:.0f}, update rows {v['update_rows']:.0f}, update prompts {v['update_prompts']:.0f} for {kg:.0f} groups")
    if v["ppo_mini_batches"] != math.ceil(kr / 128):
        why.append(f"step {s}: {v['ppo_mini_batches']:.0f} PPO mini-batches for {kr:.0f} rows")
    if v["empty"] == 0 and (abs(v["adv_mean"]) > 1e-3 or abs(v["adv_var"] - 1) > 0.02):
        why.append(f"step {s}: advantage mean {v['adv_mean']} / variance {v['adv_var']} over the kept tokens (not whitened on the kept rows)")
    g = groups.get(s, [])
    if len(g) != ng or sum(r["kept"] for r in g) != kg or sum(r["spread"] for r in g) != v["nonzero_groups"] or any(r["kept"] and not r["spread"] for r in g if not r["empty_step"]):
        why.append(f"step {s}: vardrop.jsonl has {len(g)} groups, {sum(r['kept'] for r in g)} kept, {sum(r['spread'] for r in g)} with a spread")
    rp = os.path.join(d, "smoke", f"rollouts_{s}.jsonl")
    dump[s] = sum(1 for _ in open(rp)) if os.path.exists(rp) else None
    if dump[s] != kr:
        why.append(f"step {s}: the rollout dump has {dump[s]} rows, expected the {kr:.0f} kept")
json.dump({"by_step": st, "dump_rows": dump}, open(os.path.join(d, "r3_by_step.json"), "w"))
open(os.path.join(d, "r3_check.txt"), "w").write("; ".join(why))
PY
  [ -f $D/r3_check.txt ] || finish FAIL "r3: the checks did not run"
  why=$(cat $D/r3_check.txt)
  [ -z "$why" ] || finish FAIL "r3: $why"
  KG=$(python3 -c "import json, sys; s = json.load(open(sys.argv[1]))['by_step']; print('/'.join(f\"{v['kept_groups']:.0f}\" for _, v in sorted(s.items(), key=lambda x: int(x[0]))))" $D/r3_by_step.json)
  ZV=$(python3 -c "import json, sys; s = json.load(open(sys.argv[1]))['by_step']; print('/'.join(f\"{v['zero_var_frac']:.2f}\" for _, v in sorted(s.items(), key=lambda x: int(x[0]))))" $D/r3_by_step.json)
  finish PASS "${PEV_STEPS:-3}-step smoke ok, peak ${peak} MiB; kept groups by step $KG of ${PEV_R3_GROUPS:-128} (zero-variance share $ZV); rows, the whitening, the rollout dump and vardrop.jsonl checked"
fi
if [ $VAR = 8k ] || [ $VAR = 16k ]; then   # [v2.5.8h]
  N=$(( ${VAR%k} * 1024 ))
  grep -q "^max_response_length: $N$" $D/smoke/config.yaml || finish FAIL "the smoke's config does not name max_response_length $N"
  LENS=$(grep -oE "response_length/mean:[0-9.]+" $D/smoke/train.log | cut -d: -f2 | xargs | tr ' ' '/')
  finish PASS "${PEV_STEPS:-3}-step smoke ok at the $N limit, peak ${peak} MiB, response_length/mean by step ${LENS:-none}"
fi
finish PASS "${PEV_STEPS:-3}-step smoke ok, peak ${peak} MiB"
