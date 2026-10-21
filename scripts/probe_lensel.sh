#!/usr/bin/env bash
# PROBE-LENSEL (runbook [v2.5.8m], Q-030): the gate of the length-selected arms E10b-P5lenS / -P5lenL / -P5lenR.
#  1. the unit tests of code/patches/test_opd_lensel.py on the node (the venv-fa python, a plain runner, no GPU), with
#     GOPD_RAY_TRAINER = the installed ray_trainer.py: the hook's anchor is found once and the patched copy compiles
#  2. a 2-step smoke of E10b-P5lenS on ONE GPU (SMOKE=1: no checkpoint, upload, validation; nvidia-smi every 5 s): exit 0,
#     no OOM, peak < 260 GB, the smoke's config naming 32 prompts x 16 with the short selection; in train.log at every step
#     lensel/n_all 512, lensel/n_selected 128, lensel/groups 32 and lensel/sel_len_mean <= lensel/all_len_mean; lensel.jsonl
#     with 32 rows per step, each with 16 lengths and 4 kept; the installed ray_trainer.py holding the hook exactly once.
# One gate for the three arms (they differ only in OPD_LENSEL; the selection itself is covered by the unit tests).
# report.json, to logs/<node>/probe_lensel/<task> (private), an event, .done. ~20 min on one GPU.
# Usage: bash probe_lensel.sh <gpu>
set -uo pipefail
G=$1; T=PROBE-LENSEL; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_LENSEL $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    vals = [float(v.split(",")[1]) for v in open(os.path.join(d, "mem.csv")) if "," in v]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
p = os.path.join(d, "checks.json")
if os.path.exists(p):
    r["checks"] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/unit_tests.txt $D/up/ 2> /dev/null
  [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_lensel/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_lensel_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
# 1. unit tests (the arms train in venv-fa)
PY=${PLS_PY:-$OPD_ROOT/venv-fa/bin/python}; [ -x "$PY" ] || PY=$OPD_ROOT/venv/bin/python
RT=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ray_trainer').origin)" 2> /dev/null)
[ -n "$RT" ] && [ -f "$RT" ] || finish FAIL "verl's ray_trainer.py not found by $PY"
step "start gpu=$G unit tests with $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_RAY_TRAINER=$RT $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_opd_lensel")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_opd_lensel":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
# 2. the 2-step smoke of the short arm
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PLS_WAIT_STEP:-10}
done
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PLS_STEPS:-2} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10b-P5lenS > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $OPD_ROOT/runs/$T-smoke/lensel.jsonl $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PLS_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
why=$(python3 - "$D" "${PLS_STEPS:-2}" "$RT" <<'PY'
import json, os, re, sys
d, nsteps, rt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sm = os.path.join(d, "smoke"); why = []
cfg = open(os.path.join(sm, "config.yaml"), errors="ignore").read() if os.path.exists(os.path.join(sm, "config.yaml")) else ""
if "rollouts: 32 prompts x 16 per step, length selection short: 4 kept" not in cfg:
    why.append("the smoke's config does not name 32 x 16 with the short selection")
text = open(os.path.join(sm, "train.log"), errors="ignore").read() if os.path.exists(os.path.join(sm, "train.log")) else ""
steps = {}
for st, body in re.findall(r"(?:^|[ )\]])step:(\d+) - (.*?)(?=(?:^|[ )\]])step:\d+ - |\Z)", text, re.S):
    steps[int(st)] = {k: float(v) for k, v in re.findall(r"(lensel/[a-z_]+):([-0-9.e]+)", body)}
if sorted(steps) != list(range(1, nsteps + 1)):
    why.append(f"train.log steps {sorted(steps)}, expected 1..{nsteps}")
for st, m in sorted(steps.items()):
    if (m.get("lensel/n_all"), m.get("lensel/n_selected"), m.get("lensel/groups")) != (512.0, 128.0, 32.0):
        why.append(f"step {st}: lensel n_all / n_selected / groups {m.get('lensel/n_all')} / {m.get('lensel/n_selected')} / {m.get('lensel/groups')}")
    elif not m.get("lensel/sel_len_mean", 1e9) <= m.get("lensel/all_len_mean", -1):
        why.append(f"step {st}: kept mean length {m.get('lensel/sel_len_mean')} > all {m.get('lensel/all_len_mean')}")
rows = [json.loads(x) for x in open(os.path.join(sm, "lensel.jsonl"))] if os.path.exists(os.path.join(sm, "lensel.jsonl")) else []
if len(rows) != 32 * nsteps or any(len(r["len"]) != 16 or len(r["kept"]) != 4 for r in rows):
    why.append(f"lensel.jsonl: {len(rows)} rows (expected {32 * nsteps}, each 16 lengths and 4 kept)")
hooks = open(rt, errors="ignore").read().count("# [opd-variants patch v2] Q-030 length-selected OPD")
if hooks != 1:
    why.append(f"the installed ray_trainer.py holds the hook {hooks} times")
json.dump({"steps": steps, "jsonl_rows": len(rows), "hooks": hooks}, open(os.path.join(d, "checks.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "unit tests ok; ${PLS_STEPS:-2}-step smoke ok (512 rollouts, 128 kept per step), peak ${peak} MiB"
