#!/usr/bin/env bash
# PROBE-TAIL (runbook [v2.5.9n], T4): the gate of the tail-swap arm E10b-P5tail.
#  1. the unit tests of code/patches/test_opd_tailswap.py on the node (the arm's python, with torch, a plain runner, no GPU),
#     with GOPD_RAY_TRAINER = the installed ray_trainer.py: the hook's anchor is found once (alone and with the
#     length-selection hook), the patched text compiles, a second application changes nothing; the swap; second_pass with
#     stand-ins for verl (P2's prompt for the second pass only, P5's inputs back, the swap after the box)
#  2. a 3-step smoke of E10b-P5tail on ONE GPU (SMOKE=1: no checkpoint, upload, validation; nvidia-smi every 5 s): exit 0,
#     no OOM, peak < 260 GB; the smoke's config naming the tail swap; every step in train.log with critic/score/mean in
#     [0, 1] and the tail/* metrics: both readings finite on every valid token, responses with a box, positions swapped,
#     none at or before a box; tailswap.jsonl one line per step; the installed ray_trainer.py holding the hook once.
# report.json (per step: rows with a box, share swapped, the mean change of the teacher's log-prob), to
# logs/<node>/probe_tail/<task> (private), an event, .done. ~30 min on one GPU (two reference passes a step).
# [v2.5.9z73] (Q-061) PTL_PROBE=<task> PTL_ARM=<arm> PTL_BUD=<N>: the same gate for a budget arm (PROBE-TAIL-BUD: E10b-P5bud2k,
# N = 2048): the smoke's config must name the budget, and each step's tail/* must show the budget at N, no row's first swap beyond
# N (budget_late 0), the switched rows = by a box + by the budget, the switch positions p50 / p90 / max <= N; over the smoke at
# least one row switched by the budget (else the budget never bound and the arm would be the plain tail swap). report.json adds
# the switch-position summary (rows by box / by budget, p10 / p50 / p90 / max per step).
# [v2.5.9z83] (Q-073) PTL_PROBE=PROBE-SOFTMIX PTL_ARM=E10b-P5softmix50 PTL_SOFTMIX=<w>: the gate of the soft mix: the same unit tests and
# hook check; the smoke's config must name the soft mix at w, and each step's tail/* must show softmix_w = w, both readings and the merge
# finite on every valid token, the merge between the two readings (no swap, so the box / swap / violation checks do not apply).
# Usage: bash probe_tail.sh <gpu>
set -uo pipefail
G=$1; T=${PTL_PROBE:-PROBE-TAIL}; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv   # [v2.5.9z11] this run's memory samples only (a rerun keeps the directory: PROBE-STOPMERGE's 4th peak was the 2nd run's)
step() { echo "PROBE_TAIL $(date -u +%T) $*" | tee -a $LOG; }
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
  [ -f $D/smoke/tailswap.jsonl ] && cp $D/smoke/tailswap.jsonl $D/up/
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_tail/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_tail_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
# 1. unit tests with the python the arm trains in (P5's venv)
PY=${PTL_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
RT=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ray_trainer').origin)" 2> /dev/null)
[ -n "$RT" ] && [ -f "$RT" ] || finish FAIL "verl's ray_trainer.py not found by $PY"
step "start gpu=$G unit tests with $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_RAY_TRAINER=$RT $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_opd_tailswap")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_opd_tailswap":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
try:
    import torch  # noqa: F401  (the torch tests return early without it: on the node they must run)
except ImportError:
    fails += 1; print("FAILED torch not importable: the swap and second_pass tests did not run")
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
# 2. the 3-step smoke
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PTL_WAIT_STEP:-10}
done
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PTL_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G ${PTL_ARM:-E10b-P5tail} > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $OPD_ROOT/runs/$T-smoke/tailswap.jsonl $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PTL_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
why=$(python3 - "$D" "${PTL_STEPS:-3}" "$RT" <<'PY'
import json, os, re, sys
d, nsteps, rt = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sm = os.path.join(d, "smoke"); why = []
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
smx = float(os.environ.get("PTL_SOFTMIX", "0") or 0)   # [v2.5.9z83] (Q-073) the soft mix's gate
if smx:
    if f"soft-mixed: {os.environ.get('PTL_SOFTMIX')} * the thinking reading" not in rd(os.path.join(sm, "config.yaml")):
        why.append(f"the smoke's config does not name the soft mix at {smx}")
elif "tail-swapped: the non-thinking reading after the first box" not in rd(os.path.join(sm, "config.yaml")):
    why.append("the smoke's config does not name the tail swap")
steps = {}
for st, body in re.findall(r"(?:^|[ )\]])step:(\d+) - (.*)", rd(os.path.join(sm, "train.log"))):
    steps[int(st)] = {k: float(v) for k, v in re.findall(r"((?:tail/[a-z_0-9]+|critic/score/mean)):([-0-9.e]+)", body)}
if sorted(steps) != list(range(1, nsteps + 1)):
    why.append(f"train.log steps {sorted(steps)}, expected 1..{nsteps}")
for st, m in sorted(steps.items()):
    if not 0.0 <= m.get("critic/score/mean", -1) <= 1.0:
        why.append(f"step {st}: score {m.get('critic/score/mean')} outside [0, 1]")
    if m.get("tail/lp_think_finite") != 1.0 or m.get("tail/lp_plain_finite") != 1.0:
        why.append(f"step {st}: a reading not finite on every valid token (thinking {m.get('tail/lp_think_finite')}, non-thinking {m.get('tail/lp_plain_finite')})")
    if smx:   # [v2.5.9z83] the merge
        if m.get("tail/softmix_w") != smx:
            why.append(f"step {st}: tail/softmix_w {m.get('tail/softmix_w')}, expected {smx}")
        if m.get("tail/merged_finite") != 1.0 or m.get("tail/merged_between") != 1.0:
            why.append(f"step {st}: the merge finite {m.get('tail/merged_finite')}, between the readings {m.get('tail/merged_between')}")
    elif m.get("tail/violations", 1) != 0.0:
        why.append(f"step {st}: {m.get('tail/violations')} swaps at or before a box")
if not smx and sum(m.get("tail/rows_box", 0) for m in steps.values()) == 0:
    why.append("no response with a complete box was found")
if not smx and sum(m.get("tail/swapped", 0) for m in steps.values()) == 0:
    why.append("no position was swapped")
rows = [json.loads(x) for x in rd(os.path.join(sm, "tailswap.jsonl")).splitlines() if x.strip()]
if len(rows) < nsteps:
    why.append(f"tailswap.jsonl has {len(rows)} lines, expected {nsteps}")
bud = int(os.environ.get("PTL_BUD", "0") or 0); budsum = None
if bud:   # [v2.5.9z73] (Q-061) the switch positions
    if f"budget: the switch at min(the first box, {bud} tokens)" not in rd(os.path.join(sm, "config.yaml")):
        why.append(f"the smoke's config does not name the budget {bud}")
    for st, m in sorted(steps.items()):
        if m.get("tail/budget") != float(bud):
            why.append(f"step {st}: tail/budget {m.get('tail/budget')}, expected {bud}")
        if m.get("tail/budget_late", 1) != 0.0:
            why.append(f"step {st}: {m.get('tail/budget_late')} rows first swapped beyond the budget")
        if m.get("tail/rows_switched", -1) != m.get("tail/rows_by_box", 0) + m.get("tail/rows_by_budget", 0):
            why.append(f"step {st}: switched rows {m.get('tail/rows_switched')} != by box {m.get('tail/rows_by_box')} + by budget {m.get('tail/rows_by_budget')}")
        if max(m.get("tail/switch_p50", 0), m.get("tail/switch_p90", 0), m.get("tail/switch_max", 0)) > bud:
            why.append(f"step {st}: a switch position beyond {bud} (p50 {m.get('tail/switch_p50')}, p90 {m.get('tail/switch_p90')}, max {m.get('tail/switch_max')})")
    if sum(m.get("tail/rows_by_budget", 0) for m in steps.values()) == 0:
        why.append(f"no row was switched by the budget {bud} in the smoke (the budget never bound)")
    budsum = {"budget": bud, "rows_by_box": sum(m.get("tail/rows_by_box", 0) for m in steps.values()),
              "rows_by_budget": sum(m.get("tail/rows_by_budget", 0) for m in steps.values()),
              "per_step": {st: {k: m.get(f"tail/{k}") for k in ("switch_p10", "switch_p50", "switch_p90", "switch_max", "rows_by_box", "rows_by_budget")} for st, m in sorted(steps.items())}}
hooks = rd(rt).count("# [opd-variants patch v2] [v2.5.9n] tail-swap OPD")
if hooks != 1:
    why.append(f"the installed ray_trainer.py holds the hook {hooks} times")
tot = lambda k: sum(m.get(k, 0) for m in steps.values())
summ = {"rows": tot("tail/rows"), "rows_box_share": round(tot("tail/rows_box") / max(tot("tail/rows"), 1), 4),
        "swapped_frac_mean": round(tot("tail/swapped_frac") / max(len(steps), 1), 4),
        "delta_mean": round(sum(m.get("tail/delta_mean", 0) * m.get("tail/swapped", 0) for m in steps.values()) / max(tot("tail/swapped"), 1), 4)}
if smx:   # [v2.5.9z83] the mean (non-thinking - thinking) over every valid token
    summ = {"rows": tot("tail/rows"), "softmix_w": smx, "valid": tot("tail/valid"),
            "delta_mean": round(sum(m.get("tail/delta_mean", 0) * m.get("tail/valid", 0) for m in steps.values()) / max(tot("tail/valid"), 1), 4)}
json.dump({"steps": steps, "tail": summ, "hooks": hooks, "jsonl_lines": len(rows), "budget": budsum}, open(os.path.join(d, "checks.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
[ -n "${PTL_SOFTMIX:-}" ] && finish PASS "unit tests ok; ${PTL_STEPS:-3}-step smoke ok, peak ${peak} MiB; $(python3 -c "import json, sys; s = json.load(open(sys.argv[1]))['tail']; print(f\"soft mix w {s['softmix_w']}: the merge between the two readings and finite on every step, mean (non-thinking - thinking) {s['delta_mean']} over {s['valid']} tokens\")" $D/checks.json)"   # [v2.5.9z83]
finish PASS "unit tests ok; ${PTL_STEPS:-3}-step smoke ok, peak ${peak} MiB; $(python3 -c "import json, sys; s = json.load(open(sys.argv[1]))['tail']; print(f\"{s['rows_box_share']:.0%} of responses with a box, {s['swapped_frac_mean']:.1%} of tokens swapped, mean change of the teacher's log-prob there {s['delta_mean']}\")" $D/checks.json)"
