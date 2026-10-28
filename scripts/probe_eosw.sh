#!/usr/bin/env bash
# PROBE-EOSW (runbook [v2.5.9z102]; Q-087, Fable 10-27 15:5x UTC; ops-lead's A1 conditions: the unit tests with every weight 1, a
# single-task smoke as the gate): the gate of E10b-P5-eosw. On ONE GPU:
#  1. the unit tests (the arm's python, with torch, no GPU): code/patches/test_opd_eosw.py (the last box, the window in numpy and torch,
#     the loss with every weight 1 equal to the unweighted one - allclose rtol 1e-6 / atol 1e-7 -, the advantage factor equal to a
#     per-token loss factor under the clip and dual clip, the window's tokens getting w times the gradient) and test_eosw_patch.py against
#     the INSTALLED ref_input_utils.py / dp_actor.py (every anchor once, the patched text compiles, idempotent, the hooks in place); then
#     eosw_patch.py on the installed files
#  2. a 3-step smoke of P5 (E10b-P5, the step-time reference) and one of E10b-P5-eosw on the same GPU (SMOKE=1: no checkpoint, upload,
#     validation; nvidia-smi every 5 s)
#  3. PASS = both exit 0, no OOM, peak < 260 GB; the eosw smoke's config naming the weighting; every step in its train.log with a
#     finite actor/grad_norm and eosw/weighted_frac in (0, PEW_FRAC_MAX 0.05]; its eosw.jsonl with rows that have a box and weighted
#     positions; steps 2-3 mean step time <= PEW_RATIO (1.2) x the P5 smoke's; the installed files holding the hooks (1 in
#     ref_input_utils.py, 2 in dp_actor.py)
# report.json, to logs/<node>/probe_eosw/ (private), an event, .done. ~25-35 min on one GPU. Usage: bash probe_eosw.sh <gpu>
set -uo pipefail
G=$1; T=${PEW_PROBE:-PROBE-EOSW}; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv
MARK="# [opd-variants patch v2] [v2.5.9z102] position-weighted OPD"
step() { echo "PROBE_EOSW $(date -u +%T) $*" | tee -a $LOG; }
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
  for s in p5 eosw; do [ -f $D/$s/train.log ] && tail -c 200000 $D/$s/train.log > $D/up/train_tail_$s.log; done
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_eosw/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_eosw_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
PY=${PEW_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
RI=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ref_input_utils').origin)" 2> /dev/null)
AC=$($PY -c "import importlib.util as u; print(u.find_spec('verl.workers.actor.dp_actor').origin)" 2> /dev/null)
[ -n "$RI" ] && [ -f "$RI" ] && [ -n "$AC" ] && [ -f "$AC" ] || finish FAIL "verl's ref_input_utils.py / dp_actor.py not found by $PY"
step "start gpu=$G unit tests with $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_REF_INPUT_UTILS=$RI GOPD_DP_ACTOR=$AC $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
for modname in ("test_opd_eosw", "test_eosw_patch"):
    mod = importlib.import_module(modname)
    for name, f in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and f.__module__ == modname:
            n += 1
            try:
                f()
            except Exception:
                fails += 1; print("FAILED", modname, name); traceback.print_exc()
try:
    import torch  # noqa: F401  (the torch tests return early without it: on the node they must run)
except ImportError:
    fails += 1; print("FAILED torch not importable: the torch tests did not run")
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
GOPD_REF_INPUT_UTILS=$RI GOPD_DP_ACTOR=$AC $PY $RUNBOOK/code/patches/eosw_patch.py >> $LOG 2>&1 || finish FAIL "eosw_patch.py failed on the installed files"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PEW_WAIT_STEP:-10}
done
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
for pair in p5:E10b-P5 eosw:E10b-P5-eosw; do
  s=${pair%%:*}; arm=${pair#*:}
  rm -rf $OPD_ROOT/runs/$T-$s
  step "smoke $s ($arm, ${PEW_STEPS:-3} steps)"
  SMOKE=1 SMOKE_STEPS=${PEW_STEPS:-3} PROBE_TASK=$T-$s bash $RUNBOOK/scripts/run_arm.sh $G $arm > $D/smoke_$s.out 2>&1; rc=$?
  mkdir -p $D/$s; cp $OPD_ROOT/runs/$T-$s/train.log $OPD_ROOT/runs/$T-$s/config.yaml $OPD_ROOT/runs/$T-$s/eosw.jsonl $D/$s/ 2> /dev/null
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
  [ $rc = 0 ] || finish FAIL "smoke $s exited $rc: $(tail -c 300 $D/smoke_$s.out | tr '\n' ' ')"
  grep -q "CUDA out of memory\|OutOfMemoryError" $D/$s/train.log 2> /dev/null && finish FAIL "out of memory in the $s smoke"
done
kill $MP 2> /dev/null
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ "${peak:-0}" -lt ${PEW_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
why=$(python3 - "$D" "${PEW_STEPS:-3}" "$RI" "$AC" "${PEW_RATIO:-1.2}" "${PEW_FRAC_MAX:-0.05}" "$MARK" <<'PY'
import json, math, os, re, sys
d, nsteps, ri, ac, rmax, fmax, mark = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6]), sys.argv[7]
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
why = []
def steps(path):
    out = {}
    for st, body in re.findall(r"(?:^|[ )\]])step:(\d+) - (.*)", rd(path)):
        out[int(st)] = {k: float(v) for k, v in re.findall(r"((?:eosw/[a-z_]+|actor/grad_norm|timing_s/step|critic/score/mean)):([-0-9.e]+|nan|inf)", body)}
    return out
p5, ew = steps(os.path.join(d, "p5", "train.log")), steps(os.path.join(d, "eosw", "train.log"))
if "position-weighted: the " not in rd(os.path.join(d, "eosw", "config.yaml")):
    why.append("the eosw smoke's config does not name the weighting")
for name, S in (("p5", p5), ("eosw", ew)):
    if sorted(S) != list(range(1, nsteps + 1)):
        why.append(f"the {name} smoke's train.log steps {sorted(S)}, expected 1..{nsteps}")
for st, m in sorted(ew.items()):
    if not math.isfinite(m.get("actor/grad_norm", float("nan"))):
        why.append(f"step {st}: grad_norm {m.get('actor/grad_norm')} not finite")
    f = m.get("eosw/weighted_frac")
    if f is None or not 0 < f <= fmax:
        why.append(f"step {st}: eosw/weighted_frac {f} not in (0, {fmax}]")
log = [json.loads(l) for l in rd(os.path.join(d, "eosw", "eosw.jsonl")).splitlines() if l.strip()]
rows_box, weighted = sum(r.get("rows_box", 0) for r in log), sum(r.get("weighted", 0) for r in log)
if not log or not rows_box or not weighted:
    why.append(f"eosw.jsonl: {len(log)} calls, {rows_box} rows with a box, {weighted} positions weighted")
late = lambda S: [m["timing_s/step"] for st, m in S.items() if st >= 2 and "timing_s/step" in m]
tp, te = late(p5), late(ew)
ratio = (sum(te) / len(te)) / (sum(tp) / len(tp)) if tp and te else None
if ratio is None or ratio > rmax:
    why.append(f"step time ratio eosw / P5 {ratio} above {rmax}")
hooks = (rd(ri).count(mark), rd(ac).count(mark))
if hooks != (1, 2):
    why.append(f"the installed files hold the hooks {hooks} times, expected (1, 2)")
fr = [m["eosw/weighted_frac"] for m in ew.values() if "eosw/weighted_frac" in m]
json.dump({"p5": p5, "eosw": ew, "step_ratio": ratio, "hooks": hooks, "calls": len(log), "rows_box": rows_box, "weighted": weighted,
           "weighted_frac_mean": sum(fr) / len(fr) if fr else None}, open(os.path.join(d, "checks.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "unit tests ok; smokes ok, peak ${peak} MiB; $(python3 -c "import json, sys; c = json.load(open(sys.argv[1])); print(f\"weighted share {c['weighted_frac_mean']:.4f} of the tokens, {c['rows_box']} rows with a box over {c['calls']} calls, step time x{c['step_ratio']:.2f} of P5\")" $D/checks.json)"
