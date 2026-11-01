#!/usr/bin/env bash
# PROBE-FRET (runbook [v2.5.9z118]; research_program 3q, Fable 08:3x UTC, lit_6pro_consensus_20261101 3): the gate of the three beyond
# arms E10b-P5fret / -P5topd / -P5fret-topd. On ONE GPU:
#  1. the unit tests (the arm's python, with torch, no GPU): code/patches/test_opd_shape.py (the suffix credit against its definition
#     position by position, TOP-D's bounded reward and its range, no overflow, both switches off = the advantage untouched, topd before
#     the credit) and test_shape_patch.py against the INSTALLED dp_actor.py (the anchor once, compiles, idempotent, the hook just
#     before the policy loss); then shape_patch.py on the installed file
#  2. a 3-step smoke of P5 (E10b-P5, the step-time reference) and one of each arm on the same GPU (SMOKE=1: no checkpoint, upload,
#     validation; nvidia-smi every 5 s)
#  3. PASS = every smoke exits 0, no out of memory; each arm's config naming its shaping; every step of every smoke in its train.log
#     with a finite actor/grad_norm; the fret arms with shape/fret_abs_mean at every step, the P5 smoke with no shape/ metric; the
#     topd arms' reward range at every step: shape/topd_min >= log(1 - a) and shape/topd_max <= log(a rho_max + 1 - a) (verl averages
#     the micro-batches' values; each one meets the bounds, so the means do); each arm's steps 2-3 mean step time <= PFR_RATIO (1.2) x
#     the P5 smoke's; the installed dp_actor.py holding the hook once
# report.json, to logs/<node>/probe_fret/ (private), an event, .done. ~40-50 min on one GPU. Usage: bash probe_fret.sh <gpu>
set -uo pipefail
G=$1; T=${PFR_PROBE:-PROBE-FRET}; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv
MARK="# [opd-variants patch v2] [v2.5.9z118] fret / topd reward shaping"
ARMS="fret:E10b-P5fret topd:E10b-P5topd fret-topd:E10b-P5fret-topd"
step() { echo "PROBE_FRET $(date -u +%T) $*" | tee -a $LOG; }
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
  for s in p5 fret topd fret-topd; do [ -f $D/$s/train.log ] && tail -c 200000 $D/$s/train.log > $D/up/train_tail_$s.log; done
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_fret/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_fret_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
PY=${PFR_PY:-$OPD_ROOT/venv/bin/python}; [ "${VENV:-}" = fa ] && [ -x $OPD_ROOT/venv-fa/bin/python ] && PY=${PFR_PY:-$OPD_ROOT/venv-fa/bin/python}   # the arm's venv (H800: fa)
[ -x "$PY" ] || PY=python3
AC=$($PY -c "import importlib.util as u; print(u.find_spec('verl.workers.actor.dp_actor').origin)" 2> /dev/null)
[ -n "$AC" ] && [ -f "$AC" ] || finish FAIL "verl's dp_actor.py not found by $PY"
step "start gpu=$G unit tests with $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_DP_ACTOR=$AC $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, tempfile, pathlib, traceback
sys.path.insert(0, ".")
n = fails = 0
for modname in ("test_opd_shape", "test_shape_patch"):
    mod = importlib.import_module(modname)
    for name, f in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and f.__module__ == modname:
            n += 1
            try:
                f(pathlib.Path(tempfile.mkdtemp())) if "tmp_path" in inspect.signature(f).parameters else f()
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
GOPD_DP_ACTOR=$AC $PY $RUNBOOK/code/patches/shape_patch.py >> $LOG 2>&1 || finish FAIL "shape_patch.py failed on the installed dp_actor.py"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PFR_WAIT_STEP:-10}
done
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
for pair in p5:E10b-P5 $ARMS; do
  s=${pair%%:*}; arm=${pair#*:}
  rm -rf $OPD_ROOT/runs/$T-$s
  step "smoke $s ($arm, ${PFR_STEPS:-3} steps)"
  SMOKE=1 SMOKE_STEPS=${PFR_STEPS:-3} PROBE_TASK=$T-$s bash $RUNBOOK/scripts/run_arm.sh $G $arm > $D/smoke_$s.out 2>&1; rc=$?
  mkdir -p $D/$s; cp $OPD_ROOT/runs/$T-$s/train.log $OPD_ROOT/runs/$T-$s/config.yaml $D/$s/ 2> /dev/null
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
  [ $rc = 0 ] || finish FAIL "smoke $s exited $rc: $(tail -c 300 $D/smoke_$s.out | tr '\n' ' ')"
  grep -q "CUDA out of memory\|OutOfMemoryError" $D/$s/train.log 2> /dev/null && finish FAIL "out of memory in the $s smoke"
done
kill $MP 2> /dev/null
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
why=$(python3 - "$D" "${PFR_STEPS:-3}" "$AC" "${PFR_RATIO:-1.2}" "${TOPD_ALPHA:-0.2}" "$MARK" <<'PY'
import json, math, os, re, sys
d, nsteps, ac, rmax, alpha, mark = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
why = []
def steps(path):
    out = {}
    for st, body in re.findall(r"(?:^|[ )\]])step:(\d+) - (.*)", rd(path)):
        out[int(st)] = {k: float(v) for k, v in re.findall(r"((?:shape/[a-z_]+|actor/grad_norm|timing_s/step|critic/score/mean)):([-+0-9.e]+|nan|inf)", body)}
    return out
S = {s: steps(os.path.join(d, s, "train.log")) for s in ("p5", "fret", "topd", "fret-topd")}
for s, M in S.items():
    if sorted(M) != list(range(1, nsteps + 1)):
        why.append(f"the {s} smoke's train.log steps {sorted(M)}, expected 1..{nsteps}")
    for st, m in sorted(M.items()):
        if not math.isfinite(m.get("actor/grad_norm", float("nan"))):
            why.append(f"{s} step {st}: grad_norm {m.get('actor/grad_norm')} not finite")
        has = sorted(k for k in m if k.startswith("shape/"))
        if s == "p5" and has:
            why.append(f"p5 step {st}: shaping metrics {has} with both switches off")
        if "fret" in s and "shape/fret_abs_mean" not in m:
            why.append(f"{s} step {st}: no shape/fret_abs_mean")
        if "topd" in s:
            lo, hi, rho = m.get("shape/topd_min"), m.get("shape/topd_max"), m.get("shape/rho_max")
            if lo is None or hi is None or rho is None:
                why.append(f"{s} step {st}: no topd range metrics")
            elif lo < math.log(1 - alpha) - 1e-4 or hi > math.log(alpha * rho + 1 - alpha) + 1e-3:
                why.append(f"{s} step {st}: topd range [{lo}, {hi}] outside [log {1 - alpha:g}, log({alpha:g} * {rho:g} + {1 - alpha:g})]")
        if "topd" not in s and "shape/topd_min" in m:
            why.append(f"{s} step {st}: topd metrics without OPD_TOPD_ALPHA")
cfg = {s: rd(os.path.join(d, s, "config.yaml")) for s in ("fret", "topd", "fret-topd")}
for s, c in cfg.items():
    if "reward-shaped: " not in c or ("fret" in s) != ("suffix credit" in c) or ("topd" in s) != ("TOP-D" in c):
        why.append(f"the {s} smoke's config does not name its shaping")
late = lambda M: [m["timing_s/step"] for st, m in M.items() if st >= 2 and "timing_s/step" in m]
tp = late(S["p5"]); ratios = {}
for s in ("fret", "topd", "fret-topd"):
    te = late(S[s])
    ratios[s] = (sum(te) / len(te)) / (sum(tp) / len(tp)) if tp and te else None
    if ratios[s] is None or ratios[s] > rmax:
        why.append(f"step time ratio {s} / P5 {ratios[s]} above {rmax}")
hooks = rd(ac).count(mark)
if hooks != 1:
    why.append(f"the installed dp_actor.py holds the hook {hooks} times, expected 1")
json.dump({"smokes": S, "step_ratio": ratios, "hooks": hooks, "alpha": alpha}, open(os.path.join(d, "checks.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "unit tests ok; four smokes ok, peak ${peak} MiB; $(python3 -c "import json, sys; c = json.load(open(sys.argv[1])); print('step time / P5 ' + ', '.join(f'{k} {v:.2f}' for k, v in c['step_ratio'].items()))" $D/checks.json)"
