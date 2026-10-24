#!/usr/bin/env bash
# PROBE-DENSE (runbook [v2.5.9z76]; Q-065, Fable 10-24 02:0x Beijing): the gate of the dense arms E10b-P5-dense / E10b-P5-akl. On ONE GPU:
#  1. the unit tests (the arm's python, with torch, no GPU): code/patches/test_opd_dense.py (the divergences against brute force over the
#     vocabulary, numpy and torch, gradients) and test_dense_patch.py against the INSTALLED dp_actor.py / fsdp_workers.py (every anchor
#     once, the patched text compiles, idempotent); then dense_patch.py on the installed files
#  2. a 3-step smoke of P5 (E10b-P5, the step-time reference) and one of E10b-P5-dense on the same GPU (SMOKE=1: no checkpoint, upload,
#     validation; nvidia-smi every 5 s)
#  3. PASS = both exit 0, no OOM, peak < 260 GB; the dense smoke's config naming the dense loss; every step in its train.log with a finite
#     actor/grad_norm and dense/rkl, dense/corr_sampled (the Pearson r between the dense and the sampled reverse KL over the valid tokens)
#     above PDN_CORR (0.3; Fable's A1: a single-sample estimate correlates only loosely with the exact KL) on average over the steps,
#     the ratio of the token means dense/rkl / dense/rkl_sampled (the same expectation) in [PDN_RLO 0.5, PDN_RHI 2] over the steps
#     (catches a sign or index error), and its steps 2-3 mean step time <= PDN_RATIO (1.5) x the P5 smoke's; the
#     installed files holding the hooks (8 in dp_actor.py, 1 in fsdp_workers.py)
# report.json (per step: dense/rkl, dense/fkl, dense/mass_topk, dense/corr_sampled, the step times, the ratio), to
# logs/<node>/probe_dense/ (private), an event, .done. ~30-40 min on one GPU. Usage: bash probe_dense.sh <gpu>
set -uo pipefail
G=$1; T=PROBE-DENSE; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv
step() { echo "PROBE_DENSE $(date -u +%T) $*" | tee -a $LOG; }
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
  for s in p5 dense; do [ -f $D/$s/train.log ] && tail -c 200000 $D/$s/train.log > $D/up/train_tail_$s.log; done
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_dense/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_dense_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
PY=${PDN_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
AC=$($PY -c "import importlib.util as u; print(u.find_spec('verl.workers.actor.dp_actor').origin)" 2> /dev/null)
WK=$($PY -c "import importlib.util as u; print(u.find_spec('verl.workers.fsdp_workers').origin)" 2> /dev/null)
[ -n "$AC" ] && [ -f "$AC" ] && [ -n "$WK" ] && [ -f "$WK" ] || finish FAIL "verl's dp_actor.py / fsdp_workers.py not found by $PY"
step "start gpu=$G unit tests with $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_DP_ACTOR=$AC GOPD_FSDP_WORKERS=$WK $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
for modname in ("test_opd_dense", "test_dense_patch"):
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
GOPD_DP_ACTOR=$AC GOPD_FSDP_WORKERS=$WK $PY $RUNBOOK/code/patches/dense_patch.py >> $LOG 2>&1 || finish FAIL "dense_patch.py failed on the installed files"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PDN_WAIT_STEP:-10}
done
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT
for pair in p5:E10b-P5 dense:E10b-P5-dense; do
  s=${pair%%:*}; arm=${pair#*:}
  rm -rf $OPD_ROOT/runs/$T-$s
  step "smoke $s ($arm, ${PDN_STEPS:-3} steps)"
  SMOKE=1 SMOKE_STEPS=${PDN_STEPS:-3} PROBE_TASK=$T-$s bash $RUNBOOK/scripts/run_arm.sh $G $arm > $D/smoke_$s.out 2>&1; rc=$?
  mkdir -p $D/$s; cp $OPD_ROOT/runs/$T-$s/train.log $OPD_ROOT/runs/$T-$s/config.yaml $D/$s/ 2> /dev/null
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
  [ $rc = 0 ] || finish FAIL "smoke $s exited $rc: $(tail -c 300 $D/smoke_$s.out | tr '\n' ' ')"
  grep -q "CUDA out of memory\|OutOfMemoryError" $D/$s/train.log 2> /dev/null && finish FAIL "out of memory in the $s smoke"
done
kill $MP 2> /dev/null
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ "${peak:-0}" -lt ${PDN_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
why=$(python3 - "$D" "${PDN_STEPS:-3}" "$AC" "$WK" "${PDN_CORR:-0.3}" "${PDN_RATIO:-1.5}" "${PDN_RLO:-0.5}" "${PDN_RHI:-2}" <<'PY'
import json, math, os, re, sys
d, nsteps, ac, wk, cmin, rmax = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6])
rlo, rhi = float(sys.argv[7]), float(sys.argv[8])
rd = lambda p: open(p, errors="ignore").read() if os.path.exists(p) else ""
why = []
def steps(path):
    out = {}
    for st, body in re.findall(r"(?:^|[ )\]])step:(\d+) - (.*)", rd(path)):
        out[int(st)] = {k: float(v) for k, v in re.findall(r"((?:dense/[a-z_]+|actor/grad_norm|timing_s/step|critic/score/mean)):([-0-9.e]+|nan|inf)", body)}
    return out
p5, dn = steps(os.path.join(d, "p5", "train.log")), steps(os.path.join(d, "dense", "train.log"))
if "dense rkl over the teacher's top-" not in rd(os.path.join(d, "dense", "config.yaml")):
    why.append("the dense smoke's config does not name the dense loss")
for name, S in (("p5", p5), ("dense", dn)):
    if sorted(S) != list(range(1, nsteps + 1)):
        why.append(f"the {name} smoke's train.log steps {sorted(S)}, expected 1..{nsteps}")
for st, m in sorted(dn.items()):
    if not math.isfinite(m.get("actor/grad_norm", float("nan"))):
        why.append(f"step {st}: grad_norm {m.get('actor/grad_norm')} not finite")
    if not math.isfinite(m.get("dense/rkl", float("nan"))):
        why.append(f"step {st}: dense/rkl {m.get('dense/rkl')} not finite")
corr = [m["dense/corr_sampled"] for m in dn.values() if "dense/corr_sampled" in m]
cm = sum(corr) / len(corr) if corr else None
if cm is None or not cm > cmin:
    why.append(f"dense / sampled correlation {cm} not above {cmin}")
dm = [m["dense/rkl"] for m in dn.values() if "dense/rkl" in m and "dense/rkl_sampled" in m]
sm = [m["dense/rkl_sampled"] for m in dn.values() if "dense/rkl" in m and "dense/rkl_sampled" in m]
mratio = (sum(dm) / len(dm)) / (sum(sm) / len(sm)) if dm and sum(sm) else None
if mratio is None or not rlo <= mratio <= rhi:
    why.append(f"dense / sampled mean ratio {mratio} outside [{rlo}, {rhi}]")
late = lambda S: [m["timing_s/step"] for st, m in S.items() if st >= 2 and "timing_s/step" in m]
tp, td = late(p5), late(dn)
ratio = (sum(td) / len(td)) / (sum(tp) / len(tp)) if tp and td else None
if ratio is None or ratio > rmax:
    why.append(f"step time ratio dense / P5 {ratio} above {rmax}")
hooks = (rd(ac).count("# [opd-variants patch v2] [v2.5.9z76] dense OPD"), rd(wk).count("# [opd-variants patch v2] [v2.5.9z76] dense OPD"))
if hooks != (8, 1):
    why.append(f"the installed files hold the hooks {hooks} times, expected (8, 1)")
json.dump({"p5": p5, "dense": dn, "corr_mean": cm, "mean_ratio": mratio, "step_ratio": ratio, "hooks": hooks,
           "mass_topk_mean": (sum(m.get("dense/mass_topk", 0) for m in dn.values()) / max(len(dn), 1))}, open(os.path.join(d, "checks.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "unit tests ok; smokes ok, peak ${peak} MiB; $(python3 -c "import json, sys; c = json.load(open(sys.argv[1])); print(f\"dense / sampled r {c['corr_mean']:.3f}, mean ratio {c['mean_ratio']:.2f}, step time x{c['step_ratio']:.2f} of P5, teacher top-K mass {c['mass_topk_mean']:.3f}\")" $D/checks.json)"
