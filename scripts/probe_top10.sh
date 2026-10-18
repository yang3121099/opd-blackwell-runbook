#!/usr/bin/env bash
# PROBE-TOP10 (runbook v2.5.7h; lab Q-014): the gate of the token-selective arms E10b-P5top10 / E10b-P5rand10m.
#  1. the advantage-variant unit tests on the node (code/patches/test_opd_variants.py, test_variant_dispatch.py, with
#     torch: the venv-fa python, else the venv in use; pytest when installed, else a plain runner of the test_ functions)
#  2. a 3-step smoke of each arm (SMOKE=1, the node's VENV / FIXED_MICRO_BSZ / GC): exit 0, and in train.log at every step
#     opd/sel_kept_frac in [0.08, 0.12] (the 10% selection reached the update through the P6 hook) and the token classes
#     (opd/sel_eos_frac ...); P5rand10m also opd/rand_scale_mean >= 1
# report.json: result PASS / FAIL with the reasons; run_arm.sh starts the two arms only after PASS. report and logs go to
# logs/<node>/probe_top10/ (private); an event probe_top10_done; .done. ~15 min on one GPU.
# Usage: bash probe_top10.sh <gpu>
set -uo pipefail
G=$1; T=PROBE-TOP10; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_TOP10 $(date -u +%T) $*" | tee -a $LOG; }
PY=$OPD_ROOT/venv-fa/bin/python; [ -x $PY ] || PY=$OPD_ROOT/venv/bin/python
step "start gpu=$G unit tests with $(basename $(dirname $(dirname $PY)))"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
for m in ("test_opd_variants", "test_variant_dispatch"):
    mod = importlib.import_module(m)
    for name, f in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and f.__module__ == m:
            n += 1
            try:
                f()
            except Exception:
                fails += 1; print("FAILED", m, name); traceback.print_exc()
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
); echo $? > $D/unit_tests.rc
step "unit tests: $(tail -1 $D/unit_tests.txt)"
for A in P5top10 P5rand10m; do
  rm -rf $OPD_ROOT/runs/$T-$A
  step "smoke E10b-$A (3 steps)"
  SMOKE=1 SMOKE_STEPS=3 PROBE_TASK=$T-$A bash $RUNBOOK/scripts/run_arm.sh $G E10b-$A > $D/$A.out 2>&1; echo $? > $D/$A.rc
  cp $OPD_ROOT/runs/$T-$A/train.log $D/$A.train.log 2> /dev/null
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
  step "smoke E10b-$A exit $(cat $D/$A.rc)"
done
python3 - $D > $D/report.json <<'PY'
import json, os, re, sys
d = sys.argv[1]
def rd(p, default=""):
    try:
        return open(os.path.join(d, p), errors="ignore").read()
    except OSError:
        return default
r = {"unit_tests": {"rc": rd("unit_tests.rc", "1").strip(), "summary": (rd("unit_tests.txt").strip().splitlines() or [""])[-1]}, "arms": {}, "reasons": []}
if r["unit_tests"]["rc"] != "0":
    r["reasons"].append("unit tests failed")
for a in ("P5top10", "P5rand10m"):
    text = rd(f"{a}.train.log")
    steps = []
    for st, body in re.findall(r"(?:^|[ )])step:(\d+) - (.*?)(?=(?:^|[ )])step:\d+ - |\Z)", text, re.S):
        v = {k: float(x) for k, x in re.findall(r"(opd/(?:sel_[a-z_]+|rand_scale_mean|eos_adv_mean)):([-0-9.e]+)", body)}
        steps.append(dict(v, step=int(st)))
    x = {"rc": rd(f"{a}.rc", "1").strip(), "steps": steps}
    kept = [s.get("opd/sel_kept_frac") for s in steps]
    if x["rc"] != "0":
        r["reasons"].append(f"{a}: smoke exit {x['rc']}")
    if len(steps) < 3 or any(k is None or not 0.08 <= k <= 0.12 for k in kept):
        r["reasons"].append(f"{a}: opd/sel_kept_frac {kept} (want 3 steps in [0.08, 0.12])")
    if not steps or any("opd/sel_eos_frac" not in s for s in steps):
        r["reasons"].append(f"{a}: no token classes (opd/sel_eos_frac)")
    if a == "P5rand10m" and (not steps or any(s.get("opd/rand_scale_mean", 0) < 1 for s in steps)):
        r["reasons"].append(f"{a}: opd/rand_scale_mean missing or < 1")
    r["arms"][a] = x
r["result"] = "FAIL" if r["reasons"] else "PASS"
print(json.dumps(r, indent=1))
PY
RES=$(python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print(r['result'], '; '.join(r['reasons']))" $D/report.json 2> /dev/null || echo "FAIL report missing")
mkdir -p $D/up; cp $D/report.json $D/probe.log $D/unit_tests.txt $D/*.rc $D/up/ 2> /dev/null
for A in P5top10 P5rand10m; do tail -c 200000 $D/$A.train.log > $D/up/$A.train.log 2> /dev/null; done
python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_top10/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
python3 $RUNBOOK/scripts/control.py event $T probe_top10_done "$RES" > /dev/null 2>&1
step "DONE $RES"; touch $D/.done
