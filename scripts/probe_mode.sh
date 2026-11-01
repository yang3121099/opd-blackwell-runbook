#!/usr/bin/env bash
# PROBE-MODE (runbook [v2.5.9z108]; Q-083, research_program §3i): the gate of E10b-P5mode (a visible mode token per prompt).
#  1. the unit tests of code/patches/test_opd_mode.py and test_opd_route.py on the node (the arm's python, with torch: route's
#     sampler-order test runs too), GOPD_REF_INPUT_UTILS = the installed ref_input_utils.py (route's hook then the mode line, once
#     each, the patched file compiles);
#  2. mode_data.py on the node's dapo_train.parquet: every row written, the '[long]' share in [0.45, 0.55] (mix's coin, OPD_SEED);
#  3. a 3-step smoke of E10b-P5mode on the arm's GPUs (#gpus PROBE-MODE = the arm's), then over its route.jsonl: rows over all
#     calls = steps x batch (128) and each call's rows = its batch, think + plain = rows, no prompt without a token, every decision
#     as its token ([long] -> think), stripped = rows in every call (the teacher saw no token), both tokens in every call with a
#     '[long]' share in [0.3, 0.7]; the config names the mode token; mode_len.py over the smoke's rollout dump finds both tokens in
#     every step. report.json (per call think / plain / stripped, the data share, the per-token lengths), to logs/<node>/probe_mode/
#     (private), an event, .done. PASS = all of it. Usage: bash probe_mode.sh <gpu>[,<gpu>]
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-MODE; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_MODE $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k in ("data_check", "mode_check", "len_check"):
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/unit_tests.txt $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  cp $D/smoke/route.jsonl $D/smoke/len_by_token.json $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_mode >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_mode_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
# 1. unit tests with the python the arm trains in
PY=${PMD_PY:-$OPD_ROOT/venv/bin/python}; [ "${VENV:-}" = fa ] && [ -x $OPD_ROOT/venv-fa/bin/python ] && PY=${PMD_PY:-$OPD_ROOT/venv-fa/bin/python}
[ -x "$PY" ] || PY=python3
RU=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ref_input_utils').origin)" 2> /dev/null)
[ -n "$RU" ] && [ -f "$RU" ] || finish FAIL "verl's ref_input_utils.py not found by $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_REF_INPUT_UTILS=$RU $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, os, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
for name in ("test_opd_mode", "test_opd_route"):
    mod = importlib.import_module(name)
    for fn, f in inspect.getmembers(mod, inspect.isfunction):
        if fn.startswith("test_") and f.__module__ == name:
            n += 1
            try:
                f()
            except Exception:
                fails += 1; print("FAILED", name, fn); traceback.print_exc()
try:
    import torch  # noqa: F401  (route's sampler-order test returns early without it: on the node it must run)
except ImportError:
    if os.environ.get("PMD_REQUIRE_TORCH", "1") == "1":
        fails += 1; print("FAILED torch not importable: the sampler-order test did not run")
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
# 2. the training parquet with the tokens
mkdir -p $D/data
out=$($PY $RUNBOOK/scripts/mode_data.py $OPD_ROOT/data/dapo_train.parquet $D/data/train_mode.parquet --seed ${OPD_SEED:-1234} 2>&1 | tail -1)
step "data: $out"
python3 - "$D/data/train_mode.parquet.json" "$D/data_check.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1])) if __import__("os").path.exists(sys.argv[1]) else {}
why = [] if h else ["mode_data.py wrote no header"]
if h and not (h["rows"] > 0 and h["long"] + h["short"] == h["rows"] and 0.45 <= h["share"] <= 0.55):
    why.append(f"rows {h['rows']} long {h['long']} short {h['short']} share {h['share']}")
json.dump(dict(h, why=why), open(sys.argv[2], "w"), indent=1)
PY
WHY=$(python3 -c "import json, sys; print('; '.join(json.load(open(sys.argv[1]))['why']))" $D/data_check.json 2> /dev/null || echo "the data check did not run")
[ -z "$WHY" ] || finish FAIL "data: $WHY"
# 3. the smoke on the arm's GPUs
for g in $(echo $G | tr , ' '); do
  for _w in $(seq 19); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
    [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $g not empty after 180 s"; sleep ${PMD_WAIT_STEP:-10}
  done
done
step "start smoke gpu=$G"
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PMD_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10b-P5mode > $D/smoke.out 2>&1; rc=$?
S=$OPD_ROOT/runs/$T-smoke; mkdir -p $D/smoke; cp $S/train.log $S/config.yaml $S/route.jsonl $D/smoke/ 2> /dev/null
for g in $(echo $G | tr , ' '); do python3 $RUNBOOK/scripts/control.py reap $g $T > /dev/null 2>&1 || true; done
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
grep -q "mode token per prompt" $D/smoke/config.yaml 2> /dev/null || finish FAIL "the smoke's config does not name the mode token"
python3 - "$D/smoke/route.jsonl" "$D/mode_check.json" "${PMD_STEPS:-3}" "${PMD_TBS:-128}" <<'PY'
import json, os, sys
log_p, out, steps, tbs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
calls = [json.loads(x) for x in open(log_p)] if os.path.exists(log_p) else []
why, per = [], []
if sum(c["rows"] for c in calls) != steps * tbs:
    why.append(f"{sum(c['rows'] for c in calls)} teacher rows over {len(calls)} calls, expected {steps} x {tbs}")
for c in calls:
    bad = [d for d in c["decisions"] if d[1] not in ("[long]", "[short]") or d[2] != (d[1] == "[long]")]
    per.append({"call": c["call"], "rows": c["rows"], "think": c["think"], "plain": c["plain"], "unknown": c["unknown"],
                "stripped": c.get("stripped"), "mismatch": len(bad)})
    if c["mode"] != "prefix":
        why.append(f"call {c['call']}: mode {c['mode']}")
    if c["rows"] != c["batch"] or c["think"] + c["plain"] != c["rows"]:
        why.append(f"call {c['call']}: rows {c['rows']} batch {c['batch']} think {c['think']} plain {c['plain']}")
    if c["unknown"] or bad:
        why.append(f"call {c['call']}: {c['unknown']} prompts without a token, {len(bad)} decisions not as their token")
    if c.get("stripped") != c["rows"]:
        why.append(f"call {c['call']}: the teacher saw {c['rows'] - (c.get('stripped') or 0)} prompts with their token")
    if not (0.3 <= c["think"] / max(c["rows"], 1) <= 0.7):
        why.append(f"call {c['call']}: [long] share {c['think']}/{c['rows']}")
json.dump({"per_call": per, "why": why}, open(out, "w"), indent=1)
PY
WHY=$(python3 -c "import json, sys; print('; '.join(json.load(open(sys.argv[1]))['why']))" $D/mode_check.json 2> /dev/null || echo "the mode checks did not run")
[ -z "$WHY" ] || finish FAIL "mode: $WHY"
$PY $RUNBOOK/scripts/mode_len.py $S $T-smoke >> $LOG 2>&1; cp $S/len_by_token.json $D/smoke/ 2> /dev/null
python3 - "$D/smoke/len_by_token.json" "$D/len_check.json" "${PMD_STEPS:-3}" <<'PY'
import json, os, sys
p, out, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = json.load(open(p)) if os.path.exists(p) else {"rows": []}
why = [] if len(r["rows"]) == steps else [f"len_by_token has {len(r['rows'])} steps, expected {steps}"]
for row in r["rows"]:
    if not (row.get("long", {}).get("n") and row.get("short", {}).get("n")) or row.get("none", {}).get("n"):
        why.append(f"step {row['step']}: tokens {sorted(k for k in row if k != 'step')}")
json.dump({"rows": r["rows"], "why": why}, open(out, "w"), indent=1)
PY
WHY=$(python3 -c "import json, sys; print('; '.join(json.load(open(sys.argv[1]))['why']))" $D/len_check.json 2> /dev/null || echo "the length check did not run")
[ -z "$WHY" ] || finish FAIL "lengths: $WHY"
finish PASS "${PMD_STEPS:-3}-step smoke ok; long/short per call $(python3 -c "import json; print(' '.join(f\"{c['think']}/{c['plain']}\" for c in json.load(open('$D/mode_check.json'))['per_call']))"); stripped = rows; data share $(python3 -c "import json; print(json.load(open('$D/data_check.json'))['share'])")"
