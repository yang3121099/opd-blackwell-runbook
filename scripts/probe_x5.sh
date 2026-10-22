#!/usr/bin/env bash
# PROBE-X5 (runbook [v2.5.9z46]; Q-057, the reviewer's gate 10-22 14:4x Beijing, ruling A 10:0x UTC): the gate of E16-X5 (and E16-X5e),
# one GPU, a student without a thinking mode (Qwen2.5-Math-1.5B-Instruct) and its teacher DeepSeek-R1-Distill-Qwen-1.5B:
#  0. both models in ($OPD_ROOT/models/<name>/.fetch_ok), else WAIT_INPUTS (waiting, no attempt);
#  1. the unit tests of code/patches/test_opd_x5.py with the python the arm trains in (torch: the map_ids test runs;
#     GOPD_REF_INPUT_UTILS = the installed ref_input_utils.py: the hooks go into the real file); a failure or no torch is a FAIL;
#  2. code/analysis/x5_probe.py student: the student's own answers to the first 64 MATH-500 problems at 3072 tokens (its 4096
#     positions): the shares ending in 151645 / 151643 / truncated, its accuracy;
#  3. x5_probe.py teacher: the teacher's own answers to the same problems on its thinking template at 16K (completion, accuracy) and its
#     p(151643) at the student's stops after a prompt ending in '<think>\n' (the arm's) and in an empty thought (E16-X5e's); the
#     reviewer's condition for E16-X5e: think < 0.3 and empty >= 0.5 (reported, x5e);
#  4. a 3-step smoke of E16-X5, per call (runs/PROBE-X5-smoke/x5.jsonl): rows = batch = 128 (one teacher pass per sample), mapped =
#     terminated (the stop id at every terminated response's end read as the teacher's EOS), no other special id in a response (the
#     reviewer: the count should be 0), the teacher prompt's tail '<｜Assistant｜><think>\n' with one '<think>' (no empty thought); the
#     smoke's config names x5; no out of memory.
# report.json (the student / teacher / smoke numbers) to logs/<node>/probe_x5 (private), an event, .done. PASS = 1 and 4 hold and 2, 3 ran.
# Usage: bash probe_x5.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-X5; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
S=${STUDENT_X5:-$OPD_ROOT/models/Qwen2.5-Math-1.5B-Instruct}; TE=${TEACHER_X5:-$OPD_ROOT/models/DeepSeek-R1-Distill-Qwen-1.5B}
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
for m in $S $TE; do [ -f $m/.fetch_ok ] || { echo "WAIT_INPUTS $T: $(basename $m) not in (no .fetch_ok)"; exit 1; }; done
step() { echo "PROBE_X5 $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k, f in (("student", "student_summary.json"), ("teacher", "teacher.json"), ("smoke", "smoke_check.json")):
    p = os.path.join(d, f)
    if os.path.exists(p):
        r[k] = json.load(open(p))
r["x5e"] = (r.get("teacher") or {}).get("x5e")
try:
    r["unit_tests"] = open(os.path.join(d, "unit_tests.txt")).read().strip().splitlines()[-1]
except (OSError, IndexError):
    pass
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/unit_tests.txt $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  [ -f $D/smoke/x5.jsonl ] && cp $D/smoke/x5.jsonl $D/up/
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_x5 >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_x5_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
# 1. unit tests with the python the arm trains in
PY=${PX5_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
RU=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ref_input_utils').origin)" 2> /dev/null)
[ -n "$RU" ] && [ -f "$RU" ] || finish FAIL "verl's ref_input_utils.py not found by $PY"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_REF_INPUT_UTILS=$RU $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, os, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_opd_x5")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_opd_x5":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
try:
    import torch  # noqa: F401  (the map_ids test returns early without it: here it must run)
except ImportError:
    if os.environ.get("PX5_REQUIRE_TORCH", "1") == "1":
        fails += 1; print("FAILED torch not importable: the map_ids test did not run")
print(f"{n - fails} passed, {fails} failed, 0 skipped")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PX5_WAIT_STEP:-10}
done
# 2. / 3. the student's answers, then the teacher's (each its own process)
XP=${PX5_PROBE:-$RUNBOOK/code/analysis/x5_probe.py}
step "start: student answers gpu=$G"
CUDA_VISIBLE_DEVICES=$G $PY $XP student --model $S --out $D --n ${PX5_N:-64} >> $LOG 2>&1 || finish FAIL "the student's answers did not run (see probe.log)"
step "student: $(cat $D/student_summary.json 2> /dev/null | tr -d '\n ')"
CUDA_VISIBLE_DEVICES=$G $PY $XP teacher --model $TE --out $D --n ${PX5_N:-64} >> $LOG 2>&1 || finish FAIL "the teacher's measurements did not run (see probe.log)"
step "teacher: $(cat $D/teacher.json 2> /dev/null | tr -d '\n ')"
# 4. the smoke
step "start smoke gpu=$G"
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PX5_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E16-X5 > $D/smoke.out 2>&1; rc=$?
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $OPD_ROOT/runs/$T-smoke/x5.jsonl $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
grep -q "^x5: " $D/smoke/config.yaml 2> /dev/null || finish FAIL "the smoke's config does not name x5"
python3 - "$D/smoke/x5.jsonl" "$D/smoke_check.json" "${PX5_STEPS:-3}" "${PX5_TBS:-128}" <<'PY'
import json, os, sys
log_p, out, steps, tbs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
calls = [json.loads(x) for x in open(log_p)] if os.path.exists(log_p) else []
why, per = [], []
if len(calls) != steps:
    why.append(f"{len(calls)} reference builds for {steps} steps")
for c in calls:
    per.append({k: c.get(k) for k in ("call", "rows", "terminated", "term_151645", "term_151643", "mapped", "truncated", "masked")})
    if c["rows"] != tbs:
        why.append(f"call {c['call']}: {c['rows']} rows, expected {tbs} (one teacher pass per sample)")
    if c["mapped"] != c["terminated"] or c["terminated"] != c["term_151645"] + c["term_151643"]:
        why.append(f"call {c['call']}: mapped {c['mapped']} != terminated {c['terminated']}")
    if c["masked"]:
        why.append(f"call {c['call']}: special ids inside responses {c['masked']} (expected none)")
    tail = c.get("prompt_tail") or ""
    if not tail.endswith("<｜Assistant｜><think>\n") or tail.endswith("<think>\n<think>\n") or c.get("empty_think"):
        why.append(f"call {c['call']}: teacher prompt tail {tail!r}")
tot = {k: sum(c.get(k) or 0 for c in calls) for k in ("rows", "terminated", "term_151645", "term_151643", "truncated")}
json.dump({"per_call": per, "total": tot, "why": why}, open(out, "w"), indent=1)
print("; ".join(why))
PY
WHY=$(python3 -c "import json, sys; print('; '.join(json.load(open(sys.argv[1]))['why']))" $D/smoke_check.json 2> /dev/null || echo "the smoke checks did not run")
[ -z "$WHY" ] || finish FAIL "smoke: $WHY"
finish PASS "unit tests $(tail -1 $D/unit_tests.txt); $(python3 - $D <<'PY'
import json, os, sys
d = sys.argv[1]; s = json.load(open(os.path.join(d, "student_summary.json"))); t = json.load(open(os.path.join(d, "teacher.json")))
k = json.load(open(os.path.join(d, "smoke_check.json")))["total"]
print(f"student 64 at 3072: ends 151645 {s['share_151645']}, 151643 {s['share_151643']}, acc {s['acc']}; teacher at 16K: completion "
      f"{t['completion']}, acc {t['accuracy']}; teacher p(151643) at the student's stops: think {t['p_stop_think']['mean']}, empty "
      f"{t['p_stop_empty']['mean']} (n {t['p_stop_think']['n']}; x5e {t['x5e']}); smoke {k['rows']} rows, {k['terminated']} terminated "
      f"({k['term_151645']} / {k['term_151643']}) all mapped, no special id inside")
PY
)"
