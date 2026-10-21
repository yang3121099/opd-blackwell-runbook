#!/usr/bin/env bash
# PROBE-TOOL (runbook [v2.5.9b]; research_program §1.7, reviewer 07:2x UTC 10-21): the tool-call OPD code path, end to end on
# the 10 synthetic samples (code/agent/data/synthetic_tools.jsonl) until the real sets are placed.
#  1. the unit tests of code/agent (grader, BFCL-style match, budget curve, prep) on the node, with the student's real chat
#     template (QWEN_TEMPLATE = its tokenizer_config.json)
#  2. prep_tools_parquet.py with the real student (1.7B) and teacher (8B) tokenizers: the tool block as a system message
#     renders byte for byte like the native tools= form for both (student non-thinking, teacher thinking), x40 rows for a
#     full batch; one rendered prompt kept (sample.txt)
#  3. a 3-step smoke of E10b-P5tool (P5's recipe on these prompts, the rule grader as the logged reward) on ONE GPU: exit 0,
#     no OOM, every step in train.log with critic/score/mean, its config naming the tool data
#  4. tool_eval.py on the 10 prompts (n 4, budgets 512 / 1024 / 1536 / 4096): the 1.7B student (non-thinking) and the 8B
#     teacher (thinking): rule score, legal-call rate, AST match, forbidden calls, mean tokens per budget
# PASS = 1-4 ran (the scores are readings, not conditions). report.json, to logs/<node>/probe_tool/<task> (private), an
# event, .done. ~20 min on one GPU.
# Usage: bash probe_tool.sh <gpu>
set -uo pipefail
G=$1; T=PROBE-TOOL; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_TOOL $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, re, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k in ("eval_student", "eval_teacher"):
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k] = json.load(open(p))
p = os.path.join(d, "smoke", "train.log")
if os.path.exists(p):
    r["smoke_score_per_step"] = [float(x) for x in re.findall(r"critic/score/mean:([-0-9.e]+)", open(p, errors="ignore").read())]
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/unit_tests.txt $D/tools_sample.txt $D/up/ 2> /dev/null
  [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  for k in eval_student eval_teacher; do [ -f $D/$k.jsonl ] && cp $D/$k.jsonl $D/up/; done
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_tool/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_tool_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
PY=${PT_PY:-$OPD_ROOT/venv/bin/python}
A=$RUNBOOK/code/agent
# 1. unit tests
step "start gpu=$G unit tests"
( cd $A && CUDA_VISIBLE_DEVICES= QWEN_TEMPLATE=$STUDENT_17B/tokenizer_config.json $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_tool_agent")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_tool_agent":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
) || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
step "unit tests: $(tail -1 $D/unit_tests.txt)"
# 2. prep with the real tokenizers
( cd $A && $PY prep_tools_parquet.py data/synthetic_tools.jsonl $D/tools $STUDENT_17B $TEACHER --repeat ${PT_REPEAT:-40} ) > $D/prep.out 2>&1 \
  || finish FAIL "prep: $(tail -c 300 $D/prep.out | tr '\n' ' ')"
grep -q "^TOOLS_PREP_OK" $D/prep.out || finish FAIL "prep did not report TOOLS_PREP_OK"
step "prep: $(grep '^TOOLS_PREP_OK' $D/prep.out)"
# 3. the 3-step smoke
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PT_WAIT_STEP:-10}
done
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PT_STEPS:-3} PROBE_TASK=$T-smoke TOOL_DATA=$D/tools bash $RUNBOOK/scripts/run_arm.sh $G E10b-P5tool > $D/smoke.out 2>&1; rc=$?
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
grep -q "^data: tool calls" $D/smoke/config.yaml 2> /dev/null || finish FAIL "the smoke's config does not name the tool data"
ns=$(grep -c "critic/score/mean:" $D/smoke/train.log 2> /dev/null); [ "${ns:-0}" -ge ${PT_STEPS:-3} ] || finish FAIL "train.log has ${ns:-0} steps with a score, expected ${PT_STEPS:-3}"
step "smoke: ${PT_STEPS:-3} steps"
# 4. the evaluation, student and teacher
( cd $A && CUDA_VISIBLE_DEVICES=$G $PY tool_eval.py --model $STUDENT_17B --data $D/tools_val.parquet --out $D/eval_student.json \
    --samples $D/eval_student.jsonl --n 4 --mode no_think ) > $D/eval_student.out 2>&1 || finish FAIL "student eval: $(tail -c 300 $D/eval_student.out | tr '\n' ' ')"
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
( cd $A && CUDA_VISIBLE_DEVICES=$G $PY tool_eval.py --model $TEACHER --data $D/tools_val.parquet --out $D/eval_teacher.json \
    --samples $D/eval_teacher.jsonl --n 4 --mode think ) > $D/eval_teacher.out 2>&1 || finish FAIL "teacher eval: $(tail -c 300 $D/eval_teacher.out | tr '\n' ' ')"
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
s=$(python3 -c "import json, sys; a, b = (json.load(open(p))['budgets'] for p in sys.argv[1:3]); k = max(a, key=int); print(f\"student {a[k]['score']} / teacher {b[k]['score']} at {k}\")" $D/eval_student.json $D/eval_teacher.json)
finish PASS "unit tests, prep, ${PT_STEPS:-3}-step smoke and evaluation ok; rule score $s"
