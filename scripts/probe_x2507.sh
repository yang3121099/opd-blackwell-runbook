#!/usr/bin/env bash
# PROBE-X2507-<X1|X2> (runbook [v2.5.8i], Q-029; reviewer 17:4x UTC 10-20): the gate of E10d-X1 / E10d-X2 (the student
# Qwen3-4B-Instruct-2507 taught by Qwen3-4B-Thinking-2507 / by the hybrid Qwen3-4B on the thinking template).
# 1. Templates, CPU, in the training's venv: both prompts rendered as G-OPD does (the student's with its tokenizer and
#    enable_thinking=False; the teacher's with the teacher's own tokenizer, plus enable_thinking=True and '<think>\n' when
#    OPD_REF_THINK=1, the P5 patch). PASS needs the teacher prompt to hold exactly one <think> and end with '<think>\n', the
#    student prompt none, and the two vocabularies to be identical (the student's response ids are fed to the teacher).
# 2. A 3-step smoke of the arm on ONE GPU (no checkpoint, upload, validation; nvidia-smi every 5 s): exit 0, no OOM, peak
#    < 260 GB, and the smoke's config naming the 2507 student (and for X1 the 2507 teacher).
# report.json, to logs/<node>/probe_x2507/<task> (private), an event, .done. Usage: bash probe_x2507.sh <gpu> <X1|X2>
set -uo pipefail
G=$1; X=$2; source "$(dirname "$0")/env.sh"
case $X in X1|X2) ;; *) echo "unknown arm $X"; exit 2;; esac
T=PROBE-X2507-$X; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_X2507 $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" "$X" <<'PY'
import json, os, sys
d, res, why, x = sys.argv[1:5]
r = {"result": res, "reason": why, "arm": x}
try:
    vals = [float(v.split(",")[1]) for v in open(os.path.join(d, "mem.csv")) if "," in v]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
p = os.path.join(d, "templates.json")
if os.path.exists(p):
    r["templates"] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_x2507/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_x2507_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
# 1. templates (the venv the arm trains in: 4B seed 42 = the venv in use unless E10D_FA=1)
V=$OPD_ROOT/venv; [ "${E10D_FA:-0}" = 1 ] && V=$OPD_ROOT/venv-fa
SM=${STUDENT_I2507:-Qwen/Qwen3-4B-Instruct-2507}; TM=${TEACHER_T2507:-Qwen/Qwen3-4B-Thinking-2507}; RT=0
[ $X = X2 ] && { TM=$STUDENT_4B; RT=1; }
${PXT_PY:-$V/bin/python} - "$SM" "$TM" "$RT" "$D/templates.json" >> $LOG 2>&1 <<'PY'
import json, os, sys
sm, tm, rt, out = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4]
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
loc = lambda m: m if os.path.isdir(m) else snapshot_download(m, local_files_only=True)
st, tt = AutoTokenizer.from_pretrained(loc(sm)), AutoTokenizer.from_pretrained(loc(tm))
msgs = [{"role": "user", "content": "What is 1 + 1? Please reason step by step, and put your final answer within \\boxed{}."}]
sp = st.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
tp = tt.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=rt) + ("<think>\n" if rt else "")
why = []
if tp.count("<think>") != 1 or not tp.endswith("<think>\n"):
    why.append(f"the teacher prompt holds {tp.count('<think>')} <think> and ends with {tp[-20:]!r}")
if "<think>" in sp:
    why.append("the student prompt holds <think>")
if st.get_vocab() != tt.get_vocab():
    why.append("student and teacher vocabularies differ")
json.dump({"student_prompt_tail": sp[-60:], "teacher_prompt_tail": tp[-60:], "ref_think": rt, "vocab_size": len(st.get_vocab()), "problems": why},
          open(out, "w"), indent=1)
PY
[ -f $D/templates.json ] || finish FAIL "the template check did not run (see probe.log)"
why=$(python3 -c "import json, sys; print('; '.join(json.load(open(sys.argv[1]))['problems']))" $D/templates.json)
[ -z "$why" ] || finish FAIL "templates: $why"
step "templates ok (teacher prompt ends with one <think>, student none, same vocabulary)"
# 2. the 3-step smoke
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PXT_WAIT_STEP:-10}
done
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${PXT_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10d-$X > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PXT_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
grep -q "^student: .*Instruct-2507.* (no_think)$" $D/smoke/config.yaml || finish FAIL "the smoke's student is not Instruct-2507"
[ $X = X2 ] || grep -q "^teacher: .*Thinking-2507.* (ref; think-template=0;" $D/smoke/config.yaml || finish FAIL "the smoke's teacher is not Thinking-2507 with think-template 0"
finish PASS "templates ok; ${PXT_STEPS:-3}-step smoke ok, peak ${peak} MiB"
