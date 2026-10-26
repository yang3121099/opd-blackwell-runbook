#!/usr/bin/env bash
# PROBE-VLM-COMPLETE (runbook [v2.5.9z89]; Q-080a, Fable's pre-registration research_program 3f and design ruling 10-26 23:0x UTC):
# the gate of E15-VLM-P2lo, on ONE GPU, from venv-vl (~20 min). The TERM probe's VLM form:
#  a2. (no GPU) the student's (Qwen3-VL-2B-Instruct) and the Instruct teacher's (Qwen3-VL-4B-Instruct) processors and tokenizers
#      equal field by field, and the teacher's prompt = the student's with NO suffix (vlm_teacher_check.py same, PVL_EXPECT_SUFFIX='');
#  s.  the student answers the 300 rows of MMK12 validation once (T=1.0, 1536 tokens, its own template);
#  c.  the Instruct teacher scores each finished answer (no suffix): p(stop) and whether the end token is the argmax where the
#      student stopped; the student scores its own answers the same way (its own p_stop_last); and, a reading only, the
#      thinking teacher Qwen3-VL-4B-Thinking with the empty think block (its non-thinking reading: Q-080b's teacher after the box);
#  v.  PASS when the completion rate (the share of finished answers with the end token the teacher's argmax) >= 0.9; also the
#      teacher's mean p_stop_last, the student's finished share (of 300) and its own p_stop_last, the thinking teacher's reading;
#      a student finished share under 0.5 is noted, not a FAIL (Q-080a trains on every rollout).
# report.json: complete.json's numbers; to logs/<node>/probe_vlm_complete/ (private), an event, .done. Usage: bash probe_vlm_complete.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-VLM-COMPLETE; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_VLM_COMPLETE $(date -u +%T) $*" | tee -a $LOG; }
PY=${PVC_PY:-$OPD_ROOT/venv-vl/bin/python}; S=${VL_STUDENT:-$OPD_ROOT/models/Qwen3-VL-2B-Instruct}
TI=${VL_TEACHER_INSTRUCT:-$OPD_ROOT/models/Qwen3-VL-4B-Instruct}; TT=${VL_TEACHER:-$OPD_ROOT/models/Qwen3-VL-4B-Thinking}
CHK="$RUNBOOK/code/analysis/vlm_teacher_check.py"; DATA=$OPD_ROOT/data/mmk12_val.parquet; N=${PVC_N:-300}
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k in ("same", "complete"):
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/complete*.jsonl $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_vlm_complete >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_vlm_complete_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
[ -x "$PY" ] || { [[ $PY != */* ]] && command -v "$PY" > /dev/null; } || finish FAIL "no venv-vl (BUILD-VENV-VL first)"
for m in $S $TI $TT; do [ -f $m/.fetch_ok ] || { echo "WAIT_INPUTS $T: $(basename $m) not in (no .fetch_ok)"; exit 1; }; done   # waiting, no attempt
# the 300 MMK12 validation rows E15-VLM-P5 validated on: made from the node's copy when missing (prep_mmk12.py, as run_vlm.sh)
[ -f $DATA ] || $PY $RUNBOOK/code/train/prep_mmk12.py $OPD_ROOT/data/hf/FanqingM__MMK12 $OPD_ROOT/data >> $LOG 2>&1 || finish FAIL "MMK12 could not be prepared"
# a2. no GPU
CUDA_VISIBLE_DEVICES= PVL_EXPECT_SUFFIX= $PY $CHK same $D --student $S --teacher $TI >> $LOG 2>&1 || finish FAIL "a2 check did not run: $(tail -c 300 $LOG | tr '\n' ' ')"
python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" $D/same.json || finish FAIL "a2: $(python3 -c "import json; r = json.load(open('$D/same.json')); print({k: r[k] for k in ('preprocessor_equal', 'vocab_equal', 'special_equal', 'template_teacher_is_student_plus_think')})")"
step "a2 ok: the Instruct teacher's processor, tokenizer and template = the student's"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PVC_WAIT_STEP:-10}
done
gpu_step() {   # $1 = what, then the env and the check's arguments
  local what=$1; shift
  CUDA_VISIBLE_DEVICES=$G env "$@" >> $LOG 2>&1 || finish FAIL "the $what step failed: $(tail -c 300 $LOG | tr '\n' ' ')"
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
}
gpu_step student PVL_STUDENT_MAX=1536 $PY $CHK student $D --student $S --teacher $TI --data $DATA --n $N
gpu_step teacher PVL_SUFFIX= PVL_OUT=complete_teacher.jsonl $PY $CHK complete $D --student $S --teacher $TI --data $DATA --n $N
gpu_step self PVL_SUFFIX= PVL_OUT=complete_self.jsonl $PY $CHK complete $D --student $S --teacher $S --data $DATA --n $N
gpu_step thinking PVL_SUFFIX=$'<think>\n\n</think>\n\n' PVL_OUT=complete_think.jsonl $PY $CHK complete $D --student $S --teacher $TT --data $DATA --n $N
$PY $CHK complete_verdict $D --student $S --teacher $TI >> $LOG 2>&1
[ -f $D/complete.json ] || finish FAIL "v: no complete.json: $(tail -c 300 $LOG | tr '\n' ' ')"
SUM=$(python3 -c "import json; r = json.load(open('$D/complete.json')); t = r.get('thinking_teacher_nonthinking') or {}; print(f\"completion rate {r['completion_rate']} of {r['scored']} (>= {r['threshold']}); teacher p_stop_last {r['teacher_p_stop_last_mean']}; student finished {r['student_finished']} ({r['student_finished_n']}/{r['n']}), own p_stop_last {r['student_p_stop_last_mean']}; thinking teacher non-thinking p_stop_last {t.get('p_stop_last_mean')}{'; ' + r['note'] if r['note'] else ''}\")")
python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" $D/complete.json || finish FAIL "v: $SUM"
finish PASS "$SUM"
