#!/usr/bin/env bash
# PROBE-VLM (runbook v2.5.9z; Q-051, the reviewer's gate research_program §3c): the gate of E15-VLM-P5, on ONE GPU, from venv-vl.
#  0. the unit tests of code/patches/test_opd_vlmref.py with venv-vl's python (torch: the torch path runs; GOPD_REF_INPUT_UTILS = the
#     installed ref_input_utils.py: the hook's anchor found once, the patched file compiles);
#  a2. (no GPU) the student's and teacher's preprocessor configs and tokenizers equal field by field, and the teacher's prompt for the
#     same image question = the student's + '<think>\n' (code/analysis/vlm_teacher_check.py same);
#  b. teacher reliability on the first 64 Geometry3K-train problems: the student answers (T=1.0, 4096), the teacher answers itself
#     (thinking, T=0.6, 16384): its finished share >= 0.7; the teacher's p(stop) where the student stopped, read with two suffixes
#     ([v2.5.9z1] '<think>\n' and the empty think block '<think>\n\n</think>\n\n'): either mean >= 0.5; check_b.json names the
#     suffix the arm uses (think if it passes, else empty = E15-VLM-P5e);
#  c. a 1-step OPD smoke of E15-VLM-P5 (run_vlm.sh SMOKE=1): exit 0, no out of memory, step 1 logged with a finite actor loss.
# report.json: each part's numbers; to logs/<node>/probe_vlm/ (private), an event, .done. PASS = all four. Usage: bash probe_vlm.sh <gpu>
# [v2.5.9z28] (the reviewer's ruling 10-22 04:4x UTC; a recorded deviation of Q-051's pre-registration): the teacher is Qwen3-VL-4B-Thinking (was 8B-Thinking).
# The 8B's vision tower has head_dim 1152 / 16 = 72, which this flash-attn build refuses ("does not support headdim not being a
# multiple of 32": the first PROBE-VLM, 03:28 UTC, at b's teacher step); the 4B's (like the 2B student's) is 1024 / 16 = 64.
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-VLM; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_VLM $(date -u +%T) $*" | tee -a $LOG; }
PY=${PVL_PY:-$OPD_ROOT/venv-vl/bin/python}; S=${VL_STUDENT:-$OPD_ROOT/models/Qwen3-VL-2B-Instruct}; TE=${VL_TEACHER:-$OPD_ROOT/models/Qwen3-VL-4B-Thinking}   # [v2.5.9z28] was 8B-Thinking
CHK="$RUNBOOK/code/analysis/vlm_teacher_check.py"; DATA=$OPD_ROOT/data/geo3k_train.parquet
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k in ("same", "check_b", "smoke"):
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/unit_tests.txt $D/*.jsonl $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_vlm >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_vlm_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
[ -x "$PY" ] || { [[ $PY != */* ]] && command -v "$PY" > /dev/null; } || finish FAIL "no venv-vl (BUILD-VENV-VL first)"
for m in $S $TE; do [ -f $m/.fetch_ok ] || { echo "WAIT_INPUTS $T: $(basename $m) not in (no .fetch_ok)"; exit 1; }; done   # [v2.5.9z3] waiting, no attempt
[ -f $DATA ] || [ -f $OPD_ROOT/data/hf/hiyouga__geometry3k/.fetch_ok ] || { echo "WAIT_INPUTS $T: Geometry3K not in (no .fetch_ok)"; exit 1; }
# 0. unit tests
RU=$($PY -c "import importlib.util as u; print(u.find_spec('verl.trainer.ppo.ref_input_utils').origin)" 2> /dev/null)
[ -n "$RU" ] && [ -f "$RU" ] || finish FAIL "verl's ref_input_utils.py not found by venv-vl"
( cd $RUNBOOK/code/patches && CUDA_VISIBLE_DEVICES= GOPD_REF_INPUT_UTILS=$RU $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, os, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_opd_vlmref")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_opd_vlmref":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
try:
    import torch  # noqa: F401
except ImportError:
    if os.environ.get("PVL_REQUIRE_TORCH", "1") == "1":
        fails += 1; print("FAILED torch not importable in venv-vl: the torch path did not run")
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
# a2. the same processors and tokenizers, the template prefix
CUDA_VISIBLE_DEVICES= $PY $CHK same $D --student $S --teacher $TE >> $LOG 2>&1 || finish FAIL "a2 check did not run: $(tail -c 300 $LOG | tr '\n' ' ')"
python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" $D/same.json || finish FAIL "a2: $(python3 -c "import json; r = json.load(open('$D/same.json')); print({k: r[k] for k in ('preprocessor_equal', 'vocab_equal', 'special_equal', 'template_teacher_is_student_plus_think')})")"
step "a2 ok: processors, tokenizers and the '<think>' prefix as assumed"
# b. teacher reliability (data made from the node's Geometry3K when missing)
[ -f $DATA ] || $PY $RUNBOOK/code/train/prep_geo3k.py $OPD_ROOT/data/hf/hiyouga__geometry3k $OPD_ROOT/data >> $LOG 2>&1 || finish FAIL "Geometry3K could not be prepared"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PVL_WAIT_STEP:-10}
done
for st in student teacher pstop; do
  CUDA_VISIBLE_DEVICES=$G $PY $CHK $st $D --student $S --teacher $TE --data $DATA --n ${PVL_N:-64} >> $LOG 2>&1 || finish FAIL "b: the $st step failed: $(tail -c 300 $LOG | tr '\n' ' ')"
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
done
$PY $CHK verdict $D --student $S --teacher $TE >> $LOG 2>&1
python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" $D/check_b.json 2> /dev/null || finish FAIL "b: $(cat $D/check_b.json 2> /dev/null | tr '\n' ' ' | cut -c1-400)"
step "b ok: $(python3 -c "import json; r = json.load(open('$D/check_b.json')); print('teacher finished', r['teacher_finished'], 'acc', r['teacher_acc'], '| p(stop)', {k: v['mean'] for k, v in r['pstop'].items()}, '| suffix', r['suffix'])")"
# c. a 1-step OPD smoke
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=1 PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_vlm.sh $G E15-VLM-P5 > $D/smoke.out 2>&1; rc=$?
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log $D/smoke.out 2> /dev/null && finish FAIL "c: out of memory in the 1-step smoke"
[ $rc = 0 ] || finish FAIL "c: smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
python3 - "$D/smoke/train.log" "$D/smoke.json" <<'PY'
import json, math, re, sys
log = open(sys.argv[1], errors="ignore").read() if __import__("os").path.exists(sys.argv[1]) else ""
r = {}
for line in log.splitlines():
    if re.search(r"(?:^|[ )\]])step:1 - ", line):
        for k in ("actor/pg_loss", "actor/kl_loss", "response_length/mean", "timing_s/step"):
            m = re.search(re.escape(k) + r":([-+0-9.eEnaif]+)", line)
            if m:
                r[k] = m.group(1)
r["vlmref_line"] = "[opd_vlmref] ref inputs" in log
ok = "actor/pg_loss" in r and math.isfinite(float(r["actor/pg_loss"]))
r["pass"] = bool(ok)
json.dump(r, open(sys.argv[2], "w"), indent=1)
PY
python3 -c "import json, sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" $D/smoke.json || finish FAIL "c: no finite actor loss at step 1: $(cat $D/smoke.json | tr '\n' ' ')"
finish PASS "a2 ok; b: $(python3 -c "import json; r = json.load(open('$D/check_b.json')); print('teacher finished', r['teacher_finished'], 'p(stop) think', r['pstop']['think']['mean'], 'empty', r['pstop']['empty']['mean'], '-> suffix', r['suffix'])"); c: step 1 pg_loss $(python3 -c "import json; print(json.load(open('$D/smoke.json'))['actor/pg_loss'])")"
