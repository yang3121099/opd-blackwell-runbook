#!/usr/bin/env bash
# [v2.5.9z] Stub tests of probe_vlm.sh (no GPU, network or HF): the real unit tests (numpy path) against a fake installed
# ref_input_utils.py, a fake vlm_teacher_check.py and run_vlm.sh. Usage: bash scripts/test_probe_vlm.sh -> PROBE_VLM_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/rb/code/analysis $T/rb/code/train $T/root/runs $T/root/data $T/bin $T/py/verl/trainer/ppo
  mkdir -p $T/root/models/Qwen3-VL-2B-Instruct $T/root/models/Qwen3-VL-4B-Thinking; : > $T/root/models/Qwen3-VL-2B-Instruct/.fetch_ok; : > $T/root/models/Qwen3-VL-4B-Thinking/.fetch_ok
  : > $T/root/data/geo3k_train.parquet
  cp $SRC/scripts/probe_vlm.sh $T/rb/scripts/
  cp $SRC/code/patches/opd_vlmref.py $SRC/code/patches/vlmref_patch.py $SRC/code/patches/route_patch.py $SRC/code/patches/test_opd_vlmref.py $T/rb/code/patches/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  : > $T/py/verl/__init__.py; : > $T/py/verl/trainer/__init__.py; : > $T/py/verl/trainer/ppo/__init__.py
  printf "def prepare_ref_model_inputs(\n    batch,\n    ref_tokenizer,\n    apply_chat_template_kwargs=None,\n):\n    if apply_chat_template_kwargs is None:\n        apply_chat_template_kwargs = {}\n    return batch\n" > $T/py/verl/trainer/ppo/ref_input_utils.py
  cat > $T/rb/code/analysis/vlm_teacher_check.py <<'PY'
import json, os, sys
step, out = sys.argv[1], sys.argv[2]
open(os.environ["CALLS"], "a").write("check " + step + " CUDA=" + os.environ.get("CUDA_VISIBLE_DEVICES", "") + "\n")
if step == "same":
    json.dump({"pass": os.environ.get("FAKE_SAME", "1") == "1", "preprocessor_equal": True, "vocab_equal": True, "special_equal": True,
               "template_teacher_is_student_plus_think": os.environ.get("FAKE_SAME", "1") == "1"}, open(os.path.join(out, "same.json"), "w"))
elif step == "verdict":
    tf = float(os.environ.get("FAKE_TF", "0.84")); ps = float(os.environ.get("FAKE_PS", "0.71"))
    pe = float(os.environ.get("FAKE_PE", "0.2")); suf = "think" if ps >= 0.5 else ("empty" if pe >= 0.5 else None)
    json.dump({"pass": tf >= 0.7 and suf is not None, "teacher_finished": tf, "teacher_acc": 0.6, "pstop": {"think": {"mean": ps}, "empty": {"mean": pe}}, "suffix": suf}, open(os.path.join(out, "check_b.json"), "w"))
else:
    open(os.path.join(out, step + ".jsonl"), "w").write("{}\n")
PY
  cat > $T/rb/scripts/run_vlm.sh <<'SH'
echo "run_vlm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; echo "task: $2" > $d/config.yaml
echo "[opd_vlmref] ref inputs from the student's ids + 2 suffix tokens: (128, 6146), positions (128, 4, 6146)" >> $d/train.log
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }
echo "(TaskRunner pid=1) step:1 - actor/pg_loss:${FAKE_LOSS:-0.0213} - actor/kl_loss:0.0 - response_length/mean:812.4 - timing_s/step:420" >> $d/train.log; exit 0
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  # [v2.5.9z31] g: a fake torch for the NVRTC check only (the unit tests keep their no-torch path); FAKE_NVRTC_FAIL raises as on 06:09 UTC
  mkdir -p $T/faketorch/torch
  cat > $T/faketorch/torch/__init__.py <<'PY'
import os
int64 = "int64"
class _L(list):
    def tolist(self):
        return list(self)
    def all(self):
        return all(self)
def tensor(x, device=None, dtype=None):
    return x
def prod(x, dim=1):
    if os.environ.get("FAKE_NVRTC_FAIL") == "1":
        raise RuntimeError("\n__global__ void reduction_prod_kernel(ReduceJitOp r){\nnvrtc: error: invalid value for --gpu-architecture (-arch)")
    out = []
    for row in x:
        p = 1
        for v in row:
            p *= v
        out.append(p)
    return _L(out)
def linspace(a, b, n, device=None):
    return [a + (b - a) * i / (n - 1) for i in range(n)]
def isfinite(y):
    return _L(True for _ in y)
class special:
    @staticmethod
    def bessel_j0(y):
        return y
class cuda:
    @staticmethod
    def get_device_capability(i):
        return (10, 3)
PY
  printf '#!/usr/bin/env bash\necho "vlpy $* CUDA=$CUDA_VISIBLE_DEVICES" | head -c 200 >> $CALLS; echo >> $CALLS\nif [ "$1" = - ] && [ "${2##*/}" = gpu_check.json ]; then PYTHONPATH=%s/faketorch:$PYTHONPATH exec python3 "$@"; fi\nexec python3 "$@"\n' $T > $T/bin/vlpy; chmod +x $T/bin/vlpy
  export CALLS=$T/calls PATH=$T/bin:$PATH PVL_WAIT_STEP=0 PVL_PY=$T/bin/vlpy PVL_REQUIRE_TORCH=0 PYTHONPATH=$T/py; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-VLM/report.json "$1"; }
setup; bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1
ok "PASS: unit tests, a2, b (student / teacher / pstop on GPU 3, verdict), c (1-step smoke), upload, event, .done" '[ "$(res result)" = PASS ] && grep -q "passed, 0 failed" $T/root/runs/PROBE-VLM/unit_tests.txt && grep -q "^check same CUDA=$" $CALLS && grep -q "^check student CUDA=3$" $CALLS && grep -q "^check pstop CUDA=3$" $CALLS && grep -q "^run_vlm 3 E15-VLM-P5 SMOKE=1 STEPS=1 PROBE_TASK=PROBE-VLM-smoke$" $CALLS && [[ "$(res reason)" == *"teacher finished 0.84 p(stop) think 0.71 empty 0.2 -> suffix think; c: step 1 pg_loss 0.0213"* ]] && grep -q "^control event PROBE-VLM probe_vlm_pass" $CALLS && grep -q "^remote put $T/root/runs/PROBE-VLM/up logs/abcd1234/probe_vlm" $CALLS && [ -f $T/root/runs/PROBE-VLM/.done ]'
rm -rf $T
check() { ok "$1" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"'"$2"'"* ]]'; }
setup; FAKE_SAME=0 bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1; check "a2: processors / template not as assumed is FAIL" "a2:"; ok "no GPU step after a2 fails" '! grep -q "^check student" $CALLS'; rm -rf $T
setup; FAKE_TF=0.55 bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1; check "b: the teacher finishing 55% is FAIL" "b: "; ok "no smoke after b fails" '! grep -q "^run_vlm" $CALLS'; rm -rf $T
setup; FAKE_PS=0.3 bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1; check "b: p(stop) 0.3 with both suffixes is FAIL" "b: "; rm -rf $T
setup; FAKE_PS=0.3 FAKE_PE=0.62 bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1; ok "[v2.5.9z1] b: the empty think block passing alone is PASS with suffix empty (E15-VLM-P5e)" '[ "$(res result)" = PASS ] && [[ "$(res reason)" == *"-> suffix empty"* ]]'; rm -rf $T
setup; FAKE_LOSS=nan bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1; check "c: a NaN loss is FAIL" "no finite actor loss"; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1; check "c: out of memory is FAIL" "out of memory"; rm -rf $T
setup; PVL_PY=$T/root/venv-vl/bin/python bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1; check "no venv-vl is FAIL" "no venv-vl"; rm -rf $T
setup; rm $T/root/models/Qwen3-VL-4B-Thinking/.fetch_ok; bash $T/rb/scripts/probe_vlm.sh 3 > $T/o 2>&1; ok "[v2.5.9z3] the teacher not in: WAIT_INPUTS, no .done" 'grep -q "^WAIT_INPUTS PROBE-VLM: Qwen3-VL-4B-Thinking not in" $T/o && [ ! -f $T/root/runs/PROBE-VLM/.done ]'; rm -rf $T
unset PYTHONPATH
# [v2.5.9z31] g: the NVRTC check on the GPU first; its failure is a FAIL with its text before the unit tests, a2, b or the smoke
setup; bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1
ok "[v2.5.9z31] g runs first on GPU 3 (before the unit tests), gpu_check in the report (prod [24, 30], sm 10.3)" '[ "$(res result)" = PASS ] && grep -n "" $CALLS | grep "vlpy - .*gpu_check.json" | head -1 | grep -q "CUDA=3" && python3 -c "import json, sys; r = json.load(open(sys.argv[1]))[\"gpu_check\"]; assert r[\"prod\"] == [24, 30] and r[\"capability\"] == \"10.3\" and r[\"pass\"], r" $T/root/runs/PROBE-VLM/report.json && [ "$(grep -n "gpu_check.json" $CALLS | head -1 | cut -d: -f1)" -lt "$(grep -n "check same" $CALLS | head -1 | cut -d: -f1)" ]'
rm -rf $T
setup; FAKE_NVRTC_FAIL=1 bash $T/rb/scripts/probe_vlm.sh 3 > /dev/null 2>&1
check "[v2.5.9z31] NVRTC refusing sm_103 is FAIL at g with its text" "g: NVRTC on the GPU: RuntimeError: nvrtc: error: invalid value for --gpu-architecture (-arch)"
ok "[v2.5.9z31] nothing after g: no a2 / b check, no smoke" '! grep -q "^check " $CALLS && ! grep -q "^run_vlm" $CALLS'; rm -rf $T
[ $fails = 0 ] && echo "PROBE_VLM_TEST PASS ($n checks)" || echo "PROBE_VLM_TEST FAIL ($fails of $n)"
