#!/usr/bin/env bash
# [v2.5.9y1] Stub tests of probe_vllm_vl.sh (no GPU, network or HF): a fake venv-vl python answers or fails; inputs present or
# missing. Usage: bash scripts/test_probe_vllm_vl.sh -> PROBE_VLLM_VL_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/bin $T/root/models/Qwen3-VL-2B-Instruct $T/root/data/hf/hiyouga__geometry3k
  cp $SRC/scripts/probe_vllm_vl.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  : > $T/root/models/Qwen3-VL-2B-Instruct/.fetch_ok; : > $T/root/data/hf/hiyouga__geometry3k/.fetch_ok
  # the venv-vl python: [v2.5.9z20] given the check as a file ($1 = runs/PROBE-VLLM-VL/gen_check.py, no stdin); writes gen.json ($4)
  # unless told to fail
  cat > $T/bin/fakepy <<'SH'
#!/usr/bin/env bash
echo "fakefile $1" >> $CALLS; echo "fakepy $2 $3 CUDA=$CUDA_VISIBLE_DEVICES" >> $CALLS
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory"; exit 1; }
[ "${FAKE_RC:-0}" != 0 ] && { echo "ValueError: Model architectures ['Qwen3VLForConditionalGeneration'] are not supported"; exit ${FAKE_RC}; }
printf '{"vllm": "0.11.0", "torch": "2.8.0+cu128", "capability": "10.3", "load_s": 41.2, "gen_s": 3.1, "answer": "%s", "answer_tokens": 87, "finish_reason": "stop", "reference": "12"}' "${FAKE_ANSWER-The answer is 12.}" > $4
echo GEN_OK
SH
  chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\nshift; exec "$@"\n' > $T/bin/timeout; chmod +x $T/bin/timeout
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PVV_WAIT_STEP=0 PVV_PY=$T/bin/fakepy; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-VLLM-VL/report.json "$1"; }
setup; bash $T/rb/scripts/probe_vllm_vl.sh 5 > /dev/null 2>&1
ok "[v2.5.9z20] the check runs from runs/PROBE-VLLM-VL/gen_check.py (not stdin): it compiles and calls main() only under __main__ (vLLM's spawned engine core re-imports it)" 'grep -q "^fakefile $T/root/runs/PROBE-VLLM-VL/gen_check.py$" $CALLS && python3 -m py_compile $T/root/runs/PROBE-VLLM-VL/gen_check.py && grep -q "^if __name__ == \"__main__\":$" $T/root/runs/PROBE-VLLM-VL/gen_check.py && ! grep -q "^llm = " $T/root/runs/PROBE-VLLM-VL/gen_check.py'
ok "PASS: the model and the dataset given, on GPU 5; versions, capability, answer in the report; upload, event, .done" '[ "$(res result)" = PASS ] && grep -q "^fakepy $T/root/models/Qwen3-VL-2B-Instruct $T/root/data/hf/hiyouga__geometry3k CUDA=5$" $CALLS && [ "$(res capability)" = 10.3 ] && [ "$(res answer)" = "The answer is 12." ] && [[ "$(res reason)" == *"0.11.0 on sm 10.3, load 41.2 s"* ]] && grep -q "^remote put $T/root/runs/PROBE-VLLM-VL/up logs/abcd1234/probe_vllm_vl" $CALLS && grep -q "^control event PROBE-VLLM-VL probe_vllm_vl_pass" $CALLS && [ -f $T/root/runs/PROBE-VLLM-VL/.done ]'
rm -rf $T
check() { ok "$1" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"'"$2"'"* ]]'; }
setup; PVV_PY=$T/root/venv-vl/bin/python bash $T/rb/scripts/probe_vllm_vl.sh 5 > /dev/null 2>&1; check "no venv-vl is FAIL" "no venv-vl"; rm -rf $T
setup; rm $T/root/models/Qwen3-VL-2B-Instruct/.fetch_ok; bash $T/rb/scripts/probe_vllm_vl.sh 5 > $T/o 2>&1; rc=$?
ok "[v2.5.9z3] VL-2B not in: WAIT_INPUTS, exit 1, no report or .done, no generation" '[ $rc = 1 ] && grep -q "^WAIT_INPUTS PROBE-VLLM-VL: Qwen3-VL-2B-Instruct not in" $T/o && [ ! -f $T/root/runs/PROBE-VLLM-VL/.done ] && [ ! -f $T/root/runs/PROBE-VLLM-VL/report.json ] && ! grep -q "^fakepy" $CALLS'; rm -rf $T
setup; rm $T/root/data/hf/hiyouga__geometry3k/.fetch_ok; bash $T/rb/scripts/probe_vllm_vl.sh 5 > $T/o 2>&1; ok "[v2.5.9z3] Geometry3K not in: WAIT_INPUTS, no .done" 'grep -q "^WAIT_INPUTS PROBE-VLLM-VL: Geometry3K not in" $T/o && [ ! -f $T/root/runs/PROBE-VLLM-VL/.done ]'; rm -rf $T
setup; FAKE_ANSWER="" bash $T/rb/scripts/probe_vllm_vl.sh 5 > /dev/null 2>&1; check "an empty answer is FAIL" "an empty answer"; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_vllm_vl.sh 5 > /dev/null 2>&1; check "out of memory is FAIL" "out of memory"; rm -rf $T
setup; FAKE_RC=1 bash $T/rb/scripts/probe_vllm_vl.sh 5 > /dev/null 2>&1; check "an unsupported architecture is FAIL with its message" "not supported"; rm -rf $T
[ $fails = 0 ] && echo "PROBE_VLLM_VL_TEST PASS ($n checks)" || echo "PROBE_VLLM_VL_TEST FAIL ($fails of $n)"
