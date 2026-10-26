#!/usr/bin/env bash
# [v2.5.9z89] Stub tests of probe_vlm_complete.sh (no GPU, network, HF or model): a fake vlm_teacher_check.py records each step with its
# environment; complete_verdict writes complete.json from FAKE_RATE. Usage: bash scripts/test_probe_vlm_complete.sh -> PROBE_VLM_COMPLETE_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/analysis $T/rb/code/train $T/root/runs $T/root/data $T/bin
  for m in Qwen3-VL-2B-Instruct Qwen3-VL-4B-Instruct Qwen3-VL-4B-Thinking; do mkdir -p $T/root/models/$m; : > $T/root/models/$m/.fetch_ok; done
  : > $T/root/data/mmk12_val.parquet
  cp $SRC/scripts/probe_vlm_complete.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/code/analysis/vlm_teacher_check.py <<'PY'
import json, os, sys
step, out = sys.argv[1], sys.argv[2]
a = sys.argv[3:]; teacher = a[a.index("--teacher") + 1].split("/")[-1]
open(os.environ["CALLS"], "a").write(f"check {step} teacher={teacher} CUDA={os.environ.get('CUDA_VISIBLE_DEVICES', '')} expect={os.environ.get('PVL_EXPECT_SUFFIX', 'unset')!r} max={os.environ.get('PVL_STUDENT_MAX', '')} suffix={os.environ.get('PVL_SUFFIX', 'unset')!r} out={os.environ.get('PVL_OUT', '')} n={a[a.index('--n') + 1] if '--n' in a else ''}\n")
if step == "same":
    json.dump({"pass": os.environ.get("FAKE_SAME", "1") == "1", "preprocessor_equal": True, "vocab_equal": True, "special_equal": True,
               "template_teacher_is_student_plus_think": os.environ.get("FAKE_SAME", "1") == "1"}, open(os.path.join(out, "same.json"), "w"))
elif step == "complete_verdict":
    r = float(os.environ.get("FAKE_RATE", "0.95"))
    json.dump({"pass": r >= 0.9, "completion_rate": r, "scored": 250, "threshold": 0.9, "teacher_p_stop_last_mean": 0.81, "student_finished": 0.83,
               "student_finished_n": 250, "n": 300, "student_p_stop_last_mean": 0.6, "thinking_teacher_nonthinking": {"p_stop_last_mean": 0.4},
               "note": ""}, open(os.path.join(out, "complete.json"), "w"))
elif os.environ.get("FAKE_STEP_FAIL") == step:
    sys.exit(1)
else:
    open(os.path.join(out, os.environ.get("PVL_OUT") or (step + ".jsonl")), "w").write("{}\n")
PY
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PVC_WAIT_STEP=0 PVC_PY=python3; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-VLM-COMPLETE/report.json "$1"; }
setup; bash $T/rb/scripts/probe_vlm_complete.sh 3 > /dev/null 2>&1
ok "PASS: report, event, upload, .done" '[ "$(res result)" = PASS ] && [[ "$(res reason)" == *"completion rate 0.95 of 250"*"thinking teacher non-thinking p_stop_last 0.4"* ]] && grep -q "^control event PROBE-VLM-COMPLETE probe_vlm_complete_pass" $CALLS && grep -q "^remote put .*/up logs/abcd1234/probe_vlm_complete$" $CALLS && [ -f $T/root/runs/PROBE-VLM-COMPLETE/.done ]'
ok "a2 without a GPU and with no expected suffix (the Instruct teacher)" 'grep -q "^check same teacher=Qwen3-VL-4B-Instruct CUDA= expect='"''"' " $CALLS'
ok "the student: 1536 tokens, 300 rows, on the GPU" 'grep -q "^check student teacher=Qwen3-VL-4B-Instruct CUDA=3 .* max=1536 .* n=300$" $CALLS'
ok "the three scorers: the Instruct teacher and the student with no suffix, the thinking teacher with the empty think block" 'grep -q "^check complete teacher=Qwen3-VL-4B-Instruct CUDA=3 .*suffix='"''"' out=complete_teacher.jsonl n=300$" $CALLS && grep -q "^check complete teacher=Qwen3-VL-2B-Instruct CUDA=3 .*suffix='"''"' out=complete_self.jsonl n=300$" $CALLS && grep -qF "check complete teacher=Qwen3-VL-4B-Thinking CUDA=3 expect='"'"'unset'"'"' max= suffix='"'"'<think>\n\n</think>\n\n'"'"' out=complete_think.jsonl" $CALLS'
ok "each GPU step reaped; in order" '[ "$(grep -c "^control reap 3 PROBE-VLM-COMPLETE" $CALLS)" = 4 ] && [ "$(grep "^check" $CALLS | cut -d" " -f2 | tr "\n" " ")" = "same student complete complete complete complete_verdict " ]'
rm -rf $T
setup; FAKE_RATE=0.85 bash $T/rb/scripts/probe_vlm_complete.sh 3 > /dev/null 2>&1
ok "a completion rate under 0.9 is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"completion rate 0.85"* ]]'
rm -rf $T
setup; FAKE_SAME=0 bash $T/rb/scripts/probe_vlm_complete.sh 3 > /dev/null 2>&1
ok "a2 failing stops before the GPU" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == a2:* ]] && ! grep -q "^check student" $CALLS'
rm -rf $T
setup; FAKE_STEP_FAIL=student bash $T/rb/scripts/probe_vlm_complete.sh 3 > /dev/null 2>&1
ok "a failed step is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == "the student step failed"* ]]'
rm -rf $T
setup; rm $T/root/models/Qwen3-VL-4B-Instruct/.fetch_ok; out=$(bash $T/rb/scripts/probe_vlm_complete.sh 3 2>&1); rc=$?
ok "a missing model waits (no attempt), nothing written" '[ $rc = 1 ] && [[ "$out" == *"WAIT_INPUTS PROBE-VLM-COMPLETE: Qwen3-VL-4B-Instruct not in"* ]] && [ ! -f $T/root/runs/PROBE-VLM-COMPLETE/.done ]'
rm -rf $T
setup; FAKE_MEM=9000 bash $T/rb/scripts/probe_vlm_complete.sh 3 > /dev/null 2>&1
ok "a busy GPU is FAIL before the student" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"not empty"* ]] && ! grep -q "^check student" $CALLS'
rm -rf $T
[ $fails = 0 ] && echo "PROBE_VLM_COMPLETE_TEST PASS ($n checks)" || echo "PROBE_VLM_COMPLETE_TEST FAIL ($fails of $n)"
