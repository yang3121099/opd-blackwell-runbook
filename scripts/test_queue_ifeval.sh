#!/usr/bin/env bash
# [v2.5.9z16] Stub tests of run_queue_eval.sh's ifeval rows (no GPU, network or HF). gen_eval.py and venv-ifeval's python are fakes;
# the fake grader writes <stem>.ifeval.json and folds avg_acc / ifeval into the metrics.
# Checked:
# - WAIT_INPUTS (exit 1, nothing generated) without venv-ifeval or without IFEval's #fetch copy;
# - a reading: greedy, the row's budget and mode, graded after the GPU release with no GPU visible; IFEVAL_DONE, folded metrics,
#   .done;
# - a rerun leaves a graded reading alone;
# - a failed grading fails the task, and the next attempt only grades (no generation);
# - the venv gone after the generation: WAIT_INPUTS from the grading step.
# Usage: bash scripts/test_queue_ifeval.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); T=$(cd $T && pwd -P); mkdir -p $T/rb/scripts $T/rb/code/eval $T/rb/tasks $T/root/runs $T/root/venv/bin $T/root/models/m
  cp $SRC/scripts/run_queue_eval.sh $SRC/scripts/control_rules.py $T/rb/scripts/; : > $T/root/venv/bin/activate; : > $T/root/models/m/model.safetensors
  cat > $T/rb/scripts/env.sh <<EOF
export OPD_ROOT=$T/root RUNBOOK=$T/rb CKPT_ROOT=$T/root STUDENT_17B=$T/m17
has_weights() { [ -f "\$1/model.safetensors" ]; }
gpu_wait_free() { return 0; }
EOF
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["repo"] and print("me/pub")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
  cat > $T/rb/code/eval/gen_eval.py <<'PY'
import json, os, sys
a = sys.argv; g = lambda k: a[a.index(k) + 1]
open(os.environ["CALLS"], "a").write("gen_eval " + " ".join(a[1:]) + "\n")
stem = os.path.join(g("--out"), f"{g('--tag')}_{g('--bench')}_{g('--mode')}_t{float(g('--temperature'))}")
open(stem + ".jsonl", "w").write(json.dumps({"pid": "ifeval-1", "sample": 0, "correct": None, "text": "x"}) + "\n")
json.dump({"bench": g("--bench"), "mode": g("--mode"), "avg_acc": None, "pass_at_k": {}, "ifeval": None}, open(stem + ".metrics.json", "w"))
PY
  mkdir -p $T/root/venv-ifeval/bin $T/root/data/hf/google__IFEval; : > $T/root/data/hf/google__IFEval/.fetch_ok
  cat > $T/root/venv-ifeval/bin/python <<EOF
#!/usr/bin/env bash
echo "grade cuda=[\${CUDA_VISIBLE_DEVICES-unset}] released=\$(ls \$OUTDIR/.gpu_released 2> /dev/null | wc -l | tr -d ' ') \$*" >> \$CALLS
[ -n "\${FAKE_GRADE_FAIL:-}" ] && { echo "Traceback: boom" >&2; exit 1; }
exec python3 - "\$@" <<'PY'
import json, sys
a = sys.argv; g = lambda k: a[a.index(k) + 1]
m = json.load(open(g("--metrics"))); m.update(avg_acc=61.2, ifeval={"prompt_strict": 61.2, "prompt_loose": 66.0, "inst_strict": 70.1, "inst_loose": 74.0})
json.dump(m, open(g("--metrics"), "w")); json.dump({"prompt_strict": 61.2}, open(g("--out"), "w"))
print("IFEVAL x prompt_strict=61.2 prompt_loose=66.0 inst_strict=70.1 inst_loose=74.0 think_unclosed=0")
PY
EOF
  chmod +x $T/root/venv-ifeval/bin/python
  cat > $T/rb/tasks/queue.txt <<'EOF'
version: 2
#eval EVAL-if g_ifeval_T0_4k local:models/m ifeval 1 0 4096 no_think
#eval EVAL-if g_math_T0.6 local:models/m math500 4 0.6
#eval EVAL-ift g_ifeval_T0_16k local:models/m ifeval 1 0 16384 think
EOF
  export CALLS=$T/calls OUTDIR=$T/root/runs/EVAL-if; : > $CALLS
}
R=runs/EVAL-if; S=$R/g_ifeval_T0_4k_ifeval_no_think_t0.0
run() { bash $T/rb/scripts/run_queue_eval.sh 3 ${1:-EVAL-if} > $T/out 2>&1; echo $?; }
# 1. no venv-ifeval: WAIT_INPUTS, exit 1, nothing generated
setup; mv $T/root/venv-ifeval $T/root/vi.off
rc=$(run)
ok "venv-ifeval not built: WAIT_INPUTS, exit 1, no generation" '[ $rc = 1 ] && grep -q "^WAIT_INPUTS EVAL-if: venv-ifeval not built (BUILD-VENV-IFEVAL)$" $T/out && ! grep -q "^gen_eval" $CALLS && [ ! -f $T/root/$R/.done ]'
# 2. no IFEval copy: WAIT_INPUTS
mv $T/root/vi.off $T/root/venv-ifeval; rm $T/root/data/hf/google__IFEval/.fetch_ok; : > $CALLS; rc=$(run)
ok "IFEval not in: WAIT_INPUTS, exit 1, no generation" '[ $rc = 1 ] && grep -q "^WAIT_INPUTS EVAL-if: google__IFEval not in (no .fetch_ok)$" $T/out && ! grep -q "^gen_eval" $CALLS'
# 3. both in: greedy 4K no_think, graded after the GPU release with no GPU visible, metrics folded, .done
: > $T/root/data/hf/google__IFEval/.fetch_ok; : > $CALLS; rc=$(run)
ok "the reading: --bench ifeval greedy (T0, top_p 1, top_k 0) 4096 no_think" 'grep -q "^gen_eval .*--bench ifeval --mode no_think --n 1 --temperature 0 --top_p 1.0 --top_k 0 --max_tokens 4096 .*--tag g_ifeval_T0_4k" $CALLS'
ok "graded once, after GPU_RELEASED, CUDA_VISIBLE_DEVICES empty, -I, the data / jsonl / metrics / out paths" '[ "$(grep -c "^grade " $CALLS)" = 1 ] && grep -q "^grade cuda=\[\] released=1 -I $T/rb/code/eval/ifeval_grade.py --data $T/root/data/hf/google__IFEval/ifeval_input_data.jsonl --jsonl $T/root/$S.jsonl --metrics $T/root/$S.metrics.json --out $T/root/$S.ifeval.json$" $CALLS && [ $(grep -n "^GPU_RELEASED" $T/out | cut -d: -f1) -lt $(grep -n "^IFEVAL_DONE" $T/out | cut -d: -f1) ]'
ok "IFEVAL_DONE with the four numbers; avg_acc and ifeval in the metrics; upload; .done" '[ $rc = 0 ] && grep -q "^IFEVAL_DONE g_ifeval_T0_4k: prompt_strict=61.2 prompt_loose=66.0 inst_strict=70.1 inst_loose=74.0 think_unclosed=0$" $T/out && python3 -c "import json, sys; m = json.load(open(sys.argv[1])); assert m[\"avg_acc\"] == 61.2 and m[\"ifeval\"][\"inst_loose\"] == 74.0" $T/root/$S.metrics.json && grep -q "^remote put $T/root/$R runs/EVAL-if$" $CALLS && [ -f $T/root/$R/.done ] && ! grep -q "^grade.*g_math" $CALLS'
# 4. rerun: a graded reading is left alone (no generation, no grading)
rm $T/root/$R/.done; : > $CALLS; rc=$(run)
ok "rerun: DONE_ALREADY and IFEVAL_DONE_ALREADY, nothing generated or graded, .done" '[ $rc = 0 ] && grep -q "^DONE_ALREADY g_ifeval_T0_4k$" $T/out && grep -q "^IFEVAL_DONE_ALREADY g_ifeval_T0_4k$" $T/out && ! grep -q "^gen_eval\|^grade " $CALLS && [ -f $T/root/$R/.done ]'
# 5. the grading fails: the task fails; the next attempt grades only
rm -f $T/root/$R/.done $T/root/$S.* ; : > $CALLS; rc=$(FAKE_GRADE_FAIL=1 run)
ok "a failed grading: IFEVAL_FAIL, no .done (an attempt), the reading kept" 'grep -q "^IFEVAL_FAIL g_ifeval_T0_4k (see ifeval.log)$" $T/out && [ ! -f $T/root/$R/.done ] && [ -f $T/root/$S.jsonl ] && grep -q "Traceback: boom" $T/root/$R/ifeval.log'
: > $CALLS; rc=$(run)
ok "the next attempt: no generation, one grading, .done" '[ $rc = 0 ] && ! grep -q "^gen_eval.*ifeval" $CALLS && [ "$(grep -c "^grade " $CALLS)" = 1 ] && [ -f $T/root/$S.ifeval.json ] && [ -f $T/root/$R/.done ]'
# 6. venv-ifeval gone after the generation: WAIT_INPUTS from the grading step (no attempt), nothing graded
rm -f $T/root/$R/.done $T/root/$S.ifeval.json; mv $T/root/venv-ifeval $T/root/vi.off; : > $CALLS; rc=$(run)
ok "the venv gone after the generation: WAIT_INPUTS after GPU_RELEASED, exit 1, no .done" '[ $rc = 1 ] && grep -q "^WAIT_INPUTS EVAL-if: venv-ifeval not built (BUILD-VENV-IFEVAL)$" $T/out && grep -q "^GPU_RELEASED" $T/out && [ ! -f $T/root/$R/.done ]'
mv $T/root/vi.off $T/root/venv-ifeval
# 7. a thinking-mode reference: 16K think, its stem graded
export OUTDIR=$T/root/runs/EVAL-ift; : > $CALLS; rc=$(run EVAL-ift)
ok "think row: --mode think --max_tokens 16384; its _ifeval_think_t0.0 reading graded" '[ $rc = 0 ] && grep -q "^gen_eval .*--bench ifeval --mode think --n 1 --temperature 0 .*--max_tokens 16384 .*--tag g_ifeval_T0_16k" $CALLS && grep -q "^grade .*--jsonl $T/root/runs/EVAL-ift/g_ifeval_T0_16k_ifeval_think_t0.0.jsonl " $CALLS'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "QUEUE_IFEVAL_TEST PASS ($n checks)" || echo "QUEUE_IFEVAL_TEST FAIL ($fails of $n)"
