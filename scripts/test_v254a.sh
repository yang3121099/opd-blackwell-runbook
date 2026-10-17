#!/usr/bin/env bash
# Stub tests of runbook v2.5.4a (no GPU, network or HF): task.sh's out-of-memory retry, run_tasks.sh's .oom marker,
# run_queue_eval.sh (EVAL-<name> from #eval lines of the queue). Usage: bash scripts/test_v254a.sh -> V254A_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0
ok() { if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/eval $T/rb/tasks $T/root/runs $T/root/logs $T/root/venv/bin
  cp $SRC/scripts/task.sh $SRC/scripts/run_queue_eval.sh $SRC/scripts/control_rules.py $T/rb/scripts/
  : > $T/root/venv/bin/activate
  cat > $T/rb/scripts/env.sh <<EOF
export OPD_ROOT=$T/root RUNBOOK=$T/rb CKPT_ROOT=$T/root STUDENT_06B=$T/m06 STUDENT_17B=$T/m17 TEACHER=$T/m8
has_weights() { [ -f "\$1/model.safetensors" ]; }
EOF
  printf 'echo "run_arm FIXED=${FIXED_MICRO_BSZ:-unset}"\n' > $T/rb/scripts/run_arm.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
  cat > $T/rb/code/eval/gen_eval.py <<'PY'
import sys, os, json
a = sys.argv; tag = a[a.index("--tag") + 1]; out = a[a.index("--out") + 1]; b = a[a.index("--bench") + 1]
open(os.environ["CALLS"], "a").write("gen_eval " + " ".join(a[1:]) + "\n")
json.dump({"avg_acc": 50.0}, open(os.path.join(out, f"{tag}_{b}_no_think_t0.6.metrics.json"), "w"))
PY
  export CALLS=$T/calls; : > $CALLS
}
# 1. task.sh: .oom -> FIXED_MICRO_BSZ=4; no .oom -> unset; a key already set wins
setup
mkdir -p $T/root/runs/E10b-P2; touch $T/root/runs/E10b-P2/.oom
ok "oom sets 4" '[[ "$(FIXED_MICRO_BSZ= bash $T/rb/scripts/task.sh 0 E10b-P2 | tail -1)" == "run_arm FIXED=4" ]]'
ok "no oom stays unset" '[[ "$(FIXED_MICRO_BSZ= bash $T/rb/scripts/task.sh 0 E10b-P3 | tail -1)" == "run_arm FIXED=unset" ]]'
ok "set key wins" '[[ "$(FIXED_MICRO_BSZ=8 bash $T/rb/scripts/task.sh 0 E10b-P2 | tail -1)" == "run_arm FIXED=8" ]]'
ok "EVAL name check" '! bash $T/rb/scripts/task.sh 0 "EVAL-a;b" > /dev/null 2>&1'
rm -rf $T
# 2. run_tasks.sh launch(): an out-of-memory failure writes .oom and an event; another failure does not
setup
mkdir -p $T/bin; printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; chmod +x $T/bin/setsid   # setsid is Linux-only (the node has it)
export PATH=$T/bin:$PATH
for case in oom trainlog other; do
  rm -f $T/root/runs/X-$case/.oom
  if [ $case = oom ]; then printf 'echo "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32 GiB"; exit 1\n' > $T/rb/scripts/task.sh
  elif [ $case = trainlog ]; then printf 'mkdir -p %s/root/runs/X-trainlog; echo "torch.OutOfMemoryError: CUDA out of memory." > %s/root/runs/X-trainlog/train.log; echo "trainer exited"; exit 1\n' "$T" "$T" > $T/rb/scripts/task.sh
  else printf 'echo "RuntimeError: something else"; exit 1\n' > $T/rb/scripts/task.sh; fi
  ( OPD_ROOT=$T/root RUNBOOK=$T/rb; attempts() { echo 0; }; eval "$(sed -n '/^launch() {/,/^}/p' $SRC/scripts/run_tasks.sh)"; launch 3 X-$case )
  for _ in $(seq 30); do [ -f $T/root/runs/X-$case/.failed ] && break; sleep 0.2; done; sleep 0.5
done
ok "oom marker" '[ -f $T/root/runs/X-oom/.oom ] && grep -q "control event X-oom oom_retry_fixed_micro" $CALLS'
ok "oom only in train.log" '[ -f $T/root/runs/X-trainlog/.oom ]'
ok "no marker for other failures" '[ ! -f $T/root/runs/X-other/.oom ] && ! grep -q "X-other oom_retry" $CALLS'
rm -rf $T
# 3. run_queue_eval.sh: two good specs run (one already done is skipped), a bad model is refused, no .done then
setup
mkdir -p $T/root/adhoc_models/runs/E12-P5c05/global_step_40/actor/huggingface && touch $T/root/adhoc_models/runs/E12-P5c05/global_step_40/actor/huggingface/model.safetensors
cat > $T/rb/tasks/queue.txt <<'EOF'
version: 2
#eval EVAL-bridge e12_P5c05_s40_math500_T0.6 yang31210999/opd-lab-ckpts/runs/E12-P5c05/global_step_40/actor/huggingface math500 4 0.6
#eval EVAL-bridge e12_x_amc23_T1.0 yang31210999/opd-lab-ckpts/runs/E12-P5c05/global_step_40/actor/huggingface amc23 8 1.0
#eval EVAL-other e12_y_math500_T0.6 /etc/passwd math500 4 0.6
EOF
bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/out 2>&1
ok "two evals ran" '[ "$(grep -c "^gen_eval" $CALLS)" = 2 ] && grep -q -- "--temperature 1.0 --top_p 1.0 --top_k 0" $CALLS && grep -q -- "--chunk 10" $CALLS'
ok "upload + done" 'grep -q "remote put $T/root/runs/EVAL-bridge runs/EVAL-bridge" $CALLS && [ -f $T/root/runs/EVAL-bridge/.done ]'
rm -f $T/root/runs/EVAL-bridge/.done; : > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/out2 2>&1
ok "finished evals skipped" '[ "$(grep -c "^gen_eval" $CALLS)" = 0 ] && grep -q DONE_ALREADY $T/out2'
bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-other > $T/out3 2>&1
ok "bad model refused" 'grep -q REFUSED $T/out3 && [ ! -f $T/root/runs/EVAL-other/.done ]'
rm -rf $T
[ $fails = 0 ] && echo "V254A_TEST PASS (12 checks)" || echo "V254A_TEST FAIL ($fails)"
