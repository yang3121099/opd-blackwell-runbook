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
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["repo"] and print("me/pub")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n"); sys.exit(1 if "missing" in sys.argv[2] else 0)\n' > $T/rb/scripts/fetch_model.py
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
printf 'echo "run_eval $*"\n' > $T/rb/scripts/run_eval.sh
ok "E11b seed arm accepted" '[[ "$(bash $T/rb/scripts/task.sh 0 E11b-P5-s43 | tail -1)" == "run_eval 0 E11b-P5-s43" ]]'
ok "E11b unknown seed refused" '! bash $T/rb/scripts/task.sh 0 E11b-P5-s45 > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E11b-P5-s43x > /dev/null 2>&1'
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
# 5. v2.5.4e prefetch: every opd-lab-ckpts model through fetch_model.py, .inputs_ok = hash of the #eval lines (run_tasks'
#    inputs_ok agrees), a changed spec list invalidates it, a failed fetch writes none; an eval whose fetch fails is not run
setup
cat > $T/rb/tasks/queue.txt <<'EOF2'
version: 2
#eval EVAL-bridge e12_P5c05_s40_math500_T0.6 yang31210999/opd-lab-ckpts/runs/E12-P5c05/global_step_40/actor/huggingface math500 4 0.6
#eval EVAL-bridge e12_P5c05_s20_math500_T0.6 yang31210999/opd-lab-ckpts/runs/E12-P5c05/global_step_20/actor/huggingface math500 4 0.6
#eval EVAL-gone e12_g_math500_T0.6 yang31210999/opd-lab-ckpts/runs/missing/global_step_1/actor/huggingface math500 4 0.6
#eval EVAL-res e12b_P0_s100_math500_T0.6 result:runs/E12b-P0/global_step_100/actor/huggingface math500 4 0.6
EOF2
bash $T/rb/scripts/run_queue_eval.sh prefetch EVAL-bridge > $T/p1 2>&1; rc1=$?
H=$( Q=$T/rb/tasks/queue.txt; eval "$(sed -n '/^eval_hash() {/p;/^inputs_ok() {/p' $SRC/scripts/run_tasks.sh)"; OPD_ROOT=$T/root; inputs_ok EVAL-bridge && echo yes || echo no )
ok "prefetch ok + inputs_ok" '[ $rc1 = 0 ] && [ "$H" = yes ] && [ "$(grep -c "^fetch yang31210999/opd-lab-ckpts runs/E12-P5c05" $CALLS)" = 2 ] && ! grep -q "^gen_eval" $CALLS'
echo "#eval EVAL-bridge e12_P5c05_s60_math500_T0.6 yang31210999/opd-lab-ckpts/runs/E12-P5c05/global_step_60/actor/huggingface math500 4 0.6" >> $T/rb/tasks/queue.txt
H2=$( Q=$T/rb/tasks/queue.txt; eval "$(sed -n '/^eval_hash() {/p;/^inputs_ok() {/p' $SRC/scripts/run_tasks.sh)"; OPD_ROOT=$T/root; inputs_ok EVAL-bridge && echo yes || echo no )
ok "new spec invalidates" '[ "$H2" = no ]'
bash $T/rb/scripts/run_queue_eval.sh prefetch EVAL-gone > $T/p2 2>&1; rc2=$?
ok "failed fetch: no inputs_ok" '[ $rc2 = 1 ] && [ ! -f $T/root/runs/EVAL-gone/.inputs_ok ] && grep -q PREFETCH_FAILED $T/p2'
: > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-gone > $T/p3 2>&1
bash $T/rb/scripts/run_queue_eval.sh prefetch EVAL-res > $T/p4 2>&1; rc4=$?
ok "result: model from the public repository" '[ $rc4 = 0 ] && grep -q "^fetch me/pub runs/E12b-P0/global_step_100/actor/huggingface $T/root/adhoc_models/result" $CALLS'
ok "eval with failed fetch not run" '! grep -q "^gen_eval" $CALLS && grep -q MODEL_FETCH_FAILED $T/p3 && [ ! -f $T/root/runs/EVAL-gone/.done ]'
rm -rf $T
# 6. v2.5.6f E10c (0.6B, V3-c1): task.sh names; run_arm.sh resolves student / arm / seed / batching (DRYRUN), and a
#    dyn16k task stays dynamic even when an out-of-memory retry has set FIXED_MICRO_BSZ=4
setup
cp $SRC/scripts/run_arm.sh $T/rb/scripts/run_arm.sh
ok "E10c names" '[[ "$(DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10c-P4lowlr-fix4 | tail -1)" == DRYRUN* ]] && ! bash $T/rb/scripts/task.sh 0 E10c-P4lowlr > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E10c-P9-fix4 > /dev/null 2>&1'
d1=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10c-P4lowlr-fix4 | tail -1)
d2=$(FIXED_MICRO_BSZ=4 DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10c-P4lowlr-dyn16k | tail -1)
d3=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P4lowlr | tail -1)
ok "E10c fix4" '[[ "$d1" == *"arm=P4 lr=6e-7 seed=42 student=$T/m06 "* && "$d1" == *"use_dynamic_bsz=False"*"ppo_micro_batch_size_per_gpu=4"* ]]'
ok "E10c dyn16k ignores the OOM retry key" '[[ "$d2" == *"student=$T/m06 "* && "$d2" == *"use_dynamic_bsz=True"*"ppo_max_token_len_per_gpu=16384"* ]]'
ok "E10b unchanged" '[[ "$d3" == *"student=$T/m17 "* && "$d3" == *"use_dynamic_bsz=True"* ]]'
rm -rf $T
# 4. v2.5.4d free_gpus(): a GPU whose task.sh (or ad-hoc evaluation) runs is not free, even with nothing on it yet
T=$(mktemp -d); mkdir -p $T/bin $T/x/scripts
printf '#!/usr/bin/env bash\ncase "$*" in *compute-apps*) ;; *) echo 0;; esac\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
printf 'sleep 30\n' > $T/x/scripts/task.sh; printf 'sleep 30\n' > $T/x/scripts/run_adhoc_eval.sh
bash $T/x/scripts/task.sh 5 FA-PROBE & P1=$!; bash $T/x/scripts/run_adhoc_eval.sh 13 41 & P2=$!; sleep 0.5
got=$( PATH=$T/bin:$PATH NGPU=14; eval "$(sed -n '/^free_gpus() {/,/; }$/p' $SRC/scripts/run_tasks.sh)"; free_gpus | tr '\n' ' ' )
kill $P1 $P2 2> /dev/null
ok "running tasks keep their GPUs" '[[ " $got" == *" 1 "* && " $got" == *" 6 "* && " $got" != *" 5 "* && " $got" != *" 13 "* ]]'   # 13 must not hide 1
rm -rf $T
[ $fails = 0 ] && echo "V254A_TEST PASS (24 checks)" || echo "V254A_TEST FAIL ($fails)"
