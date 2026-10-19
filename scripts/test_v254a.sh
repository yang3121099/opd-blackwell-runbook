#!/usr/bin/env bash
# Stub tests of runbook v2.5.4a (no GPU, network or HF): task.sh's out-of-memory retry, run_tasks.sh's .oom marker,
# run_queue_eval.sh (EVAL-<name> from #eval lines of the queue). Usage: bash scripts/test_v254a.sh -> V254A_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0
ok() { if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/eval $T/rb/tasks $T/root/runs $T/root/logs $T/root/venv/bin
  cp $SRC/scripts/task.sh $SRC/scripts/run_queue_eval.sh $SRC/scripts/control_rules.py $SRC/scripts/dr.sh $T/rb/scripts/
  mkdir -p $T/rb/code/leakage; printf 'import sys, os; open(os.environ["CALLS"], "a").write("twr " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/code/leakage/twr_alpha.py
  : > $T/root/venv/bin/activate
  cat > $T/rb/scripts/env.sh <<EOF
export OPD_ROOT=$T/root RUNBOOK=$T/rb CKPT_ROOT=$T/root STUDENT_06B=$T/m06 STUDENT_17B=$T/m17 TEACHER=$T/m8 STUDENT_4B=$T/m4 TEACHER_14B=$T/m14
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
open(os.path.join(out, f"{tag}_{b}_no_think_t0.6.jsonl"), "w").write("{}\n")
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
#twr EVAL-bridge e12_P5c05_s40_math500_T0.6
#eval EVAL-loc e12b_s25_math500_T0.6_32k local:runs/E12b-X/ckpt/global_step_25/actor/huggingface math500 4 0.6 32768
#eval EVAL-loc e12b_s50_math500_T0.6_32k local:runs/E12b-X/ckpt/global_step_50/actor/huggingface math500 4 0.6 32768
EOF
mkdir -p $T/root/runs/E11b-P0; : > $T/root/runs/E11b-P0/e11b_P0_math500_T0.6_math500_no_think_t0.6.jsonl
for tg in e12_P5c05_s40_math500_T0.6 e12_x_amc23_T1.0; do :; done
bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/out 2>&1; cp $CALLS $T/calls.bridge
ok "two evals ran" '[ "$(grep -c "^gen_eval" $CALLS)" = 2 ] && grep -q -- "--temperature 1.0 --top_p 1.0 --top_k 0" $CALLS && grep -q -- "--chunk 10" $CALLS'
ok "TWR after the readings" 'grep -q "^twr --judge $T/m17 --bench math500 --ref $T/root/runs/E11b-P0/e11b_P0_math500_T0.6_math500_no_think_t0.6.jsonl --score $T/root/runs/EVAL-bridge/e12_P5c05_s40_math500_T0.6_" $CALLS && grep -q -- "--model_k $T/root/adhoc_models/runs/E12-P5c05/global_step_40/actor/huggingface" $CALLS'
ok "upload + done" 'grep -q "remote put $T/root/runs/EVAL-bridge runs/EVAL-bridge" $CALLS && [ -f $T/root/runs/EVAL-bridge/.done ]'
rm -f $T/root/runs/EVAL-bridge/.done; : > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/out2 2>&1
ok "finished evals skipped" '[ "$(grep -c "^gen_eval" $CALLS)" = 0 ] && grep -q DONE_ALREADY $T/out2'
bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-other > $T/out3 2>&1
ok "bad model refused" 'grep -q REFUSED $T/out3 && [ ! -f $T/root/runs/EVAL-other/.done ]'
ok "v2.5.7q archive / result models read as auto" '[ "$(grep -c -- "--dtype auto " $T/calls.bridge)" = 2 ]'
mkdir -p $T/root/runs/E12b-X/ckpt/global_step_25/actor/huggingface; touch $T/root/runs/E12b-X/ckpt/global_step_25/actor/huggingface/model.safetensors
: > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-loc > $T/out8 2>&1
ok "v2.5.7q local: (a training's fp32 save) read as bfloat16; a missing step is MODEL_MISSING, no .done" 'grep -q -- "--model $T/root/runs/E12b-X/ckpt/global_step_25/actor/huggingface .*--max_tokens 32768 --chunk 50 --dtype bfloat16 " $CALLS && [ "$(grep -c "^gen_eval" $CALLS)" = 1 ] && grep -q "MODEL_MISSING e12b_s50_math500_T0.6_32k" $T/out8 && [ ! -f $T/root/runs/EVAL-loc/.done ]'
rm -rf $T
# 3b. v2.5.6r "#pair": a finished reading paired against references under runs/ (problem-level bootstrap, CPU only)
setup
mkdir -p $T/rb/code/analysis $T/root/runs/EVAL-pp $T/root/runs/E11b-P0; cp $SRC/code/analysis/paired_bootstrap.py $T/rb/code/analysis/
cat > $T/rb/tasks/queue.txt <<'EOF'
version: 2
#eval EVAL-pp pp_s100_math500_T0.6 result:runs/E12b-PP/global_step_100/actor/huggingface math500 4 0.6
#pair EVAL-pp pp_s100_math500_T0.6 E11b-P0/base.jsonl E11b-P9/missing.jsonl ../etc/passwd.jsonl archive:runs/e10/e10_P5_s50.jsonl archive:../x.jsonl
EOF
echo '{"avg_acc": 1}' > $T/root/runs/EVAL-pp/pp_s100_math500_T0.6_math500_no_think_t0.6.metrics.json
python3 -c "
import json, sys
a, b = open(sys.argv[1], 'w'), open(sys.argv[2], 'w')
for p in range(20):
    for s in range(4):
        a.write(json.dumps({'pid': p, 'sample': s, 'correct': p % 2 == 0 or s == 0}) + chr(10)); b.write(json.dumps({'pid': p, 'sample': s, 'correct': p % 2 == 0}) + chr(10))
" $T/root/runs/EVAL-pp/pp_s100_math500_T0.6_math500_no_think_t0.6.jsonl $T/root/runs/E11b-P0/base.jsonl
mkdir -p $T/root/adhoc_models/runs/e10; cp $T/root/runs/E11b-P0/base.jsonl $T/root/adhoc_models/runs/e10/e10_P5_s50.jsonl
bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-pp > $T/out4 2>&1
PJ=$T/root/runs/EVAL-pp/pp_s100_math500_T0.6.paired.json
ok "paired CI written" 'python3 -c "import json, sys; r = json.load(open(sys.argv[1]))[\"pairs\"]; sys.exit(0 if r[0][\"diff\"] == 12.5 and r[0][\"problems\"] == 20 and r[0][\"wins\"] == 10 and r[0][\"ref\"] == \"E11b-P0/base.jsonl\" and r[1][\"error\"] == \"missing\" and r[2][\"ref\"] == \"e10/e10_P5_s50.jsonl\" and len(r) == 3 else 1)" $PJ'
ok "bad pair ref refused, no absolute path in the result" 'grep -q "PAIR_REF_REFUSED ../etc/passwd.jsonl" $T/out4 && ! grep -q "$T" $PJ && grep -q "PAIR_DONE pp_s100_math500_T0.6" $T/out4'
ok "v2.5.7r archive: pair ref fetched from the lab archive and paired; archive:.. refused" 'grep -q "^fetch yang31210999/opd-lab-ckpts runs/e10/e10_P5_s50.jsonl $T/root/adhoc_models --task EVAL-pp" $CALLS && grep -q "PAIR_REF_REFUSED archive:../x.jsonl" $T/out4 && [ "$(python3 -c "import json, sys; print(sum(1 for p in json.load(open(sys.argv[1]))[\"pairs\"] if \"diff\" in p))" $PJ)" = 2 ]'
rm -rf $T
# 3c. v2.5.7g: the 7th field max_tokens (checked; default 16384) and "#budget" (budget_curve.py on this task's tags and
#     references under runs/; a bad reference refused, a missing tag listed)
setup
mkdir -p $T/m17; touch $T/m17/model.safetensors
mkdir -p $T/rb/code/analysis; printf 'import sys, os; open(os.environ["CALLS"], "a").write("budget " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/code/analysis/budget_curve.py
cat > $T/rb/tasks/queue.txt <<'EOF'
version: 2
#eval EVAL-bud bud_base17_32k Qwen/Qwen3-1.7B math500 4 0.6 32768
#eval EVAL-bud bud_x Qwen/Qwen3-1.7B math500 4 0.6
#eval EVAL-bud bud_y Qwen/Qwen3-1.7B math500 4 0.6 12000
#budget EVAL-bud budget_curve bud_base17_32k bud_x E11b-P0/base.jsonl nosuch ../etc/x.jsonl
EOF
bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bud > $T/out5 2>&1
ok "max_tokens 32768 / default 16384 / 12000 refused" 'grep -q "^gen_eval .*--max_tokens 32768 .*--tag bud_base17_32k " $CALLS && grep -q "^gen_eval .*--max_tokens 16384 .*--tag bud_x " $CALLS && ! grep -q "tag bud_y" $CALLS && grep -q "REFUSED bud_y: .* 12000" $T/out5'
O=$T/root/runs/EVAL-bud
ok "#budget: tags, runs/ reference, missing tag listed, bad reference refused" 'grep -q "^budget --out $O/budget_curve.json --tokenizer $T/m17 --forced_dir $O $O/bud_base17_32k_math500_no_think_t0.6.jsonl $O/bud_x_math500_no_think_t0.6.jsonl $T/root/runs/E11b-P0/base.jsonl $O/nosuch.missing.jsonl$" $CALLS && grep -q "BUDGET_REF_REFUSED ../etc/x.jsonl" $T/out5 && grep -q "BUDGET_DONE budget_curve" $T/out5'
rm -rf $T
# 3d. v2.5.7h: the 8th field mode (think / no_think, checked) and "#forced" (forced_stop.py budB on a tag of this task and
#     on a reading under runs/, with the model that wrote it; a missing reading blocks .done; done ones are skipped)
setup
mkdir -p $T/m17 $T/root/runs/EVAL-e12b-P5; touch $T/m17/model.safetensors
printf 'import sys, os; open(os.environ["CALLS"], "a").write("forced " + " ".join(sys.argv[1:]) + "\\n"); a = sys.argv; open(a[a.index("--out") + 1], "w").write("{}")\n' > $T/rb/code/eval/forced_stop.py
: > $T/root/runs/EVAL-e12b-P5/e12b_P5.jsonl; : > $T/root/runs/EVAL-e12b-P5/e12b_P5.metrics.json
cat > $T/rb/tasks/queue.txt <<'EOF'
version: 2
#eval EVAL-bb bb_think Qwen/Qwen3-1.7B math500 4 0.6 32768 think
#eval EVAL-bb bb_bad Qwen/Qwen3-1.7B math500 4 0.6 32768 deep
#forced EVAL-bb bb_think Qwen/Qwen3-1.7B think
#forced EVAL-bb EVAL-e12b-P5/e12b_P5.jsonl Qwen/Qwen3-1.7B no_think
#forced EVAL-bb EVAL-e12b-P0/gone.jsonl Qwen/Qwen3-1.7B no_think
#forced EVAL-bb bb_think Qwen/Qwen3-1.7B sideways
EOF
bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bb > $T/out6 2>&1
O=$T/root/runs/EVAL-bb
ok "mode think passed to gen_eval, a bad mode refused" 'grep -q "^gen_eval .*--mode think .*--max_tokens 32768 .*--tag bb_think " $CALLS && ! grep -q "tag bb_bad" $CALLS && grep -q "REFUSED bb_bad" $T/out6'
ok "#forced: own tag (think) and a runs/ reading (no_think), budB, 32 tokens; missing reading and bad mode refused, no .done" 'grep -q "^forced --model $T/m17 --jsonl $O/bb_think_math500_no_think_t0.6.jsonl --bench math500 --mode think --force_preset budB --max_new 32 --budgets 2048,4096,8192,16384,32768 --out $O/bb_think_math500_no_think_t0.6.forced.json --samples $O/bb_think_math500_no_think_t0.6.forced.jsonl" $CALLS && grep -q "^forced --model $T/m17 --jsonl $T/root/runs/EVAL-e12b-P5/e12b_P5.jsonl .*--mode no_think .*--out $O/e12b_P5.forced.json" $CALLS && grep -q "FORCED_SKIPPED EVAL-e12b-P0/gone.jsonl" $T/out6 && grep -q "FORCED_REFUSED bb_think: Qwen/Qwen3-1.7B sideways" $T/out6 && [ ! -f $O/.done ]'
mkdir -p $T/rb/code/analysis; printf 'import sys, os; open(os.environ["CALLS"], "a").write("budget " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/code/analysis/budget_curve.py
: > $O/bb_think_math500_no_think_t0.6.forced.jsonl; echo "#budget EVAL-bb bc bb_think" >> $T/rb/tasks/queue.txt
: > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bb > $T/out7 2>&1
ok "finished forced answers are not redone; a tag never resolves to its forced-answer file" '! grep -q "^forced" $CALLS && grep -q "FORCED_DONE_ALREADY e12b_P5" $T/out7 && grep -q "^budget .* $O/bb_think_math500_no_think_t0.6.jsonl$" $CALLS'
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
R1=$( Q=$T/rb/tasks/queue.txt; eval "$(sed -n '/^result_only() {/p' $SRC/scripts/run_tasks.sh)"; result_only EVAL-res && echo yes || echo no )
R2=$( Q=$T/rb/tasks/queue.txt; eval "$(sed -n '/^result_only() {/p' $SRC/scripts/run_tasks.sh)"; result_only EVAL-bridge && echo yes || echo no )
ok "result-only EVAL ready without inputs_ok" '[ "$R1" = yes ] && [ "$R2" = no ]'
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
d8=$(FIXED_MICRO_BSZ=4 DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10c-P4lowlr-dyn6144 | tail -1); d9=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10c-P4lowlr-fix4-s43 | tail -1)
ok "v2.5.7g E10c dyn6144, fix4-s43 (data seed 43)" '[[ "$d8" == *"arm=P4 lr=6e-7 seed=42 student=$T/m06 "*"use_dynamic_bsz=True"*"ppo_max_token_len_per_gpu=6144"* && "$d9" == *"arm=P4 lr=6e-7 seed=43 student=$T/m06 "*"ppo_micro_batch_size_per_gpu=4"* ]] && [[ "$(DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10c-P5-dyn6144-s43 | tail -1)" == DRYRUN* ]] && ! bash $T/rb/scripts/task.sh 0 E10c-P5-dyn6144-s44 > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E10c-P5-dyn8192 > /dev/null 2>&1'
ok "E10b unchanged" '[[ "$d3" == *"student=$T/m17 "* && "$d3" == *"use_dynamic_bsz=True"* ]]'
d4=$(FUSED=1 GC=0 DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P2 | tail -1)   # v2.5.6o: SPEED-PROBE knobs
d5=$(GC=0 DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-P5 | tail -1)   # v2.5.7b: the 4B student, checkpointing on
mkdir -p $T/root/venv-fa/bin; : > $T/root/venv-fa/bin/activate   # v2.5.7c: E10c stays on the venv in use, E10b takes venv-fa
d6=$(VENV=fa DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10c-P4lowlr-fix4 | tail -1); d7=$(VENV=fa DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P2 | tail -1)
ok "VENV=fa: E10c pinned, E10b on venv-fa" '[[ "$d6" == *"venv=venv "* && "$d7" == *"venv=venv-fa "* ]]'
d8=$(VENV=fa DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-P2 | tail -1); d9=$(VENV=fa E10D_FA=1 DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-P2 | tail -1)
mkdir -p $T/root/runs/PROBE-4B; ( bash $T/rb/scripts/run_arm.sh 0 E10d-P2 > $T/g1 2>&1 ); r1=$?; echo '{"result": "PASS"}' > $T/root/runs/PROBE-4B/report.json
ok "E10d gated on PROBE-4B PASS" '[ $r1 = 3 ] && grep -q "PROBE-4B has not passed" $T/g1'
t1=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5top10 | tail -1); t2=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5rand10m | tail -1)
t3=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P6 | tail -1); t4=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5 | tail -1)
ok "v2.5.7h P5top10 / P5rand10m: through the P6 hook, P5 teacher prompt, 50 steps; P6 / P5 unchanged" '[[ "$t1" == *"arm=P5top10 lr=3e-6 "*"variant=P6 override=P5top10 refthink=1 steps=50" && "$t2" == *"variant=P6 override=P5rand10m refthink=1 steps=50" && "$t3" == *"variant=P6 override= refthink=0 steps=150" && "$t4" == *"variant=P5 override= refthink=1 steps=150" ]]'
( bash $T/rb/scripts/run_arm.sh 0 E10b-P5top10 > $T/g2 2>&1 ); r2=$?
ok "v2.5.7h token-selective arms gated on PROBE-TOP10 PASS" '[ $r2 = 3 ] && grep -q "PROBE-TOP10 has not passed" $T/g2'
ok "E10d on SDPA unless E10D_FA=1" '[[ "$d8" == *"venv=venv "* && "$d9" == *"venv=venv-fa "* ]]'
s1=$(VENV= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-P5-s43 | tail -1); s2=$(VENV=fa DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-P2-s43 | tail -1)
ok "v2.5.7o E10d-P5-s43 / P2-s43: 4B, data seed 43, venv-fa whatever VENV; -s44 refused" '[[ "$s1" == *"arm=P5 lr=3e-6 seed=43 student=$T/m4 "*"venv=venv-fa "* && "$s2" == *"arm=P2 lr=3e-6 seed=43 student=$T/m4 "*"venv=venv-fa "* ]] && [[ "$(DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10d-P5-s43 | tail -1)" == DRYRUN* ]] && ! bash $T/rb/scripts/task.sh 0 E10d-P5-s44 > /dev/null 2>&1'
# v2.5.7p (Q-015): the -t14 arms = E10b-P5 / P2 with the 14B teacher, not offloaded (a node-wide TEACHER_OFFLOAD does not
# reach them), gated on PROBE-T14; E10b-P5 keeps the 8B teacher
u1=$(TEACHER_OFFLOAD=1 DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5-t14 | tail -1); u2=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P2-t14 | tail -1)
u3=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5 | tail -1)
ok "v2.5.7p E10b-P5-t14 / P2-t14: 14B teacher, offload 0, seed 42, 150 steps; E10b-P5 keeps the 8B (offloaded)" '[[ "$u1" == *"arm=P5 lr=3e-6 seed=42 student=$T/m17 "*"teacher=$T/m14 offload=0 variant=P5 "*"refthink=1 steps=150" && "$u2" == *"arm=P2 lr=3e-6 seed=42 "*"teacher=$T/m14 offload=0 variant=P2 "*"refthink=0 steps=150" && "$u3" == *"teacher=$T/m8 offload=1 variant=P5 "* ]]'
( bash $T/rb/scripts/run_arm.sh 0 E10b-P5-t14 > $T/g3 2>&1 ); r3=$?
ok "v2.5.7p -t14 arms gated on PROBE-T14 PASS" '[ $r3 = 3 ] && grep -q "PROBE-T14 has not passed" $T/g3'
ok "v2.5.7p task ids: E10b / E11b -t14 for P2 and P5 only" '[[ "$(DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10b-P2-t14 | tail -1)" == DRYRUN* ]] && ! bash $T/rb/scripts/task.sh 0 E10b-P3-t14 > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E10b-P5-t14-s43 > /dev/null 2>&1'
# v2.5.7r (Q-016): E10b-P5think = P5 with a thinking student: 8K responses, the teacher on the shared template (OPD_REF_THINK 0),
# variant P5, 150 steps, gated on PROBE-P5THINK; P5 itself unchanged (non-thinking, 4K, refthink 1)
v1=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5think | tail -1)
ok "v2.5.7r E10b-P5think: thinking student, 8K, refthink 0, variant P5, 50 steps; E10b-P5 unchanged" '[[ "$v1" == *"arm=P5 lr=3e-6 seed=42 student=$T/m17 "*"sthink=1 maxresp=8192 teacher=$T/m8 offload=1 variant=P5 override= refthink=0 steps=50" && "$u3" == *"sthink=0 maxresp=4096 teacher=$T/m8 offload=1 variant=P5 override= refthink=1 steps=150" ]]'
( bash $T/rb/scripts/run_arm.sh 0 E10b-P5think > $T/g4 2>&1 ); r4=$?
ok "v2.5.7r P5think gated on PROBE-P5THINK PASS; E11b / E12b-P5think refused" '[ $r4 = 3 ] && grep -q "PROBE-P5THINK has not passed" $T/g4 && ! bash $T/rb/scripts/task.sh 0 E11b-P5think > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E12b-P5think > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E10b-P2think > /dev/null 2>&1'
# v2.5.7v (Q-017): E10b-P5self = P5 with the student's base as the teacher (thinking template), 50 steps, gated on PROBE-P5SELF
w1=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5self | tail -1)
ok "v2.5.7v E10b-P5self: teacher = the 1.7B base, refthink 1, variant P5, 4K, 50 steps" '[[ "$w1" == *"arm=P5 lr=3e-6 seed=42 student=$T/m17 "*"sthink=0 maxresp=4096 teacher=$T/m17 offload=1 variant=P5 override= refthink=1 steps=50" ]]'
( bash $T/rb/scripts/run_arm.sh 0 E10b-P5self > $T/g5 2>&1 ); r5=$?
ok "v2.5.7v P5self gated on PROBE-P5SELF PASS; E11b / E12b-P5self refused" '[ $r5 = 3 ] && grep -q "PROBE-P5SELF has not passed" $T/g5 && ! bash $T/rb/scripts/task.sh 0 E11b-P5self > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E12b-P5self > /dev/null 2>&1'
ok "E10d-P5: 4B student, P5, GC on" '[[ "$d5" == *"arm=P5 lr=3e-6 seed=42 student=$T/m4 "* && "$d5" == *"gc=True"* ]] && ! bash $T/rb/scripts/task.sh 0 E10d-P3 > /dev/null 2>&1'
ok "FUSED=1 GC=0" '[[ "$d4" == *"kern=[actor_rollout_ref.model.use_fused_kernels=True actor_rollout_ref.model.fused_kernel_options.impl_backend=torch] gc=False"* && "$d3" == *"kern=[] gc=True"* ]]'
printf 'echo "speed_probe $*"\n' > $T/rb/scripts/speed_probe.sh
ok "SPEED-PROBE task id" '[[ "$(bash $T/rb/scripts/task.sh 3 SPEED-PROBE | tail -1)" == "speed_probe 3" ]]'
rm -rf $T
# 7. v2.5.6j: E12b-P5-s43 (100 steps), E01 / E02 task ids, the OPSA name from the HF id (not from a local base path)
setup
printf 'echo "run_e12 $*"\n' > $T/rb/scripts/run_e12.sh
cp $SRC/scripts/run_zero.sh $T/rb/scripts/run_zero.sh; mkdir -p $T/rb/code/weights
printf 'import sys, os; open(os.environ["CALLS"], "a").write("wops " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/code/weights/wops.py
ok "E12b-P5-s43 / -s44 accepted" '[[ "$(bash $T/rb/scripts/task.sh 0 E12b-P5-s43 | tail -1)" == "run_e12 0 E12b-P5-s43" ]] && [[ "$(bash $T/rb/scripts/task.sh 0 E12b-P5-s44 | tail -1)" == "run_e12 0 E12b-P5-s44" ]] && ! bash $T/rb/scripts/task.sh 0 E12b-P5-s45 > /dev/null 2>&1'
ok "run_e12 gives P5-s43 / P5-s44 / P2-s43 100 steps" 'grep -q "P0|P1|P2|P3|P5|P5-s43|P5-s44|P2-s43) STEPS=100" $SRC/scripts/run_e12.sh'
ok "v2.5.7o E12d-P0 / P5, E12b-P2-s43 accepted" '[[ "$(bash $T/rb/scripts/task.sh 0 E12d-P0 | tail -1)" == "run_e12 0 E12d-P0" ]] && [[ "$(bash $T/rb/scripts/task.sh 0 E12d-P5 | tail -1)" == "run_e12 0 E12d-P5" ]] && [[ "$(bash $T/rb/scripts/task.sh 0 E12b-P2-s43 | tail -1)" == "run_e12 0 E12b-P2-s43" ]] && ! bash $T/rb/scripts/task.sh 0 E12d-P2 > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E12b-P2-s44 > /dev/null 2>&1'
bash $T/rb/scripts/task.sh 0 E02-opsa17-c2 > $T/z1 2>&1; bash $T/rb/scripts/task.sh 0 E01-opsa17-think > $T/z2 2>&1
ok "E02 OPSA name from the HF id" 'grep -q "^wops scale --base $T/m17 --ft Tuwhy/Qwen3-1.7B-OPSA --c 2 " $CALLS && grep -q "e02_opsa17_c2_aime24_nothink" $CALLS'
bash $T/rb/scripts/task.sh 0 E05-fstop-base06 > $T/z3 2>&1
ok "E05-p3 forced stop" 'grep -q -- "--model $T/m06 --bench aime25 --mode think --n 8 --temperature 0.6 --top_p 0.95 --top_k 20 --max_tokens 32768 --tag e05_base06_aime25_think" $CALLS && ! bash $T/rb/scripts/task.sh 0 E05-fstop-x > /dev/null 2>&1'
ok "E01 remainder" 'grep -q -- "--model Tuwhy/Qwen3-1.7B-OPSA --bench aime24 --mode think --n 32 --temperature 0.7 --top_p 0.8 --top_k 20 --max_tokens 32768 --tag e01_opsa17_think_aime24" $CALLS && ! bash $T/rb/scripts/task.sh 0 E01-opsa9-think > /dev/null 2>&1'
rm -rf $T
# 3c. v2.5.7w fetch_pass(): "#fetch Qwen/Qwen3-<n>B" fetched to $OPD_ROOT/models/<name> in the background (no GPU); others refused
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/logs
printf 'import sys, os, time; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n"); os.makedirs(sys.argv[3], exist_ok=True); open(os.path.join(sys.argv[3], ".fetch_ok"), "w").write("ok")\n' > $T/rb/scripts/fetch_model.py
printf 'version: 2\n#fetch Qwen/Qwen3-32B\n#fetch evil/model\n' > $T/q.txt
export CALLS=$T/calls; : > $CALLS
mkdir -p $T/bin; printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; printf '#!/usr/bin/env bash\nexec python3 "$@"\n' > $T/bin/python; chmod +x $T/bin/setsid $T/bin/python   # macOS has no setsid
( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb Q=$T/q.txt; eval "$(sed -n '/^fetch_pass() {/,/^  done; }$/p' $SRC/scripts/run_tasks.sh)"; fetch_pass; sleep 2; fetch_pass; sleep 1 )
ok "v2.5.7w #fetch: Qwen3-32B fetched once to models/, evil/model ignored" '[ "$(grep -c "^fetch Qwen/Qwen3-32B  $T/root/models/Qwen3-32B --task FETCH-Qwen3-32B" $CALLS)" = 1 ] && [ -f $T/root/models/Qwen3-32B/.fetch_ok ] && ! grep -q evil $CALLS'
rm -rf $T
# 3d. v2.5.7za heartbeat hb_to(): a timed-out upload (exit 124) is logged and sent as an event, the loop goes on; others pass through
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/logs $T/bin
printf '#!/usr/bin/env bash\nshift; [ "$3" = hang ] && exit 124; "$@"\n' > $T/bin/timeout; chmod +x $T/bin/timeout
printf 'import sys, os; open(os.environ["CALLS"], "a").write("py " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py; cp $T/rb/scripts/remote.py $T/rb/scripts/control.py
export CALLS=$T/calls; : > $CALLS
( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb; python() { python3 "$@"; }; export -f python
  eval "$(sed -n '/^TO=""; command -v timeout/,/^  return \$rc; }$/p' $SRC/scripts/heartbeat.sh)"
  hb_to python $RUNBOOK/scripts/remote.py put a status/x.json; echo "rc1 $?" >> $CALLS
  hb_to python $RUNBOOK/scripts/remote.py hang a; echo "rc2 $?" >> $CALLS )
ok "v2.5.7za hb_to: normal upload passes, a timeout is logged + evented and returns 124" 'grep -q "^py put a status/x.json" $CALLS && grep -q "^rc1 0" $CALLS && grep -q "^rc2 124" $CALLS && grep -q "UPLOAD_TIMEOUT .* remote.py hang" $T/root/logs/heartbeat.err && grep -q "^py event node heartbeat_upload_timeout remote.py hang" $CALLS'
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
# v2.5.7j: no task script deletes a final bf16 copy (Runming: final weights keep a local copy after upload)
ok "final bf16 copies are never deleted by run_e12.sh / run_arm.sh / run_eval.sh" '! grep -nE "rm -rf[^#]*huggingface_bf16" $SRC/scripts/run_e12.sh $SRC/scripts/run_arm.sh $SRC/scripts/run_eval.sh'
[ $fails = 0 ] && echo "V254A_TEST PASS (48 checks)" || echo "V254A_TEST FAIL ($fails)"
