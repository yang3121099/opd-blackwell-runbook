#!/usr/bin/env bash
# Stub tests of runbook v2.5.4a (no GPU, network or HF): task.sh's out-of-memory retry, run_tasks.sh's .oom marker,
# run_queue_eval.sh (EVAL-<name> from #eval lines of the queue). Usage: bash scripts/test_v254a.sh -> V254A_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0
n=0; ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }   # [v2.5.8a] counted (the total was a fixed 74)
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
# v2.5.7zj: an optional judge / reference on a #twr line (4B: the 4B base and EVAL-e10d-P0); a bad one is refused, nothing runs
mkdir -p $T/root/runs/EVAL-e10d-P0; : > $T/root/runs/EVAL-e10d-P0/e10d_P0_math500_T0.6_math500_no_think_t0.6.jsonl
sed -i.bak 's|^#twr EVAL-bridge e12_P5c05_s40_math500_T0.6$|#twr EVAL-bridge e12_P5c05_s40_math500_T0.6 Qwen/Qwen3-4B EVAL-e10d-P0/e10d_P0_math500_T0.6_math500_no_think_t0.6.jsonl|' $T/rb/tasks/queue.txt
rm -f $T/root/runs/EVAL-bridge/twr.json $T/root/runs/EVAL-bridge/.done; : > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/tw1 2>&1
ok "v2.5.7zj #twr with judge Qwen3-4B and reference EVAL-e10d-P0" 'grep -q "^twr --judge $T/m4 --bench math500 --ref $T/root/runs/EVAL-e10d-P0/e10d_P0_math500_T0.6_math500_no_think_t0.6.jsonl --score $T/root/runs/EVAL-bridge/e12_P5c05_s40_math500_T0.6_" $CALLS'
sed -i.bak 's| Qwen/Qwen3-4B EVAL-e10d-P0/| Qwen/Qwen3-8B ../etc/|' $T/rb/tasks/queue.txt
rm -f $T/root/runs/EVAL-bridge/twr.json $T/root/runs/EVAL-bridge/.done; : > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/tw2 2>&1
ok "v2.5.7zj #twr: a judge / reference outside the list refused, no TWR run" '! grep -q "^twr " $CALLS && grep -q "TWR_REFUSED judge Qwen/Qwen3-8B" $T/tw2 && grep -q "TWR_REFUSED reference" $T/tw2'
sed -i.bak 's| Qwen/Qwen3-8B ../etc/[^ ]*$| Qwen/Qwen3-4B|' $T/rb/tasks/queue.txt
rm -f $T/root/runs/EVAL-bridge/twr.json $T/root/runs/EVAL-bridge/.done; : > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/tw3 2>&1
ok "[v2.5.8] #twr: a judge without a reference refused" '! grep -q "^twr " $CALLS && grep -q "TWR_REFUSED judge Qwen/Qwen3-4B without a reference" $T/tw3'
echo "#twr EVAL-bridge e12_P5c05_s40_math500_T0.6" >> $T/rb/tasks/queue.txt
rm -f $T/root/runs/EVAL-bridge/twr.json $T/root/runs/EVAL-bridge/.done; : > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/tw4 2>&1
ok "[v2.5.8] #twr: a second #twr line for the task refused, no TWR run" '! grep -q "^twr " $CALLS && grep -q "TWR_REFUSED EVAL-bridge has several #twr lines" $T/tw4'
sed -i.bak '$d' $T/rb/tasks/queue.txt; sed -i.bak 's|^#twr EVAL-bridge e12_P5c05_s40_math500_T0.6 Qwen/Qwen3-4B$|#twr EVAL-bridge e12_P5c05_s40_math500_T0.6 Qwen/Qwen3-4B EVAL-e10d-P0/e10d_P0_math500_T0.6_math500_no_think_t0.6.jsonl|' $T/rb/tasks/queue.txt
printf 'import sys, os, json\nopen(os.environ["CALLS"], "a").write("twr " + " ".join(sys.argv[1:]) + "\\n")\nprint("ALPHA_REFUSED hidden size: judge 2560 != model_k 2048")\nout = sys.argv[sys.argv.index("--out") + 1]; json.dump({"alpha_error": "x"}, open(out, "w"))\n' > $T/rb/code/leakage/twr_alpha.py
rm -f $T/root/runs/EVAL-bridge/twr.json $T/root/runs/EVAL-bridge/.done; : > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-bridge > $T/tw5 2>&1
ok "[v2.5.8] #twr: a judge of another size is one clear line in the task log" 'grep -q "^twr " $CALLS && grep -q "^TWR_e12_P5c05_s40_math500_T0.6 ALPHA_REFUSED hidden size: judge 2560 != model_k 2048" $T/tw5'
got=$( OPD_ROOT=$T/root; eval "$(sed -n '/^run_rel() {/p' $SRC/scripts/run_queue_eval.sh)"; for x in A/b.jsonl ../b.jsonl ./b.jsonl A/..jsonl; do run_rel "$x" && printf "%s=1 " $x || printf "%s=0 " $x; done )
ok "[v2.5.8] run_rel: <task>/<file>.jsonl only, never .. / ." '[ "$got" = "A/b.jsonl=1 ../b.jsonl=0 ./b.jsonl=0 A/..jsonl=1 " ]'
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
# 5b. v2.5.7zd: result: reads the training's own save on this node first (bf16 copy, else fp32 read in bf16); the
#     result repository only when there is none; the prefetch fetches nothing for a local save
setup
mkdir -p $T/root/runs/E12d-P0/ckpt/global_step_100/actor/huggingface $T/root/runs/E12b-X/ckpt/global_step_50/actor/huggingface_bf16
touch $T/root/runs/E12d-P0/ckpt/global_step_100/actor/huggingface/model.safetensors $T/root/runs/E12b-X/ckpt/global_step_50/actor/huggingface_bf16/model.safetensors
mkdir -p $T/root/adhoc_models/result/runs/E12b-Y/global_step_50/actor/huggingface; touch $T/root/adhoc_models/result/runs/E12b-Y/global_step_50/actor/huggingface/model.safetensors   # as fetched
cat > $T/rb/tasks/queue.txt <<'EOF2'
version: 2
#eval EVAL-lf e12d_P0_s100_math500_T0.6 result:runs/E12d-P0/global_step_100/actor/huggingface math500 4 0.6
#eval EVAL-lf e12b_X_s50_math500_T0.6 result:runs/E12b-X/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-lf e12b_Y_s50_math500_T0.6 result:runs/E12b-Y/global_step_50/actor/huggingface math500 4 0.6
EOF2
bash $T/rb/scripts/run_queue_eval.sh prefetch EVAL-lf > $T/p5 2>&1
ok "v2.5.7zd prefetch: local saves fetch nothing, the third is fetched" '[ "$(grep -c PREFETCH_LOCAL $T/p5)" = 2 ] && [ "$(grep -c "^fetch " $CALLS)" = 1 ] && grep -q "^fetch me/pub runs/E12b-Y/" $CALLS'
: > $CALLS; bash $T/rb/scripts/run_queue_eval.sh 2 EVAL-lf > $T/p6 2>&1
ok "v2.5.7zd result: -> local fp32 (bfloat16) / local bf16 (auto) / fetched" 'grep -q "^gen_eval .*--model $T/root/runs/E12d-P0/ckpt/global_step_100/actor/huggingface .*--dtype bfloat16" $CALLS && grep -q "^gen_eval .*--model $T/root/runs/E12b-X/ckpt/global_step_50/actor/huggingface_bf16 .*--dtype auto" $CALLS && grep -q "^gen_eval .*--model $T/root/adhoc_models/result/runs/E12b-Y/global_step_50/actor/huggingface " $CALLS && [ "$(grep -c "^fetch " $CALLS)" = 1 ]'
rm -rf $T
# 5c. v2.5.7zd env.sh: an inherited HF id (a loop started before the fetch) gives way to the fetched local copy; an
#     explicit local path is kept; no .fetch_ok keeps the HF id
T=$(mktemp -d); mkdir -p $T/models/Qwen3-14B; touch $T/models/Qwen3-14B/.fetch_ok
got=$( OPD_ROOT=$T; eval "$(grep -E '^case "\$\{(STUDENT_4B|TEACHER_14B):-' $SRC/scripts/env.sh)"
  a=$TEACHER_14B; TEACHER_14B=Qwen/Qwen3-14B; eval "$(grep -E '^case "\$\{TEACHER_14B:-' $SRC/scripts/env.sh)"; b=$TEACHER_14B
  TEACHER_14B=/x/own; eval "$(grep -E '^case "\$\{TEACHER_14B:-' $SRC/scripts/env.sh)"; c=$TEACHER_14B
  STUDENT_4B=Qwen/Qwen3-4B; eval "$(grep -E '^case "\$\{STUDENT_4B:-' $SRC/scripts/env.sh)"; echo "$a|$b|$c|$STUDENT_4B" )
ok "v2.5.7zd env.sh: local 14B wins over an unset or inherited HF id, an own path stays, no .fetch_ok keeps the id" '[ "$got" = "$T/models/Qwen3-14B|$T/models/Qwen3-14B|/x/own|Qwen/Qwen3-4B" ]'
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
d10=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10c-P5-fix4 | tail -1)   # [v2.5.8a]
ok "[v2.5.8a] E10c-P5-fix4: 0.6B P5 (lr 3e-6, fix4, 150 steps); run_arm and run_tasks upload s100 and s150" '[[ "$d10" == *"arm=P5 lr=3e-6 seed=42 student=$T/m06 "*"ppo_micro_batch_size_per_gpu=4"*"steps=150" ]] && grep -q "UW=\$NSTEPS; \[ \$TASK = E10c-P5-fix4 \] && UW=100,\$NSTEPS" $SRC/scripts/run_arm.sh && grep -q "UPLOAD_WEIGHTS=\${UPLOAD_WEIGHTS:-\$UW} setsid" $SRC/scripts/run_arm.sh && grep -q "E10c-P5-fix4) env UPLOAD_WEIGHTS=\${UPLOAD_WEIGHTS:-100,150}" $SRC/scripts/run_tasks.sh'
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
ok "v2.5.7v P5self gated on PROBE-P5SELF PASS; E11b-P5self refused" '[ $r5 = 3 ] && grep -q "PROBE-P5SELF has not passed" $T/g5 && ! bash $T/rb/scripts/task.sh 0 E11b-P5self > /dev/null 2>&1'
# [v2.5.8e] E10b-P5self-s43 (data seed 43, the same gate PROBE-P5SELF) and E10d-P5self (4B, its own base as the teacher, PROBE-P5SELF-4B)
w2=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5self-s43 | tail -1); w3=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-P5self | tail -1)
ok "[v2.5.8e] E10b-P5self-s43: P5self with data seed 43, 50 steps" '[[ "$w2" == *"arm=P5 lr=3e-6 seed=43 student=$T/m17 "*"teacher=$T/m17 offload=1 variant=P5 override= refthink=1 steps=50" ]]'
ok "[v2.5.8e] E10d-P5self: the 4B student taught by its own base (thinking template), SDPA as E10d-P5, 50 steps" '[[ "$w3" == *"arm=P5 lr=3e-6 seed=42 student=$T/m4 "*"gc=True venv=venv "*"teacher=$T/m4 offload=1 variant=P5 override= refthink=1 steps=50" ]]'
( bash $T/rb/scripts/run_arm.sh 0 E10b-P5self-s43 > $T/g6 2>&1 ); r6=$?; ( bash $T/rb/scripts/run_arm.sh 0 E10d-P5self > $T/g7 2>&1 ); r7=$?
mkdir -p $T/mi $T/mt $T/m4   # [v2.5.8i] E10d-X1 / X2: the 2507 student, the 2507 or hybrid teacher, the template flag, 50 steps, the gate
x1=$(STUDENT_I2507=$T/mi TEACHER_T2507=$T/mt FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-X1 | tail -1)
x2=$(STUDENT_I2507=$T/mi TEACHER_T2507=$T/mt FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-X2 | tail -1)
ok "[v2.5.8i] E10d-X1: Instruct-2507 taught by Thinking-2507, think-template 0 (its own template opens <think>), 50 steps" '[[ "$x1" == *"arm=P5 lr=3e-6 seed=42 student=$T/mi "*"teacher=$T/mt offload=1 variant=P5 override= refthink=0 steps=50" ]]'
ok "[v2.5.8i] E10d-X2: Instruct-2507 taught by the hybrid 4B on the thinking template (refthink 1)" '[[ "$x2" == *"student=$T/mi "*"teacher=$T/m4 offload=1 variant=P5 override= refthink=1 steps=50" ]]'
( STUDENT_I2507=$T/mi TEACHER_T2507=$T/mt bash $T/rb/scripts/run_arm.sh 0 E10d-X1 > $T/g8 2>&1 ); r8=$?
ok "[v2.5.8i] X1 gated on PROBE-X2507-X1; task names X1 / X2 run, X3 refused; uploads s25 / s50" '[ $r8 = 3 ] && grep -q "PROBE-X2507-X1 has not passed: E10d-X1" $T/g8 && [[ "$(STUDENT_I2507=$T/mi TEACHER_T2507=$T/mt DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10d-X2 | tail -1)" == DRYRUN* ]] && ! bash $T/rb/scripts/task.sh 0 E10d-X3 > /dev/null 2>&1 && grep -q "E10d-X1|E10d-X2|E10d-X4) env UPLOAD_WEIGHTS=\${UPLOAD_WEIGHTS:-25,50}" $SRC/scripts/run_tasks.sh && grep -q "\[ -n \"\$XARM\" \] && UW=25,\$NSTEPS" $SRC/scripts/run_arm.sh'
# [v2.5.8l] (Q-032) E10d-X4: the hybrid 4B student (as E10d-P5) taught by Instruct-2507, think-template 0, 50 steps, its gate
x4=$(STUDENT_I2507=$T/mi TEACHER_T2507=$T/mt FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10d-X4 | tail -1)
ok "[v2.5.8l] E10d-X4: the hybrid 4B taught by Instruct-2507 (refthink 0), 50 steps" '[[ "$x4" == *"arm=P5 lr=3e-6 seed=42 student=$T/m4 "*"teacher=$T/mi offload=1 variant=P5 override= refthink=0 steps=50" ]]'
( STUDENT_I2507=$T/mi TEACHER_T2507=$T/mt bash $T/rb/scripts/run_arm.sh 0 E10d-X4 > $T/g9 2>&1 ); r9=$?
ok "[v2.5.8l] X4 gated on PROBE-X2507-X4; task name X4 runs, X5 refused" '[ $r9 = 3 ] && grep -q "PROBE-X2507-X4 has not passed: E10d-X4" $T/g9 && [[ "$(STUDENT_I2507=$T/mi TEACHER_T2507=$T/mt DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10d-X4 | tail -1)" == DRYRUN* ]] && ! bash $T/rb/scripts/task.sh 0 E10d-X5 > /dev/null 2>&1'
# [v2.5.8m] (Q-030) E10b-P5lenS / L / R: P5 with 32 prompts x 16 rollouts, 4 kept (shortest / longest / random), one mini-batch
# of 8 prompts (= 128 kept), venv-fa, 50 steps; gated on PROBE-LENSEL; the trainer arguments carry the batch shape
ls1=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5lenS | tail -1); ls2=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5lenL | tail -1); ls3=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5lenR | tail -1)
ok "[v2.5.8m] E10b-P5lenS / L / R: P5, venv-fa, short / long / rand, 32 x 16, mini-batch 8, refthink 1, 50 steps" '[[ "$ls1" == *"arm=P5 lr=3e-6 seed=42 student=$T/m17 "*"venv=venv-fa "*"lensel=short/32x16/mb8 sthink=0 "*"variant=P5 override= refthink=1 steps=50" && "$ls2" == *"lensel=long/32x16/mb8 "* && "$ls3" == *"lensel=rand/32x16/mb8 "* ]]'
ls0=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5 | tail -1)
ok "[v2.5.8m] other arms keep 128 x 1 and no selection" '[[ "$ls0" == *"lensel=none/128x1/mb128 "* ]] && grep -q "data.train_batch_size=\$TBS " $SRC/scripts/run_arm.sh && grep -q "ppo_mini_batch_size=\$MBS " $SRC/scripts/run_arm.sh && grep -q "rollout.n=\$RN " $SRC/scripts/run_arm.sh'
( bash $T/rb/scripts/run_arm.sh 0 E10b-P5lenL > $T/g10 2>&1 ); r10=$?
ok "[v2.5.8m] gated on PROBE-LENSEL; names P5lenS / L / R run, P5lenX refused; uploads s50" '[ $r10 = 3 ] && grep -q "PROBE-LENSEL has not passed: E10b-P5lenL" $T/g10 && [[ "$(DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10b-P5lenR | tail -1)" == DRYRUN* ]] && ! bash $T/rb/scripts/task.sh 0 E10b-P5lenX > /dev/null 2>&1 && grep -q "E10b-P2self|E10b-P5len\[SLR\]) env UPLOAD_WEIGHTS=\${UPLOAD_WEIGHTS:-50}" $SRC/scripts/run_tasks.sh'
# [v2.5.9l] (Q-038) E10b-P5stopmerge: P5 (refthink 1, variant P5), 50 steps, no fused kernels even with FUSED=1, OPD_STOPMERGE=1
# and its log; gated on PROBE-STOPMERGE; uploads s50
sm=$(FUSED=1 FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5stopmerge); ( bash $T/rb/scripts/run_arm.sh 0 E10b-P5stopmerge > $T/gsm 2>&1 ); rsm=$?
ok "[v2.5.9l] E10b-P5stopmerge: P5, kern [] under FUSED=1, OPD_STOPMERGE=1 + stopmerge.jsonl, 50 steps; gated on PROBE-STOPMERGE; uploads s50" '[[ "$(echo "$sm" | tail -1)" == *"arm=P5 lr=3e-6 "*"kern=[] "*"variant=P5 override= refthink=1 steps=50" ]] && echo "$sm" | grep -q "^DRYRUN_STOPMERGE OPD_STOPMERGE=1 log=stopmerge.jsonl kern=\[\]$" && [ $rsm = 3 ] && grep -q "PROBE-STOPMERGE has not passed: E10b-P5stopmerge" $T/gsm && [[ "$(DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10b-P5stopmerge | tail -1)" == DRYRUN* ]] && grep -q "        E10b-P5stopmerge|E10b-P5top10|" $SRC/scripts/run_tasks.sh'
# [v2.5.9a] a GPU list: trainings (and PROBE-2GPU) take it, run_arm with ngpu=2; an evaluation and a malformed list are refused
g2=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/task.sh 2,3 E10b-P5 | tail -1); g3=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 2,3 E10b-P5 | tail -1); g4=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 2 E10b-P5 | tail -1)
( bash $T/rb/scripts/task.sh 2,3 EVAL-x > $T/g11 2>&1 ); r11=$?; ( bash $T/rb/scripts/task.sh '2;3' E12b-P5 > $T/g12 2>&1 ); r12=$?
ok "[v2.5.9a] GPU lists: task.sh passes 2,3 to an E10b arm (ngpu=2; 1 alone), EVAL and a bad list refused, n_gpus_per_node from the list" '[[ "$g2" == *" ngpu=2 "* && "$g3" == *" ngpu=2 "* && "$g4" == *" ngpu=1 "* ]] && [ $r11 = 2 ] && grep -q "EVAL-x takes one GPU" $T/g11 && [ $r12 = 2 ] && grep -q "bad gpu" $T/g12 && grep -q "trainer.n_gpus_per_node=\$NG " $SRC/scripts/run_e12.sh && grep -q "trainer.n_gpus_per_node=\$NG " $SRC/scripts/run_arm.sh && grep -q "RAY_TMPDIR=/tmp/ray_gpu\${GPU//,/_}" $SRC/scripts/run_arm.sh'
# [v2.5.9b] E10b-P5tool: P5 on a tool-call parquet (TOOL_DATA), prompts 2048, the rule grader; refused without its data; not a queue task
mkdir -p $T/td; : > $T/td/t_train.parquet; : > $T/td/t_val.parquet
tl=$(TOOL_DATA=$T/td/t FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5tool | tail -1); tm=$(FIXED_MICRO_BSZ= DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5 | tail -1)
( DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P5tool > $T/g13 2>&1 ); r13=$?
ok "[v2.5.9b] E10b-P5tool: tool data, maxp 2048, tool_grader, P5 refthink 1, 50 steps; math arms unchanged; no data: exit 2" '[[ "$tl" == *"arm=P5 "*" tool=1 maxp=2048 reward=tool_grader.py "*"refthink=1 steps=50" && "$tm" == *" tool=0 maxp=1024 reward=reward_mathverify.py "* ]] && [ $r13 = 2 ] && grep -q TOOL_DATA_MISSING $T/g13 && ! bash $T/rb/scripts/task.sh 0 E10b-P5tool > /dev/null 2>&1 && grep -q "data.train_files=\$TRAIN data.val_files=\$VAL" $SRC/scripts/run_arm.sh && grep -q "custom_reward_function.path=\$REWARD " $SRC/scripts/run_arm.sh'
ok "[v2.5.8e] gates: P5self-s43 on PROBE-P5SELF, E10d-P5self on PROBE-P5SELF-4B" '[ $r6 = 3 ] && grep -q "PROBE-P5SELF has not passed: E10b-P5self-s43" $T/g6 && [ $r7 = 3 ] && grep -q "PROBE-P5SELF-4B has not passed: E10d-P5self" $T/g7'
ok "[v2.5.8e] task names: E10b-P5self-s43, E10d-P5self, E12b-P5self-s43 run; -s44 / E10d-P2self refused" '[[ "$(DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10d-P5self | tail -1)" == DRYRUN* ]] && [[ "$(DRYRUN=1 bash $T/rb/scripts/task.sh 0 E10b-P5self-s43 | tail -1)" == DRYRUN* ]] && ! bash $T/rb/scripts/task.sh 0 E10b-P5self-s44 > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E10d-P2self > /dev/null 2>&1 && grep -q "|P5self|P5self-s43|P5-kl|P5-stop|P5-stop2|P5-8k|P5-16k)" $SRC/scripts/task.sh'
# v2.5.7zg (Q-019): E10b-P2self = P2 with the student's base as the teacher (non-thinking), 50 steps, venv-fa, gated on PROBE-P2SELF
w2=$(DRYRUN=1 bash $T/rb/scripts/run_arm.sh 0 E10b-P2self | tail -1)
ok "v2.5.7zg E10b-P2self: teacher = the 1.7B base, refthink 0, variant P2, 4K, 50 steps, venv-fa" '[[ "$w2" == *"arm=P2 lr=3e-6 seed=42 student=$T/m17 "*"venv=venv-fa "*"sthink=0 maxresp=4096 teacher=$T/m17 offload=1 variant=P2 override= refthink=0 steps=50" ]]'
( bash $T/rb/scripts/run_arm.sh 0 E10b-P2self > $T/g6 2>&1 ); r6=$?
ok "v2.5.7zg P2self gated on PROBE-P2SELF PASS; E11b / E12b-P2self refused" '[ $r6 = 3 ] && grep -q "PROBE-P2SELF has not passed" $T/g6 && ! bash $T/rb/scripts/task.sh 0 E11b-P2self > /dev/null 2>&1 && ! bash $T/rb/scripts/task.sh 0 E12b-P2self > /dev/null 2>&1'
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
ok "run_e12 gives P5-s43 / P5-s44 / P2-s43 / [v2.5.8a] P5self / [v2.5.8e] P5self-s43 100 steps" 'grep -q "P0|P1|P2|P3|P5|P5-s43|P5-s44|P2-s43|P5self|P5self-s43) STEPS=100" $SRC/scripts/run_e12.sh && grep -q "case \$A in P5self|P5self-s43) SS=50;; esac" $SRC/scripts/run_e12.sh'
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
# 3c2. [v2.5.9e] fetch_pass: dataset:<owner>/<name> -> data/hf/<owner>__<name> with --dataset, Qwen3-VL ids, niced, a 6 h cap; no new
#      pull while uploads time out; malformed ids refused
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/logs $T/root/control $T/bin
printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + " T=" + os.environ.get("FETCH_TIMEOUT", "") + "\\n"); os.makedirs(sys.argv[3], exist_ok=True); open(os.path.join(sys.argv[3], ".fetch_ok"), "w").write("ok")\n' > $T/rb/scripts/fetch_model.py
printf 'version: 2\n#fetch dataset:BytedTsinghua-SIA/DAPO-Math-17k\n#fetch Qwen/Qwen3-VL-2B-Instruct\n#fetch dataset:bad;name/x\n' > $T/q.txt
export CALLS=$T/calls; : > $CALLS
printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; printf '#!/usr/bin/env bash\nexec python3 "$@"\n' > $T/bin/python; chmod +x $T/bin/setsid $T/bin/python
fp() { ( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb Q=$T/q.txt CDIR=$T/root/control; eval "$(sed -n '/^fetch_pass() {/,/^  done; }$/p' $SRC/scripts/run_tasks.sh)"; fetch_pass; sleep 1.5 ); }
echo "UPLOAD_TIMEOUT 10:00:00 x y rc=124" > $T/root/logs/heartbeat.err; fp
ok "[v2.5.9e] no pull while result uploads time out" '[ ! -s $CALLS ]'
rm -f $T/root/logs/heartbeat.err; fp; fp; fp
ok "[v2.5.9e] a dataset to data/hf/<owner>__<name> with --dataset and a 6 h cap, then the VL model; a bad id ignored" 'grep -q "^fetch BytedTsinghua-SIA/DAPO-Math-17k  $T/root/data/hf/BytedTsinghua-SIA__DAPO-Math-17k --dataset --task FETCH-BytedTsinghua-SIA__DAPO-Math-17k T=21600$" $CALLS && grep -q "^fetch Qwen/Qwen3-VL-2B-Instruct  $T/root/models/Qwen3-VL-2B-Instruct --task FETCH-Qwen3-VL-2B-Instruct T=21600$" $CALLS && [ "$(grep -c ^fetch $CALLS)" = 2 ] && ! grep -q bad $CALLS'
ok "[v2.5.9e] background pulls are niced" 'grep -q "nice -n 19 \$io python \$RUNBOOK/scripts/fetch_model.py" $SRC/scripts/run_tasks.sh && grep -q "^fetch_progress() {" $SRC/scripts/run_tasks.sh'
rm -rf $T
# 3d. v2.5.7za heartbeat hb_to(): a timed-out upload (exit 124) is logged and sent as an event, the loop goes on; others pass through
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/logs $T/bin
printf '#!/usr/bin/env bash\necho "timeout $*" >> %s; [ "$1" = -k ] && shift 2; shift; [ "$3" = hang ] && exit 124; "$@"\n' $T/targs > $T/bin/timeout; chmod +x $T/bin/timeout
printf 'import sys, os; open(os.environ["CALLS"], "a").write("py " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py; cp $T/rb/scripts/remote.py $T/rb/scripts/control.py
export CALLS=$T/calls; : > $CALLS
( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb; python() { python3 "$@"; }; export -f python
  eval "$(sed -n '/^TO=""; command -v timeout/,/^  return \$rc; }$/p' $SRC/scripts/heartbeat.sh)"
  hb_to python $RUNBOOK/scripts/remote.py put a status/x.json; echo "rc1 $?" >> $CALLS
  hb_to python $RUNBOOK/scripts/remote.py hang a; echo "rc2 $?" >> $CALLS )
ok "[v2.5.8] hb_to runs its uploads under timeout -k 30" 'grep -q "^timeout -k 30 180 python .*remote.py put a status/x.json" $T/targs'
ok "v2.5.7za hb_to: normal upload passes, a timeout is logged + evented and returns 124" 'grep -q "^py put a status/x.json" $CALLS && grep -q "^rc1 0" $CALLS && grep -q "^rc2 124" $CALLS && grep -q "UPLOAD_TIMEOUT .* remote.py hang" $T/root/logs/heartbeat.err && grep -q "^py event node heartbeat_upload_timeout remote.py hang" $CALLS'
rm -rf $T
# 3e. v2.5.7zb run_tasks bounded(): a hung command (git pull, control.py apply) is cut at its timeout (124), logged, and
#     the loop goes on; a normal command passes through (a GNU-timeout stub, since macOS has none)
T=$(mktemp -d); mkdir -p $T/root/logs $T/rb/scripts $T/bin
cat > $T/bin/timeout <<'PY'
#!/usr/bin/env python3
import os, signal, subprocess, sys
a = sys.argv[3:] if sys.argv[1] == "-k" else sys.argv[1:]   # [v2.5.8] timeout -k <s> <limit> ...
p = subprocess.Popen(a[1:], start_new_session=True)
try:
    sys.exit(p.wait(timeout=float(a[0])))
except subprocess.TimeoutExpired:
    os.killpg(p.pid, signal.SIGTERM); sys.exit(124)
PY
chmod +x $T/bin/timeout; printf 'import sys, os; open(os.environ["CALLS"], "a").write("py " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
export CALLS=$T/calls; : > $CALLS
( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb; python() { python3 "$@"; }; export -f python
  eval "$(sed -n '/^bounded() {/,/^  return \$rc; }$/p' $SRC/scripts/run_tasks.sh)"
  t0=$(date +%s); bounded 2 sleep 30; echo "rc_hang $? $(( $(date +%s) - t0 ))" >> $CALLS
  bounded 5 true; echo "rc_ok $?" >> $CALLS; sleep 1 )
ok "v2.5.7zb bounded: a hung call returns 124 at its timeout and is logged + evented; a normal one passes" 'grep -qE "^rc_hang 124 [23]$" $CALLS && grep -q "^rc_ok 0" $CALLS && grep -q "LOOP_TIMEOUT .* 2s rc=124 sleep 30" $T/root/logs/control.log && grep -q "^py event node loop_timeout sleep 30" $CALLS'
ok "[v2.5.8] bounded() uses timeout -k 30 and takes 137 (KILLed) as a timeout too" 'grep -q "timeout -k 30 \$s \"\$@\"" $SRC/scripts/run_tasks.sh && grep -q "\[ \$rc = 137 \]" $SRC/scripts/run_tasks.sh'
ok "v2.5.7zb run_tasks bounds its git pull and apply; the loop writes its liveness file and starts the watchdog" 'grep -q "bounded \${PULL_TIMEOUT:-120} git -C" $SRC/scripts/run_tasks.sh && grep -q "bounded \${APPLY_TIMEOUT:-300} python \$RUNBOOK/scripts/control.py apply" $SRC/scripts/run_tasks.sh && grep -q "loop_run_tasks" $SRC/scripts/run_tasks.sh && grep -q "ensure_watchdog" $SRC/scripts/run_tasks.sh'
rm -rf $T
# 3m. v2.5.7zl run_eval.sh (E11b) #pair: pairs before the upload; a pairs-only rerun skips the readings and the TWR there
setup; cp $SRC/scripts/run_eval.sh $T/rb/scripts/; mkdir -p $T/rb/code/analysis $T/root/runs/E10b-P2-t14/ckpt/global_step_150/actor/huggingface_bf16
touch $T/root/runs/E10b-P2-t14/ckpt/global_step_150/actor/huggingface_bf16/model.safetensors
printf 'import sys, os; open(os.environ["CALLS"], "a").write("pair " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/code/analysis/paired_bootstrap.py
printf 'version: 2\n#pair E11b-P2-t14 e11b_P2-t14_math500_T0.6 E11b-P2/e11b_P2_math500_T0.6_math500_no_think_t0.6.jsonl E11b-P0/x.jsonl ../etc/y.jsonl\n' > $T/rb/tasks/queue.txt
mkdir -p $T/root/runs/E11b-P2-t14; for TT in T0.6 T1.0; do : > $T/root/runs/E11b-P2-t14/e11b_P2-t14_math500_${TT}_math500_no_think_t${TT#T}.jsonl; : > $T/root/runs/E11b-P2-t14/e11b_P2-t14_math500_${TT}_math500_no_think_t${TT#T}.metrics.json; done
echo '{}' > $T/root/runs/E11b-P2-t14/twr.json; mkdir -p $T/root/runs/E11b-P0; : > $T/root/runs/E11b-P0/e11b_P0_math500_T0.6_math500_no_think_t0.6.jsonl
( cd $T/rb && PATH=$PATH bash scripts/run_eval.sh 0 E11b-P2-t14 > $T/re1 2>&1 )
ok "v2.5.7zl run_eval pairs: two refs, ../ refused, no generation, no TWR, uploaded" 'grep -q "^pair --pairs $T/root/runs/E11b-P2-t14/e11b_P2-t14_math500_T0.6.paired.json $T/root/runs/E11b-P2-t14/e11b_P2-t14_math500_T0.6_math500_no_think_t0.6.jsonl $T/root/runs/E11b-P2/e11b_P2_math500_T0.6_math500_no_think_t0.6.jsonl $T/root/runs/E11b-P0/x.jsonl$" $CALLS && grep -q "PAIR_REF_REFUSED ../etc/y.jsonl" $T/re1 && ! grep -q "^gen_eval\|^twr" $CALLS && grep -q "remote put $T/root/runs/E11b-P2-t14 runs/E11b-P2-t14" $CALLS && [ -f $T/root/runs/E11b-P2-t14/.done ]'
printf 'version: 2\n#pair E11b-P2-t14 e11b_P2-t14_math500_T0.6 ../etc/y.jsonl\n' > $T/rb/tasks/queue.txt; rm -f $T/root/runs/E11b-P2-t14/.done; : > $CALLS
( cd $T/rb && PATH=$PATH bash scripts/run_eval.sh 0 E11b-P2-t14 > $T/re2 2>&1 )
ok "[v2.5.8] run_eval pairs: all refs refused -> PAIR_SKIPPED, no paired_bootstrap call, still uploaded" '! grep -q "^pair " $CALLS && grep -q "PAIR_SKIPPED e11b_P2-t14_math500_T0.6 (no usable reference)" $T/re2 && [ -f $T/root/runs/E11b-P2-t14/.done ]'
rm -rf $T
# 3f. [v2.5.8] prefetch_pass(): up to PREFETCH_PAR prefetches of different tasks at once (was one), none twice, each
#     under PREFETCH_TIMEOUT (a timeout stub that records its limit)
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/root/logs $T/bin
printf 'T=$2; echo "prefetch $T" >> $CALLS; sleep 4; touch $OPD_ROOT/runs/$T/.done\n' > $T/rb/scripts/run_queue_eval.sh
printf '#!/usr/bin/env bash\n[ "$1" = -k ] && shift 2; echo "limit $1" >> $CALLS; shift; exec "$@"\n' > $T/bin/timeout
printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; chmod +x $T/bin/timeout $T/bin/setsid
printf 'version: 2\n' > $T/q.txt; for i in 1 2 3 4 5; do echo "#eval EVAL-p$i t$i result:runs/X/global_step_$i math500 4 0.6" >> $T/q.txt; done
export CALLS=$T/calls; : > $CALLS
( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb Q=$T/q.txt PREFETCH_PAR=3 PREFETCH_TIMEOUT=77; deps_of() { :; }
  eval "$(sed -n '/^eval_hash() {/,/^  done; }$/p' $SRC/scripts/run_tasks.sh)"
  prefetch_pass; sleep 1; prefetch_pass; sleep 1; echo "mid $(grep -c '^prefetch' $CALLS)" >> $CALLS; sleep 4; prefetch_pass; sleep 5 )
ok "[v2.5.8] prefetch_pass: 3 at once, none twice, the rest after them" 'grep -q "^mid 3$" $CALLS && [ "$(grep -c "^prefetch" $CALLS)" = 5 ] && [ "$(grep "^prefetch" $CALLS | sort -u | wc -l | tr -d " ")" = 5 ]'
ok "[v2.5.8] prefetch_pass: every prefetch under PREFETCH_TIMEOUT" '[ "$(grep -c "^limit 77$" $CALLS)" = 5 ]'
rm -rf $T
# 3g. [v2.5.8] heartbeat status: .paused (not running) -> "paused"; a directory no longer in the queue, not done and not
#     running -> stale_task_dirs, not in tasks; done ones and ad-hoc evaluations stay listed
T=$(mktemp -d); mkdir -p $T/root/runs/{E12d-P0,EVAL-gone,EVAL-paused,adhoc_eval_7,EVAL-old-done} $T/root/logs $T/root/control $T/rb/tasks $T/bin
touch $T/root/runs/EVAL-old-done/.done $T/root/runs/EVAL-paused/.paused; printf 'version: 9\ne12 open E12d-P0 EVAL-paused\n' > $T/rb/tasks/queue.txt
printf '#!/usr/bin/env bash\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi; cp -r $SRC/scripts $T/rb/
sed -n "/^  python - \"\$F\" <<'PY'$/,/^PY$/p" $SRC/scripts/heartbeat.sh | sed '1d;$d' > $T/hb.py
( cd $T && PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb python3 hb.py $T/out.json > /dev/null 2>&1 )
got=$(python3 -c "import json; d = json.load(open('$T/out.json')); print(' '.join(sorted(d['tasks'])), '|', ' '.join(d['stale_task_dirs']), '|', d['tasks']['EVAL-paused']['state'])" 2>&1)
ok "[v2.5.8] heartbeat: paused state, stale directories counted not listed" '[ "$got" = "E12d-P0 EVAL-old-done EVAL-paused adhoc_eval_7 | EVAL-gone | paused" ]'
rm -rf $T
# 3g2. [v2.5.9m] padding stages: run_tasks dispatches every stage not on a "#pad" line first, then the padding stages, each
#      in file order (without #pad lines: the file order); the heartbeat marks their tasks pad: true and lists a task
#      written as name:dep (it was called stale before)
T=$(mktemp -d); printf 'version: 2\n# a comment\na  open X1\np1 open P1a P1b\nb  auto Y1\np2 open P2a\nc  open Z1:Y1\n#pad p1\n#pad p2 p9\n' > $T/q.txt
grep -v '^#pad' $T/q.txt > $T/q0.txt
sl() { ( eval "$(sed -n '/^stage_lines() {/,/; }$/p' $SRC/scripts/run_tasks.sh)"; stage_lines "$1" | cut -d' ' -f1 | tr '\n' ' ' ); }
o1=$(sl $T/q.txt); o0=$(sl $T/q0.txt)
ok "[v2.5.9m] stage_lines: formal stages first (file order), then the padding ones (file order); no #pad = the file order (o1=$o1 o0=$o0)" '[ "$o1" = "a b c p1 p2 " ] && [ "$o0" = "a p1 b p2 c " ] && grep -q "done < <(stage_lines \"\$Q\")" $SRC/scripts/run_tasks.sh'
rm -rf $T
T=$(mktemp -d); mkdir -p $T/root/runs/{E12d-P0,EVAL-pad1,EVAL-wait,EVAL-formal} $T/root/logs $T/root/control $T/rb/tasks $T/bin
printf 'version: 9\ne12 open E12d-P0\nfill open EVAL-pad1 EVAL-wait:E12d-P0\nlate open EVAL-formal\n#pad fill\n' > $T/rb/tasks/queue.txt
printf '#!/usr/bin/env bash\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi; cp -r $SRC/scripts $T/rb/
sed -n "/^  python - \"\$F\" <<'PY'$/,/^PY$/p" $SRC/scripts/heartbeat.sh | sed '1d;$d' > $T/hb.py
( cd $T && PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb python3 hb.py $T/out.json > /dev/null 2>&1 )
got=$(python3 -c "import json; d = json.load(open('$T/out.json')); t = d['tasks']; print(' '.join(sorted(t)), '|', ' '.join(sorted(n for n in t if t[n].get('pad'))), '|', ' '.join(d['stale_task_dirs']))" 2>&1)
ok "[v2.5.9m] heartbeat: tasks of #pad stages carry pad true; a name:dep task is listed, not stale (got=$got)" '[ "$got" = "E12d-P0 EVAL-formal EVAL-pad1 EVAL-wait | EVAL-pad1 EVAL-wait | " ]'
rm -rf $T
# 3h. [v2.5.8] upload_sync ends for a paused / finally failed task that no longer runs (nothing pending); a paused task
#     still running (pause does not stop it) keeps its loop
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/venv/bin $T/root/runs/A $T/root/runs/B $T/root/runs/C $T/x/scripts
cp $SRC/scripts/upload_sync.sh $T/rb/scripts/; : > $T/root/venv/bin/activate
printf 'export OPD_ROOT=%s RUNBOOK=%s\n' $T/root $T/rb > $T/rb/scripts/env.sh
touch $T/root/runs/A/.paused $T/root/runs/C/.paused; printf 'attempts: 3\n' > $T/root/runs/B/.failed
printf 'sleep 30\n' > $T/x/scripts/task.sh; bash $T/x/scripts/task.sh 4 C & P1=$!
for t in A B C; do ( UPLOAD_DRY=1 bash $T/rb/scripts/upload_sync.sh $T/root/runs/$t > $T/up_$t.log 2>&1 & ); done; sleep 8
ok "[v2.5.8] upload_sync stops for paused A / failed B, keeps C (paused but its task.sh runs)" 'grep -q "UPLOAD_STOPPED A paused" $T/up_A.log && grep -q "UPLOAD_STOPPED B failed" $T/up_B.log && ! grep -q UPLOAD_STOPPED $T/up_C.log && grep -q "UPLOAD_SETTINGS .* C UPLOAD_WEIGHTS=all" $T/up_C.log'
pkill -f "upload_sync.sh $T/root/runs/C" 2> /dev/null; kill $P1 2> /dev/null; wait $P1 2> /dev/null
rm -rf $T
# 3h2. [v2.5.9g] upload_sync: an E12 run's rollout_stats.csv while it trains (Q-034): settled dumps only, uploaded at 10 new
#      rows; at the end complete (the fresh last step too); the steps come from the per-step cache (no tokenizer is loaded)
T=$(mktemp -d); R=$T/root/runs/E12b-T; mkdir -p $T/rb/scripts $T/rb/code/analysis $T/root/venv/bin $R/rollouts $T/bin
cp $SRC/scripts/upload_sync.sh $T/rb/scripts/; cp $SRC/code/analysis/e12_rollout_stats.py $T/rb/code/analysis/; : > $T/root/venv/bin/activate
printf 'export OPD_ROOT=%s RUNBOOK=%s STUDENT_17B=%s/no-tokenizer\n' $T/root $T/rb $T > $T/rb/scripts/env.sh
printf '#!/usr/bin/env bash\nexec python3 "$@"\n' > $T/bin/python; chmod +x $T/bin/python
command -v timeout > /dev/null || { printf '#!/usr/bin/env bash\nshift; exec "$@"\n' > $T/bin/timeout; chmod +x $T/bin/timeout; }
python3 - $R <<'PY'
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[1]), "..", "..", "rb", "code", "analysis"))
import e12_rollout_stats as E
run = sys.argv[1]
for step in range(1, 14):
    p = os.path.join(run, "rollouts", f"{step}.jsonl")
    open(p, "w").write("".join(json.dumps({"input": "a", "output": "x", "score": s}) + "\n" for s in (1.0, 0.0)))
    if step < 13:
        os.utime(p, (time.time() - 600, time.time() - 600))
E.collect(run, lambda: (lambda t: len(t.split())))   # the cache of every step, 13 (being written) included
PY
printf 'trainer: x\n' > $R/config.yaml
( PATH=$T/bin:$PATH UPLOAD_DRY=1 bash $T/rb/scripts/upload_sync.sh $R > $T/up.log 2>&1 & ); sleep 3
rows1=$(awk 'NR > 1' $R/rollout_stats.csv 2> /dev/null | grep -c .); up1=$(cat $R/.stats_uploaded 2> /dev/null)
echo "finished: now" >> $R/config.yaml; sleep 7
ok "[v2.5.9g] upload_sync: E12 stats while training (settled s1-s12, uploaded at >= 10 rows), complete at the end (13 rows), no tokenizer" '[ "$rows1" = 12 ] && [ "$up1" = 12 ] && grep -q "STATS_UP E12b-T 12 steps" $T/up.log && [ "$(awk "NR > 1" $R/rollout_stats.csv | grep -c .)" = 13 ] && grep -q "UPLOAD_DONE E12b-T" $T/up.log && ! grep -q STATS_ERR $T/up.log'
pkill -f "upload_sync.sh $R" 2> /dev/null; rm -rf $T
# 3h3. [v2.5.9i] net2_unfreeze: a pull a killed PROBE-NET2 left stopped is resumed, but not while probe_net2.sh runs (pgrep is a
#      stub reading $T/probe_running, so no timing; the stopped process is a real one this test starts)
T=$(mktemp -d); mkdir -p $T/root/runs/PROBE-NET2 $T/rb/scripts $T/bin; printf 'import sys\n' > $T/rb/scripts/control.py
printf '#!/usr/bin/env bash\n[ "$*" = "-f scripts/probe_net2\\\\.sh" ] && [ -f %s/probe_running ]\n' $T > $T/bin/pgrep; chmod +x $T/bin/pgrep
sleep 300 & SP=$!; kill -STOP $SP; echo $SP > $T/root/runs/PROBE-NET2/paused_pids
uf() { ( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb; eval "$(sed -n '/^net2_unfreeze() {/,/; }$/p' $SRC/scripts/run_tasks.sh)"; net2_unfreeze ); }
touch $T/probe_running; uf; s1=$(ps -o stat= -p $SP | cut -c1)
rm -f $T/probe_running; uf; for _i in $(seq 30); do s2=$(ps -o stat= -p $SP | cut -c1); [ "$s2" != T ] && break; sleep 0.1; done
left=$(cat $T/root/runs/PROBE-NET2/paused_pids)
ok "[v2.5.9i] net2_unfreeze: left alone while probe_net2.sh runs, resumed after; paused_pids emptied (s1=$s1 s2=$s2 left=$left)" '[ "$s1" = T ] && [ "$s2" != T ] && [ -z "$left" ]'
kill -CONT $SP 2> /dev/null; kill $SP 2> /dev/null; wait $SP 2> /dev/null; rm -rf $T   # only the process this test started
# 3i. [v2.5.8] no_fetch(): an EVAL task with only Qwen/ and local: models needs no prefetch; one with a ckpts or
#     result: model does
T=$(mktemp -d); printf 'version: 2\n#eval EVAL-a a local:runs/X/ckpt/global_step_1/actor/huggingface math500 4 0.6\n#eval EVAL-a b Qwen/Qwen3-1.7B math500 4 0.6\n#eval EVAL-b c yang31210999/opd-lab-ckpts/runs/Y/global_step_1/actor/huggingface math500 4 0.6\n#eval EVAL-c d result:runs/Z/global_step_1/actor/huggingface math500 4 0.6\n' > $T/q.txt
got=$( Q=$T/q.txt; eval "$(sed -n '/^no_fetch() {/p' $SRC/scripts/run_tasks.sh)"; for t in EVAL-a EVAL-b EVAL-c EVAL-none; do no_fetch $t && printf "%s=1 " $t || printf "%s=0 " $t; done )
ok "[v2.5.8] no_fetch: local:/Qwen only -> ready; ckpts / result: / no lines -> not" '[ "$got" = "EVAL-a=1 EVAL-b=0 EVAL-c=0 EVAL-none=0 " ]'
ok "[v2.5.8] the dispatch takes no_fetch tasks" 'grep -q "inputs_ok \$t || result_only \$t || no_fetch \$t || continue" $SRC/scripts/run_tasks.sh'
rm -rf $T
# 3j. [v2.5.8] the CPU lane: STRICT-C1 is a CPU task; a running "task.sh cpu <task>" counts as running and fills the
#     lane (CPU_MAX 1); the dispatch launches it with gpu cpu before any GPU is taken, and launch skips the GPU reap
T=$(mktemp -d); mkdir -p $T/x/scripts; printf 'sleep 30\n' > $T/x/scripts/task.sh
got=$( eval "$(sed -n '/^running() {/p;/^cpu_task() {/p;/^cpu_busy() {/p' $SRC/scripts/run_tasks.sh)"
  cpu_task STRICT-C1 && printf "c1 "; cpu_task EVAL-x || printf "notcpu "; cpu_busy || printf "free "
  bash $T/x/scripts/task.sh cpu STRICT-C1 & P=$!; sleep 0.5
  running STRICT-C1 && printf "running "; cpu_busy && printf "busy "; kill $P 2> /dev/null; wait $P 2> /dev/null )
ok "[v2.5.8] CPU lane: STRICT-C1 cpu task, running + busy while its task.sh cpu runs" '[ "$got" = "c1 notcpu free running busy " ]'
ok "[v2.5.8] dispatch: cpu tasks launch with gpu cpu before the GPU check; launch skips the reap for cpu" 'grep -q "launch cpu \$t" $SRC/scripts/run_tasks.sh && grep -q "\[ \$1 = cpu \] || for _g in .*control.py reap" $SRC/scripts/run_tasks.sh && [ $(grep -n "launch cpu \$t" $SRC/scripts/run_tasks.sh | cut -d: -f1) -lt $(grep -n "take_gpus \$t || continue; g=\$G" $SRC/scripts/run_tasks.sh | head -1 | cut -d: -f1) ]'
rm -rf $T
# 3k. [v2.5.8] gpu_wait_free: returns once the GPU's memory falls to <= 2048 MiB; times out (1, GPU_NOT_FREE) if it stays
#     up; the eval scripts call it before every engine
T=$(mktemp -d); mkdir -p $T/bin; echo 0 > $T/n
printf '#!/usr/bin/env bash\nn=$(cat %s); echo $((n + 1)) > %s; [ $n -ge 2 ] && [ "$STICK" != 1 ] && echo 100 || echo 70000\n' $T/n $T/n > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
got=$( PATH=$T/bin:$PATH; eval "$(sed -n '/^gpu_wait_free() {/,/; }$/p' $SRC/scripts/env.sh)"; gpu_wait_free 3 10 > /dev/null && printf "freed "
  echo 0 > $T/n; STICK=1 gpu_wait_free 3 2 | grep -q GPU_NOT_FREE && printf "timeout " )
ok "[v2.5.8] gpu_wait_free: waits until free, times out when it stays full" '[ "$got" = "freed timeout " ]'
ok "[v2.5.8] gpu_wait_free before every engine (gen_eval, forced_stop, E11b)" '[ "$(grep -c "gpu_wait_free \$G" $SRC/scripts/run_queue_eval.sh)" -ge 2 ] && grep -q "gpu_wait_free \$G" $SRC/scripts/run_eval.sh'
rm -rf $T
# 3l. [v2.5.8] ensure_loop: starts a missing watchdog / uploader; leaves a current one alone; restarts one whose script
#     changed since it started (control/<name>.sha)
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/control $T/root/logs $T/bin
printf 'echo $$ >> %s/starts\nwhile true; do sleep 1; done\n' $T > $T/rb/scripts/uploader.sh
printf 'import sys\n' > $T/rb/scripts/control.py
printf '#!/usr/bin/env bash\nexec python3 -c "import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" "$@"\n' > $T/bin/setsid
printf '#!/usr/bin/env bash\nexec python3 "$@"\n' > $T/bin/python; chmod +x $T/bin/setsid $T/bin/python
command -v sha1sum > /dev/null || printf '#!/usr/bin/env bash\nshasum "$@"\n' > $T/bin/sha1sum; chmod +x $T/bin/sha1sum 2> /dev/null
( PATH=$T/bin:$PATH RUNBOOK=$T/rb OPD_ROOT=$T/root CDIR=$T/root/control; eval "$(sed -n '/^ensure_loop() {/,/^  (cd \$RUNBOOK && setsid nohup bash scripts\/\$1.sh .*; }$/p' $SRC/scripts/run_tasks.sh)"
  ensure_loop uploader; sleep 1; ensure_loop uploader; sleep 1; echo "# changed" >> $T/rb/scripts/uploader.sh; ensure_loop uploader; sleep 2 )
ok "[v2.5.8] ensure_loop: started once, left alone, restarted after its script changed" '[ "$(grep -c . $T/starts)" = 2 ] && ! kill -0 $(head -1 $T/starts) 2> /dev/null && kill -0 $(tail -1 $T/starts) 2> /dev/null'
for p in $(cat $T/starts); do kill $p 2> /dev/null; done; rm -rf $T   # only the pids this test started
# 4. v2.5.4d free_gpus(): a GPU whose task.sh (or ad-hoc evaluation) runs is not free, even with nothing on it yet
T=$(mktemp -d); mkdir -p $T/bin $T/x/scripts
printf '#!/usr/bin/env bash\ncase "$*" in *compute-apps*) ;; *) echo 0;; esac\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
printf 'sleep 30\n' > $T/x/scripts/task.sh; printf 'sleep 30\n' > $T/x/scripts/run_adhoc_eval.sh
bash $T/x/scripts/task.sh 5 FA-PROBE & P1=$!; bash $T/x/scripts/run_adhoc_eval.sh 13 41 & P2=$!; sleep 0.5
got=$( PATH=$T/bin:$PATH NGPU=14; eval "$(sed -n '/^free_gpus() {/,/; }$/p' $SRC/scripts/run_tasks.sh)"; free_gpus | tr '\n' ' ' )
kill $P1 $P2 2> /dev/null
ok "running tasks keep their GPUs" '[[ " $got" == *" 1 "* && " $got" == *" 6 "* && " $got" != *" 5 "* && " $got" != *" 13 "* ]]'   # 13 must not hide 1
rm -rf $T
# [v2.5.9a] #gpus: a task on a GPU list keeps every GPU of it; take_gpus gives n GPUs, holds the free ones in turns of
# GPUS_HOLD_MIN, gives one GPU otherwise (an invalid n counts as 1) and clears its wait when it starts
T=$(mktemp -d); mkdir -p $T/bin $T/x/scripts $T/root/runs
printf '#!/usr/bin/env bash\ncase "$*" in *compute-apps*) ;; *) echo 0;; esac\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
printf 'sleep 30\n' > $T/x/scripts/task.sh
bash $T/x/scripts/task.sh 1,2 E12b-P5-16k & P1=$!; sleep 0.5
got=$( PATH=$T/bin:$PATH NGPU=4; eval "$(sed -n '/^free_gpus() {/,/; }$/p' $SRC/scripts/run_tasks.sh)"; free_gpus | tr '\n' ' ' )
kill $P1 2> /dev/null
ok "[v2.5.9a] a task on gpus 1,2 keeps both" '[ "$got" = "0 3 " ]'
printf '#gpus A 2\n#gpus B 3\n#gpus C 9\n' > $T/q; mkdir -p $T/root/logs $T/root/control
echo '{"tasks": {"X": {"state": "running", "eta_min": 12}}}' > $T/root/logs/status_n.json   # a GPU frees within 30 min
tg=$( Q=$T/q OPD_ROOT=$T/root; eval "$(sed -n '/^gpus_of() {/p' $SRC/scripts/run_tasks.sh)"; eval "$(sed -n '/^soon_free() {/,/^}$/p' $SRC/scripts/run_tasks.sh)"
  eval "$(sed -n '/^take_gpus() {/,/; }$/p' $SRC/scripts/run_tasks.sh)"
  FREE=(3 5 6); take_gpus A; printf "a=$G:${FREE[*]} "
  FREE=(4 7); take_gpus C; printf "c=$G:${FREE[*]} "
  FREE=(4); take_gpus A > /dev/null; printf "hold=$?:${#FREE[@]} "; [ -f $T/root/runs/A/.gpus_wait ] && printf "wait "
  python3 -c "import json, sys; h = json.load(open(sys.argv[1])); print('held=%s:%s' % (h['task'], h['gpus']), end=' ')" $T/root/control/held.json
  echo $(( $(date +%s) - 31 * 60 )) > $T/root/runs/A/.gpus_wait; FREE=(4); take_gpus A; printf "release=$?:${#FREE[@]} "
  echo $(( $(date +%s) - 61 * 60 )) > $T/root/runs/A/.gpus_wait; FREE=(4); take_gpus A; printf "again=${#FREE[@]} "
  echo '{"tasks": {"X": {"state": "running", "eta_min": null}, "Y": {"state": "done", "eta_min": 5}}}' > $T/root/logs/status_n.json
  FREE=(4); take_gpus A; printf "noeta=${#FREE[@]} "
  FREE=(4 9); take_gpus A; printf "go=$G "; [ -f $T/root/runs/A/.gpus_wait ] || printf "cleared "
  FREE=(); take_gpus B; printf "none=$?" )
ok "[v2.5.9a/c] take_gpus: n GPUs, n invalid = 1; too few: held (held.json) while a GPU frees within 30 min, released in turns, not held without an ETA; the wait cleared on start" '[ "$tg" = "a=3,5:6 c=4:7 hold=1:0 wait held=A:[4] release=1:1 again=0 noeta=1 go=4,9 cleared none=1" ]'
rm -rf $T
# v2.5.7j: no task script deletes a final bf16 copy (Runming: final weights keep a local copy after upload)
ok "final bf16 copies are never deleted by run_e12.sh / run_arm.sh / run_eval.sh" '! grep -nE "rm -rf[^#]*huggingface_bf16" $SRC/scripts/run_e12.sh $SRC/scripts/run_arm.sh $SRC/scripts/run_eval.sh'
[ $fails = 0 ] && echo "V254A_TEST PASS ($n checks)" || echo "V254A_TEST FAIL ($fails of $n)"
