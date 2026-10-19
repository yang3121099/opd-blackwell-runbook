#!/usr/bin/env bash
# v2.5.4a: queue task EVAL-<name>, evaluations that take a GPU in queue order (an `eval` request only gets the GPUs the
# queue leaves free). Its evaluations are comment lines of tasks/queue.txt (so they need no other file):
#   #eval EVAL-<name> <tag> <model> <bench> <n> <temp>
# <tag> names the output files (e.g. the lab's e12_P5c05_s40_math500_T0.6), <model>/<bench>/<n>/<temp> are checked with the
# rules of the `eval` command (control_rules.check_eval_args). They run one after the other on this GPU with the settings
# of run_adhoc_eval.sh (no_think, T=0.6: top_p 0.95 / T=1.0: top_p 1.0, top_k 0, 16K, resumable blocks); results in
# runs/EVAL-<name>/ locally and in the public result repository; .done once every listed evaluation has its metrics.
# v2.5.7g (reviewer 18:4x UTC, budget-dependence readings A): an optional 7th field = the generation budget max_tokens
# (2048 / 4096 / 8192 / 16384 / 32768; default 16384, as before):  #eval EVAL-<name> <tag> <model> <bench> <n> <temp> [<max_tokens>]
# v2.5.7h (budget readings B, lab Q-013): an optional 8th field = the mode, think or no_think (default no_think, as before)
# v2.5.4e: models from yang31210999/opd-lab-ckpts come through fetch_model.py (complete by size and sha256, or not used);
# `prefetch` mode fetches every model of the task without a GPU and writes runs/EVAL-<name>/.inputs_ok (the hash of the
# task's #eval lines); run_tasks.sh starts prefetches for all EVAL tasks of the queue and gives a GPU only after .inputs_ok.
# Usage: bash run_queue_eval.sh <gpu> <EVAL-name>  |  bash run_queue_eval.sh prefetch <EVAL-name>
set -uo pipefail
source "$(dirname "$0")/env.sh"; source ${VENV_EVAL:-$OPD_ROOT/venv}/bin/activate
G=$1; T=$2; OUT=$OPD_ROOT/runs/$T; Q=${QUEUE_FILE:-$RUNBOOK/tasks/queue.txt}; mkdir -p $OUT
[ -f $OUT/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
SPECS=$(awk -v t="$T" '$1 == "#eval" && $2 == t {print $3, $4, $5, $6, $7, $8, $9}' "$Q")
[ -n "$SPECS" ] || { echo "no #eval lines for $T in the queue"; exit 1; }
HASH=$(awk -v t="$T" '$1 == "#eval" && $2 == t' "$Q" | cksum | cut -d' ' -f1)
fetch_ckpt() {   # $1 = path inside yang31210999/opd-lab-ckpts; 0 when complete
  python $RUNBOOK/scripts/fetch_model.py yang31210999/opd-lab-ckpts "$1" "$CKPT_ROOT/adhoc_models" --task $T >> $OUT/fetch.log 2>&1; }
fetch_result() {   # v2.5.6h: $1 = path inside this node's public result repository (result:runs/<task>/global_step_N/...;
  # E12b keeps no local weights after its bf16 upload, so its readings read them back from there). The task can start
  # as soon as the training has ended: the bf16 copy reaches the repository a few minutes later, so wait up to 20 min.
  local i; for i in $(seq 1 ${RESULT_WAIT_MIN:-20}); do
    python $RUNBOOK/scripts/fetch_model.py "$(python $RUNBOOK/scripts/remote.py repo public)" "$1" "$CKPT_ROOT/adhoc_models/result" --task $T >> $OUT/fetch.log 2>&1 && return 0
    [ "$G" = prefetch ] && return 1; sleep ${RESULT_WAIT_STEP:-60}; done; return 1; }
local_save() {   # v2.5.7zd (reviewer 15:2x UTC, 10-19) $1 = runs/<task>/global_step_N/actor/huggingface (a result: path) -> the
  # training's own save on this node, if it has weights: its bf16 copy, else its fp32 HF save (read in bf16); "" if none.
  # result: downloads stalled three times on 10-19 (P5self 10:39, EVAL-e12d-P0 14:49, EVAL-e12b-P2-s43 15:04) with
  # the GPU idle; the local save is the same weights (upload_sync's bf16 copy is this save cast to bf16)
  [[ $1 =~ ^runs/([A-Za-z0-9._-]+)/global_step_([0-9]+)/actor/huggingface$ ]] || return 0
  local r b
  for r in $OPD_ROOT $CKPT_ROOT; do
    b=$r/runs/${BASH_REMATCH[1]}/ckpt/global_step_${BASH_REMATCH[2]}/actor
    has_weights $b/huggingface_bf16 && { echo "$b/huggingface_bf16 auto"; return 0; }
    has_weights $b/huggingface && { echo "$b/huggingface bfloat16"; return 0; }
  done; }
resolve_model() {   # $1 = a model as on an #eval line -> M (fetched if needed); 1 = the fetch failed, 2 = no weights
  DT=auto   # v2.5.7q: local: = a training's own HF save (fp32 for E12b / E10b steps), read in bf16 as its upload would be
  case $1 in
    Qwen/Qwen3-0.6B) M=$STUDENT_06B;; Qwen/Qwen3-1.7B) M=$STUDENT_17B;; Qwen/Qwen3-8B) M=$TEACHER;; Qwen/*) M=$1;;
    yang31210999/opd-lab-ckpts/*) M=$CKPT_ROOT/adhoc_models/${1#yang31210999/opd-lab-ckpts/}; fetch_ckpt "${1#yang31210999/opd-lab-ckpts/}" || return 1;;
    result:*) L=$(local_save "${1#result:}")   # v2.5.7zd: local first, the result repository as the fallback
      if [ -n "$L" ]; then M=${L% *}; DT=${L##* }; echo "LOCAL_FIRST $1 -> $M ($DT)"
      else M=$CKPT_ROOT/adhoc_models/result/${1#result:}; fetch_result "${1#result:}" || return 1; fi;;
    *) M=$OPD_ROOT/${1#local:}; has_weights $M || M=$CKPT_ROOT/${1#local:}; DT=bfloat16;;
  esac
  has_weights $M || [ "${M#Qwen/}" != "$M" ] || [ -f $M/config.json ] || return 2
}
ref_path() {   # a reading: a tag of this task -> its jsonl, or <task dir>/<file>.jsonl under $OPD_ROOT/runs; "" if refused
  if [[ $1 =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.jsonl$ ]]; then echo "$OPD_ROOT/runs/$1"
  elif [[ $1 =~ ^[A-Za-z0-9._-]+$ ]]; then F=$(ls $OUT/${1}_*.jsonl 2>/dev/null | grep -v '\.forced\.jsonl$' | head -1); echo "${F:-$OUT/$1.missing.jsonl}"; fi
}
if [ "$G" = prefetch ]; then
  pok=1
  while read -r TAG MODEL BENCH N TEMP MAXT MODE; do
    case $MODEL in yang31210999/opd-lab-ckpts/*) fetch_ckpt "${MODEL#yang31210999/opd-lab-ckpts/}" || { echo "PREFETCH_FAILED $TAG $MODEL"; pok=0; };;
      result:*) [ -n "$(local_save "${MODEL#result:}")" ] && { echo "PREFETCH_LOCAL $TAG $MODEL"; continue; }   # v2.5.7zd
        fetch_result "${MODEL#result:}" || { echo "PREFETCH_FAILED $TAG $MODEL"; pok=0; };; esac
  done <<< "$SPECS"
  [ $pok = 1 ] && { echo $HASH > $OUT/.inputs_ok; rm -f $OUT/.prefetch_fail; echo "PREFETCH_DONE $T"; exit 0; }
  rm -f $OUT/.inputs_ok; exit 1
fi
ok=1; touch $OUT/.models   # v2.5.6i: "<tag> <resolved model>" per reading, for a #twr line (kept across attempts)
while read -r TAG MODEL BENCH N TEMP MAXT MODE; do
  MAXT=${MAXT:-16384}; MODE=${MODE:-no_think}
  [[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "REFUSED bad tag $TAG"; ok=0; continue; }
  ls $OUT/${TAG}_*.metrics.json > /dev/null 2>&1 && { echo "DONE_ALREADY $TAG"; continue; }
  python -c "import sys; sys.path.insert(0, sys.argv[1]); import control_rules as R; R.check_eval_args(sys.argv[2:8])" \
    "$RUNBOOK/scripts" "$MODEL" "$BENCH" "$N" "$TEMP" "$MAXT" "$MODE" || { echo "REFUSED $TAG: $MODEL $BENCH $N $TEMP $MAXT $MODE"; ok=0; continue; }
  resolve_model "$MODEL"; rc=$?
  [ $rc = 1 ] && { echo "MODEL_FETCH_FAILED $TAG: $MODEL (see $OUT/fetch.log)"; ok=0; continue; }
  [ $rc = 2 ] && { echo "MODEL_MISSING $TAG: $MODEL"; ok=0; continue; }
  case $TEMP in 1.0) SAMP="--temperature 1.0 --top_p 1.0 --top_k 0";; *) SAMP="--temperature 0.6 --top_p 0.95 --top_k 0";; esac
  case $BENCH in aime24|aime25|amc23) K=10;; *) K=50;; esac
  echo "EVAL_START $(date -u +%T) $TAG"
  CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/eval/gen_eval.py --model $M --bench $BENCH --mode $MODE --n $N $SAMP \
    --max_tokens $MAXT --chunk $K --dtype $DT --tag $TAG --out $OUT >> $OUT/gen_eval.log 2>&1 || { echo "EVAL_FAIL $TAG"; ok=0; python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1; continue; }
  echo "EVAL_DONE $(date -u +%T) $TAG"; echo "$TAG $M" >> $OUT/.models
  python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true   # v2.5.4b: release the GPU before the next engine
done <<< "$SPECS"
# v2.5.6i: "#twr EVAL-<name> <tag>" = TWR / alpha of that reading, as E11b does it (run_eval.sh): judge the base 1.7B,
# reference E11b-P0's own T=0.6 generations, 4 per problem; a failure is logged and does not hold back the readings
TWR_TAG=$(awk -v t="$T" '$1 == "#twr" && $2 == t {print $3}' "$Q" | head -1)
if [ -n "$TWR_TAG" ] && [ ! -f $OUT/twr.json ]; then
  REF=$(ls $OPD_ROOT/runs/E11b-P0/e11b_P0_math500_T0.6_*.jsonl 2>/dev/null | head -1); SC=$(ls $OUT/${TWR_TAG}_*.jsonl 2>/dev/null | grep -v '\.forced\.jsonl$' | head -1)
  M=$(awk -v t="$TWR_TAG" '$1 == t {print $2}' $OUT/.models | tail -1)
  if [ -n "$REF" ] && [ -n "$SC" ] && [ -n "$M" ]; then
    CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/leakage/twr_alpha.py --judge $STUDENT_17B --bench math500 --ref $REF --score $SC \
      --model_k $M --per_problem 4 --out $OUT/twr.json >> $OUT/twr.log 2>&1 && echo "TWR_DONE $TWR_TAG" || echo "TWR_FAIL $TWR_TAG (see twr.log)"
    python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
  else echo "TWR_SKIPPED $TWR_TAG (ref ${REF:-missing}, score ${SC:-missing}, model ${M:-missing})"; fi
fi
# v2.5.7h (budget readings B, lab Q-013): "#forced EVAL-<name> <reading> <model> <think|no_think>" = forced answers of that
# reading at 2K / 4K / 8K / 16K / 32K (up to its own max_tokens): every sample not finished within B is cut at B, the
# stop string appended ("\n\n**Final Answer**\n\\boxed{", thinking: "\n</think>\n\n\\boxed{"), at most 32 more tokens
# (code/eval/forced_stop.py --force_preset budB, E05's mechanism); <model> is the model that wrote the reading (as on an
# #eval line); out: <reading stem>.forced.json / .forced.jsonl in this task, which #budget then reads
while read -r FR FMODEL FMODE; do
  [ -n "$FR" ] || continue
  RP=$(ref_path "$FR"); [ -n "$RP" ] || { echo "FORCED_REF_REFUSED $FR"; continue; }
  [ -f "$RP" ] && [ -f "${RP%.jsonl}.metrics.json" ] || { echo "FORCED_SKIPPED $FR (no finished reading)"; ok=0; continue; }
  python -c "import sys; sys.path.insert(0, sys.argv[1]); import control_rules as R; R.check_eval_args([sys.argv[2], 'math500', '1', '0.6', '16384', sys.argv[3]])" \
    "$RUNBOOK/scripts" "$FMODEL" "$FMODE" || { echo "FORCED_REFUSED $FR: $FMODEL $FMODE"; ok=0; continue; }
  STEM=$(basename "${RP%.jsonl}")
  [ -f $OUT/$STEM.forced.json ] && { echo "FORCED_DONE_ALREADY $STEM"; continue; }
  resolve_model "$FMODEL" || { echo "FORCED_MODEL_MISSING $FR: $FMODEL"; ok=0; continue; }
  CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/eval/forced_stop.py --model $M --jsonl "$RP" --bench math500 --mode $FMODE --force_preset budB \
    --max_new 32 --budgets 2048,4096,8192,16384,32768 --out $OUT/$STEM.forced.json --samples $OUT/$STEM.forced.jsonl >> $OUT/forced.log 2>&1 \
    && echo "FORCED_DONE $STEM" || { echo "FORCED_FAIL $STEM (see forced.log)"; ok=0; rm -f $OUT/$STEM.forced.json; }
  python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
done <<< "$(awk -v t="$T" '$1 == "#forced" && $2 == t {print $3, $4, $5}' "$Q")"
# v2.5.6r (reviewer A1 06:3x UTC): "#pair EVAL-<name> <tag> <ref> [<ref> ...]" = problem-level paired bootstrap of that
# reading against each reference (code/analysis/paired_bootstrap.py, the lab's method: per-problem accuracy, 10000
# resamples, seed 0, 95% percentile CI, wins / losses); a ref is <task dir>/<file>.jsonl under $OPD_ROOT/runs; the result
# is <tag>.paired.json next to the reading; CPU only; a missing reference is listed, the readings are not affected
while read -r PTAG PREFS; do
  [ -n "$PTAG" ] || continue
  SC=$(ls $OUT/${PTAG}_*.jsonl 2>/dev/null | grep -v '\.forced\.jsonl$' | head -1)   # v2.5.7h: never a forced-answer file
  [ -n "$SC" ] || { echo "PAIR_SKIPPED $PTAG (no reading)"; continue; }
  RP=(); for R in $PREFS; do
    if [[ $R =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.jsonl$ ]]; then RP+=("$OPD_ROOT/runs/$R")
    # v2.5.7r: archive:<path>.jsonl = a lab reading in yang31210999/opd-lab-ckpts (fetch_model.py: size + sha256), e.g. P5 s50
    elif [[ $R =~ ^archive:[A-Za-z0-9._/-]+\.jsonl$ && $R != *..* ]]; then P=${R#archive:}; fetch_ckpt "$P" && RP+=("$CKPT_ROOT/adhoc_models/$P") || echo "PAIR_REF_MISSING $R"
    else echo "PAIR_REF_REFUSED $R"; fi; done
  python3 $RUNBOOK/code/analysis/paired_bootstrap.py --pairs $OUT/${PTAG}.paired.json "$SC" "${RP[@]}" > /dev/null 2>> $OUT/pair.log \
    && echo "PAIR_DONE $PTAG (${#RP[@]} refs)" || echo "PAIR_FAIL $PTAG (see pair.log)"
done <<< "$(awk -v t="$T" '$1 == "#pair" && $2 == t {$1 = ""; $2 = ""; sub(/^  /, ""); print}' "$Q")"
# v2.5.7g (reviewer 18:4x UTC, readings A): "#budget EVAL-<name> <out> <base> <reading> [<reading> ...]" = accuracy and
# truncation rate of each reading at the budgets 2K / 4K / 8K / 16K / 32K, read off one long generation (a sample counts
# at budget B as it would have been graded had it been cut at B tokens: unchanged if it ended within B, else its first B
# tokens graded again), every reading paired against <base> at every budget (problem-level bootstrap), and the crossing
# budget (the smallest B where reading - base turns positive); code/analysis/budget_curve.py, CPU only, <out>.json
# (reviewer A1 19:2x UTC: <out> = budget_curve, i.e. runs/EVAL-bud*/budget_curve.json; the table in <out>.txt).
# A reading is a tag of this task or <task dir>/<file>.jsonl under $OPD_ROOT/runs; a missing one is listed.
while read -r BOUT BREFS; do
  [ -n "$BOUT" ] || continue
  [[ $BOUT =~ ^[A-Za-z0-9._-]+$ ]] || { echo "BUDGET_REFUSED bad name $BOUT"; continue; }
  BP=(); for R in $BREFS; do F=$(ref_path "$R"); [ -n "$F" ] && BP+=("$F") || echo "BUDGET_REF_REFUSED $R"; done
  python $RUNBOOK/code/analysis/budget_curve.py --out $OUT/$BOUT.json --tokenizer $STUDENT_17B --forced_dir $OUT "${BP[@]}" > $OUT/$BOUT.txt 2>> $OUT/budget.log \
    && echo "BUDGET_DONE $BOUT (${#BP[@]} readings)" || echo "BUDGET_FAIL $BOUT (see budget.log)"
done <<< "$(awk -v t="$T" '$1 == "#budget" && $2 == t {$1 = ""; $2 = ""; sub(/^  /, ""); print}' "$Q")"
python $RUNBOOK/scripts/remote.py put $OUT runs/$T || { echo "upload failed (results kept in $OUT)"; ok=0; }
[ $ok = 1 ] && touch $OUT/.done
echo "TASK_FINISHED $T ok=$ok"
