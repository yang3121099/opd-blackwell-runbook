#!/usr/bin/env bash
# v2.5.4a: queue task EVAL-<name>, evaluations that take a GPU in queue order (an `eval` request only gets the GPUs the
# queue leaves free). Its evaluations are comment lines of tasks/queue.txt (so they need no other file):
#   #eval EVAL-<name> <tag> <model> <bench> <n> <temp>
# <tag> names the output files (e.g. the lab's e12_P5c05_s40_math500_T0.6), <model>/<bench>/<n>/<temp> are checked with the
# rules of the `eval` command (control_rules.check_eval_args). They run one after the other on this GPU with the settings
# of run_adhoc_eval.sh (no_think, T=0.6: top_p 0.95 / T=1.0: top_p 1.0, top_k 0, 16K, resumable blocks); results in
# runs/EVAL-<name>/ locally and in the public result repository; .done once every listed evaluation has its metrics.
# v2.5.4e: models from yang31210999/opd-lab-ckpts come through fetch_model.py (complete by size and sha256, or not used);
# `prefetch` mode fetches every model of the task without a GPU and writes runs/EVAL-<name>/.inputs_ok (the hash of the
# task's #eval lines); run_tasks.sh starts prefetches for all EVAL tasks of the queue and gives a GPU only after .inputs_ok.
# Usage: bash run_queue_eval.sh <gpu> <EVAL-name>  |  bash run_queue_eval.sh prefetch <EVAL-name>
set -uo pipefail
source "$(dirname "$0")/env.sh"; source ${VENV_EVAL:-$OPD_ROOT/venv}/bin/activate
G=$1; T=$2; OUT=$OPD_ROOT/runs/$T; Q=${QUEUE_FILE:-$RUNBOOK/tasks/queue.txt}; mkdir -p $OUT
[ -f $OUT/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
SPECS=$(awk -v t="$T" '$1 == "#eval" && $2 == t {print $3, $4, $5, $6, $7}' "$Q")
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
if [ "$G" = prefetch ]; then
  pok=1
  while read -r TAG MODEL BENCH N TEMP; do
    case $MODEL in yang31210999/opd-lab-ckpts/*) fetch_ckpt "${MODEL#yang31210999/opd-lab-ckpts/}" || { echo "PREFETCH_FAILED $TAG $MODEL"; pok=0; };;
      result:*) fetch_result "${MODEL#result:}" || { echo "PREFETCH_FAILED $TAG $MODEL"; pok=0; };; esac
  done <<< "$SPECS"
  [ $pok = 1 ] && { echo $HASH > $OUT/.inputs_ok; rm -f $OUT/.prefetch_fail; echo "PREFETCH_DONE $T"; exit 0; }
  rm -f $OUT/.inputs_ok; exit 1
fi
ok=1; touch $OUT/.models   # v2.5.6i: "<tag> <resolved model>" per reading, for a #twr line (kept across attempts)
while read -r TAG MODEL BENCH N TEMP; do
  [[ $TAG =~ ^[A-Za-z0-9._-]+$ ]] || { echo "REFUSED bad tag $TAG"; ok=0; continue; }
  ls $OUT/${TAG}_*.metrics.json > /dev/null 2>&1 && { echo "DONE_ALREADY $TAG"; continue; }
  python -c "import sys; sys.path.insert(0, sys.argv[1]); import control_rules as R; R.check_eval_args(sys.argv[2:6])" \
    "$RUNBOOK/scripts" "$MODEL" "$BENCH" "$N" "$TEMP" || { echo "REFUSED $TAG: $MODEL $BENCH $N $TEMP"; ok=0; continue; }
  case $MODEL in
    Qwen/Qwen3-0.6B) M=$STUDENT_06B;; Qwen/Qwen3-1.7B) M=$STUDENT_17B;; Qwen/Qwen3-8B) M=$TEACHER;; Qwen/*) M=$MODEL;;
    yang31210999/opd-lab-ckpts/*)
      P=${MODEL#yang31210999/opd-lab-ckpts/}; M=$CKPT_ROOT/adhoc_models/$P
      fetch_ckpt "$P" || { echo "MODEL_FETCH_FAILED $TAG: $MODEL (see $OUT/fetch.log)"; ok=0; continue; };;
    result:*)
      P=${MODEL#result:}; M=$CKPT_ROOT/adhoc_models/result/$P
      fetch_result "$P" || { echo "MODEL_FETCH_FAILED $TAG: $MODEL (see $OUT/fetch.log)"; ok=0; continue; };;
    *) M=$OPD_ROOT/${MODEL#local:}; has_weights $M || M=$CKPT_ROOT/${MODEL#local:};;
  esac
  has_weights $M || [ "${M#Qwen/}" != "$M" ] || [ -f $M/config.json ] || { echo "MODEL_MISSING $TAG: $MODEL"; ok=0; continue; }
  case $TEMP in 1.0) SAMP="--temperature 1.0 --top_p 1.0 --top_k 0";; *) SAMP="--temperature 0.6 --top_p 0.95 --top_k 0";; esac
  case $BENCH in aime24|aime25|amc23) K=10;; *) K=50;; esac
  echo "EVAL_START $(date -u +%T) $TAG"
  CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/eval/gen_eval.py --model $M --bench $BENCH --mode no_think --n $N $SAMP \
    --max_tokens 16384 --chunk $K --tag $TAG --out $OUT >> $OUT/gen_eval.log 2>&1 || { echo "EVAL_FAIL $TAG"; ok=0; python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1; continue; }
  echo "EVAL_DONE $(date -u +%T) $TAG"; echo "$TAG $M" >> $OUT/.models
  python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true   # v2.5.4b: release the GPU before the next engine
done <<< "$SPECS"
# v2.5.6i: "#twr EVAL-<name> <tag>" = TWR / alpha of that reading, as E11b does it (run_eval.sh): judge the base 1.7B,
# reference E11b-P0's own T=0.6 generations, 4 per problem; a failure is logged and does not hold back the readings
TWR_TAG=$(awk -v t="$T" '$1 == "#twr" && $2 == t {print $3}' "$Q" | head -1)
if [ -n "$TWR_TAG" ] && [ ! -f $OUT/twr.json ]; then
  REF=$(ls $OPD_ROOT/runs/E11b-P0/e11b_P0_math500_T0.6_*.jsonl 2>/dev/null | head -1); SC=$(ls $OUT/${TWR_TAG}_*.jsonl 2>/dev/null | head -1)
  M=$(awk -v t="$TWR_TAG" '$1 == t {print $2}' $OUT/.models | tail -1)
  if [ -n "$REF" ] && [ -n "$SC" ] && [ -n "$M" ]; then
    CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/leakage/twr_alpha.py --judge $STUDENT_17B --bench math500 --ref $REF --score $SC \
      --model_k $M --per_problem 4 --out $OUT/twr.json >> $OUT/twr.log 2>&1 && echo "TWR_DONE $TWR_TAG" || echo "TWR_FAIL $TWR_TAG (see twr.log)"
    python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
  else echo "TWR_SKIPPED $TWR_TAG (ref ${REF:-missing}, score ${SC:-missing}, model ${M:-missing})"; fi
fi
# v2.5.6r (reviewer A1 06:3x UTC): "#pair EVAL-<name> <tag> <ref> [<ref> ...]" = problem-level paired bootstrap of that
# reading against each reference (code/analysis/paired_bootstrap.py, the lab's method: per-problem accuracy, 10000
# resamples, seed 0, 95% percentile CI, wins / losses); a ref is <task dir>/<file>.jsonl under $OPD_ROOT/runs; the result
# is <tag>.paired.json next to the reading; CPU only; a missing reference is listed, the readings are not affected
while read -r PTAG PREFS; do
  [ -n "$PTAG" ] || continue
  SC=$(ls $OUT/${PTAG}_*.jsonl 2>/dev/null | head -1)
  [ -n "$SC" ] || { echo "PAIR_SKIPPED $PTAG (no reading)"; continue; }
  RP=(); for R in $PREFS; do [[ $R =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.jsonl$ ]] && RP+=("$OPD_ROOT/runs/$R") || echo "PAIR_REF_REFUSED $R"; done
  python3 $RUNBOOK/code/analysis/paired_bootstrap.py --pairs $OUT/${PTAG}.paired.json "$SC" "${RP[@]}" > /dev/null 2>> $OUT/pair.log \
    && echo "PAIR_DONE $PTAG (${#RP[@]} refs)" || echo "PAIR_FAIL $PTAG (see pair.log)"
done <<< "$(awk -v t="$T" '$1 == "#pair" && $2 == t {$1 = ""; $2 = ""; sub(/^  /, ""); print}' "$Q")"
python $RUNBOOK/scripts/remote.py put $OUT runs/$T || { echo "upload failed (results kept in $OUT)"; ok=0; }
[ $ok = 1 ] && touch $OUT/.done
echo "TASK_FINISHED $T ok=$ok"
