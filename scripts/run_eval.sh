#!/usr/bin/env bash
# E11b: 16K re-evaluation of one checkpoint on MATH500 (avg@4, T=0.6 / top-p 0.95 and T=1.0 / top-p 1.0, non-thinking),
# then TWR/alpha with the base Qwen3-1.7B as judge (reference = E11b-P0's own T=0.6 generations).
# E11b-P0 evaluates the base Qwen3-1.7B (the TWR reference). Outputs go to runs/<task>/ and are uploaded at the end.
# Status: syntax-checked; same evaluation code as the lab's H100 runs (gen_eval.py, twr_alpha.py), not yet run on Blackwell.
# Usage: bash run_eval.sh <gpu> <task id>      e.g. E11b-P2
set -euo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=$2; A=${T#E11b-}; OUT=$OPD_ROOT/runs/$T; mkdir -p $OUT
[ -f $OUT/.done ] && exit 0
source $OPD_ROOT/venv/bin/activate; export CUDA_VISIBLE_DEVICES=$G
M=$STUDENT_17B   # v2.4: E10b keeps its step-150 weights locally as bf16 only (upload_sync); fp32 if an older run kept it
ST=150; case ${A%-s4[34]} in P5top10|P5rand10m) ST=50;; esac   # v2.5.7h: the 50-step token-selective arms (Q-014)
if [ $A != P0 ]; then M=$OPD_ROOT/runs/E10b-$A/ckpt/global_step_$ST/actor/huggingface_bf16; has_weights $M || M=${M%_bf16}; fi
[ $A = P0 ] || has_weights $M || { echo "missing checkpoint $M"; exit 1; }
cd $RUNBOOK
for t in "T0.6 --temperature 0.6 --top_p 0.95 --top_k 0" "T1.0 --temperature 1.0 --top_p 1.0 --top_k 0"; do
  set -- $t; tag=e11b_${A}_math500_$1; shift
  ls $OUT/${tag}_*.metrics.json >/dev/null 2>&1 || \
    python code/eval/gen_eval.py --model $M --bench math500 --mode no_think --n 4 "$@" --max_tokens 16384 --tag $tag --out $OUT
  python scripts/control.py reap $G $T > /dev/null 2>&1 || true   # v2.5.4b: the finished engine must not hold the GPU for the next one
  gpu_wait_free $G || true   # [v2.5.8] and its memory has fallen before the next engine starts
done
# [v2.5.9z103] (Fable 10-28 02:2x UTC: outcomes for every reading is a 2.6.0 acceptance item; as run_queue_eval.sh z101) each
# finished reading also writes <stem>.outcomes.jsonl (no text; code/analysis/outcomes.py); a failure is listed, the readings are not
# affected
for J in $OUT/*.jsonl; do
  case $J in *.outcomes.jsonl|*.forced.jsonl) continue;; esac
  [ -f ${J%.jsonl}.metrics.json ] || continue   # a finished reading only
  [ -f ${J%.jsonl}.outcomes.jsonl ] || python3 $RUNBOOK/code/analysis/outcomes.py reading $J ${J%.jsonl}.outcomes.jsonl > /dev/null 2>> $OUT/outcomes.log \
    || echo "OUTCOMES_FAIL $(basename $J) (see outcomes.log)"
done
if [ $A != P0 ]; then
  REF=$(ls $OPD_ROOT/runs/E11b-P0/e11b_P0_math500_T0.6_*.jsonl | grep -Ev '\.(forced|outcomes)\.jsonl$' | head -1)
  SC=$(ls $OUT/e11b_${A}_math500_T0.6_*.jsonl | grep -Ev '\.(forced|outcomes)\.jsonl$' | head -1)
  [ -f $OUT/twr.json ] || python code/leakage/twr_alpha.py --judge $STUDENT_17B --bench math500 --ref $REF --score $SC \
                                --model_k $M --per_problem 4 --out $OUT/twr.json
fi
# v2.5.7zl (reviewer 21:1x UTC): "#pair E11b-<arm> <tag> <ref> [<ref> ...]" as in run_queue_eval.sh (paired_bootstrap.py; a ref
# is <task dir>/<file>.jsonl under $OPD_ROOT/runs, no "." / ".." part), before the upload; a rerun for the pairs skips
# the readings and the TWR already there. E11b-P2-t14 had no pairs (run_eval.sh had none).
Q=${QUEUE_FILE:-$RUNBOOK/tasks/queue.txt}
while read -r PTAG PREFS; do
  [ -n "$PTAG" ] || continue
  SC=$(ls $OUT/${PTAG}_*.jsonl 2> /dev/null | grep -Ev '\.(forced|outcomes)\.jsonl$' | head -1 || true)
  [ -n "$SC" ] || { echo "PAIR_SKIPPED $PTAG (no reading)"; continue; }
  RP=(); for R in $PREFS; do
    if [[ $R =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.jsonl$ && /$R/ != */../* && /$R/ != */./* ]]; then RP+=("$OPD_ROOT/runs/$R"); else echo "PAIR_REF_REFUSED $R"; fi
  done
  [ ${#RP[@]} -gt 0 ] || { echo "PAIR_SKIPPED $PTAG (no usable reference)"; continue; }   # [v2.5.8] (reviewer 21:2x UTC: an empty
  # array under set -u fails on older bash, and paired_bootstrap needs a reference)
  python3 code/analysis/paired_bootstrap.py --pairs $OUT/${PTAG}.paired.json "$SC" "${RP[@]}" > /dev/null 2>> $OUT/pair.log \
    && echo "PAIR_DONE $PTAG (${#RP[@]} refs)" || echo "PAIR_FAIL $PTAG (see pair.log)"
done <<< "$(awk -v t="$T" '$1 == "#pair" && $2 == t {$1 = ""; $2 = ""; sub(/^  /, ""); print}' "$Q")"
# .done only after the upload (v2.3): a failed upload makes run_tasks retry, and the retry only re-uploads
python scripts/remote.py put $OUT runs/$T && touch $OUT/.done && echo "UP_OK $T"
