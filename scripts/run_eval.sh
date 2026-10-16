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
M=Qwen/Qwen3-1.7B; [ $A = P0 ] || M=$OPD_ROOT/runs/E10b-$A/ckpt/global_step_150/actor/huggingface
[ $A = P0 ] || [ -f $M/model.safetensors ] || { echo "missing checkpoint $M"; exit 1; }
cd $RUNBOOK
for t in "T0.6 --temperature 0.6 --top_p 0.95 --top_k 0" "T1.0 --temperature 1.0 --top_p 1.0 --top_k 0"; do
  set -- $t; tag=e11b_${A}_math500_$1; shift
  ls $OUT/${tag}_*.metrics.json >/dev/null 2>&1 || \
    python code/eval/gen_eval.py --model $M --bench math500 --mode no_think --n 4 "$@" --max_tokens 16384 --tag $tag --out $OUT
done
if [ $A != P0 ]; then
  REF=$(ls $OPD_ROOT/runs/E11b-P0/e11b_P0_math500_T0.6_*.jsonl | head -1)
  SC=$(ls $OUT/e11b_${A}_math500_T0.6_*.jsonl | head -1)
  [ -f $OUT/twr.json ] || python code/leakage/twr_alpha.py --judge Qwen/Qwen3-1.7B --bench math500 --ref $REF --score $SC \
                                --model_k $M --per_problem 4 --out $OUT/twr.json
fi
touch $OUT/.done
python scripts/remote.py put $OUT runs/$T && echo "UP_OK $T"
