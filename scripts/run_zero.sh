#!/usr/bin/env bash
# Zero-training items (no training, inference only):
#   E03-MI-l<lam>   : MATH500 avg@4 (thinking template, T=0.6 / top-p 0.95 / top-k 20, 32K) of the weight interpolation
#                     lam*Qwen3-4B-Thinking-2507 + (1-lam)*Qwen3-4B-Instruct-2507 (lam 1.0 = Thinking-2507 itself)
#   E02-opsa<17|4>-c<c> : AIME24 avg@16 (non-thinking, T=0.7 / top-p 0.8 / top-k 20, 32K) of Base + c*(OPSA - Base)
#                     for Qwen3-1.7B / Qwen3-4B with the public OPSA checkpoints (Tuwhy/Qwen3-*-OPSA)
# The merged weights are built on CPU (code/weights/wops.py) in $OPD_ROOT/tmpm and deleted afterwards.
# Status: syntax-checked; same code as the lab's runs on A100/H100, not yet run on Blackwell.
# Usage: bash run_zero.sh <gpu> <task id>
set -euo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=$2; OUT=$OPD_ROOT/runs/$T; TMP=$OPD_ROOT/tmpm; mkdir -p $OUT $TMP
[ -f $OUT/.done ] && exit 0
source $OPD_ROOT/venv/bin/activate; export CUDA_VISIBLE_DEVICES=$G; cd $RUNBOOK
case $T in
  E03-MI-l*)
    L=${T#E03-MI-l}; M=Qwen/Qwen3-4B-Thinking-2507
    if [ $L != 1.0 ]; then M=$TMP/mi_$L
      [ -d $M ] || python code/weights/wops.py interp --a Qwen/Qwen3-4B-Thinking-2507 --b Qwen/Qwen3-4B-Instruct-2507 --lam $L --out $M; fi
    python code/eval/gen_eval.py --model $M --bench math500 --mode think --n 4 --temperature 0.6 --top_p 0.95 --top_k 20 \
      --max_tokens 32768 --tag e03_mi_${L}_math500 --out $OUT;;
  E02-opsa*)
    S=${T#E02-opsa}; S=${S%%-c*}; C=${T##*-c}; B=Qwen/Qwen3-4B; [ $S = 17 ] && B=Qwen/Qwen3-1.7B
    M=$TMP/opsa${S}_c$C
    [ -d $M ] || python code/weights/wops.py scale --base $B --ft Tuwhy/${B#Qwen/}-OPSA --c $C --out $M
    python code/eval/gen_eval.py --model $M --bench aime24 --mode no_think --n 16 --temperature 0.7 --top_p 0.8 --top_k 20 \
      --max_tokens 32768 --tag e02_opsa${S}_c${C}_aime24_nothink --out $OUT;;
  *) echo "unknown task $T"; exit 2;;
esac
rm -rf $TMP/mi_* $TMP/opsa*
touch $OUT/.done
python scripts/remote.py put $OUT runs/$T && echo "UP_OK $T"
