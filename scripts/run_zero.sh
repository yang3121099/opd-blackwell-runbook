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
    ls $OUT/e03_mi_${L}_math500_*.metrics.json > /dev/null 2>&1 || \
    python code/eval/gen_eval.py --model $M --bench math500 --mode think --n 4 --temperature 0.6 --top_p 0.95 --top_k 20 \
      --max_tokens 32768 --tag e03_mi_${L}_math500 --out $OUT;;
  E02-opsa*)
    S=${T#E02-opsa}; S=${S%%-c*}; C=${T##*-c}
    # v2.5.6j: the OPSA name comes from the HF id, not from $STUDENT_17B (a local path on a node: "Tuwhy//<path>-OPSA")
    case $S in 17) B=$STUDENT_17B; FT=Tuwhy/Qwen3-1.7B-OPSA;; 4) B=Qwen/Qwen3-4B; FT=Tuwhy/Qwen3-4B-OPSA;; *) echo "unknown size $S"; exit 2;; esac
    M=$TMP/opsa${S}_c$C
    [ -d $M ] || python code/weights/wops.py scale --base $B --ft $FT --c $C --out $M
    ls $OUT/e02_opsa${S}_c${C}_aime24_nothink_*.metrics.json > /dev/null 2>&1 || \
    python code/eval/gen_eval.py --model $M --bench aime24 --mode no_think --n 16 --temperature 0.7 --top_p 0.8 --top_k 20 \
      --max_tokens 32768 --tag e02_opsa${S}_c${C}_aime24_nothink --out $OUT;;
  E05-fstop-*)   # v2.5.6l (queue v3 V3-c3 c): E05-p3 forced stop. AIME25 avg@8, thinking template, 32K (E05's setup),
    # then code/eval/forced_stop.py at 8K / 16K / 32K: unterminated samples cut there, the E05 stop string appended, <= 64 tokens
    W=${T#E05-fstop-}
    case $W in base06) M=$STUDENT_06B;; art06) M=taki555/Qwen3-0.6B-Art;; *) echo "unknown model $W"; exit 2;; esac
    ls $OUT/e05_${W}_aime25_think_*.metrics.json > /dev/null 2>&1 || \
    python code/eval/gen_eval.py --model $M --bench aime25 --mode think --n 8 --temperature 0.6 --top_p 0.95 --top_k 20 \
      --max_tokens 32768 --tag e05_${W}_aime25_think --out $OUT
    [ -f $OUT/e05_fstop_${W}.json ] || python code/eval/forced_stop.py --model $M --jsonl "$(ls $OUT/e05_${W}_aime25_think_*.jsonl | head -1)" \
      --bench aime25 --budgets 8192,16384,32768 --out $OUT/e05_fstop_${W}.json;;
  E01-*)   # v2.5.6j (queue v3 V3-c3 d): E01 remainder, AIME24 avg@32 at 32K with OPSA's sampling (T=0.7, top-p 0.8, top-k 20)
    W=${T#E01-}; MODE=${W##*-}; NAME=${W%-*}
    case $NAME in opsa17) M=Tuwhy/Qwen3-1.7B-OPSA;; base17) M=$STUDENT_17B;; *) echo "unknown model $NAME"; exit 2;; esac
    case $MODE in think) GM=think;; nothink) GM=no_think;; *) echo "unknown mode $MODE"; exit 2;; esac
    ls $OUT/e01_${NAME}_${MODE}_aime24_*.metrics.json > /dev/null 2>&1 || \
    python code/eval/gen_eval.py --model $M --bench aime24 --mode $GM --n 32 --temperature 0.7 --top_p 0.8 --top_k 20 \
      --max_tokens 32768 --tag e01_${NAME}_${MODE}_aime24 --out $OUT;;
  *) echo "unknown task $T"; exit 2;;
esac
rm -rf $TMP/mi_* $TMP/opsa*
# .done only after the upload (v2.3); a retry skips the finished generation (metrics file present) and only re-uploads
python scripts/remote.py put $OUT runs/$T && touch $OUT/.done && echo "UP_OK $T"
