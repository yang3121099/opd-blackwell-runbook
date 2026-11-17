#!/usr/bin/env bash
# [v3.8] sft_warm.sh <gpu>: the task SFT-WARM (Fable 11-18; Runming: "a real training that starts fast": a borrowed machine is
# taken back when no real training runs on it, and the stack - vLLM, verl, flash-attn, the verify smoke - takes an hour to
# install through its proxy). One plain SFT process per GPU (code/train/sft_warm.py: torch + transformers, LoRA on the base
# model), started as soon as an environment imports torch and transformers; it waits for one by itself. A real training, not a
# registered experiment. It takes every GPU (SFT_WARM_GPUS=<a,b,..> narrows it): end it with `kill SFT-WARM` before the
# queue's tasks need the cards.
set -uo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=SFT-WARM; R=$OPD_ROOT/runs/$T; mkdir -p $R
PY=""
for i in $(seq 1 ${SFT_WARM_WAIT_MIN:-90}); do
  for c in $OPD_ROOT/venv/bin/python $OPD_ROOT/venv-fa/bin/python; do
    [ -x $c ] && $c -c 'import torch, transformers' 2> /dev/null && { PY=$c; break 2; }; done
  echo "SFT_WARM $(date -u +%T) waiting for an environment with torch and transformers (minute $i)"; sleep 60; done
[ -n "$PY" ] || { echo "SFT_WARM no environment with torch and transformers after ${SFT_WARM_WAIT_MIN:-90} min"; exit 1; }
M=""; for m in ${SFT_WARM_MODEL:-Qwen3-8B} Qwen3-1.7B Qwen3-0.6B; do [ -f $OPD_ROOT/models/$m/config.json ] && ls $OPD_ROOT/models/$m/*.safetensors > /dev/null 2>&1 && [ ! -e $OPD_ROOT/models/$m/*.aria2 ] && { M=$OPD_ROOT/models/$m; break; }; done
[ -n "$M" ] || { echo "SFT_WARM no complete base model under models/"; exit 1; }
D="$OPD_ROOT/data"; for r in $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do [ -d $r/data ] && D="$D $r/data"; done
GP=${SFT_WARM_GPUS:-all}; [ "$GP" = all ] && GP=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
echo "SFT_WARM $(date -u +%T) python $($PY -c 'import torch, transformers; print(torch.__version__, transformers.__version__)') model $(basename $M) gpus $GP"
pids=""
for g in $(echo $GP | tr , ' '); do
  CUDA_VISIBLE_DEVICES=$g TOKENIZERS_PARALLELISM=false $PY $RUNBOOK/code/train/sft_warm.py --model $M --out $R/gpu$g --data $D --hours ${SFT_WARM_HOURS:-6} --seed $g > $R/train_gpu$g.log 2>&1 &
  pids="$pids $!"; sleep 2; done
trap 'kill $pids 2> /dev/null' TERM INT
rc=0; for p in $pids; do wait $p || rc=1; done
echo "SFT_WARM $(date -u +%T) ended rc=$rc: $(tail -q -n 1 $R/train_gpu*.log 2> /dev/null | cut -c1-120 | tr '\n' ' ')"
[ $rc = 0 ] && touch $R/.done; exit $rc
