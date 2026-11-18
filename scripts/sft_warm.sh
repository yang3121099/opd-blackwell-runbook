#!/usr/bin/env bash
# [v3.8] sft_warm.sh <gpu>: the task SFT-WARM (Fable 11-18; Runming: "a real training that starts fast": a borrowed machine is
# taken back when no real training runs on it, and the stack - vLLM, verl, flash-attn, the verify smoke - takes an hour to
# install through its proxy). One plain SFT process per GPU (code/train/sft_warm.py: torch + transformers, LoRA on the base
# model), started as soon as an environment imports torch and transformers; it waits for one by itself. A real training, not a
# registered experiment. It takes every GPU (SFT_WARM_GPUS=<a,b,..> narrows it): end it with `kill SFT-WARM` before the
# queue's tasks need the cards.
set -uo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=SFT-WARM; R=$OPD_ROOT/runs/$T; mkdir -p $R
PY=""; PP=""
# [v3.8c] an earlier install's venv usually has its packages intact on the shared disk while its bin/python points at an
# interpreter of a container that is gone (11-18: "-x venv/bin/python" failed and the old venv was taken for unusable). The right
# use: this node's own Python of the same minor version with the old venv's site-packages on PYTHONPATH (for this stopgap only).
try_env() {   # $1 = python, $2 = PYTHONPATH or ""
  [ -x "$1" ] || return 1
  PYTHONPATH="$2" timeout 420 "$1" -c 'import torch, transformers; assert torch.cuda.is_available()' >> $R/env_try.log 2>&1; }
for i in $(seq 1 ${SFT_WARM_WAIT_MIN:-90}); do
  for c in $OPD_ROOT/venv/bin/python $OPD_ROOT/venv-fa/bin/python; do try_env $c "" && { PY=$c; break 2; }; done
  for r in $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do for v in venv-fa venv; do
    for sp in $r/$v/lib/python3.*/site-packages; do [ -d "$sp/torch" ] || continue
      mv_=$(basename $(dirname $sp))   # python3.10
      for c in $r/$v/bin/python $OPD_ROOT/venv-ctl/bin/python $OPD_ROOT/venv/bin/python $(command -v $mv_ 2> /dev/null); do
        [ -x "$c" ] && [ "$($c -c 'import sys; print("python%d.%d" % sys.version_info[:2])' 2> /dev/null)" = "$mv_" ] || continue
        echo "SFT_WARM $(date -u +%T) trying the earlier install's packages ($v, $mv_) with $(basename $(dirname $(dirname $c)))'s interpreter"
        try_env $c "$sp" && { PY=$c; PP=$sp; break 5; }
      done; done; done; done
  echo "SFT_WARM $(date -u +%T) waiting for an environment with torch and transformers (minute $i): $(tail -n 1 $R/env_try.log 2> /dev/null | cut -c1-160)"; sleep 45; done
[ -n "$PY" ] || { echo "SFT_WARM no environment with torch and transformers after ${SFT_WARM_WAIT_MIN:-90} min"; exit 1; }
[ -n "$PP" ] && export PYTHONPATH="$PP"
# [v3.10c] the base model: local if it is complete; else copied from the long-term store / an earlier install on this machine
# (history_scan.py) - on a fresh machine models/ is empty until setup has copied or downloaded it, and a real training must not
# wait for that; else wait for setup's download (SFT_WARM_MODEL_WAIT_MIN, 60)
model_ok() { [ -f $1/config.json ] && ls $1/*.safetensors > /dev/null 2>&1 && ! ls $1/*.aria2 > /dev/null 2>&1; }
M=""
for i in $(seq 0 ${SFT_WARM_MODEL_WAIT_MIN:-60}); do
  for m in ${SFT_WARM_MODEL:-Qwen3-8B} Qwen3-1.7B Qwen3-0.6B; do
    model_ok $OPD_ROOT/models/$m && { M=$OPD_ROOT/models/$m; break 2; }
    for r in ${OPD_STORE:-} $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do model_ok $r/models/$m || continue
      echo "SFT_WARM $(date -u +%T) copying $m from the long-term store / an earlier install on this machine"
      rm -rf $OPD_ROOT/models/$m.warm.tmp; mkdir -p $OPD_ROOT/models/$m.warm.tmp
      if timeout ${SFT_WARM_COPY_MAX_S:-300} cp -rL $r/models/$m/. $OPD_ROOT/models/$m.warm.tmp/ 2>> $R/env_try.log && model_ok $OPD_ROOT/models/$m.warm.tmp && [ ! -e $OPD_ROOT/models/$m ]; then
        mv $OPD_ROOT/models/$m.warm.tmp $OPD_ROOT/models/$m; M=$OPD_ROOT/models/$m; break 3; fi
      # [v3.11] too slow (or setup made the directory meanwhile): this training reads the model where it is
      rm -rf $OPD_ROOT/models/$m.warm.tmp; model_ok $OPD_ROOT/models/$m && { M=$OPD_ROOT/models/$m; break 3; }
      echo "SFT_WARM $(date -u +%T) the copy took over ${SFT_WARM_COPY_MAX_S:-300} s: $m is read where it is"; M=$r/models/$m; break 3
    done
    # [v3.11] a complete copy anywhere else on this machine's disks (history_scan.py: a pretrained folder, a Hugging Face cache);
    # this stopgap does not need the published-weights check the registered tasks get
    hm=$(python3 $RUNBOOK/scripts/history_scan.py --model $m 2> /dev/null) && [ -n "$hm" ] && model_ok $hm && { echo "SFT_WARM $(date -u +%T) $m found on this machine's disks, read where it is"; M=$hm; break 2; }
    [ "$m" = "${SFT_WARM_MODEL:-Qwen3-8B}" ] && [ $i -lt ${SFT_WARM_MODEL_WAIT_MIN:-60} ] && break   # the wanted model first: smaller ones only after the wait
  done
  echo "SFT_WARM $(date -u +%T) waiting for a complete base model under models/ (minute $i)"; sleep 60; done
[ -n "$M" ] || { echo "SFT_WARM no complete base model under models/ after ${SFT_WARM_MODEL_WAIT_MIN:-60} min"; exit 1; }
D="$OPD_ROOT/data"; for r in $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do [ -d $r/data ] && D="$D $r/data"; done
GP=${SFT_WARM_GPUS:-all}; [ "$GP" = all ] && GP=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
echo "SFT_WARM $(date -u +%T) python $($PY -c 'import torch, transformers; print(torch.__version__, transformers.__version__)') model $(basename $M) gpus $GP"
pids=""
for g in $(echo $GP | tr , ' '); do
  CUDA_VISIBLE_DEVICES=$g TOKENIZERS_PARALLELISM=false $PY $RUNBOOK/code/train/sft_warm.py --model $M --out $R/gpu$g --data $D --hours ${SFT_WARM_HOURS:-6} --seed $g > $OPD_ROOT/logs/SFT-WARM_gpu$g.log 2>&1 &   # [v3.8d] under logs/: `tail SFT-WARM_gpu0.log`
  pids="$pids $!"; sleep 2; done
trap 'kill $pids 2> /dev/null' TERM INT
rc=0; for p in $pids; do wait $p || rc=1; done
echo "SFT_WARM $(date -u +%T) ended rc=$rc: $(tail -q -n 1 $OPD_ROOT/logs/SFT-WARM_gpu*.log 2> /dev/null | cut -c1-120 | tr '\n' ' ')"
[ $rc = 0 ] && touch $R/.done; exit $rc
