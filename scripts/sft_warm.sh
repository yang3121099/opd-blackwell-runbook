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
  [ -s $OPD_ROOT/control/history_secret.env ] && source $OPD_ROOT/control/history_secret.env   # [v3.12] the scan may end after this task started
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
  [ -s $OPD_ROOT/control/history_secret.env ] && source $OPD_ROOT/control/history_secret.env
  for m in ${SFT_WARM_MODEL:-Qwen3-8B} Qwen3-1.7B Qwen3-0.6B; do
    model_ok $OPD_ROOT/models/$m && { M=$OPD_ROOT/models/$m; break 2; }
    for r in ${OPD_STORE:-} $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do model_ok $r/models/$m || continue
      echo "SFT_WARM $(date -u +%T) copying $m from the long-term store / an earlier install on this machine"
      # [v3.12] (Runming 11-18: "use it directly first") read where it is; setup copies the model to the local disk for the registered
      # tasks anyway. SFT_WARM_COPY=1: the time-boxed copy first, as in v3.11
      [ "${SFT_WARM_COPY:-0}" = 1 ] || { echo "SFT_WARM $(date -u +%T) $m is read where it is (the store / an earlier install on this machine)"; M=$r/models/$m; break 3; }
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
# [v3.12] the GPUs: SFT_WARM_GPUS (setenv; `all` or a list), else every GPU but the first - setup's flash-attn test and the queue's
# first task use GPU 0, and a full-parameter 8B worker leaves little room beside itself. The scheduler does not count the GPU this
# task was started on as held (.gpu_released), and its workers give way card by card when the queue's tasks start (run_tasks.sh
# sft_yield: yield_gpu<g>); the task is done when every worker has ended by itself or has given way.
echo "$G" > $R/.gpu_released; rm -f $R/yield_gpu* $R/gpu*.pid
GP=${SFT_WARM_GPUS:-}; [ "$GP" = all ] && GP=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
[ -n "$GP" ] || GP=$(nvidia-smi --query-gpu=index --format=csv,noheader | grep -vx 0 | paste -sd, -)
[ -n "$GP" ] || GP=0
echo "SFT_WARM $(date -u +%T) python $($PY -c 'import torch, transformers; print(torch.__version__, transformers.__version__)') model $(basename $M) gpus $GP"
# [v3.12] one worker per GPU. Each is watched by a small wrapper: a full-parameter worker that ends with an error (not because it gave
# way) is started once more in LoRA mode - the mode that ran on 11-18 - so the card is not left without a training. gpu<g>.pid always
# holds the pid of the python process on the card (what nvidia-smi lists; run_tasks.sh's sft_only / sft_yield read it).
worker() {   # <gpu> <mode>: starts the python process, writes its pid, waits for it
  CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false SFT_WARM_MODE=$2 $PY $RUNBOOK/code/train/sft_warm.py --model $M --out $R/gpu$1 --data $D --hours ${SFT_WARM_HOURS:-6} --seed $1 >> $OPD_ROOT/logs/SFT-WARM_gpu$1.log 2>&1 &
  echo $! > $R/gpu$1.pid; wait $!; }
wpids=""
for g in $(echo $GP | tr , ' '); do
  : > $OPD_ROOT/logs/SFT-WARM_gpu$g.log
  ( trap 'kill $(cat $R/gpu$g.pid 2> /dev/null) 2> /dev/null' TERM INT
    worker $g ${SFT_WARM_MODE:-full}; rc=$?
    if [ $rc != 0 ] && [ ! -f $R/yield_gpu$g ] && [ "${SFT_WARM_MODE:-full}" = full ]; then
      echo "SFT_WARM $(date -u +%T) gpu $g: the full-parameter worker ended with exit $rc; started again in LoRA mode"
      worker $g lora; rc=$?; fi
    exit $rc ) &
  wpids="$wpids $g:$!"; sleep 2; done
trap 'for x in $wpids; do kill ${x#*:} 2> /dev/null; done' TERM INT
rc=0; for x in $wpids; do wait ${x#*:} || { [ -f $R/yield_gpu${x%%:*} ] || rc=1; }; done
echo "SFT_WARM $(date -u +%T) ended rc=$rc: $(tail -q -n 1 $OPD_ROOT/logs/SFT-WARM_gpu*.log 2> /dev/null | cut -c1-120 | tr '\n' ' ')"
[ $rc = 0 ] && touch $R/.done; exit $rc
