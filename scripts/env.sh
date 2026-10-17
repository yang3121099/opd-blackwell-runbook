# Common settings, sourced by every script. Override any of them in your shell before running.
export OPD_ROOT=${OPD_ROOT:-$HOME/opd-run}
export RUNBOOK=${RUNBOOK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# model locations (v2.2): download_models.sh writes the three models' local paths to $OPD_ROOT/models.env (Hugging Face
# or ModelScope route); a value already set in your shell wins; without the file the Hugging Face IDs are used
if [ -f $OPD_ROOT/models.env ]; then
  while IFS='=' read -r k v; do k=${k#export }; case "$k" in STUDENT_06B|STUDENT_17B|TEACHER) [ -z "${!k:-}" ] && export "$k=$v";; esac; done < $OPD_ROOT/models.env
fi
# v2.4: checkpoints live on a large local disk (B300: /home has far more room than the 48 GB of $OPD_ROOT's disk).
# CKPT_ROOT is fixed once in $OPD_ROOT/paths.env: <writable directory under /home or $HOME whose filesystem has the most
# free space>/opd-ckpt (a value set in the shell wins). runs/<task>/ckpt is a symlink to $CKPT_ROOT/<task>/ckpt
# (ckpt_link below), so every script still reads and writes runs/<task>/ckpt.
if [ -z "${CKPT_ROOT:-}" ] && [ -f $OPD_ROOT/paths.env ]; then source $OPD_ROOT/paths.env; fi
if [ -z "${CKPT_ROOT:-}" ]; then
  _best=""; _free=0
  for _d in "$HOME" /home/*/ /home; do _d=${_d%/}; [ -d "$_d" ] && [ -w "$_d" ] || continue
    _f=$(df -Pk "$_d" 2>/dev/null | awk 'NR==2{print $4}'); [ -n "$_f" ] && [ "$_f" -gt "$_free" ] && { _best=$_d; _free=$_f; }; done
  CKPT_ROOT=${_best:+$_best/opd-ckpt}; CKPT_ROOT=${CKPT_ROOT:-$OPD_ROOT/ckpt}
  mkdir -p "$CKPT_ROOT" && echo "CKPT_ROOT=$CKPT_ROOT" > $OPD_ROOT/paths.env
fi
export CKPT_ROOT
ckpt_link() {   # <task dir>: its ckpt becomes a symlink into $CKPT_ROOT; a real ckpt dir from an older run stays as it is
  local t=$(basename "$1"); mkdir -p "$CKPT_ROOT/$t/ckpt"
  [ -e "$1/ckpt" ] || [ -L "$1/ckpt" ] || ln -s "$CKPT_ROOT/$t/ckpt" "$1/ckpt"; }
export STUDENT_06B=${STUDENT_06B:-Qwen/Qwen3-0.6B} STUDENT_17B=${STUDENT_17B:-Qwen/Qwen3-1.7B} TEACHER=${TEACHER:-Qwen/Qwen3-8B}
export GOPD_COMMIT=37371a4c31ad7947746200d234161769191f4748
export UPLOAD_BACKEND=${UPLOAD_BACKEND:-hf}          # hf | modelscope
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export RAY_prestart_worker_first_driver=${RAY_prestart_worker_first_driver:-0}
export RAY_num_workers_soft_limit=${RAY_num_workers_soft_limit:-8}
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
mkdir -p $OPD_ROOT/logs $OPD_ROOT/runs $OPD_ROOT/data
# HF weights present in a folder: model.safetensors or shards model-0000k-of-0000n.safetensors (every 1.7B checkpoint is
# sharded; v2.3: run_e12/run_eval/run_sft used to look for model.safetensors only and failed on every 1.7B start)
has_weights() { ls "$1"/*.safetensors > /dev/null 2>&1; }
# B300 (compute capability 10.3, v2.3): the Triton in torch 2.8 ships the ptxas of CUDA 12.8, which may fail on sm_103
# ("PTXASError: Internal Triton PTX codegen error"). Order of fixes (README section 2b): 12.8 as is, then the CUDA 12.9
# ptxas ($B300_PTXAS, installed by install_uv.sh / install_conda.sh on a 10.3 card), then the CUDA 13 route.
# overnight.sh (tier B) or the manual step writes $OPD_ROOT/triton.env; every script picks it up here, so a restarted
# run_tasks.sh keeps the same ptxas. Delete the file to go back to Triton's own ptxas. v2.3.1: it may also set
# VLLM_ATTENTION_BACKEND and VLLM_USE_FLASHINFER_SAMPLER=0 (FlashInfer's JIT sampler needs an nvcc that knows sm_103).
export B300_PTXAS="nvidia-cuda-nvcc-cu12==12.9.86"
gpu_cap() { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1; }
ptxas_rel() { [ -x "${1:-}" ] && "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p'; }
if [ -f $OPD_ROOT/triton.env ]; then source $OPD_ROOT/triton.env; fi
# v2.4 (ops-1): variables set remotely with `setenv` (control.py; whitelisted keys only), sourced last
if [ -f $OPD_ROOT/control.env ]; then source $OPD_ROOT/control.env; fi
