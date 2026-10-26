# Common settings, sourced by every script. Override any of them in your shell before running.
export OPD_ROOT=${OPD_ROOT:-$HOME/opd-run}
export RUNBOOK=${RUNBOOK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# v2.5.4c: setproctitle (vLLM's VLLM::EngineCore, Ray's ray::...) must not write the title over the environment area, so
# /proc/<pid>/environ keeps OPD_ROOT and CUDA_VISIBLE_DEVICES for control.py reap (titles stay within the command line's length)
export SPT_NOENV=${SPT_NOENV:-1}
# model locations (v2.2): download_models.sh writes the three models' local paths to $OPD_ROOT/models.env (Hugging Face
# or ModelScope route); a value already set in your shell wins; without the file the Hugging Face IDs are used
if [ -f $OPD_ROOT/models.env ]; then
  while IFS='=' read -r k v; do k=${k#export }; case "$k" in STUDENT_06B|STUDENT_17B|STUDENT_4B|TEACHER) [ -z "${!k:-}" ] && export "$k=$v";; esac; done < $OPD_ROOT/models.env
fi
# v2.4: checkpoints live on a large local disk (B300: /home has far more room than the 48 GB of $OPD_ROOT's disk).
# CKPT_ROOT is fixed once in $OPD_ROOT/paths.env: <writable directory under /home or $HOME whose filesystem has the most
# free space>/opd-ckpt (a value set in the shell wins). runs/<task>/ckpt is a symlink to $CKPT_ROOT/<task>/ckpt
# (ckpt_link below), so every script still reads and writes runs/<task>/ckpt.
# v2.5: every candidate base B holds the same tree, B/opd-ckpt/<task>/ckpt/global_step_<N>/..., and CKPT_ROOT is always
# <one candidate>/opd-ckpt. Candidates (found at run time, nothing machine-specific in this repository): $HOME, the
# directory above $OPD_ROOT, the same relative path on every other mounted data disk, and CKPT_CANDIDATES_EXTRA from
# paths.env (control: set CKPT_CANDIDATES+=<dir>). control: set CKPT_ROOT=<candidate> switches for tasks started later.
ckpt_candidates() {
  local base mnt rel m out
  base=$(dirname "$OPD_ROOT"); out="$HOME:$base"
  mnt=$(df -P "$base" 2>/dev/null | awk 'NR==2{print $6}'); rel=${base#$mnt}
  if [ -n "$mnt" ] && [ "$mnt" != / ] && [ -r /proc/mounts ]; then
    for m in $(awk '$3 !~ /^(proc|sysfs|tmpfs|devtmpfs|cgroup2?|overlay|devpts|mqueue|securityfs|debugfs|tracefs|fusectl|configfs|pstore|bpf|autofs|hugetlbfs|nsfs|squashfs|ramfs|rpc_pipefs|binfmt_misc)$/ {print $2}' /proc/mounts); do
      [ "$m" = "$mnt" ] || [ "$m" = / ] && continue
      [ -d "$m$rel" ] && [ -w "$m$rel" ] && out="$out:$m$rel"; done
  fi
  echo "$out:${CKPT_CANDIDATES_EXTRA:-}" | tr ':' '\n' | awk 'NF && !seen[$0]++' | paste -sd: -; }
# paths.env holds CKPT_ROOT, CKPT_CANDIDATES_EXTRA and ARCHIVE_ROOT. A CKPT_ROOT read from it is marked
# (CKPT_ROOT_FROM=paths.env) and re-read at every source, so `set CKPT_ROOT=` reaches tasks started later even though
# run_tasks.sh exported the old value; a CKPT_ROOT set by hand in the shell still wins.
if [ -f $OPD_ROOT/paths.env ]; then
  if [ -z "${CKPT_ROOT:-}" ] || [ "${CKPT_ROOT_FROM:-}" = paths.env ]; then
    unset CKPT_ROOT; source $OPD_ROOT/paths.env; [ -n "${CKPT_ROOT:-}" ] && CKPT_ROOT_FROM=paths.env
  else _ck=$CKPT_ROOT; source $OPD_ROOT/paths.env; CKPT_ROOT=$_ck; fi
fi
if [ -z "${CKPT_ROOT:-}" ]; then   # first use: the writable directory under $HOME or /home with the most free space
  _best=""; _free=0
  for _d in "$HOME" /home/*/ /home; do _d=${_d%/}; [ -d "$_d" ] && [ -w "$_d" ] || continue
    _f=$(df -Pk "$_d" 2>/dev/null | awk 'NR==2{print $4}'); [ -n "$_f" ] && [ "$_f" -gt "$_free" ] && { _best=$_d; _free=$_f; }; done
  CKPT_ROOT=${_best:-$(dirname "$OPD_ROOT")}/opd-ckpt
  mkdir -p "$CKPT_ROOT" && { grep -v '^CKPT_ROOT=' $OPD_ROOT/paths.env 2>/dev/null; echo "CKPT_ROOT=$CKPT_ROOT"; } > $OPD_ROOT/paths.env.new \
    && mv $OPD_ROOT/paths.env.new $OPD_ROOT/paths.env; CKPT_ROOT_FROM=paths.env
fi
export CKPT_ROOT CKPT_ROOT_FROM=${CKPT_ROOT_FROM:-} CKPT_CANDIDATES_EXTRA=${CKPT_CANDIDATES_EXTRA:-} ARCHIVE_ROOT=${ARCHIVE_ROOT:-}
# [v2.5.9z88] (ops-lead 10-26 02:0x Beijing: another user filled the shared work disk to 69 GB; Runming "no new writes to the shared
# disk") the caches that only save time are written on the checkpoint disk: hf_xet's chunk cache (HF_XET_CACHE), and no __pycache__
# next to any .py (existing .pyc files are still read). HF_HOME stays: the token is there, and the hub / datasets caches under it
# hold what the evals load by repo id (gen_eval's load_dataset) and the 2507 arms find with local_files_only
[ -n "${CKPT_ROOT:-}" ] && export HF_XET_CACHE=${HF_XET_CACHE:-$CKPT_ROOT/hf_cache/xet}   # never an empty or root-level path
export PYTHONDONTWRITEBYTECODE=1
export CKPT_CANDIDATES=$(ckpt_candidates)   # colon list of candidate bases (v2.5)
ckpt_link() {   # <task dir>: its ckpt becomes a symlink into $CKPT_ROOT; a real ckpt dir from an older run stays as it is
  local t=$(basename "$1"); mkdir -p "$CKPT_ROOT/$t/ckpt"
  [ -e "$1/ckpt" ] || [ -L "$1/ckpt" ] || ln -s "$CKPT_ROOT/$t/ckpt" "$1/ckpt"; }
# [v2.5.9z48] (ops-lead 10-22: the shared work disk) a new run's per-step rollout dumps go to $CKPT_ROOT as its ckpt does: an absent
# or empty rollouts dir becomes a symlink to $CKPT_ROOT/<task>/rollouts; a run that already dumped (a running or resumed one) keeps
# its real dir. upload_sync.sh reads $RUN/rollouts through the link (the result repository's path unchanged); du does not follow it
rollouts_link() {
  local t=$(basename "$1"); mkdir -p "$CKPT_ROOT/$t/rollouts"
  [ -d "$1/rollouts" ] && [ ! -L "$1/rollouts" ] && [ -z "$(ls -A "$1/rollouts")" ] && rmdir "$1/rollouts"
  [ -e "$1/rollouts" ] || [ -L "$1/rollouts" ] || ln -s "$CKPT_ROOT/$t/rollouts" "$1/rollouts"; }
# v2.5.7b: the 4B student (V3-c5) from $OPD_ROOT/models/Qwen3-4B once PROBE-4B has fetched it, else its HF id
# v2.5.7p (Q-015): the 14B teacher of the -t14 arms from $OPD_ROOT/models/Qwen3-14B once PROBE-T14 has fetched it
# v2.5.7zd (10-19 15:5x UTC): also when the HF id was inherited. run_tasks, started before PROBE-T14 wrote .fetch_ok,
# exported TEACHER_14B=Qwen/Qwen3-14B; its tasks kept that id, and the -t14 arms of 15:17 / 15:33 UTC hung on a hub
# download (ProxyError) with the complete 29.6 GB local copy unused
case "${STUDENT_4B:-Qwen/Qwen3-4B}" in Qwen/Qwen3-4B) [ -f "$OPD_ROOT/models/Qwen3-4B/.fetch_ok" ] && STUDENT_4B=$OPD_ROOT/models/Qwen3-4B;; esac
case "${TEACHER_14B:-Qwen/Qwen3-14B}" in Qwen/Qwen3-14B) [ -f "$OPD_ROOT/models/Qwen3-14B/.fetch_ok" ] && TEACHER_14B=$OPD_ROOT/models/Qwen3-14B;; esac
export TEACHER_14B=${TEACHER_14B:-Qwen/Qwen3-14B}
export STUDENT_06B=${STUDENT_06B:-Qwen/Qwen3-0.6B} STUDENT_17B=${STUDENT_17B:-Qwen/Qwen3-1.7B} TEACHER=${TEACHER:-Qwen/Qwen3-8B} STUDENT_4B=${STUDENT_4B:-Qwen/Qwen3-4B}
export GOPD_COMMIT=37371a4c31ad7947746200d234161769191f4748
export UPLOAD_BACKEND=${UPLOAD_BACKEND:-hf}          # hf | modelscope
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export RAY_prestart_worker_first_driver=${RAY_prestart_worker_first_driver:-0}
export RAY_num_workers_soft_limit=${RAY_num_workers_soft_limit:-8}
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
# v2.5.7a: once persist-env has made <the directory above $OPD_ROOT>/opd-env/hf on the share, Hugging Face downloads
# by name (OPSA / Art models of the zero stage, ...) are cached there and survive a container rebuild
[ -d "$(dirname "$OPD_ROOT")/opd-env/hf" ] && export HF_HOME=${HF_HOME:-$(dirname "$OPD_ROOT")/opd-env/hf}
mkdir -p $OPD_ROOT/logs $OPD_ROOT/runs $OPD_ROOT/data
# HF weights present in a folder: model.safetensors or shards model-0000k-of-0000n.safetensors (every 1.7B checkpoint is
# sharded; v2.3: run_e12/run_eval/run_sft used to look for model.safetensors only and failed on every 1.7B start)
has_weights() { ls "$1"/*.safetensors > /dev/null 2>&1; }
# [v2.5.9z88d] (ops-lead 10-26) a scan that keys on a script name must not count CORPUS-RECORD's sandbox processes: a sandboxed
# task.sh runs the bash scripts it dispatches for real, and every sandbox process inherits DRYRUN_SB (dryrun_json.py's child_env).
# ours_pgrep <pgrep args>: pgrep's matches (pids, or -a lines) without those; it reads only the matches' /proc/<pid>/environ
ours_pargs() { local p; for p in $(ours_pgrep "$@"); do ps -o args= -p ${p%% *} 2> /dev/null; done; }   # their command lines (portable: no pgrep -a)
ours_pgrep() { local l; pgrep "$@" | while read -r l; do [ -n "$l" ] || continue; tr '\0' '\n' < /proc/${l%% *}/environ 2> /dev/null | grep -q '^DRYRUN_SB=' || echo "$l"; done; }
# [v2.5.8] (10-19 03:0x UTC: E11b engine-init failures) gpu_wait_free <gpu> [seconds]: before the next engine on a GPU, wait
# until its memory has fallen to <= GPU_FREE_MIB (2048) MiB, i.e. the previous engine and its workers are gone; 1 on timeout
gpu_wait_free() {
  command -v nvidia-smi > /dev/null || return 0
  local i u; for i in $(seq ${2:-${GPU_WAIT_FREE_S:-180}}); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2> /dev/null | tr -d ' ')
    [[ $u =~ ^[0-9]+$ ]] && [ $u -le ${GPU_FREE_MIB:-2048} ] && return 0; sleep 1; done
  echo "GPU_NOT_FREE gpu $1 still ${u:-?} MiB after ${2:-${GPU_WAIT_FREE_S:-180}} s"; return 1; }
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
# v2.5.7i: a probe's own settings (PROBE_ENV="KEY=VAL ...", set by fa_speed.sh for its variants) win over control.env.
# (FA-SPEED-TWIN 2026-10-18 19:5x UTC: `env VENV= ...` was overwritten by control.env's VENV=fa when run_arm.sh sourced
# this file, so both twins ran flash-attn; the token-budget variants ran 4 per micro-batch the same way.)
for _kv in ${PROBE_ENV:-}; do [[ $_kv =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_.,:/=+-]*$ ]] && export "$_kv"; done
