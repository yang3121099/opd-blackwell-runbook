# Common settings, sourced by every script. Override any of them in your shell before running.
export OPD_ROOT=${OPD_ROOT:-$HOME/opd-run}
export RUNBOOK=${RUNBOOK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
# model locations (v2.2): download_models.sh writes the three models' local paths to $OPD_ROOT/models.env (Hugging Face
# or ModelScope route); a value already set in your shell wins; without the file the Hugging Face IDs are used
if [ -f $OPD_ROOT/models.env ]; then
  while IFS='=' read -r k v; do k=${k#export }; case "$k" in STUDENT_06B|STUDENT_17B|TEACHER) [ -z "${!k:-}" ] && export "$k=$v";; esac; done < $OPD_ROOT/models.env
fi
export STUDENT_06B=${STUDENT_06B:-Qwen/Qwen3-0.6B} STUDENT_17B=${STUDENT_17B:-Qwen/Qwen3-1.7B} TEACHER=${TEACHER:-Qwen/Qwen3-8B}
export GOPD_COMMIT=37371a4c31ad7947746200d234161769191f4748
export UPLOAD_BACKEND=${UPLOAD_BACKEND:-hf}          # hf | modelscope
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export RAY_prestart_worker_first_driver=${RAY_prestart_worker_first_driver:-0}
export RAY_num_workers_soft_limit=${RAY_num_workers_soft_limit:-8}
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
mkdir -p $OPD_ROOT/logs $OPD_ROOT/runs $OPD_ROOT/data
