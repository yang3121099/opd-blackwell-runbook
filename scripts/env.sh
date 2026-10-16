# Common settings, sourced by every script. Override any of them in your shell before running.
export OPD_ROOT=${OPD_ROOT:-$HOME/opd-run}
export RUNBOOK=${RUNBOOK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export GOPD_COMMIT=37371a4c31ad7947746200d234161769191f4748
export UPLOAD_BACKEND=${UPLOAD_BACKEND:-hf}          # hf | modelscope
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export RAY_prestart_worker_first_driver=${RAY_prestart_worker_first_driver:-0}
export RAY_num_workers_soft_limit=${RAY_num_workers_soft_limit:-8}
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
mkdir -p $OPD_ROOT/logs $OPD_ROOT/runs $OPD_ROOT/data
