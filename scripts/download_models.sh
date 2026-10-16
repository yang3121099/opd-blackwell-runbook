#!/usr/bin/env bash
# Student, teacher and benchmark data. USE_MODELSCOPE=1 downloads the models from ModelScope (same IDs);
# HF_ENDPOINT=https://hf-mirror.com works for the public datasets.
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-8B; do
  if [ "${USE_MODELSCOPE:-0}" = 1 ]; then python -c "from modelscope import snapshot_download; print(snapshot_download('$m'))"
  else python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','*.jinja']))"; fi \
    && echo "OK $m" || echo "FAIL $m"
done
for d in HuggingFaceH4/MATH-500 BytedTsinghua-SIA/DAPO-Math-17k; do
  python -c "from huggingface_hub import snapshot_download; snapshot_download('$d', repo_type='dataset')" && echo "OK $d" || echo "FAIL $d"
done
