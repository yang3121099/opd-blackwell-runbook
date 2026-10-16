#!/usr/bin/env bash
# Blackwell stack with uv: torch 2.8.0 (CUDA 12.8 wheels) + vLLM 0.10.2 + G-OPD (verl 0.6.1 fork) at a fixed commit
# + the lab patches; then models and data. UNTESTED on Blackwell until section 8 of README.md is filled in.
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd $OPD_ROOT
command -v uv >/dev/null || pip install -q uv
[ -d venv ] || uv venv -p 3.10 venv
source venv/bin/activate
uv pip install "torch==2.8.0" "vllm==0.10.2" "transformers>=4.55.2,<4.57" accelerate "datasets>=3.0,<4.1" peft hf-transfer \
  "numpy<2.0.0" "pyarrow>=15.0.0" pandas "ray[default]" codetiming hydra-core pylatexenc dill pybind11 \
  "tensordict>=0.8,<0.10" torchdata liger-kernel mathruler pytest math-verify safetensors "flashinfer-python==0.3.0" \
  "huggingface_hub>=0.34" modelscope
# B300 (compute capability 10.3, v2.3): put the CUDA 12.9 ptxas in place for the second fix (README section 2b);
# it is used only after $OPD_ROOT/triton.env points TRITON_PTXAS_PATH at it
if [ "$(gpu_cap)" = 10.3 ]; then uv pip install "$B300_PTXAS"; fi
bash $RUNBOOK/scripts/install_gopd.sh
