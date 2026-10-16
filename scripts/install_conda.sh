#!/usr/bin/env bash
# Same stack as install_uv.sh, in a conda environment named opd-e10b. UNTESTED on Blackwell.
set -euo pipefail
source "$(dirname "$0")/env.sh"
conda create -y -n opd-e10b python=3.10
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate opd-e10b
pip install "torch==2.8.0" "vllm==0.10.2" "transformers>=4.55.2,<4.57" accelerate "datasets>=3.0,<4.1" peft hf-transfer \
  "numpy<2.0.0" "pyarrow>=15.0.0" pandas "ray[default]" codetiming hydra-core pylatexenc dill pybind11 \
  "tensordict>=0.8,<0.10" torchdata liger-kernel mathruler pytest math-verify safetensors "flashinfer-python==0.3.0" \
  "huggingface_hub>=0.34" modelscope
mkdir -p $OPD_ROOT && ln -sfn "$CONDA_PREFIX" $OPD_ROOT/venv     # scripts activate $OPD_ROOT/venv/bin/activate
bash $RUNBOOK/scripts/install_gopd.sh
