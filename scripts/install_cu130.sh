#!/usr/bin/env bash
# CUDA 13 fallback (B300 / sm_103), only when the default cu128 stack fails: verify.sh reports "no kernel image" or the
# bf16 matmul does not run. UNTESTED. The previous environment is kept as $OPD_ROOT/venv-cu128.
#   1. new venv; torch 2.9.0 + torchvision 0.24.0 from the cu130 index (vLLM v0.11.2 pins torch==2.9.0)
#   2. check that torch's arch list covers the GPU (sm_103, or at least sm_100)
#   3. vLLM v0.11.2 built from source against this torch (needs nvcc >= 13.0; MAX_JOBS parallel compile jobs)
#   4. the other packages, then install_gopd.sh (G-OPD at the fixed commit + patches + models + data), then verify.sh
# In the NGC container (nvcr.io/nvidia/pytorch:25.10-py3, torch and nvcc preinstalled) run with SKIP_TORCH=1 inside
# a venv created with --system-site-packages, so the container's torch is used.
# The G-OPD fork is verl 0.6.1; its interface with vLLM 0.11 is NOT verified. If the smoke test fails there, send the
# log back: the project adapts code/patches (see README section 2b).
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd $OPD_ROOT
nv=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.[0-9]*.*/\1/p')
[ "${nv:-0}" -ge 13 ] || { echo "nvcc ${nv:-missing} is older than 13.0: use the NGC container route (README section 2b)"; exit 1; }
[ -d venv ] && [ ! -d venv-cu128 ] && mv venv venv-cu128      # keep the cu128 environment, never reuse it here
[ -d venv ] && { echo "$OPD_ROOT/venv exists and venv-cu128 is taken: move one of them away first"; exit 1; }
if [ "${SKIP_TORCH:-0}" != 1 ]; then
  uv venv -p 3.10 venv
  source venv/bin/activate
  uv pip install "torch==2.9.0" "torchvision==0.24.0" --index-url https://download.pytorch.org/whl/cu130
else
  python -m venv --system-site-packages venv     # the NGC container's torch
  source venv/bin/activate
fi
python - <<'PY'
import torch
arch, cap = torch.cuda.get_arch_list(), torch.cuda.get_device_capability()
print("torch", torch.__version__, "cuda", torch.version.cuda, "device capability", cap, "arch list", arch)
want = f"sm_{cap[0]}{cap[1]}"
if want not in arch:
    if f"sm_{cap[0]}0" in arch:
        print(f"WARNING: {want} not in the arch list; relying on {cap[0]}.0 binaries running on {cap[0]}.{cap[1]}")
    else:
        raise SystemExit(f"this torch build has no kernels for {want}")
PY
[ -d vllm-src ] || git clone --depth 1 --branch v0.11.2 https://github.com/vllm-project/vllm.git vllm-src
(cd vllm-src && python use_existing_torch.py \
   && uv pip install -r requirements/build.txt \
   && MAX_JOBS=${MAX_JOBS:-32} uv pip install --no-build-isolation -e .)
uv pip install "transformers>=4.56,<5" accelerate "datasets>=3.0,<4.1" peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
  "ray[default]" codetiming hydra-core pylatexenc dill pybind11 "tensordict>=0.8,<0.11" torchdata liger-kernel mathruler \
  pytest math-verify safetensors "flashinfer-python==0.5.2" "huggingface_hub>=0.34" modelscope
python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)"
bash $RUNBOOK/scripts/install_gopd.sh
echo "CU130_INSTALL_DONE: now run  bash scripts/verify.sh 2>&1 | tee verify_cu130.log"
