# verl's official CUDA 12.8 / torch 2.8 base image, plus vLLM 0.10.2 and the runbook. UNTESTED on Blackwell.
#   docker build -t opd-e10b .
#   docker run --gpus all --ipc=host --shm-size=64g -v $HOME/opd-run:/root/opd-run -e HF_TOKEN -e HF_REPO -it opd-e10b
#   inside: bash scripts/install_gopd.sh   (G-OPD at the fixed commit + patches + models + data), then scripts/verify.sh
# The base image ships flash-attn 2.7.4. If it fails on Blackwell ("no kernel image"), `pip uninstall -y flash-attn`:
# the scripts then fall back to SDPA. nvidia-cuda-nvcc-cu12 brings the CUDA 12.9 ptxas that B300 may need (v2.3; used
# only after $OPD_ROOT/triton.env points TRITON_PTXAS_PATH at it, README section 2b).
FROM verlai/verl:base-verl0.6-cu128-cudnn9.8-torch2.8.0-fa2.7.4
ENV OPD_ROOT=/root/opd-run RUNBOOK=/opt/runbook
COPY . /opt/runbook
RUN pip install "vllm==0.10.2" "transformers>=4.55.2,<4.57" "datasets>=3.0,<4.1" math-verify mathruler "flashinfer-python==0.3.0" \
      "huggingface_hub>=0.34" modelscope hf-transfer pytest "nvidia-cuda-nvcc-cu12==12.9.86" && \
    python -m venv --system-site-packages $OPD_ROOT/venv
WORKDIR /opt/runbook
CMD ["bash"]
