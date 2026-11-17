#!/usr/bin/env bash
# Blackwell stack with uv: torch 2.8.0 (CUDA 12.8 wheels) + vLLM 0.10.2 + G-OPD (verl 0.6.1 fork) at a fixed commit
# + the lab patches; then models and data. UNTESTED on Blackwell until section 8 of README.md is filled in.
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd $OPD_ROOT
source "$RUNBOOK/scripts/pypi_index.sh"; pypi_index_export || true   # [v3.5] the fastest index that answers (mainland mirrors before a proxied PyPI)
command -v uv >/dev/null || pip install -q uv
[ -d venv ] || uv venv -p 3.10 venv
source venv/bin/activate
# [v3.9] (Runming 11-18: "the install is this slow - keep the environment on the long-term disk and reuse it") an earlier install of
# ours on this machine's disks (history_scan.py -> OLD_BASE_CANDS) that holds the pinned stack (torch 2.8.0, vllm 0.10.2, the same
# Python minor version) seeds this venv: its site-packages are COPIED to the local disk (the node does not run from a shared disk),
# its console scripts are copied with this venv's interpreter on their first line, and its own G-OPD link is dropped
# (install_gopd.sh installs this node's checkout). `uv pip install` below then only fetches what is missing. 11-17/18: three
# installs through a mainland proxy failed (index timeout, a wheel's download timed out after 27 min), and a fresh resolve takes
# the newest version of every unpinned dependency, not the validated ones.
seed_venv() {
  local dst mv r v sp f n
  dst=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || return 1
  mv=$(basename "$(dirname "$dst")")
  ls -d $dst/torch-2.8.0*.dist-info $dst/vllm-0.10.2*.dist-info > /dev/null 2>&1 && return 1   # the stack is here already
  for r in ${OPD_STORE:-} $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do   # [v3.10] first the long-term store's environment pack (store_sync.sh)
    f=$r/envpack/site-packages-$mv-torch2.8.0-vllm0.10.2.tar; [ -f $f ] || continue
    [ ! -f $f.sha256 ] || [ "$(sha256sum $f | cut -c1-64)" = "$(cat $f.sha256)" ] || { echo "SEED_VENV $(basename $f): sha256 differs, not used"; continue; }
    echo "SEED_VENV $(date -u +%T) unpacking the environment pack of the long-term store ($(du -h $f | cut -f1))"
    tar -xf $f -C $dst && { echo "SEED_VENV $(date -u +%T) done: $(ls $dst | wc -l) entries"; : > venv/.seeded_from_store; return 0; }
  done
  for r in $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do for v in venv venv-fa; do
    sp=$r/$v/lib/$mv/site-packages
    ls -d $sp/torch-2.8.0*.dist-info $sp/vllm-0.10.2*.dist-info > /dev/null 2>&1 || continue
    echo "SEED_VENV $(date -u +%T) copying the pinned stack's packages from an earlier install on this machine ($v, $mv)"
    ( cd $sp && tar -cf - . ) | ( cd $dst && tar -xf - ) || { echo "SEED_VENV copy failed"; continue; }
    rm -rf $dst/__editable__*verl* $dst/verl-*.dist-info $dst/__pycache__/__editable__*verl* 2> /dev/null
    n=0; for f in $r/$v/bin/*; do case $(basename $f) in python*|activate*|pip*|uv*|Activate*|deactivate*) continue;; esac
      [ -f $f ] && [ ! -e venv/bin/$(basename $f) ] && head -1 $f 2> /dev/null | grep -q '^#!.*python' || continue
      { echo "#!$OPD_ROOT/venv/bin/python"; tail -n +2 $f; } > venv/bin/$(basename $f) && chmod +x venv/bin/$(basename $f) && n=$((n + 1)); done
    echo "SEED_VENV $(date -u +%T) done: $(ls $dst | wc -l) entries, $n console scripts; torch $(python -c 'import torch; print(torch.__version__)' 2>&1 | tail -1)"
    : > venv/.seeded_from_history; return 0
  done; done; return 1; }
seed_venv || true
# [v3.9] three tries: a download that timed out through the proxy goes on from uv's cache
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT_INSTALL:-600}
stack() { uv pip install "torch==2.8.0" "vllm==0.10.2" "transformers>=4.55.2,<4.57" accelerate "datasets>=3.0,<4.1" peft hf-transfer \
  "numpy<2.0.0" "pyarrow>=15.0.0" pandas "ray[default]" codetiming hydra-core pylatexenc dill pybind11 \
  "tensordict>=0.8,<0.10" torchdata liger-kernel mathruler pytest math-verify safetensors "flashinfer-python==0.3.0" \
  "huggingface_hub>=0.34" modelscope; }
stack || { echo "INSTALL retry 2"; sleep 20; stack; } || { echo "INSTALL retry 3"; sleep 60; stack; }
# B300 (compute capability 10.3, v2.3): put the CUDA 12.9 ptxas in place for the second fix (README section 2b);
# it is used only after $OPD_ROOT/triton.env points TRITON_PTXAS_PATH at it
if [ "$(gpu_cap)" = 10.3 ]; then uv pip install "$B300_PTXAS"; fi
bash $RUNBOOK/scripts/install_gopd.sh
