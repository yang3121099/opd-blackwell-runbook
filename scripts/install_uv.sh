#!/usr/bin/env bash
# Blackwell stack with uv: torch 2.8.0 (CUDA 12.8 wheels) + vLLM 0.10.2 + G-OPD (verl 0.6.1 fork) at a fixed commit
# + the lab patches; then models and data. UNTESTED on Blackwell until section 8 of README.md is filled in.
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd $OPD_ROOT
source "$RUNBOOK/scripts/pypi_index.sh"; pypi_index_export || true   # [v3.5] the fastest index that answers (mainland mirrors before a proxied PyPI)
command -v uv >/dev/null || pip install -q uv
[ -d venv ] || uv venv -p 3.10 venv
source venv/bin/activate
# [v3.9 / v3.11] (Runming 11-18: "keep the environment on the long-term disk and reuse it"; "copy it to the local disk first; if the
# copy is too slow, use it where it is") the pinned stack (torch 2.8.0, vllm 0.10.2, the same Python minor version) is taken, in
# this order, from the long-term store's environment pack, then from an earlier install of ours on this machine's disks
# (history_scan.py -> OPD_STORE / OLD_BASE_CANDS):
#   copy  its site-packages are unpacked / copied into a temporary directory on the local disk within SEED_COPY_MAX_S (300 s) and
#         then moved into this venv (nothing half-copied is ever in the venv); console scripts get this venv's interpreter;
#   link  a copy that takes longer is dropped and this venv gets one .pth line naming the earlier install's site-packages: the
#         packages are used where they are (venv/.linked_to_history). The node then depends on that disk while it runs; a later
#         `setup 2` with a larger SEED_COPY_MAX_S makes it a copy.
# Either way the earlier install's own G-OPD link is not taken (install_gopd.sh installs this node's checkout), and the network
# is asked only for what is missing. 11-17/18: three installs through a mainland proxy failed (index timeout, a wheel's
# download timed out after 27 min), and a fresh resolve takes the newest version of every unpinned dependency.
CORE="torch vllm transformers ray tensordict datasets peft hydra codetiming pylatexenc math_verify mathruler accelerate pandas pyarrow safetensors dill torchdata"
core_missing() { python - $CORE <<'PY' 2> /dev/null
import importlib.util, sys
print(" ".join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))
PY
}
seed_venv() {
  local dst mv r v sp f n tmp=$OPD_ROOT/venv/.seed_tmp to="" e
  dst=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || return 1
  mv=$(basename "$(dirname "$dst")")
  ls -d $dst/torch-2.8.0*.dist-info $dst/vllm-0.10.2*.dist-info > /dev/null 2>&1 && return 1   # the stack is here already
  [ -f venv/.linked_to_history ] && [ "${SEED_RELINK:-0}" != 1 ] && python -c 'import torch, vllm' 2> /dev/null && return 1   # linked and working
  command -v timeout > /dev/null && to="timeout ${SEED_COPY_MAX_S:-300}"
  into_venv() {   # the temporary directory's entries into site-packages (an entry of the same name gives way)
    for e in $tmp/* $tmp/.[!.]*; do [ -e "$e" ] || continue; rm -rf "$dst/$(basename "$e")"; mv "$e" "$dst/"; done
    rm -rf $tmp $dst/__editable__*verl* $dst/verl-*.dist-info $dst/__pycache__/__editable__*verl* $dst/opd_shared_packages.pth venv/.linked_to_history 2> /dev/null; }
  for r in ${OPD_STORE:-} $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do   # 1. an environment pack (one file: the fastest to read from a shared disk)
    f=$r/envpack/site-packages-$mv-torch2.8.0-vllm0.10.2.tar; [ -f $f ] || continue
    echo "SEED_VENV $(date -u +%T) unpacking the long-term store's environment pack ($(du -h $f | cut -f1)), at most ${SEED_COPY_MAX_S:-300} s"
    rm -rf $tmp; mkdir -p $tmp
    if $to tar -xf $f -C $tmp; then
      [ ! -f $f.sha256 ] || [ "$(sha256sum $f | cut -c1-64)" = "$(cat $f.sha256)" ] || { echo "SEED_VENV $(basename $f): sha256 differs, not used"; rm -rf $tmp; continue; }
      into_venv; echo "SEED_VENV $(date -u +%T) copied: $(ls $dst | wc -l) entries"; : > venv/.seeded_from_store; return 0
    fi
    echo "SEED_VENV $(date -u +%T) the pack did not unpack within ${SEED_COPY_MAX_S:-300} s"; rm -rf $tmp
  done
  for r in $(echo "${OLD_BASE_CANDS:-}" | tr ':' ' '); do for v in venv venv-fa; do   # 2. an earlier install's venv
    sp=$r/$v/lib/$mv/site-packages
    ls -d $sp/torch-2.8.0*.dist-info $sp/vllm-0.10.2*.dist-info > /dev/null 2>&1 || continue
    echo "SEED_VENV $(date -u +%T) copying the pinned stack's packages from an earlier install on this machine ($v, $mv), at most ${SEED_COPY_MAX_S:-300} s"
    rm -rf $tmp; mkdir -p $tmp
    if $to bash -c "cd '$sp' && tar -cf - . | tar -xf - -C '$tmp'"; then
      into_venv
      n=0; for f in $r/$v/bin/*; do case $(basename $f) in python*|activate*|pip*|uv*|Activate*|deactivate*) continue;; esac
        [ -f $f ] && [ ! -e venv/bin/$(basename $f) ] && head -1 $f 2> /dev/null | grep -q '^#!.*python' || continue
        { echo "#!$OPD_ROOT/venv/bin/python"; tail -n +2 $f; } > venv/bin/$(basename $f) && chmod +x venv/bin/$(basename $f) && n=$((n + 1)); done
      echo "SEED_VENV $(date -u +%T) copied: $(ls $dst | wc -l) entries, $n console scripts; torch $(python -c 'import torch; print(torch.__version__)' 2>&1 | tail -1)"
      : > venv/.seeded_from_history; return 0
    fi
    rm -rf $tmp
    echo "$sp" > $dst/opd_shared_packages.pth; echo "$sp" > venv/.linked_to_history
    echo "SEED_VENV $(date -u +%T) the copy took over ${SEED_COPY_MAX_S:-300} s: the earlier install's packages are used where they are (venv/.linked_to_history); torch $(python -c 'import torch; print(torch.__version__)' 2>&1 | tail -1)"
    return 0
  done; done; return 1; }
seed_venv || true
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT_INSTALL:-600}
stack() { uv pip install "torch==2.8.0" "vllm==0.10.2" "transformers>=4.55.2,<4.57" accelerate "datasets>=3.0,<4.1" peft hf-transfer \
  "numpy<2.0.0" "pyarrow>=15.0.0" pandas "ray[default]" codetiming hydra-core pylatexenc dill pybind11 \
  "tensordict>=0.8,<0.10" torchdata liger-kernel mathruler pytest math-verify safetensors "flashinfer-python==0.3.0" \
  "huggingface_hub>=0.34" modelscope; }
if [ -f venv/.linked_to_history ]; then
  # linked: uv does not see the linked packages and would fetch the whole stack, so only what does not import is added, without
  # dependencies (module name -> requirement)
  MISS=$(core_missing); echo "INSTALL linked to an earlier install; missing here: ${MISS:-nothing}"
  for m in $MISS; do case $m in hydra) q=hydra-core;; math_verify) q=math-verify;; *) q=$m;; esac
    uv pip install --no-deps "$q" || echo "INSTALL $q could not be added"; done
  [ -z "$(core_missing)" ] || { echo "INSTALL the linked packages do not give the stack ($(core_missing)): a fresh install"; rm -f venv/.linked_to_history $(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/opd_shared_packages.pth; stack; }
else
  # three tries (a download that timed out through the proxy goes on from uv's cache); a seeded venv whose stack imports is
  # accepted when the index cannot be reached at all
  stack || { echo "INSTALL retry 2"; sleep 20; stack; } || { echo "INSTALL retry 3"; sleep 60; stack; } || {
    { [ -f venv/.seeded_from_history ] || [ -f venv/.seeded_from_store ]; } && [ -z "$(core_missing)" ] && echo "INSTALL the index was not reached; the seeded packages give the stack: going on" || exit 1; }
fi
# B300 (compute capability 10.3, v2.3): put the CUDA 12.9 ptxas in place for the second fix (README section 2b);
# it is used only after $OPD_ROOT/triton.env points TRITON_PTXAS_PATH at it
if [ "$(gpu_cap)" = 10.3 ]; then uv pip install "$B300_PTXAS"; fi
bash $RUNBOOK/scripts/install_gopd.sh
