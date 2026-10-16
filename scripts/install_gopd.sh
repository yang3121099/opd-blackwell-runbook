#!/usr/bin/env bash
# G-OPD at the fixed commit, installed editable without dependencies; the lab patches; models and data.
set -euo pipefail
source "$(dirname "$0")/env.sh"; cd $OPD_ROOT; source venv/bin/activate
[ -d G-OPD ] || git clone https://github.com/RUCBM/G-OPD.git
(cd G-OPD && git fetch -q origin $GOPD_COMMIT 2>/dev/null || true; git checkout -q $GOPD_COMMIT)
# install into the active environment: uv venvs have no pip of their own (a bare `pip` would hit the system Python)
(cd G-OPD/verl && { command -v uv >/dev/null && uv pip install --no-deps -e . || python -m pip install --no-deps -e .; })
python $RUNBOOK/code/patches/apply_patches.py $OPD_ROOT/G-OPD/verl
python - <<'PY'
import torch, vllm, transformers, verl
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch", torch.cuda.get_arch_list() if torch.cuda.is_available() else "no GPU")
print("vllm", vllm.__version__, "transformers", transformers.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__, "(use_remove_padding on)")
except Exception:
    print("flash_attn not installed: SDPA, use_remove_padding off")
PY
echo "INSTALL DONE. Next (README section 2a): bash scripts/scan_cache.sh, then bash scripts/download_models.sh (v2.2: the download is a separate step)"
echo INSTALL_DONE
