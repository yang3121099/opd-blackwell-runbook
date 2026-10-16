#!/usr/bin/env bash
# Student (Qwen3-0.6B, Qwen3-1.7B), teacher (Qwen3-8B) and benchmark data (MATH-500, DAPO-Math-17k). Run
# scripts/scan_cache.sh first: copies already on the machine are reused once HF_HOME / HF_HUB_CACHE / MODELSCOPE_CACHE
# point at them.
# Hugging Face route (default): downloads through the mirror https://hf-mirror.com unless HF_MIRROR=0 or HF_ENDPOINT is
# already set; a 10-second model_info check falls back to https://huggingface.co when the endpoint does not answer.
# HF_ENDPOINT is set inside this script only (it does not leak into your shell; uploads always go to huggingface.co).
# ModelScope route: USE_MODELSCOPE=1 takes the three models from ModelScope (same IDs); the two datasets still come
# from Hugging Face (mirror).
# Either way the three models' local paths are written to $OPD_ROOT/models.env (STUDENT_06B, STUDENT_17B, TEACHER),
# which env.sh sources, so every script uses these local copies. At the end it prints each item's local path and its
# weight files (all shards named in the index present); with five OK lines it builds the parquet files
# (code/train/prep_data.py) and exits 0.
# Usage: bash scripts/download_models.sh     (USE_MODELSCOPE=1 for ModelScope; HF_MIRROR=0 to skip the mirror)
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
if [ -z "${HF_ENDPOINT:-}" ] && [ "${HF_MIRROR:-1}" = 1 ]; then export HF_ENDPOINT=https://hf-mirror.com; fi
if [ -n "${HF_ENDPOINT:-}" ] && ! timeout 10 python -c "from huggingface_hub import HfApi; HfApi().model_info('Qwen/Qwen3-0.6B')" >/dev/null 2>&1; then
  echo "HF_ENDPOINT $HF_ENDPOINT did not answer within 10 s: falling back to https://huggingface.co"; export HF_ENDPOINT=https://huggingface.co
fi
echo "Hugging Face endpoint for this download: ${HF_ENDPOINT:-https://huggingface.co}; models from $([ "${USE_MODELSCOPE:-0}" = 1 ] && echo ModelScope || echo Hugging Face)"
declare -A VAR=([Qwen/Qwen3-0.6B]=STUDENT_06B [Qwen/Qwen3-1.7B]=STUDENT_17B [Qwen/Qwen3-8B]=TEACHER)
SUM=(); ENV=()
check() {   # <local dir> -> "<n present>/<n expected> weight files" ; exit 1 if incomplete
  python - "$1" <<'PY'
import json, os, sys
d = sys.argv[1]; ix = os.path.join(d, "model.safetensors.index.json")
want = sorted(set(json.load(open(ix))["weight_map"].values())) if os.path.exists(ix) else ["model.safetensors"]
have = [f for f in want if os.path.exists(os.path.join(d, f))]
print(f"{len(have)}/{len(want)} weight files"); sys.exit(0 if len(have) == len(want) else 1)
PY
}
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-8B; do
  c=""
  if [ "${USE_MODELSCOPE:-0}" = 1 ]; then p=$(python -c "from modelscope import snapshot_download; print(snapshot_download('$m'))" 2>>$OPD_ROOT/logs/download.log | tail -1)
  else p=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$m', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','*.jinja']))" 2>>$OPD_ROOT/logs/download.log | tail -1); fi
  if [ -n "$p" ] && [ -d "$p" ] && c=$(check "$p"); then SUM+=("OK   $m -> $p ($c)"); ENV+=("export ${VAR[$m]}=$p")
  else SUM+=("FAIL $m -> ${p:-no path} (${c:-download failed; see $OPD_ROOT/logs/download.log})"); fi
done
for d in HuggingFaceH4/MATH-500 BytedTsinghua-SIA/DAPO-Math-17k; do
  p=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$d', repo_type='dataset'))" 2>>$OPD_ROOT/logs/download.log | tail -1)
  if [ -n "$p" ] && [ -n "$(find "$p" -name '*.parquet' -o -name '*.jsonl' 2>/dev/null | head -1)" ]; then SUM+=("OK   $d -> $p ($(find -L "$p" -type f \( -name '*.parquet' -o -name '*.jsonl' \) | wc -l) data files)")
  else SUM+=("FAIL $d -> ${p:-no path}"); fi
done
if [ ${#ENV[@]} = 3 ]; then printf '%s\n' "# written by download_models.sh $(date -u +%FT%TZ); sourced by env.sh" "${ENV[@]}" > $OPD_ROOT/models.env; fi
echo "== summary"; printf '%s\n' "${SUM[@]}"
ok=$(printf '%s\n' "${SUM[@]}" | grep -c '^OK')
[ "$ok" = 5 ] || { echo "DOWNLOAD FAIL ($ok/5 OK)"; exit 1; }
# training and validation parquet files from the two datasets (moved here from install_gopd.sh in v2.2)
python $RUNBOOK/code/train/prep_data.py > $OPD_ROOT/logs/prep_data.log 2>&1 || { echo "PREP_DATA FAIL (see $OPD_ROOT/logs/prep_data.log)"; exit 1; }
echo "DOWNLOAD PASS (5/5; model paths in $OPD_ROOT/models.env; parquet files in $OPD_ROOT/data: $(ls $OPD_ROOT/data/*.parquet 2>/dev/null | wc -l))"; exit 0
