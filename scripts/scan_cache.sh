#!/usr/bin/env bash
# Read-only: find model and dataset copies already on this machine before downloading anything. Prints the cache
# variables; the Qwen3 / MATH-500 / DAPO entries of the usual caches; any other Hugging Face (models--Qwen--Qwen3-*) or
# ModelScope (.../Qwen/Qwen3-*) copies under /data /mnt /home /root /workspace, every other top-level mount point (from df) and any directories you pass; for each
# model copy whether its weights are complete (every shard named in model.safetensors.index.json present: 0.6B has 1
# weight file, 1.7B 2, 8B 5); and at the end the export lines that make download_models.sh reuse what was found.
# Writes nothing. Usage: bash scripts/scan_cache.sh [extra dirs ...]
set -uo pipefail
echo "== cache variables"
for v in HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE MODELSCOPE_CACHE HF_ENDPOINT; do
  echo "  $v=${!v:-<unset>}"; done
complete() {   # <model dir> -> "k/n weight files" (follows the HF snapshot symlinks)
  python3 - "$1" <<'PY'
import json, os, sys
d = sys.argv[1]; ix = os.path.join(d, "model.safetensors.index.json")
try:
    want = sorted(set(json.load(open(ix))["weight_map"].values())) if os.path.exists(ix) else ["model.safetensors"]
except Exception:
    want = ["model.safetensors"]
have = [f for f in want if os.path.exists(os.path.join(d, f))]
print(f"{len(have)}/{len(want)} weight files, {'complete' if len(have) == len(want) else 'INCOMPLETE'}")
PY
}
report() {   # <HF models--Org--Name dir | ModelScope Org/Name dir>
  local e=$1
  case "$(basename "$e")" in
    models--*) for s in "$e"/snapshots/*; do [ -d "$s" ] && echo "  $s  [$(complete "$s")]"; done;;
    datasets--*) echo "  $e  [dataset, $(find -L "$e" -type f \( -name '*.parquet' -o -name '*.jsonl' \) 2>/dev/null | wc -l) data files]";;
    *) echo "  $e  [$(complete "$e")]";;
  esac
}
echo "== usual caches"
C=()
for d in "${HF_HUB_CACHE:-}" "${HUGGINGFACE_HUB_CACHE:-}" "${HF_HOME:+$HF_HOME/hub}" "$HOME/.cache/huggingface/hub" \
         "${MODELSCOPE_CACHE:+$MODELSCOPE_CACHE/hub}" "${MODELSCOPE_CACHE:+$MODELSCOPE_CACHE/hub/models}" \
         "$HOME/.cache/modelscope/hub" "$HOME/.cache/modelscope/hub/models"; do
  [ -n "$d" ] && [ -d "$d" ] && C+=("$d"); done
E=()   # every model copy found (usual caches + find), for the export suggestions
for d in $(printf '%s\n' ${C[@]+"${C[@]}"} | sort -u); do
  echo "-- $d"
  for e in "$d"/models--Qwen--Qwen3-* "$d"/datasets--HuggingFaceH4--MATH-500 "$d"/datasets--BytedTsinghua-SIA--DAPO-Math-17k "$d"/Qwen/Qwen3-*; do
    [ -d "$e" ] || continue; report "$e"; case "$e" in *datasets--*) ;; *) E+=("$e");; esac; done
done
echo "== other copies (find, max depth 6, at most 2 min per root)"
MOUNTS=$(df -P 2>/dev/null | awk 'NR>1 && $6 ~ "^/[^/]+$" {print $6}' | grep -vE '^/(proc|sys|dev|run|boot|tmp|snap|var|usr|etc|opt|lib|bin|sbin)$' | sort -u | tr '\n' ' ')
FOUND=$(for r in /data /mnt /home /root /workspace $MOUNTS "$@"; do
  [ -d "$r" ] && timeout 120 find "$r" -maxdepth 6 \( -name 'models--Qwen--Qwen3-*' -o -path '*modelscope*/Qwen/Qwen3-*' \) -prune -print 2>/dev/null
done | sort -u)
if [ -n "$FOUND" ]; then while read -r e; do report "$e"; E+=("$e"); done <<< "$FOUND"; else echo "  (none)"; fi
echo "== to reuse a cache found above (pick the one whose 0.6B, 1.7B and 8B are complete), before download_models.sh:"
printf '%s\n' ${E[@]+"${E[@]}"} | while read -r e; do [ -n "$e" ] || continue
  case "$e" in
    */models--Qwen--Qwen3-*) echo "  export HF_HUB_CACHE=$(dirname "$e")";;
    *modelscope*) echo "  export MODELSCOPE_CACHE=${e%%/hub/*} USE_MODELSCOPE=1";;
  esac
done | sort -u
echo "  (HF_HUB_CACHE is the folder that contains models--Qwen--Qwen3-*; if that folder is <X>/hub, export HF_HOME=<X> works too)"
