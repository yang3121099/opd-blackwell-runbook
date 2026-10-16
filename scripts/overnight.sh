#!/usr/bin/env bash
# scripts/overnight.sh v3 (2026-10-16) — one unattended run on a Blackwell node (B200 / B300):
#   pull -> model downloads start in the background -> cu128 env -> data -> models.env ->
#   vLLM probe with three tiers: A) cu128 as is  B) + CUDA 12.9 ptxas for Triton (sm_103)  C) CUDA 13 env (install_cu130.sh)
#   -> verify.sh (G2 / VLLM / SMOKE must PASS) -> verify.log uploaded -> heartbeat -> wait for 1.7B -> run_tasks
# Needs in the launching shell: OPD_ROOT (default $HOME/opd-run), HF_TOKEN, HF_REPO. Every step writes "STEP n ..." to $OPD_ROOT/logs/overnight.log.
# Added to the runbook in v2.3 (2026-10-17): env.sh sources $OPD_ROOT/triton.env (tier B writes it), so scripts started
# later, including a restarted run_tasks.sh, use the same ptxas. Launch: setsid nohup bash scripts/overnight.sh > /dev/null 2>&1 &
set -o pipefail
RB=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OPD_ROOT=${OPD_ROOT:-$HOME/opd-run}
export PATH=$HOME/.local/bin:$PATH
mkdir -p $OPD_ROOT/logs $OPD_ROOT/models $OPD_ROOT/data
LOG=$OPD_ROOT/logs/overnight.log
step(){ echo "[$(date -u +%FT%TZ)] STEP $1 $2" | tee -a $LOG; }
upload_logs(){ if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_REPO:-}" ] && [ -f $OPD_ROOT/venv/bin/activate ]; then (cd $RB && source $OPD_ROOT/venv/bin/activate && python scripts/remote.py put $OPD_ROOT/logs logs/$(hostname) >>$LOG 2>&1); fi; }
fail(){ step "$1" "FAIL: $2"; upload_logs; exit 1; }

# ---------- helpers ----------
complete_dir(){ python3 - "$1" <<'PY'
import json,sys,os
d=sys.argv[1]
if not os.path.isdir(d): sys.exit(1)
idx=os.path.join(d,'model.safetensors.index.json')
need=set(json.load(open(idx))['weight_map'].values()) if os.path.exists(idx) else {'model.safetensors'}
ok=all(os.path.exists(os.path.join(d,f)) and os.path.getsize(os.path.join(d,f))>1000000 for f in need) and os.path.exists(os.path.join(d,'config.json'))
sys.exit(0 if ok else 1)
PY
}
find_model(){ m=$1
  complete_dir $OPD_ROOT/models/$m && { echo $OPD_ROOT/models/$m; return 0; }
  for root in ${HF_HUB_CACHE:-} ${HF_HOME:+$HF_HOME/hub} $HOME/.cache/huggingface/hub $OPD_ROOT/hf_home/hub; do
    [ -d "$root" ] || continue
    for s in $root/models--Qwen--$m/snapshots/*/; do complete_dir "$s" && { echo "${s%/}"; return 0; }; done
  done
  return 1
}
ensure_aria2(){ command -v aria2c >/dev/null && return 0
  apt-get install -y aria2 >>$LOG 2>&1 || sudo -n apt-get install -y aria2 >>$LOG 2>&1 || conda install -y -c conda-forge aria2 >>$LOG 2>&1 || \
  ( mkdir -p $HOME/.local/bin && cd /tmp && curl -sL -o aria2.tgz https://github.com/P3TERX/aria2-static-builds/releases/download/v1.36.0/aria2-1.36.0-static-linux-amd64.tar.gz && tar xzf aria2.tgz && cp aria2-1.36.0-static-linux-amd64/aria2c $HOME/.local/bin/ && chmod +x $HOME/.local/bin/aria2c ) >>$LOG 2>&1
  command -v aria2c >/dev/null
}
list_files(){ m=$1
  for api in https://hf-mirror.com https://huggingface.co; do
    curl -sL --max-time 60 $api/api/models/Qwen/$m | python3 -c 'import sys,json; d=json.load(sys.stdin); fs=[s["rfilename"] for s in d["siblings"] if not s["rfilename"].startswith(".")]; assert fs; print("\n".join(fs))' 2>/dev/null && return 0
  done; return 1
}
get_model(){ m=$1; d=$OPD_ROOT/models/$m; mkdir -p $d
  fs=$(list_files $m) || return 1
  if ensure_aria2; then
    ( cd $d && echo "$fs" | while read -r f; do printf '%s\t%s\n  out=%s\n' "https://hf-mirror.com/Qwen/$m/resolve/main/$f" "https://www.modelscope.cn/models/Qwen/$m/resolve/master/$f" "$f"; done > files.txt \
      && aria2c -x 16 -s 16 -j 3 -k 1M --continue=true --max-tries=0 --retry-wait=5 --summary-interval=60 -i files.txt >>$OPD_ROOT/logs/aria2_$m.log 2>&1 )
  fi
  complete_dir $d && return 0
  for ep in https://hf-mirror.com https://huggingface.co; do
    ( cd $RB && source $OPD_ROOT/venv/bin/activate && HF_ENDPOINT=$ep HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ENABLE_HF_TRANSFER=0 \
      python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/$m', local_dir='$d', allow_patterns=['*.json','*.safetensors','*.txt','tokenizer*','*.jinja'])" ) >>$OPD_ROOT/logs/hfdl_$m.log 2>&1
    complete_dir $d && return 0
  done
  return 1
}
bg_get(){ m=$1; ( get_model $m && touch $OPD_ROOT/models/$m/.complete || echo "$m download FAILED" >> $LOG ) >>$LOG 2>&1 & }
get_datasets(){ for ep in https://hf-mirror.com https://hf-mirror.com https://huggingface.co; do
    ( cd $RB && source $OPD_ROOT/venv/bin/activate && HF_ENDPOINT=$ep HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ENABLE_HF_TRANSFER=0 \
      python -c "from huggingface_hub import snapshot_download as s; [s(d, repo_type='dataset') for d in ['HuggingFaceH4/MATH-500','BytedTsinghua-SIA/DAPO-Math-17k']]" ) >>$LOG 2>&1 && return 0
    sleep 20; done; return 1
}
vllm_probe(){ tag=$1   # vLLM engine start + 4 generations on GPU 0, about 2-5 min; PASS marker VLLM_OUT
  ( cd $RB && source scripts/env.sh && source $OPD_ROOT/venv/bin/activate && CUDA_VISIBLE_DEVICES=0 timeout 1200 python scripts/vllm_check.py ) > $OPD_ROOT/logs/vllm_probe_$tag.log 2>&1
  grep -q "VLLM_OUT" $OPD_ROOT/logs/vllm_probe_$tag.log
}
probe_err(){ grep -oE "PTXASError|Internal Triton PTX codegen error|no kernel image|CUDA error[^\n]{0,60}|Error: [^\n]{0,80}" $OPD_ROOT/logs/vllm_probe_$1.log | sort -u | head -3 | tr '\n' ';'; }

# ---------- 1. code ----------
cd $RB || fail 0 "runbook dir missing: $RB"
if git pull --ff-only >>$LOG 2>&1; then step 1 "git pull OK ($(git rev-parse --short HEAD))"; else step 1 "git pull FAILED, continuing with local copy ($(git rev-parse --short HEAD))"; fi
# ---------- 2. model downloads start now, in the background (no venv needed) ----------
P17=$(find_model Qwen3-1.7B) || { P17=$OPD_ROOT/models/Qwen3-1.7B; bg_get Qwen3-1.7B; step 2 "Qwen3-1.7B downloading in background -> $P17"; }
P06=$(find_model Qwen3-0.6B) || { P06=$OPD_ROOT/models/Qwen3-0.6B; bg_get Qwen3-0.6B; step 2 "Qwen3-0.6B downloading in background -> $P06"; }
P8=$(find_model Qwen3-8B)    || { P8=$OPD_ROOT/models/Qwen3-8B;    bg_get Qwen3-8B;    step 2 "Qwen3-8B downloading/resuming in background -> $P8"; }
step 2 "model locations: 0.6B=$P06 1.7B=$P17 8B=$P8"
# ---------- 3. cu128 environment (the validated stack) ----------
if [ ! -f $OPD_ROOT/venv/bin/activate ]; then bash scripts/install_uv.sh >>$LOG 2>&1 || fail 3 "install_uv.sh (see logs/overnight.log)"; fi
source $OPD_ROOT/venv/bin/activate || fail 3 "venv activate"
step 3 "env OK ($(python -c 'import torch,vllm;print("torch",torch.__version__,"cuda",torch.version.cuda,"vllm",vllm.__version__)' 2>/dev/null))"
# ---------- 4. data ----------
get_datasets || fail 4 "datasets (MATH-500 / DAPO-17k) after 3 attempts"
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120 python code/train/prep_data.py > $OPD_ROOT/logs/prep_data.log 2>&1 || HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python code/train/prep_data.py >> $OPD_ROOT/logs/prep_data.log 2>&1 || fail 4 "prep_data.py (see logs/prep_data.log)"
step 4 "data OK: $(ls $OPD_ROOT/data/*.parquet 2>/dev/null | wc -l) parquet files"
# ---------- 5. wait for the two models the smoke needs; write models.env ----------
for i in $(seq 1 180); do complete_dir $P8 && complete_dir $P06 && break; sleep 60; done
complete_dir $P8 || fail 5 "Qwen3-8B incomplete after 3 h (see logs/aria2_Qwen3-8B.log)"; complete_dir $P06 || fail 5 "Qwen3-0.6B incomplete after 3 h (see logs/aria2_Qwen3-0.6B.log)"
printf 'STUDENT_06B=%s\nSTUDENT_17B=%s\nTEACHER=%s\n' "$P06" "$P17" "$P8" > $OPD_ROOT/models.env
step 5 "models.env: $(tr '\n' ' ' < $OPD_ROOT/models.env)"
# ---------- 6. vLLM probe, three tiers ----------
TIER=""
if vllm_probe A; then TIER="A: cu128 as is"; else
  step 6 "tier A (cu128) vLLM probe FAILED: $(probe_err A)"
  uv pip install -q "nvidia-cuda-nvcc-cu12==12.9.86" >>$LOG 2>&1 || pip install -q "nvidia-cuda-nvcc-cu12==12.9.86" >>$LOG 2>&1
  PTX=$(find $OPD_ROOT/venv/ -path '*cuda_nvcc/bin/ptxas' 2>/dev/null | head -1)
  if [ -n "$PTX" ]; then
    export TRITON_PTXAS_PATH=$PTX; rm -rf $HOME/.triton/cache; echo "export TRITON_PTXAS_PATH=$PTX" > $OPD_ROOT/triton.env
    if vllm_probe B; then TIER="B: cu128 + ptxas $($PTX --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')"; else step 6 "tier B (ptxas 12.9) vLLM probe FAILED: $(probe_err B)"; unset TRITON_PTXAS_PATH; rm -f $OPD_ROOT/triton.env; fi
  else step 6 "tier B skipped: ptxas wheel not installed"; fi
fi
if [ -z "$TIER" ]; then
  NV=$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\.[0-9]*.*/\1/p'); [ -z "$NV" ] && [ -x /usr/local/cuda/bin/nvcc ] && { export PATH=/usr/local/cuda/bin:$PATH; NV=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.[0-9]*.*/\1/p'); }
  if [ "${NV:-0}" -ge 13 ]; then
    step 6 "tier C: nvcc $NV found, building the CUDA 13 environment (install_cu130.sh, about 1 h; cu128 venv kept as venv-cu128)"
    bash scripts/install_cu130.sh >>$LOG 2>&1 || fail 6 "install_cu130.sh (see logs/overnight.log)"
    source $OPD_ROOT/venv/bin/activate
    if vllm_probe C; then TIER="C: cu130 ($(python -c 'import torch,vllm;print("torch",torch.__version__,"vllm",vllm.__version__)'))"; else fail 6 "tier C (cu130) vLLM probe FAILED: $(probe_err C) — send logs/vllm_probe_*.log back"; fi
  else fail 6 "tiers A and B failed and nvcc ${NV:-missing} < 13: the CUDA 13 route needs the NGC container (README 2b) — send logs/vllm_probe_*.log back"; fi
fi
step 6 "vLLM OK via tier $TIER"
# ---------- 7. verify (G2 tests, GPU, vLLM, 3-step smoke, upload path) ----------
for i in $(seq 1 30); do complete_dir $P17 && break; sleep 60; done
bash scripts/verify.sh > $OPD_ROOT/logs/verify.log 2>&1
if grep -q "G2 PASS" $OPD_ROOT/logs/verify.log && grep -q "VLLM PASS" $OPD_ROOT/logs/verify.log && grep -q "SMOKE PASS" $OPD_ROOT/logs/verify.log; then step 7 "verify PASS ($(grep -oE '(G2|VLLM|SMOKE|UPLOAD_CHECK)[A-Z _]*(PASS|FAIL)' $OPD_ROOT/logs/verify.log | tr '\n' ' '))"; else fail 7 "verify (see logs/verify.log: $(grep -oE '(G2|VLLM|SMOKE)[A-Z _]*(PASS|FAIL)' $OPD_ROOT/logs/verify.log | tr '\n' ' '))"; fi
if [ -z "${HF_TOKEN:-}" ] || [ -z "${HF_REPO:-}" ]; then fail 8 "HF_TOKEN / HF_REPO not exported in the launching shell"; fi
python scripts/remote.py put $OPD_ROOT/logs/verify.log logs/$(hostname)/verify.log >>$LOG 2>&1 && step 8 "verify.log uploaded to $HF_REPO" || step 8 "verify.log upload FAILED (continuing)"
# ---------- 9. heartbeat now, tasks once 1.7B is complete ----------
setsid nohup bash scripts/heartbeat.sh > $OPD_ROOT/logs/heartbeat.log 2>&1 &
step 9 "heartbeat started (status/ in $HF_REPO within 10 min)"
for i in $(seq 1 360); do complete_dir $P17 && break; sleep 60; done
complete_dir $P17 || fail 10 "Qwen3-1.7B still incomplete after 6 h (see logs/aria2_Qwen3-1.7B.log)"
setsid nohup bash scripts/run_tasks.sh > $OPD_ROOT/logs/run_tasks.log 2>&1 &
step 10 "run_tasks started with STUDENT_17B=$P17 (tier $TIER); first task log appears in $OPD_ROOT/logs/ within 5 min"
