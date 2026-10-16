#!/usr/bin/env bash
# Four checks before any task: (1) G2 unit tests of the patches, (2) accept: GPU/driver/CUDA, bf16 matmul, disk,
# vLLM start + 4 generations, (3) a 3-step P2 smoke with the 0.6B student + 8B teacher (no validation, no checkpoints),
# (4) the upload path on a sharded 1.7B checkpoint (dry run: no credentials, nothing uploaded), checked against
# reference/upload_manifest_qwen3-1.7b_bf16.json (v2.1: before, sharded checkpoints were silently not uploaded).
# Pass = every step has a finite loss, no NaN anywhere, peak memory within the card. Send the log back. About 1 hour.
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate; cd $OPD_ROOT
echo "== 1. G2 unit tests"
(cd $RUNBOOK/code/patches && python -m pytest -q -p no:cacheprovider test_opd_variants.py 2>&1 | tail -2) | tee /tmp/g2.txt
grep -q "16 passed" /tmp/g2.txt && echo "G2 PASS" || echo "G2 FAIL"
echo "== 2. accept"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo "driver CUDA: $(nvidia-smi | grep -o 'CUDA Version: [0-9.]*')"
echo "compute capability $(gpu_cap); ptxas for Triton: ${TRITON_PTXAS_PATH:-the one bundled with Triton}${TRITON_PTXAS_PATH:+ (release $(ptxas_rel $TRITON_PTXAS_PATH))}; vLLM attention: ${VLLM_ATTENTION_BACKEND:-default}, FlashInfer sampler: ${VLLM_USE_FLASHINFER_SAMPLER:-default (on)}"
python - <<'PY'
import time, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "arch list", torch.cuda.get_arch_list())
x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); torch.cuda.synchronize(); t = time.time()
for _ in range(50): x @ x
torch.cuda.synchronize(); print("bf16 matmul TFLOPS ~", round(50 * 2 * 8192**3 / (time.time() - t) / 1e12, 1))
PY
df -h $OPD_ROOT | tail -1; dd if=/dev/zero of=$OPD_ROOT/ddtest bs=1M count=2048 oflag=direct 2>&1 | tail -1; rm -f $OPD_ROOT/ddtest
CUDA_VISIBLE_DEVICES=0 python $RUNBOOK/scripts/vllm_check.py 2>&1 | grep -E "VLLM_OUT|Error" | tail -3
[ ${PIPESTATUS[0]} = 0 ] && echo "VLLM PASS" || echo "VLLM FAIL"
echo "== 3. 3-step P2 smoke: 0.6B student + 8B teacher (no validation, no checkpoints)"
t=$(date +%s)
SMOKE=1 STUDENT=$STUDENT_06B bash $RUNBOOK/scripts/run_arm.sh 0 E10b-P2 > $OPD_ROOT/logs/smoke.log 2>&1; rc=$?
grep -E "(^|[ )])step:[0-9]+ - " $OPD_ROOT/logs/smoke.log | tr ' ' '\n' | \
  grep -E '^(step:|timing_s/step:|perf/max_memory_allocated_gb:|actor/pg_loss:|actor/entropy:|response_length/mean:|opd/adv_abs_mean:)' | paste -s -d ' '
nan=$(grep -E "(^|[ )])step:[0-9]+ - " $OPD_ROOT/logs/smoke.log | grep -ciE ":(nan|inf)( |$)")
loss=$(grep -cE "(^|[ )])step:[0-9]+ - .*actor/pg_loss:-?[0-9]" $OPD_ROOT/logs/smoke.log)
if [ $rc = 0 ] && [ "$loss" -ge 3 ] && [ "$nan" = 0 ]; then echo "SMOKE PASS ($(( $(date +%s) - t ))s; 3 steps with a finite loss, no NaN)"
else echo "SMOKE FAIL (exit $rc, steps with loss $loss, lines with NaN/inf $nan)"; grep -E "Error|Traceback" $OPD_ROOT/logs/smoke.log | tail -5; fi
rm -rf $OPD_ROOT/runs/smoke-E10b-P2
echo "== 4. upload path: sharded 1.7B checkpoint, dry run (no credentials, nothing uploaded)"
t=$(date +%s); U=$OPD_ROOT/runs/smoke-upload; rm -rf $U; mkdir -p $U/ckpt/global_step_1/actor
python - $U/ckpt/global_step_1/actor/huggingface > $OPD_ROOT/logs/smoke_upload.log 2>&1 <<'PY'
import sys, torch   # the same save as verl's actor/huggingface: fp32 weights, 5 GB shards, tokenizer files, and
from transformers import AutoModelForCausalLM, AutoTokenizer   # lm_head.weight written although tied (FSDP full state dict)
import os; m = AutoModelForCausalLM.from_pretrained(os.environ["STUDENT_17B"], torch_dtype=torch.float32)
m.save_pretrained(sys.argv[1], state_dict={k: v.detach().clone() for k, v in m.state_dict().items()}, max_shard_size="5GB")
AutoTokenizer.from_pretrained(os.environ["STUDENT_17B"]).save_pretrained(sys.argv[1])
PY
echo 1 > $U/ckpt/latest_checkpointed_iteration.txt; touch -d '-5 min' $U/ckpt/global_step_1/actor/huggingface/*.safetensors
echo "finished: upload smoke" > $U/config.yaml
UPLOAD_DRY=1 timeout 900 bash $RUNBOOK/scripts/upload_sync.sh $U >> $OPD_ROOT/logs/smoke_upload.log 2>&1
grep -E "UP_DRY|BF16_FAIL|UPLOAD_DONE" $OPD_ROOT/logs/smoke_upload.log
python $RUNBOOK/scripts/check_upload.py $U/ckpt/global_step_1/.bf16 $RUNBOOK/reference/upload_manifest_qwen3-1.7b_bf16.json \
  && echo "UPLOAD_CHECK PASS ($(( $(date +%s) - t ))s)" || { echo "UPLOAD_CHECK FAIL"; grep -E "Error|Traceback" $OPD_ROOT/logs/smoke_upload.log | tail -3; }
rm -rf $U
