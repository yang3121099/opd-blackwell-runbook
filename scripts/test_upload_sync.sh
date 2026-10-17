# upload_sync.sh v2.4 path test (bash scripts/test_upload_sync.sh) (temp tree, stub remote.py, real to_bf16.py; uploads nothing, kills nothing)
set -u; V=${VENV:-${OPD_ROOT:-$HOME/opd-run}/venv}; T=$(mktemp -d); SRC=${1:-$(dirname "$0")}
mkdir -p $T/rb/scripts $T/root; ln -s $V $T/root/venv
cp $SRC/env.sh $SRC/upload_sync.sh $SRC/to_bf16.py $SRC/metrics_jsonl.py $T/rb/scripts/
cat > $T/rb/scripts/remote.py <<'PY'
import sys, os
open(os.environ["CALLS"], "a").write(" ".join(sys.argv[1:]) + "\n"); sys.exit(0)
PY
export OPD_ROOT=$T/root RUNBOOK=$T/rb CALLS=$T/calls HF_TOKEN=x HF_REPO=x; touch $CALLS
R=$T/root/runs/E10b-PX; H=$R/ckpt/global_step_150/actor/huggingface; H100=$R/ckpt/global_step_100/actor/huggingface; mkdir -p $H $H100 $R/ckpt/global_step_25/actor
source $V/bin/activate
python - $H <<'PY'
import sys, json, torch
from safetensors.torch import save_file
save_file({"w": torch.randn(4, 4, dtype=torch.float32)}, sys.argv[1] + "/model.safetensors", metadata={"format": "pt"})
json.dump({"torch_dtype": "float32"}, open(sys.argv[1] + "/config.json", "w"))
PY
cp $H/model.safetensors $H/config.json $H100/; touch -d '-5 min' $H/model.safetensors $H100/model.safetensors; echo x > $R/ckpt/global_step_25/actor/model_world_size_1_rank_0.pt; echo x > $R/ckpt/global_step_25/actor/optim_world_size_1_rank_0.pt
echo 150 > $R/ckpt/latest_checkpointed_iteration.txt; printf 'task: E10b-PX\nfinished: now\n' > $R/config.yaml; echo "step:150 - x" > $R/train.log; touch $R/.done
UPLOAD_WEIGHTS=150 timeout 120 bash $T/rb/scripts/upload_sync.sh $R > $T/out.log 2>&1
cat $T/out.log | tail -4
python - $R <<'PY'
import sys, os, torch
from safetensors.torch import load_file
R = sys.argv[1]; b = R + "/ckpt/global_step_150/actor/huggingface_bf16"
ok = [os.path.exists(b + "/model.safetensors") and load_file(b + "/model.safetensors")["w"].dtype == torch.bfloat16,
      not os.path.exists(R + "/ckpt/global_step_150/actor/huggingface/model.safetensors"),
      os.path.exists(R + "/ckpt/global_step_150/actor/huggingface/config.json"),
      not os.path.exists(R + "/ckpt/global_step_25/actor/model_world_size_1_rank_0.pt"),
      not os.path.exists(R + "/ckpt/global_step_150/.bf16"),
      not os.path.exists(R + "/ckpt/global_step_100/actor/huggingface/model.safetensors") and os.path.exists(R + "/ckpt/global_step_100/.not_uploaded"),
      not os.path.exists(R + "/ckpt/global_step_100/actor/huggingface_bf16")]
print("UPLOAD_SYNC_TEST", "PASS" if all(ok) else "FAIL", ok)
PY
echo "uploads:"; cut -d' ' -f1,3 $CALLS | sort | uniq -c | head; rm -rf $T
