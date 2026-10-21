#!/usr/bin/env bash
# PROBE-VLLM-VL (runbook v2.5.9y1; Q-051, ops-lead's A1 10-22): the GPU gate of venv-vl, about a minute on ONE GPU. vLLM 0.11.0 from
# $OPD_ROOT/venv-vl loads Qwen3-VL-2B-Instruct ($OPD_ROOT/models, its .fetch_ok) and answers one Geometry3K image question (the
# first row of its test split, from data/hf/hiyouga__geometry3k, its .fetch_ok) through the model's own processor chat template
# with the image (multi_modal_data), greedy, 512 tokens. report.json: vLLM / torch versions, the GPU's compute capability, load
# and generation seconds, the peak GPU memory (nvidia-smi every 2 s), the first 300 characters of the answer and its finish
# reason. PASS = exit 0, a non-empty answer, no out of memory. To logs/<node>/probe_vllm_vl/ (private), an event, .done.
# PROBE-VLM (Q-051 b / c) waits for it. Usage: bash probe_vllm_vl.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-VLLM-VL; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_VLLM_VL $(date -u +%T) $*" | tee -a $LOG; }
PY=${PVV_PY:-$OPD_ROOT/venv-vl/bin/python}; M=$OPD_ROOT/models/Qwen3-VL-2B-Instruct; DS=$OPD_ROOT/data/hf/hiyouga__geometry3k
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    r.update(json.load(open(os.path.join(d, "gen.json"))))
except (OSError, ValueError):
    pass
try:
    vals = [float(x.split(",")[1]) for x in open(os.path.join(d, "mem.csv")) if "," in x]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/up/ 2> /dev/null; [ -f $D/gen.out ] && tail -c 100000 $D/gen.out > $D/up/gen_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_vllm_vl >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_vllm_vl_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
[ -x "$PY" ] || finish FAIL "no venv-vl (BUILD-VENV-VL first)"
# [v2.5.9z3] inputs still downloading: waiting (run_tasks launches it again later, no attempt counted, no .done)
[ -f $M/.fetch_ok ] || { echo "WAIT_INPUTS $T: Qwen3-VL-2B-Instruct not in (no .fetch_ok)"; exit 1; }
[ -f $DS/.fetch_ok ] || { echo "WAIT_INPUTS $T: Geometry3K not in (no .fetch_ok)"; exit 1; }
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PVV_WAIT_STEP:-10}
done
step "start gpu=$G model=$(basename $M)"
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 2; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT   # [v2.5.9z3] the sampler never outlives the probe
CUDA_VISIBLE_DEVICES=$G timeout ${PVV_TIMEOUT:-900} $PY - "$M" "$DS" "$D/gen.json" > $D/gen.out 2>&1 <<'PY'
import glob, io, json, os, sys, time
m, ds, out = sys.argv[1:4]
r = {}
import torch, vllm
from PIL import Image
r.update(vllm=vllm.__version__, torch=torch.__version__, capability=".".join(map(str, torch.cuda.get_device_capability(0))))
import pyarrow.parquet as pq
f = sorted(glob.glob(os.path.join(ds, "**", "*test*.parquet"), recursive=True))[0]
row = pq.read_table(f).slice(0, 1).to_pylist()[0]
img = row["images"][0] if isinstance(row.get("images"), list) else row["images"]
img = Image.open(io.BytesIO(img["bytes"] if isinstance(img, dict) else img)).convert("RGB")
q = str(row.get("problem", row.get("question", ""))).replace("<image>", "").strip()
r.update(data_file=os.path.relpath(f, ds), image_size=list(img.size))
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained(m)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
t0 = time.time()
llm = vllm.LLM(model=m, max_model_len=8192, gpu_memory_utilization=float(os.environ.get("PVV_UTIL", "0.5")), limit_mm_per_prompt={"image": 1}, seed=0)
r["load_s"] = round(time.time() - t0, 1); t1 = time.time()
o = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": img}}], vllm.SamplingParams(temperature=0, max_tokens=512))[0].outputs[0]
r.update(gen_s=round(time.time() - t1, 1), answer=o.text[:300], answer_tokens=len(o.token_ids), finish_reason=o.finish_reason, reference=str(row.get("answer", ""))[:50])
json.dump(r, open(out, "w"), indent=1); print("GEN_OK", json.dumps(r)[:500])
PY
rc=$?
kill $MP 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
grep -q "CUDA out of memory\|OutOfMemoryError" $D/gen.out 2> /dev/null && finish FAIL "out of memory"
[ $rc = 0 ] || finish FAIL "exited $rc: $(tail -c 300 $D/gen.out | tr '\n' ' ')"
python3 -c "import json, sys; a = json.load(open(sys.argv[1])); sys.exit(0 if a.get('answer', '').strip() else 1)" $D/gen.json 2> /dev/null || finish FAIL "an empty answer"
finish PASS "vLLM $(python3 -c "import json; a = json.load(open('$D/gen.json')); print(a['vllm'], 'on sm', a['capability'] + ',', 'load', a['load_s'], 's, gen', a['gen_s'], 's,', a['answer_tokens'], 'tokens,', a['finish_reason'])")"
