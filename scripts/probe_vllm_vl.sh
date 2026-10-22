#!/usr/bin/env bash
# PROBE-VLLM-VL (runbook v2.5.9y1; Q-051, ops-lead's A1 10-22): the GPU gate of venv-vl, about a minute on ONE GPU. vLLM 0.11.0 from
# $OPD_ROOT/venv-vl loads Qwen3-VL-2B-Instruct ($OPD_ROOT/models, its .fetch_ok) and answers one Geometry3K image question (the
# first row of its test split, from data/hf/hiyouga__geometry3k, its .fetch_ok) through the model's own processor chat template
# with the image (multi_modal_data), greedy, 512 tokens. report.json: vLLM / torch versions, the GPU's compute capability, load
# and generation seconds, the peak GPU memory (nvidia-smi every 2 s), the first 300 characters of the answer and its finish
# reason. PASS = exit 0, a non-empty answer, no out of memory. To logs/<node>/probe_vllm_vl/ (private), an event, .done.
# PROBE-VLM (Q-051 b / c) waits for it. Usage: bash probe_vllm_vl.sh <gpu>
# [v2.5.9z28] (the reviewer 10-22: E15's teacher is Qwen3-VL-4B-Thinking now) every model of PVV_MODELS in turn (default the student
# Qwen3-VL-2B-Instruct, then the teacher Qwen3-VL-4B-Thinking), each in its own process: gen_<model>.json, the report's "models"; the
# report's top-level fields stay the first model's. PASS = each model answers.
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-VLLM-VL; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -f $D/mem.csv   # [v2.5.9z11] this run's memory samples only (a rerun keeps the directory: PROBE-STOPMERGE's 4th peak was the 2nd run's)
step() { echo "PROBE_VLLM_VL $(date -u +%T) $*" | tee -a $LOG; }
PY=${PVV_PY:-$OPD_ROOT/venv-vl/bin/python}; DS=$OPD_ROOT/data/hf/hiyouga__geometry3k
MODELS=${PVV_MODELS:-Qwen3-VL-2B-Instruct Qwen3-VL-4B-Thinking}; M0=${MODELS%% *}   # [v2.5.9z28] the student, then the teacher
finish() {   # $1 = PASS / FAIL, $2 = reason
  PVV_MODEL_LIST="$MODELS" python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    r.update(json.load(open(os.path.join(d, "gen.json"))))
except (OSError, ValueError):
    pass
r["models"] = {}   # [v2.5.9z28] each model's check
for m in os.environ.get("PVV_MODEL_LIST", "").split():
    try:
        r["models"][m] = json.load(open(os.path.join(d, f"gen_{m}.json")))
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
for m in $MODELS; do [ -f $OPD_ROOT/models/$m/.fetch_ok ] || { echo "WAIT_INPUTS $T: $m not in (no .fetch_ok)"; exit 1; }; done   # [v2.5.9z28] each model
[ -f $DS/.fetch_ok ] || { echo "WAIT_INPUTS $T: Geometry3K not in (no .fetch_ok)"; exit 1; }
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PVV_WAIT_STEP:-10}
done
rm -f $D/gen.json $D/gen_*.json; : > $D/gen.out   # [v2.5.9z28] this run's answers only
step "start gpu=$G models=$MODELS"
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 2; done ) &
MP=$!; trap 'kill $MP 2> /dev/null' EXIT   # [v2.5.9z3] the sampler never outlives the probe
# [v2.5.9z20] the check runs from a file, its body under __main__: vLLM 0.11's engine core is a spawned process that re-imports the
# main module by its path, and from stdin that path is '<stdin>' (10-22 01:36 UTC: FileNotFoundError, "Engine core initialization failed")
cat > $D/gen_check.py <<'PY'
import glob, io, json, os, sys, time


def main():
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


if __name__ == "__main__":
    main()
PY
for m in $MODELS; do   # [v2.5.9z28] one process per model (vLLM's memory freed in between)
  step "model $m"
  CUDA_VISIBLE_DEVICES=$G timeout ${PVV_TIMEOUT:-900} $PY $D/gen_check.py "$OPD_ROOT/models/$m" "$DS" "$D/gen_$m.json" > $D/gen_$m.out 2>&1
  rc=$?; cat $D/gen_$m.out >> $D/gen.out
  python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
  [ $m = $M0 ] && cp $D/gen_$m.json $D/gen.json 2> /dev/null
  grep -q "CUDA out of memory\|OutOfMemoryError" $D/gen_$m.out 2> /dev/null && { kill $MP 2> /dev/null; finish FAIL "$m: out of memory"; }
  [ $rc = 0 ] || { kill $MP 2> /dev/null; finish FAIL "$m exited $rc: $(tail -c 300 $D/gen_$m.out | tr '\n' ' ')"; }
  python3 -c "import json, sys; a = json.load(open(sys.argv[1])); sys.exit(0 if a.get('answer', '').strip() else 1)" $D/gen_$m.json 2> /dev/null || { kill $MP 2> /dev/null; finish FAIL "$m: an empty answer"; }
done
kill $MP 2> /dev/null
SUM=$(python3 - $D $MODELS <<'PY'
import json, sys
d, ms = sys.argv[1], sys.argv[2:]
g = {m: json.load(open(f"{d}/gen_{m}.json")) for m in ms}
a = g[ms[0]]
print(a["vllm"], "on sm", a["capability"] + ";", "; ".join(f"{m}: load {b['load_s']} s, gen {b['gen_s']} s, {b['answer_tokens']} tokens, {b['finish_reason']}" for m, b in g.items()))
PY
)
finish PASS "vLLM $SUM"
