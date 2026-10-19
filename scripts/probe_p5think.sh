#!/usr/bin/env bash
# PROBE-P5THINK (runbook v2.5.7r, Q-016; reviewer A1 07:5x UTC): the gate of E10b-P5think (P5 with the student in thinking
# mode, 8K rollouts). On ONE GPU:
#   1. the templates, with the teacher's tokenizer: the thinking render ends with "<|im_start|>assistant\n" and holds no
#      <think> (so teacher prompt + a student response that opens with <think> has exactly one); the non-thinking render
#      (P5's student) ends with the empty think block, as a check that enable_thinking reaches the template
#   2. a 3-step smoke of E10b-P5think with the node's settings (no checkpoint, upload, validation; nvidia-smi every 5 s),
#      rollouts dumped (verl trainer.rollout_data_dir)
#   3. report.json: the template check, exit code, out of memory or not, peak memory, steps 2-3 mean per timing segment,
#      the projected hours for 150 steps, and of step 1's responses the share that opens with <think> and the share that
#      closes it within 8K. PASS = templates ok, exit 0, no OOM, peak < 260 GB, >= 90% of step 1's responses open with
#      <think>; to logs/<node>/probe_p5think/ (private), an event, .done
# run_arm.sh starts E10b-P5think only after a PASS. Usage: bash probe_p5think.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
T=PROBE-P5THINK; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_P5THINK $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import glob, json, os, sys
sys.path.insert(0, os.path.join(os.environ["RUNBOOK"], "scripts"))
from speed_probe import steps_of
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    r["template"] = json.load(open(os.path.join(d, "template.json")))
except (OSError, ValueError):
    r["template"] = None
try:
    vals = [float(x.split(",")[1]) for x in open(os.path.join(d, "mem.csv")) if "," in x]
    r["mem_peak_mib"] = max(vals) if vals else None
except (OSError, ValueError, IndexError):
    r["mem_peak_mib"] = None
try:
    text = open(os.path.join(d, "smoke", "train.log"), errors="ignore").read()
except OSError:
    text = ""
st = steps_of(text); r["steps"] = len(st); r["oom"] = "CUDA out of memory" in text or "OutOfMemoryError" in text
late = [s for s in st if s["i"] >= 2]
r["mean_s2plus"] = {k: round(sum(s[k] for s in late if s.get(k) is not None) / max(1, len([s for s in late if s.get(k) is not None])), 1)
                    for k in ("gen", "old_log_prob", "ref", "update_actor", "step")} if late else {}
r["projected_hours_150"] = round(150 * r["mean_s2plus"]["step"] / 3600, 1) if r["mean_s2plus"].get("step") else None
r["response_length"] = [s.get("response_length/mean") for s in st]
r["response_clip_ratio"] = [s.get("response_length/clip_ratio") for s in st]
try:
    r["rollouts_step1"] = json.load(open(os.path.join(d, "rollouts.json")))
except (OSError, ValueError):
    r["rollouts_step1"] = None
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/template.json $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_p5think >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_p5think_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PTH_WAIT_STEP:-10}
done
step "start gpu=$G FIXED_MICRO_BSZ=${FIXED_MICRO_BSZ:-} VENV=${VENV:-} GC=${GC:-}"
# 1. the templates with the teacher's tokenizer (CPU)
${PTH_PY:-$OPD_ROOT/venv/bin/python} - "$TEACHER" "$D/template.json" >> $LOG 2>&1 <<'PY'
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
m = [{"role": "user", "content": "1+1?"}]
th = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=True)
nt = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
r = {"thinking_tail": th[-40:], "non_thinking_tail": nt[-40:],
     "ok": th.endswith("<|im_start|>assistant\n") and "<think>" not in th and nt.rstrip().endswith("</think>")}
json.dump(r, open(sys.argv[2], "w"))
PY
grep -q '"ok": true' $D/template.json 2> /dev/null || finish FAIL "template check failed (see template.json / probe.log)"
step "templates ok"
# 2. the 3-step smoke, rollouts dumped
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke $D/rollouts
SMOKE=1 SMOKE_STEPS=${PTH_STEPS:-3} PROBE_TASK=$T-smoke ROLLOUT_DUMP=$D/rollouts bash $RUNBOOK/scripts/run_arm.sh $G E10b-P5think > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
python3 - "$D" > $D/rollouts.json 2>> $LOG <<'PY'
import glob, json, os, sys
d = sys.argv[1]; fs = sorted(glob.glob(os.path.join(d, "rollouts", "*.jsonl")), key=lambda p: int(os.path.basename(p).split(".")[0]))
outs = [json.loads(x).get("output", "") for x in open(fs[0])] if fs else []
n = len(outs)
print(json.dumps({"step_file": os.path.basename(fs[0]) if fs else None, "responses": n,
                  "open_think": round(sum(o.lstrip().startswith("<think>") for o in outs) / n, 4) if n else None,
                  "close_think": round(sum("</think>" in o for o in outs) / n, 4) if n else None}))
PY
step "rollouts: $(cat $D/rollouts.json 2> /dev/null)"
peak=$(cut -d, -f2 $D/mem.csv 2> /dev/null | sort -n | tail -1)
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
[ "${peak:-0}" -lt ${PTH_PEAK_MIB:-266240} ] || finish FAIL "peak ${peak} MiB >= 260 GB"
python3 -c "import json, sys; r = json.load(open(sys.argv[1])); sys.exit(0 if (r.get('open_think') or 0) >= 0.9 else 1)" $D/rollouts.json 2> /dev/null \
  || finish FAIL "step 1's responses do not open with <think>: $(cat $D/rollouts.json 2> /dev/null)"
finish PASS "${PTH_STEPS:-3}-step smoke ok, peak ${peak} MiB, $(cat $D/rollouts.json)"
