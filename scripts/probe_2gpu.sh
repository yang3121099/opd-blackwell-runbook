#!/usr/bin/env bash
# PROBE-2GPU (runbook [v2.5.9a]; Runming via the reviewer 06:4x UTC 10-21): the gate of the two-GPU slot ("#gpus <task> 2").
# A 3-step smoke of E12b-P5-16k on the two GPUs run_tasks.sh gave it (G = a,b; SMOKE=1: no checkpoint, upload, validation;
# nvidia-smi of both every 5 s): exit 0, no OOM, both GPUs in use (peak > 20 GB each) and under 260 GB each, train.log
# with every step and its config naming both GPUs ("gpus: a,b (n_gpus_per_node 2)"). Against the one-GPU smoke of the same arm (PROBE-E12B-16K's
# train.log) it reports, step by step and as means, the step / generation / update times, response length, clip ratio,
# score, pg_loss and grad_norm, and the speedup (one-GPU mean step time / two-GPU mean step time; also per generated
# token). PASS = it ran correctly on two GPUs; the speedup is a reading for the reviewer (worth_it = speedup >= P2G_WORTH,
# default 1.3), who decides which arms take two GPUs.
# report.json, to logs/<node>/probe_2gpu/<task> (private), an event, .done. ~1 h on two GPUs.
# Usage: bash probe_2gpu.sh <gpu>,<gpu>
set -uo pipefail
G=$1; T=PROBE-2GPU; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_2GPU $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
p = os.path.join(d, "compare.json")
if os.path.exists(p):
    r.update(json.load(open(p)))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/ 2> /dev/null; [ -f $D/smoke/train.log ] && tail -c 200000 $D/smoke/train.log > $D/up/train_tail.log
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_2gpu/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_2gpu_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
[[ $G =~ ^[0-9]+,[0-9]+$ ]] || finish FAIL "needs two GPUs (got '$G': is '#gpus PROBE-2GPU 2' in the queue?)"
REF=$OPD_ROOT/runs/PROBE-E12B-16K-smoke/train.log; [ -f $REF ] || REF=$OPD_ROOT/runs/PROBE-E12B-16K/smoke/train.log
[ -f $REF ] || finish FAIL "no one-GPU smoke of E12b-P5-16k to compare with (PROBE-E12B-16K)"
for g in ${G//,/ }; do
  for _w in $(seq 19); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
    [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $g not empty after 180 s"; sleep ${P2G_WAIT_STEP:-10}
  done
done
step "start gpus=$G smoke E12b-P5-16k (${P2G_STEPS:-3} steps)"
( while true; do for g in ${G//,/ }; do echo "$(date -u +%s),$g,$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g | tr -d ' ')" >> $D/mem.csv; done; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/$T-smoke
SMOKE=1 SMOKE_STEPS=${P2G_STEPS:-3} PROBE_TASK=$T-smoke bash $RUNBOOK/scripts/run_e12.sh $G E12b-P5-16k > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/$T-smoke/train.log $OPD_ROOT/runs/$T-smoke/config.yaml $D/smoke/ 2> /dev/null
for g in ${G//,/ }; do python3 $RUNBOOK/scripts/control.py reap $g $T > /dev/null 2>&1 || true; done
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "CUDA out of memory\|OutOfMemoryError" $D/smoke/train.log 2> /dev/null && finish FAIL "out of memory in the smoke"
why=$(python3 - "$D" "$REF" "${P2G_STEPS:-3}" "$G" "${P2G_WORTH:-1.3}" "${P2G_PEAK_MIB:-266240}" <<'PY'
import json, os, re, sys
d, ref, nsteps, gl, worth, peak_max = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4].split(","), float(sys.argv[5]), float(sys.argv[6])
KEYS = {"step_s": "timing_s/step", "gen_s": "timing_s/gen", "update_s": "timing_s/update_actor", "tokens": "perf/total_num_tokens",
        "resp_len": "response_length/mean", "clip": "response_length/clip_ratio", "score": "critic/score/mean",
        "pg_loss": "actor/pg_loss", "grad_norm": "actor/grad_norm"}
def steps(path):
    text = open(path, errors="ignore").read() if os.path.exists(path) else ""
    out = {}
    for st, body in re.findall(r"(?:^|[ )\]])step:(\d+) - (.*?)(?=(?:^|[ )\]])step:\d+ - |\Z)", text, re.S):
        out[int(st)] = {k: float(m.group(1)) for k, v in KEYS.items() for m in [re.search(re.escape(v) + r":([-0-9.e]+)", body)] if m}
    return out
def mean(rows, k):
    v = [r[k] for r in rows.values() if k in r]
    return round(sum(v) / len(v), 4) if v else None
two, one = steps(os.path.join(d, "smoke", "train.log")), steps(ref)
why = []
if sorted(two) != list(range(1, nsteps + 1)):
    why.append(f"two-GPU train.log steps {sorted(two)}, expected 1..{nsteps}")
cfg = open(os.path.join(d, "smoke", "config.yaml"), errors="ignore").read() if os.path.exists(os.path.join(d, "smoke", "config.yaml")) else ""
if f"gpus: {','.join(gl)} (n_gpus_per_node {len(gl)})" not in cfg:
    why.append("the smoke's config does not name both GPUs")
peak = {}
for line in open(os.path.join(d, "mem.csv")) if os.path.exists(os.path.join(d, "mem.csv")) else []:
    f = line.strip().split(",")
    if len(f) == 3 and f[2].replace(".", "").isdigit():
        peak[f[1]] = max(peak.get(f[1], 0.0), float(f[2]))
for g in gl:
    if peak.get(g, 0) <= 20480:
        why.append(f"gpu {g} peak {peak.get(g, 0):.0f} MiB: not used")
    elif peak[g] >= peak_max:
        why.append(f"gpu {g} peak {peak[g]:.0f} MiB >= 260 GB")
c = {"two_gpu": {"gpus": gl, "steps": two, "mean": {k: mean(two, k) for k in KEYS}, "peak_mib": peak},
     "one_gpu": {"steps": one, "mean": {k: mean(one, k) for k in KEYS}}}
s1, s2 = c["one_gpu"]["mean"]["step_s"], c["two_gpu"]["mean"]["step_s"]
t1, t2 = c["one_gpu"]["mean"]["tokens"], c["two_gpu"]["mean"]["tokens"]
c["speedup"] = round(s1 / s2, 3) if s1 and s2 else None
c["speedup_per_token"] = round((s1 / t1) / (s2 / t2), 3) if s1 and s2 and t1 and t2 else None
c["worth_it"] = bool(c["speedup"] and c["speedup"] >= worth)
json.dump(c, open(os.path.join(d, "compare.json"), "w"), indent=1)
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
sp=$(python3 -c "import json, sys; c = json.load(open(sys.argv[1])); print(c['speedup'], c['speedup_per_token'], c['worth_it'])" $D/compare.json)
finish PASS "${P2G_STEPS:-3}-step smoke on gpus $G ok; speedup (step time, per token, worth it) $sp"
