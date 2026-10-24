#!/usr/bin/env bash
# PROBE-DIFFPROBE (runbook [v2.5.9z74]; Q-062, Runming 10-22 16:0x Beijing, pre-registered research_program 10-22): can a problem's
# difficulty be read off the 1.7B base's hidden states before it answers? On ONE GPU code/analysis/diff_probe.py: a forward pass of the
# base over PROBE-DIFF17's 6,400 train prompts and MATH-500's 500 (the student's non-thinking template), the last prompt token's hidden
# state at every layer; per layer an L2 logistic probe (5-fold CV on train, strength by CV) for easy = r >= 0.5, its AUC on MATH-500;
# the zero-training proxies on MATH-500: first-token entropy and the self-rated confidence after a 256-token greedy attempt; H3 = the
# route arm's MATH-500 16K mean length hard / easy with the true and the probe labels.
# Inputs: runs/PROBE-DIFF17/labels_17.jsonl (WAIT_INPUTS when absent), data/dapo_train.parquet, the route reading
# runs/EVAL-e10b-P5route/e10b_P5route_s50_math500_T0.6_math500_no_think_t0.6.jsonl (H3 left open when absent).
# PASS = exit 0, >= 6,000 train prompts matched and all 500 of MATH-500, every layer with a test AUC, both proxies' AUC. The
# hypotheses are readings, not the gate. diff_probe.json + report.json to the public result repository as runs/PROBE-DIFFPROBE
# (numbers only, path-free); probe.log and the output tail to logs/<node>/probe_diffprobe/ (private); an event; .done.
# ~10-15 min on one GPU. Usage: bash probe_diffprobe.sh <gpu>
set -uo pipefail
G=$1; T=PROBE-DIFFPROBE; source "$(dirname "$0")/env.sh"; source ${VENV_EVAL:-$OPD_ROOT/venv}/bin/activate
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_DIFFPROBE $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    r["probe"] = json.load(open(os.path.join(d, "out", "diff_probe.json")))
except (OSError, ValueError):
    pass
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/up/; [ -f $D/out/diff_probe.json ] && cp $D/out/diff_probe.json $D/up/
  python3 $RUNBOOK/scripts/remote.py put $D/up runs/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  mkdir -p $D/logs; cp $LOG $D/logs/ 2> /dev/null; tail -c 200000 $D/gen.out > $D/logs/out_tail.log 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/logs logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_diffprobe/$T >> $LOG 2>&1 || true
  python3 $RUNBOOK/scripts/control.py event $T "probe_diffprobe_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
LAB=$OPD_ROOT/runs/PROBE-DIFF17/labels_17.jsonl
[ -f $LAB ] || { echo "WAIT_INPUTS $T: runs/PROBE-DIFF17/labels_17.jsonl not on this node"; exit 1; }
[ -f $OPD_ROOT/data/dapo_train.parquet ] || finish FAIL "no data/dapo_train.parquet"
ROUTE=$OPD_ROOT/runs/EVAL-e10b-P5route/e10b_P5route_s50_math500_T0.6_math500_no_think_t0.6.jsonl
[ -f $ROUTE ] || { step "the route reading is not on this node: H3 left open"; ROUTE=""; }
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PDP_WAIT_STEP:-10}
done
step "start gpu=$G model=$(basename $STUDENT_17B)"
rm -rf $D/out; mkdir -p $D/out
CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/analysis/diff_probe.py --model $STUDENT_17B --labels $LAB --train $OPD_ROOT/data/dapo_train.parquet \
  --out $D/out $([ -n "$ROUTE" ] && echo --route $ROUTE) $([ -n "${PDP_LIMIT:-}" ] && echo --limit $PDP_LIMIT) > $D/gen.out 2>&1; rc=$?
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
[ $rc = 0 ] || finish FAIL "diff_probe exited $rc: $(tail -c 300 $D/gen.out | tr '\n' ' ')"
why=$(python3 - $D/out/diff_probe.json "${PDP_LIMIT:-0}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); lim = int(sys.argv[2]); why = []
if r.get("n_train", 0) < (6000 if not lim else 1):   # PDP_LIMIT (a smoke): any
    why.append(f"{r.get('n_train')} train prompts matched, expected >= 6000")
if not lim and r.get("n_math500") != 500:
    why.append(f"{r.get('n_math500')} MATH-500 problems, expected 500")
if not r.get("layers") or any(l.get("test_auc") is None for l in r["layers"]):
    why.append("a layer without a test AUC")
if r.get("entropy_auc") is None or r.get("confidence_auc") is None:
    why.append("a proxy without an AUC")
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "$(python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print(f\"layer {r['h1_layer']} (best train CV) MATH-500 AUC {r['h1_auc']} (H1 {r['H1']}); entropy {r['entropy_auc']}, confidence {r['confidence_auc']} (H2 {r['H2']}); H3 {r['H3']} (ratios {r['h3'].get('ratio_true')} / {r['h3'].get('ratio_probe')})\")" $D/out/diff_probe.json)"
