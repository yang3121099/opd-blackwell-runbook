#!/usr/bin/env bash
# DIAG-VLM-TRUNC (runbook [v2.5.9z89]; Q-080's first diagnostic, Fable's ruling 10-26 23:0x UTC): a CPU-lane task (no GPU, no process
# touched). The truncated samples of the VLM base's 16K readings (MathVista testmini from EVAL-e15-base, MathVision / MathVerse from
# EVAL-e15-base2) and, as the comparison, of E15-VLM-P5 s50's (EVAL-e15-vlm-s50), each in three classes (code/analysis/
# vlm_trunc_classify.py): an n-gram loop (a 20-token window three times), no box and no loop, a box and more after it. report.json:
# each input's shares; to logs/<node>/diag_vlm_trunc/ (private), an event, .done. Waits (no attempt) while an input is not on the node.
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=DIAG-VLM-TRUNC; D=$OPD_ROOT/runs/$T; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
IN="base_mathvista=EVAL-e15-base/e15_base2B_mathvista_T0.6_mathvista_no_think_t0.6.jsonl
base_mathvision=EVAL-e15-base2/e15_base2B_mathvision_T0_mathvision_no_think_t0.0.jsonl
base_mathverse=EVAL-e15-base2/e15_base2B_mathverse_T0_mathverse_no_think_t0.0.jsonl
s50_mathvista=EVAL-e15-vlm-s50/e15_P5_s50_mathvista_T0.6_mathvista_no_think_t0.6.jsonl
s50_mathvision=EVAL-e15-vlm-s50/e15_P5_s50_mathvision_T0_mathvision_no_think_t0.0.jsonl
s50_mathverse=EVAL-e15-vlm-s50/e15_P5_s50_mathverse_T0_mathverse_no_think_t0.0.jsonl"
args=()
for kv in $IN; do
  rel=${kv#*=}; [ -f $OPD_ROOT/runs/$rel ] || { echo "WAIT_INPUTS $T: runs/$rel not on this node"; exit 1; }
  args+=("${kv%%=*}=$OPD_ROOT/runs/$rel")
done
python3 $RUNBOOK/code/analysis/vlm_trunc_classify.py $D/classes.json "${args[@]}" > $D/diag.log 2>&1 || { tail -5 $D/diag.log; echo "DIAG_FAILED $T"; exit 1; }
python3 - "$D" <<'PY'
import json, os, sys
d = sys.argv[1]; c = json.load(open(os.path.join(d, "classes.json")))
json.dump({"result": "PASS", "reason": "; ".join(f"{k} loop {v['shares']['loop']:.1%} unfinished {v['shares']['unfinished']:.1%} box_more {v['shares']['box_more']:.1%} of {v['truncated']}" for k, v in c["inputs"].items()), **c},
          open(os.path.join(d, "report.json"), "w"), indent=1)
PY
mkdir -p $D/up; cp $D/report.json $D/diag.log $D/up/
python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/diag_vlm_trunc > /dev/null 2>&1 || echo "report upload failed (kept in $D)"
python3 $RUNBOOK/scripts/control.py event $T diag_vlm_trunc_done "$(python3 -c "import json; print(json.load(open('$D/report.json'))['reason'][:400])")" > /dev/null 2>&1
cat $D/diag.log; touch $D/.done; exit 0
