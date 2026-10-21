#!/usr/bin/env bash
# PROBE-DIFF17 (runbook v2.5.9x; Q-041 / Q-042, the reviewer's pre-registration research_program §3c): difficulty labels for the
# routed OPD arms. On ONE GPU code/analysis/diff_labels.py samples the 1.7B base on its non-thinking template, k=8, T=1.0, 4K, on
# the 6400 training prompts an E10b arm meets in its 50 steps (128 a step, verl's sampler order with data.seed 42, after the
# overlong-prompt filter) and on MATH-500; r = correct / 8 by the training reward (gen_eval's grader for MATH-500).
# With 4b: PROBE-DIFF17-4B, the same with the 4B base (the k_4b column for Q-050; a pad task, not a gate).
# Output runs/PROBE-DIFF17/labels_<17|4b>.jsonl (prompt key = sha1 of its user message, sampler position, k, r, mean length,
# truncation) and report_<tag>.json (histogram of k, the easy share r >= 0.5, timing) to the public result repository as
# runs/PROBE-DIFF17 (path-free: hashes and numbers); probe.log to logs/<node>/probe_diff17/ (private); an event; .done.
# PASS = exit 0, every selected prompt labelled (6400 + 500). E10b-P5route reads labels_17.jsonl. Usage: bash probe_diff17.sh <gpu> [4b]
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"; source ${VENV_EVAL:-$OPD_ROOT/venv}/bin/activate
TAG=17; T=PROBE-DIFF17; M=$STUDENT_17B
[ "${2:-}" = 4b ] && { TAG=4b; T=PROBE-DIFF17-4B; M=$STUDENT_4B; }
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; OUTD=$OPD_ROOT/runs/PROBE-DIFF17; mkdir -p $D $OUTD
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_DIFF17 $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$OUTD/report_$TAG.json" "$1" "$2" <<'PY'
import json, os, sys
d, rp, res, why = sys.argv[1:5]
r = {"result": res, "reason": why}
try:
    r.update(json.load(open(rp)))
except (OSError, ValueError):
    pass
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  if [ $1 = PASS ]; then mkdir -p $OUTD/up; cp $OUTD/labels_$TAG.jsonl $OUTD/report_$TAG.json $OUTD/up/ 2> /dev/null
    python3 $RUNBOOK/scripts/remote.py put $OUTD/up runs/PROBE-DIFF17 >> $LOG 2>&1 || step "labels upload failed (kept in $OUTD)"; fi
  mkdir -p $D/logs; cp $LOG $D/report.json $D/logs/ 2> /dev/null; tail -c 200000 $D/gen.out > $D/logs/gen_tail.log 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/logs logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_diff17/$T >> $LOG 2>&1 || true
  python3 $RUNBOOK/scripts/control.py event $T "probe_diff17_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
[ -f $OPD_ROOT/data/dapo_train.parquet ] || finish FAIL "no data/dapo_train.parquet"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PDF_WAIT_STEP:-10}
done
step "start gpu=$G model=$(basename $M) tag=$TAG"
CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/analysis/diff_labels.py --model $M --tag $TAG --train $OPD_ROOT/data/dapo_train.parquet \
  --out $OUTD --steps ${DIFF_STEPS:-50} --tbs ${DIFF_TBS:-128} --seed ${DIFF_SEED:-42} --maxp 1024 --k 8 --max_tokens 4096 \
  $([ -n "${DIFF_LIMIT:-}" ] && echo --limit $DIFF_LIMIT) > $D/gen.out 2>&1; rc=$?
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
[ $rc = 0 ] || finish FAIL "diff_labels exited $rc: $(tail -c 300 $D/gen.out | tr '\n' ' ')"
WANT=$(( ${DIFF_LIMIT:-0} > 0 ? 2 * ${DIFF_LIMIT:-0} : ${DIFF_STEPS:-50} * ${DIFF_TBS:-128} + 500 ))
GOT=$(wc -l < $OUTD/labels_$TAG.jsonl 2> /dev/null | tr -d ' ')
[ "${GOT:-0}" = $WANT ] || finish FAIL "labels_$TAG.jsonl has ${GOT:-0} prompts, expected $WANT"
finish PASS "$GOT prompts labelled; $(python3 -c "import json; r = json.load(open('$OUTD/report_$TAG.json')); print('train easy share', r['train']['easy_share'], 'mean r', r['train']['mean_r'], '| math500 easy share', r['math500']['easy_share'])")"
