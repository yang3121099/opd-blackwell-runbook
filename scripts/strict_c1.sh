#!/usr/bin/env bash
# STRICT-C1 (runbook v2.5.7o; reviewer 2026-10-19 06:0x UTC, queue v3 V3-c4 without training): every MATH500 reading on
# this node judged again under the strict delivery contract (code/analysis/strict_contract.py: correct under EVAL-v1, ended
# at EOS, exactly one complete \boxed{} in the answer); CPU only, the texts as written (no generation, no GPU work).
# Readings: every <task>/<file>.jsonl and <task>/gen/<file>.jsonl under $OPD_ROOT/runs with a math500 metrics.json.
# Pairs are comment lines of the queue (problem-level paired bootstrap under both graders):
#   #strict STRICT-C1 <a> <b>      a, b = <task dir>/<file>.jsonl under $OPD_ROOT/runs (as on a #pair line)
# Output: report.json and table.txt (readings named <task>/<file>, path-free) to the public result repository as
# runs/STRICT-C1; strict.log to logs/<node>/strict_c1/ (private); an event strict_c1_done; .done. `rerun STRICT-C1` after
# new readings arrive judges them all again.
# Usage: bash strict_c1.sh <gpu> STRICT-C1      (the GPU is not used)
set -uo pipefail
G=$1; T=$2; source "$(dirname "$0")/env.sh"; source ${VENV_EVAL:-$OPD_ROOT/venv}/bin/activate
D=$OPD_ROOT/runs/$T; LOG=$D/strict.log; Q=${QUEUE_FILE:-$RUNBOOK/tasks/queue.txt}; mkdir -p $D/up
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "STRICT_C1 $(date -u +%T) $*" | tee -a $LOG; }
rm -f $D/report.json $D/table.txt $D/up/report.json $D/up/table.txt   # v2.5.7t: a failed run never uploads an older table
step "start $T (CPU only, gpu $G not used)"
PAIRS=()
while read -r A B; do
  [ -n "$A" ] || continue
  if [[ $A =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.jsonl$ && $B =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.jsonl$ ]]; then PAIRS+=(--pair "$A" "$B")
  else step "PAIR_REFUSED $A $B"; fi
done <<< "$(awk -v t="$T" '$1 == "#strict" && $2 == t {print $3, $4}' "$Q")"
CUDA_VISIBLE_DEVICES= python $RUNBOOK/code/analysis/strict_contract.py --scan $OPD_ROOT/runs --out $D/report.json --table $D/table.txt \
  ${PAIRS[@]+"${PAIRS[@]}"} >> $LOG 2>&1; rc=$?
cp $D/report.json $D/table.txt $D/up/ 2> /dev/null
python $RUNBOOK/scripts/remote.py put $D/up runs/$T >> $LOG 2>&1 || { step "report upload failed (kept in $D)"; rc=1; }
mkdir -p $D/logs; cp $LOG $D/logs/ 2> /dev/null
python $RUNBOOK/scripts/remote.py put $D/logs logs/$(python $RUNBOOK/scripts/control.py node)/strict_c1/$T >> $LOG 2>&1 || true
S=$(python -c "import json, sys; r = json.load(open(sys.argv[1])); x = [v for v in r['readings'] if 'error' not in v]
print(f\"{len(x)} readings, mean diff {sum(v['diff'] for v in x) / max(len(x), 1):+.2f}, {len(r['pairs'])} pairs\")" $D/report.json 2> /dev/null || echo "report missing")
python $RUNBOOK/scripts/control.py event $T strict_c1_done "rc=$rc $S" > /dev/null 2>&1
step "DONE rc=$rc $S"
[ $rc = 0 ] && touch $D/.done
exit $rc
