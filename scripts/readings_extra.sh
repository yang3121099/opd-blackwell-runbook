#!/usr/bin/env bash
# READINGS-EXTRA (runbook v2.5.9x1; the reviewer's / ops-lead's CPU pad 10-22): code/analysis/readings_extra.py over the queue's
# `#split READINGS-EXTRA <label> <task>/<file>.jsonl` lines (MATH-500 16K readings by problems 1-100 / 101-500, and easy / hard by
# PROBE-DIFF17's labels when they exist) and `#frontier READINGS-EXTRA <label> <task>/<file> [<reading>]` lines (16K mean length and
# truncation for the cost-frontier table), [v2.5.9z5] #pairq and [v2.5.9z7] #coverq lines, from files already on the node (nothing generated). Each <task>/<stem>.split.json goes to
# the public result repository next to its reading, and runs/READINGS-EXTRA/{frontier_fill.json, summary.json}; run.log to
# logs/<node>/readings_extra/ (private); an event readings_extra_done; .done. The CPU lane; `rerun READINGS-EXTRA` after new lines.
# Usage: bash readings_extra.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=READINGS-EXTRA; D=$OPD_ROOT/runs/$T; LOG=$D/run.log; Q=${QUEUE_FILE:-$RUNBOOK/tasks/queue.txt}; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
rm -rf $D/up
nice -n 19 python3 $RUNBOOK/code/analysis/readings_extra.py $OPD_ROOT/runs $Q $D > $LOG 2>&1; rc=$?
# [v2.5.9z5] one upload per run directory as runs/<task>: remote.py routes by prefix, and a bare "runs" (10-21 18:56 UTC) went to the
# private repository instead of the public one
if [ $rc = 0 ]; then for d in $D/up/runs/*/; do [ -d "$d" ] || continue
  python3 $RUNBOOK/scripts/remote.py put ${d%/} runs/$(basename $d) >> $LOG 2>&1 || { echo "upload of $(basename $d) failed (kept in $D/up)" >> $LOG; rc=1; }; done; fi
python3 $RUNBOOK/scripts/remote.py put $LOG logs/$(python3 $RUNBOOK/scripts/control.py node)/readings_extra/run.log > /dev/null 2>&1 || true
S=$(python3 -c "import json, sys; s = json.load(open(sys.argv[1])); print(len(s['split']), 'splits,', len(s['frontier']), 'frontier rows,', len(s.get('pairs', {})), 'pairs,', len(s.get('coverage', {})), 'coverage rows,', len(s['missing']), 'missing')" $D/summary.json 2> /dev/null || echo "no summary")
python3 $RUNBOOK/scripts/control.py event $T readings_extra_done "rc=$rc $S" > /dev/null 2>&1
echo "READINGS_EXTRA rc=$rc $S" | tee -a $LOG
[ $rc = 0 ] && touch $D/.done
exit $rc
