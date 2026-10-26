#!/usr/bin/env bash
# [v2.6 S0] (Runming 10-25 "polish, not patches"; Fable F7 / ops-lead O7: the one structural exception on main during the v2.6
# migration) CORPUS-RECORD: the golden corpus of 2.5.9's behaviour for every task name, recorded by dryrun_json.py in sandboxes
# on the checkpoint disk (no GPU, no network, nothing written outside $CKPT_ROOT/dryrun_json.tmp and dryrun_json.out); one put
# of the index, the findings and the tarball to the private repository. CPU lane, nice 19, 60 s a task, 2 h in all.
# Usage: bash dryrun_json.sh --all | <task>...  [--profile clean|node|both] [--no-upload]   (dryrun_json.py --help)
set -uo pipefail
# [v2.5.9z88e] (10-26 06:48Z: a sandbox ran the recorder again and the nesting grew) never inside a sandbox: every nested level
# re-reads this file from the real checkout, so from this push on a new level stops here at once
if [ -n "${DRYRUN_SB:-}${DRYRUN_REAL_PY:-}" ]; then echo "CORPUS_RECORD_REFUSED inside a sandbox: no nested recording"; exit 0; fi
source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/CORPUS-RECORD
TD=(); [ "${1:-}" = --all ] && { mkdir -p $D; [ -f $D/.done ] && { echo "TASK_ALREADY_DONE CORPUS-RECORD"; exit 0; }; TD=(--task-dir $D); }
IO=(); command -v ionice > /dev/null && IO=(ionice -c 3)   # idle I/O class where the node has it
${IO[@]+"${IO[@]}"} nice -n 19 python3 $RUNBOOK/scripts/dryrun_json.py --runbook $RUNBOOK ${TD[@]+"${TD[@]}"} "$@" 2>&1 | tail -c 20000 | tee -a $OPD_ROOT/logs/corpus_record.log
rc=${PIPESTATUS[0]}
[ -f $D/report.json ] && python3 $RUNBOOK/scripts/control.py event CORPUS-RECORD "corpus_record_$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['result'].lower())" $D/report.json)" \
  "$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['reason'])" $D/report.json)" > /dev/null 2>&1
exit $rc
