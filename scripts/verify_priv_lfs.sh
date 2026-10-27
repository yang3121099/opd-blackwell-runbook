#!/usr/bin/env bash
# VERIFY-PRIV-LFS (runbook [v2.5.9z96a]; Fable 10-27 02:5x UTC): the CPU lane, read-only (verify_lfs_csv.py), at the lowest CPU and I/O
# priority so the trainings' reads come first. report.json to logs/<node>/verify_priv_lfs/ (private, a small file), an event, .done.
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=VERIFY-PRIV-LFS; D=$OPD_ROOT/runs/$T; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
LOW=""; command -v ionice > /dev/null && LOW="ionice -c3"
nice -n 19 $LOW python3 $RUNBOOK/scripts/verify_lfs_csv.py $RUNBOOK/tasks/private_repo_lfs_20261027.csv $OPD_ROOT $D/report.json > $D/verify.log 2>&1 || { tail -5 $D/verify.log; echo "VERIFY_FAILED $T"; exit 1; }
mkdir -p $D/up; cp $D/report.json $D/verify.log $D/up/
python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/verify_priv_lfs > /dev/null 2>&1 || echo "report upload failed (kept in $D)"
python3 $RUNBOOK/scripts/control.py event $T verify_priv_lfs_done "$(tail -1 $D/verify.log)" > /dev/null 2>&1
cat $D/verify.log; touch $D/.done; exit 0
