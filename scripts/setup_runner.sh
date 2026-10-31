#!/usr/bin/env bash
# [v2.5.9z106] setup_runner.sh <2-7|all> <seq>: the control command `setup` (control.py v_setup starts it in a session of its own;
# connect-first, Fable 11-01; ops-lead's A1 16:11 UTC). cold_start.sh's steps one at a time (--step N, with the arguments
# bootstrap.sh wrote to control/coldstart_args), each one's start and end in control/setup_state.json (the status; STOP with the
# step log's last 5 lines, redacted), a heartbeat round right away. One at a time: control/setup.pid (= this group, which `setup
# stop` signals). The loops are not touched, except: once the main venv imports torch, vllm and verl, control/venv_ready and
# run_tasks restarted through loops.sh (so its tasks get that python); dispatch stays frozen until `dispatch on`.
set -uo pipefail
source "$(dirname "$0")/env.sh"
CDIR=$OPD_ROOT/control; mkdir -p $CDIR $OPD_ROOT/logs
WHAT=${1:-}; SEQ=${2:-0}
[[ $WHAT =~ ^([2-7]|all)$ ]] || { echo "SETUP usage: setup_runner.sh <2-7|all> <seq>"; exit 2; }
[ -s $CDIR/coldstart_args ] || { python3 $RUNBOOK/scripts/control.py setup_state "$WHAT" "arguments" STOP; echo "SETUP no control/coldstart_args"; exit 1; }
echo $$ > $CDIR/setup.pid
trap 'rm -f $CDIR/setup.pid' EXIT
name() { case $1 in 2) echo "overnight.sh: pull, base models, venv, data, vLLM tier, verify";; 3) echo "last status: tasks done";;
  4) echo "models and venvs";; 5) echo "probe labels";; 6) echo "weights the tasks need";; 7) echo "health checks";; esac; }
STEPS=$WHAT; [ $WHAT = all ] && STEPS="2 3 4 5 6 7"
for N in $STEPS; do
  L=$OPD_ROOT/logs/setup_step_$N.log
  python3 $RUNBOOK/scripts/control.py setup_state $N "$(name $N)" running
  echo "SETUP $(date -u +%T) seq $SEQ step $N start"
  # shellcheck disable=SC2046
  COLDSTART_DRYRUN=${COLDSTART_DRYRUN:-0} bash $RUNBOOK/scripts/cold_start.sh $(cat $CDIR/coldstart_args) --step $N > $L 2>&1; rc=$?
  if [ $rc != 0 ]; then
    python3 $RUNBOOK/scripts/control.py setup_state $N "$(name $N)" STOP $L
    python3 $RUNBOOK/scripts/control.py event node setup_stop "step $N exit $rc (status.setup_state; tail logs/setup_step_$N.log)" > /dev/null 2>&1
    echo "SETUP $(date -u +%T) step $N STOP (exit $rc)"; exit 1
  fi
  python3 $RUNBOOK/scripts/control.py setup_state $N "$(name $N)" done $L
  echo "SETUP $(date -u +%T) step $N done"
  if [ ! -f $CDIR/venv_ready ] && [ -x $OPD_ROOT/venv/bin/python ] && $OPD_ROOT/venv/bin/python -c "import torch, vllm, verl" > /dev/null 2>&1; then
    date -u +%FT%TZ > $CDIR/venv_ready
    python3 $RUNBOOK/scripts/control.py event node venv_ready "main venv imports torch, vllm, verl: run_tasks restarted on it (dispatch still frozen)" > /dev/null 2>&1
    bash $RUNBOOK/scripts/loops.sh restart run_tasks >> $OPD_ROOT/logs/loops.log 2>&1
  fi
done
python3 $RUNBOOK/scripts/control.py event node setup_done "steps $STEPS done (seq $SEQ)" > /dev/null 2>&1
