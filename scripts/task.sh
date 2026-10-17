#!/usr/bin/env bash
# Whitelist dispatcher: run ONE task on ONE GPU. Only the task IDs matched below are accepted; anything else exits 2.
# Every task writes $OPD_ROOT/runs/<task>/.done when it has finished properly; run_tasks.sh treats a missing .done
# after exit as a failure. task.sh stays the parent process (no exec), so run_tasks.sh can see that a task is running.
# Usage: bash task.sh <gpu> <task id>
set -uo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=$2
ARMS="P2|P3|P4lowlr|P5|P6|P6r|P2noEOS|Pstop|P6rand|P6top5|P4EOSpen"
case $T in
  E10b-*)  [[ $T =~ ^E10b-($ARMS)(-s(43|44))?$ ]] || exit 2; bash $RUNBOOK/scripts/run_arm.sh $G $T;;
  E11b-*)  [[ $T =~ ^E11b-(P0|$ARMS)$ ]] || exit 2; bash $RUNBOOK/scripts/run_eval.sh $G $T;;
  SFT-P1)  bash $RUNBOOK/scripts/run_sft.sh $G $T;;
  E12b-*)  [[ $T =~ ^E12b-(P0|P1|$ARMS)$ ]] || exit 2; bash $RUNBOOK/scripts/run_e12.sh $G $T;;
  E03-MI-l0.0|E03-MI-l0.3|E03-MI-l0.5|E03-MI-l0.7|E03-MI-l1.0) bash $RUNBOOK/scripts/run_zero.sh $G $T;;
  E02-opsa17-c0.5|E02-opsa17-c1.5|E02-opsa4-c-1|E02-opsa4-c0.5|E02-opsa4-c1.5) bash $RUNBOOK/scripts/run_zero.sh $G $T;;
  FA-PROBE) bash $RUNBOOK/scripts/fa_probe.sh $G;;   # v2.5.3: can flash-attn 2 varlen run here (no E10b directory touched)
  *) echo "NOT_WHITELISTED $T"; exit 2;;
esac
