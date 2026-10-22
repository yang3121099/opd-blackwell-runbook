#!/usr/bin/env bash
# TOKCHECK-X5 (runbook [v2.5.9z24]; ops-lead 10-22, Q-057's X5 arm): once the three urgent pulls are in (each with .fetch_ok), the
# zero-GPU facts of the student Qwen2.5-Math-1.5B-Instruct, the teacher DeepSeek-R1-Distill-Qwen-1.5B and the base
# Qwen2.5-Math-1.5B (tok_check.py): size and pull time / rate (each and combined), vocab_size, the tokenizer's size, bos / eos / pad
# ids, <think> in the chat template, and whether the student's and the teacher's vocabularies agree id for id. While a model is not
# in: WAIT_INPUTS (waiting, no attempt). report.json to logs/<node>/tokcheck_x5/ (private), an event, .done. The CPU lane.
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=TOKCHECK-X5; D=$OPD_ROOT/runs/$T; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
MODELS="student=Qwen2.5-Math-1.5B-Instruct teacher=DeepSeek-R1-Distill-Qwen-1.5B base=Qwen2.5-Math-1.5B"
for x in $MODELS; do [ -f $OPD_ROOT/models/${x#*=}/.fetch_ok ] || { echo "WAIT_INPUTS $T: ${x#*=} not in (no .fetch_ok)"; exit 1; }; done
nice -n 19 python3 $RUNBOOK/scripts/tok_check.py $OPD_ROOT $D/report.json $MODELS > $D/check.out 2>&1 || { echo "TOKCHECK_FAIL $(tail -c 300 $D/check.out | tr '\n' ' ')"; exit 1; }
mkdir -p $D/up; cp $D/report.json $D/up/
python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/tokcheck_x5 > $D/upload.log 2>&1 || echo "report upload failed (kept in $D)"
python3 $RUNBOOK/scripts/control.py event $T tokcheck_done "$(tail -1 $D/check.out | cut -c1-400)" > /dev/null 2>&1
touch $D/.done; echo "TOKCHECK-X5 done"
