#!/usr/bin/env bash
# PROBE-WDIFF (runbook [v2.5.9z18], the reviewer's Q-031: where in the network does each method change the weights, and are
# the changes low-rank?). CPU lane: no GPU, no network. Delta = M - Qwen3-4B for five 4B models (WD_MODELS overrides):
#   P5      result:runs/E10d-P5/global_step_150      OPD with the thinking-tag teacher
#   P5self  result:runs/E10d-P5self/global_step_50   OPD with the self-teacher
#   RL      result:runs/E12d-P0/global_step_100      GRPO from the base (base -> RL)
#   T2507 / I2507  Qwen/Qwen3-4B-Thinking-2507 / Qwen/Qwen3-4B-Instruct-2507, Qwen's own releases
# Paths as run_queue_eval.sh's resolve_model / local_save find them: a training's own save runs/<task>/ckpt/global_step_N/
# actor/huggingface_bf16 (the bf16 copy that was evaluated), else its huggingface (fp32, read in bf16 as the evaluations read
# it), under $OPD_ROOT then $CKPT_ROOT,
# else the download cache $CKPT_ROOT/adhoc_models/result/...; Qwen/* (the base: $STUDENT_4B first) from $OPD_ROOT/models/<name>,
# else the Hugging Face caches (as overnight.sh's find_model). Nothing is downloaded: a model not found is a report row with
# its error, and `rerun PROBE-WDIFF` once it is there. scripts/wdiff.py (the node's venv: torch + safetensors; WD_THREADS 8
# threads, nice 10) reads one tensor at a time in row chunks (memory near a few GB): per layer and matrix type (q/k/v/o/gate/
# up/down, embeddings, norms) |Delta| / |W|, the share left unchanged, the pairwise cosines of the five Deltas, per matrix the
# share of |Delta|^2 in the top-10 singular values (the exact spectrum from the Gram matrix; checked against torch's full SVD
# on layer 0, both timed; a Gaussian null per shape). report.json (every row) and summary.json (per Delta: relative change by
# depth, the types that moved most, low-rankness; the cosines) to logs/<node>/probe_wdiff/<task> (private), an event, .done.
# A missing base: the report and a probe_wdiff_failed event, no .done. Usage: bash probe_wdiff.sh cpu
set -uo pipefail
T=PROBE-WDIFF; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_WDIFF $(date -u +%T) $*" | tee -a $LOG; }
PY=${WD_PY:-$OPD_ROOT/venv/bin/python}; TH=${WD_THREADS:-8}; BASE=${WD_BASE:-Qwen/Qwen3-4B}
MODELS=${WD_MODELS:-P5=result:runs/E10d-P5/global_step_150/actor/huggingface P5self=result:runs/E10d-P5self/global_step_50/actor/huggingface RL=result:runs/E12d-P0/global_step_100/actor/huggingface T2507=Qwen/Qwen3-4B-Thinking-2507 I2507=Qwen/Qwen3-4B-Instruct-2507}
sym() { local p=$1 v; for v in CKPT_ROOT OPD_ROOT HF_HOME HOME; do [ -n "${!v:-}" ] && [[ $p == "${!v}"/* ]] && { echo "\$$v${p#${!v}}"; return; }; done; echo "$p"; }
local_save() {   # as run_queue_eval.sh's: runs/<task>/global_step_N/actor/huggingface -> "<the training's own save> <source>"
  [[ $1 =~ ^runs/([A-Za-z0-9._-]+)/global_step_([0-9]+)/actor/huggingface$ ]] || return 0
  local r b
  for r in $OPD_ROOT $CKPT_ROOT; do
    b=$r/runs/${BASH_REMATCH[1]}/ckpt/global_step_${BASH_REMATCH[2]}/actor
    has_weights $b/huggingface_bf16 && { echo "$b/huggingface_bf16 local_bf16"; return 0; }
    has_weights $b/huggingface && { echo "$b/huggingface local_fp32"; return 0; }
  done; }
resolve() {   # a spec -> "<path> <source>", or nothing; never downloads
  local m r s L
  case $1 in
    result:runs/*) L=$(local_save "${1#result:}"); [ -n "$L" ] && { echo "$L"; return 0; }
      [[ ${1#result:} != *..* ]] && has_weights $CKPT_ROOT/adhoc_models/result/${1#result:} && echo "$CKPT_ROOT/adhoc_models/result/${1#result:} download_cache";;
    Qwen/*) m=${1#Qwen/}; [[ $m =~ ^[A-Za-z0-9._-]+$ ]] || return 0
      [ "$1" = Qwen/Qwen3-4B ] && [ -d "${STUDENT_4B:-}" ] && has_weights $STUDENT_4B && { echo "$STUDENT_4B models"; return 0; }
      has_weights $OPD_ROOT/models/$m && { echo "$OPD_ROOT/models/$m models"; return 0; }
      for r in ${HF_HUB_CACHE:-} ${HF_HOME:+$HF_HOME/hub} $HOME/.cache/huggingface/hub $OPD_ROOT/hf_home/hub; do
        for s in $r/models--Qwen--$m/snapshots/*/; do has_weights $s && { echo "${s%/} hf_cache"; return 0; }; done; done;;
  esac; }
step "start: base $BASE; $(echo $MODELS | wc -w | tr -d ' ') models; $TH threads"
: > $D/models.tsv
for kv in base=$BASE $MODELS; do
  k=${kv%%=*}; spec=${kv#*=}; L=$(resolve "$spec")
  printf '%s\t%s\t%s\t%s\n' "$k" "$spec" "${L% *}" "${L##* }" >> $D/models.tsv
  [ -n "$L" ] && step "$k: $(sym ${L% *}) (${L##* })" || step "$k: $spec not found (never downloaded)"
done
OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH WD_THREADS=$TH timeout ${WD_TIMEOUT:-21600} nice -n ${WD_NICE:-10} \
  $PY $RUNBOOK/scripts/wdiff.py $D/models.tsv $D/report.json >> $LOG 2>&1 || step "wdiff.py exited $?"
[ -s $D/report.json ] || { step "report failed"; python3 $RUNBOOK/scripts/control.py event $T probe_wdiff_failed "no report: $(tail -c 300 $LOG | tr '\n' ' ')" > /dev/null 2>&1; exit 1; }
err=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('error') or '')" $D/report.json)
summary=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('summary_line') or '')" $D/report.json)
mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/; [ -f $D/summary.json ] && cp $D/summary.json $D/up/
python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_wdiff/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
if [ -n "$err" ]; then
  python3 $RUNBOOK/scripts/control.py event $T probe_wdiff_failed "$err" > /dev/null 2>&1; step "failed: $err"; exit 1
fi
python3 $RUNBOOK/scripts/control.py event $T probe_wdiff_done "$summary" > /dev/null 2>&1
step "done; $summary"; touch $D/.done
