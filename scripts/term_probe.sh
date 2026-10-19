#!/usr/bin/env bash
# PROBE-TERM (runbook v2.5.7g; reviewer 2026-10-18 19:0x UTC): termination quality of trained students, no training,
# forward passes only (code/analysis/term_probe.py: p(<|im_end|>) at the first newline after the first complete \boxed{},
# the share of re-check phrases in what follows the answer, the 8B teacher's p(<|im_end|>) at the same point; ~2 GPU-h).
# Models and their generations are comment lines of the queue:
#   #term PROBE-TERM <tag> <model> <generations>
# <tag>: <size>_<name> (models are compared in pairs within one size); <model>: as on an #eval line (Qwen/<name>, the lab
# archive yang31210999/opd-lab-ckpts/..., result:runs/..., local:<path under $OPD_ROOT or $CKPT_ROOT>); <generations>:
# the model's MATH500 T0.6 16K gen_eval jsonl, one of <task dir>/<file>.jsonl under $OPD_ROOT/runs, archive:<path>.jsonl
# in the lab archive (fetch_model.py, size + sha256), or gen = generated here first with the EVAL-v1 settings (no_think,
# n 4, T 0.6, top-p 0.95, top-k 0, 16K, seed 0, blocks of 50) for a model whose generations are not archived (the report
# says so). A model or generation that cannot be had is listed in the report; the others are measured.
# Output: up/ = report.json, samples.jsonl (one line per generation) and gen/*.jsonl, path-free, to the public result
# repository as runs/PROBE-TERM; probe.log to logs/<node>/term_probe/ (private); an event term_probe_done; .done.
# v2.5.7k: PROBE-TERM2 = the same models and generations (the #term PROBE-TERM lines; the E14 generations PROBE-TERM made
# are reused, not drawn again) with the environment-close anchor and the 16-token maximum (term_probe.py, reviewer 23:35 UTC)
# v2.5.7m: PROBE-TERM3 (reviewer 01:0x UTC) = cross prefixes: a model on another model's generations, its own #term
# PROBE-TERM3 lines, no teacher (its values depend on the text only, and the texts are those PROBE-TERM2 measured)
# v2.5.7z: PROBE-TERM4 (reviewer 11:1x UTC) = its own #term PROBE-TERM4 lines with an optional 4th field `think` (the
# model's prompt in the thinking template), no teacher, term_probe.py --term4: p(</think>) after the first final-answer
# box, p(<|im_end|>) after the last box (the summary's)
# v2.5.7zk (reviewer 21:0x UTC): PROBE-TERM4-4B = PROBE-TERM4 on the 4B models, its own #term PROBE-TERM4-4B lines, its own
# runs/PROBE-TERM4-4B (runs/PROBE-TERM4 keeps the 1.7B numbers the logs cite)
# Usage: bash term_probe.sh <gpu> PROBE-TERM|PROBE-TERM2|PROBE-TERM3|PROBE-TERM4|PROBE-TERM4-4B
set -uo pipefail
G=$1; T=$2; source "$(dirname "$0")/env.sh"; source ${VENV_EVAL:-$OPD_ROOT/venv}/bin/activate
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; Q=${QUEUE_FILE:-$RUNBOOK/tasks/queue.txt}; mkdir -p $D/gen $D/up/gen
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "TERM_PROBE $(date -u +%T) $*" | tee -a $LOG; }
fetch_ckpt() { python $RUNBOOK/scripts/fetch_model.py yang31210999/opd-lab-ckpts "$1" "$CKPT_ROOT/adhoc_models" --task $T >> $D/fetch.log 2>&1; }
fetch_result() { python $RUNBOOK/scripts/fetch_model.py "$(python $RUNBOOK/scripts/remote.py repo public)" "$1" "$CKPT_ROOT/adhoc_models/result" --task $T >> $D/fetch.log 2>&1; }
step "start $T gpu=$G"
: > $D/inputs.tsv
QT=$T; [ $T = PROBE-TERM2 ] && QT=PROBE-TERM   # v2.5.7m: only PROBE-TERM2 reuses PROBE-TERM's lines
while read -r TAG MODEL GENS MODE; do
  [ -n "$TAG" ] || continue
  [[ $TAG =~ ^[A-Za-z0-9.-]+_[A-Za-z0-9._-]+$ ]] || { step "REFUSED bad tag $TAG"; continue; }
  python -c "import sys; sys.path.insert(0, sys.argv[1]); import control_rules as R; R.check_eval_args([sys.argv[2], 'math500'])" \
    "$RUNBOOK/scripts" "$MODEL" 2>> $LOG || { step "REFUSED $TAG: model $MODEL"; continue; }
  case $MODEL in
    Qwen/Qwen3-0.6B) M=$STUDENT_06B;; Qwen/Qwen3-1.7B) M=$STUDENT_17B;; Qwen/Qwen3-8B) M=$TEACHER;; Qwen/*) M=$MODEL;;
    yang31210999/opd-lab-ckpts/*) P=${MODEL#yang31210999/opd-lab-ckpts/}; M=$CKPT_ROOT/adhoc_models/$P; fetch_ckpt "$P" || M="";;
    result:*) P=${MODEL#result:}; M=$CKPT_ROOT/adhoc_models/result/$P; fetch_result "$P" || M="";;
    *) M=$OPD_ROOT/${MODEL#local:}; has_weights $M || M=$CKPT_ROOT/${MODEL#local:};;
  esac
  [ -n "$M" ] && { has_weights $M || [ "${M#Qwen/}" != "$M" ]; } || { step "MODEL_MISSING $TAG: $MODEL"; printf '%s\t-\t-\t%s\tmodel missing\n' $TAG $MODEL >> $D/inputs.tsv; continue; }
  case $GENS in
    gen) F=$D/gen/term_${TAG}_math500_no_think_t0.6.jsonl
         P1=$OPD_ROOT/runs/PROBE-TERM/gen/term_${TAG}_math500_no_think_t0.6.jsonl   # v2.5.7k: PROBE-TERM2 reuses PROBE-TERM's
         [ $T != PROBE-TERM ] && [ ! -f $F ] && [ -f $P1 ] && F=$P1
         if [ ! -f $F ]; then
           step "generate $TAG (EVAL-v1 MATH500 T0.6 16K n4; not archived)"
           CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/eval/gen_eval.py --model $M --bench math500 --mode no_think --n 4 \
             --temperature 0.6 --top_p 0.95 --top_k 0 --max_tokens 16384 --chunk 50 --tag term_$TAG --out $D/gen >> $D/gen_eval.log 2>&1 || step "GEN_FAIL $TAG"
           python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
         fi
         [ -f $F ] && cp $F $D/up/gen/; SRC="generated here (EVAL-v1, not archived)";;
    archive:*.jsonl) P=${GENS#archive:}; F=$CKPT_ROOT/adhoc_models/$P
         [[ $P =~ ^[A-Za-z0-9._/-]+$ && $P != *..* ]] && fetch_ckpt "$P" || F=""; SRC="lab archive $P";;
    *.jsonl) [[ $GENS =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\.jsonl$ ]] && F=$OPD_ROOT/runs/$GENS || F=""; SRC="node runs/$GENS";;
    *) F=""; SRC="refused $GENS";;
  esac
  [ -n "$F" ] && [ -f "$F" ] || { step "GENERATIONS_MISSING $TAG: $GENS"; F=-; }
  [ "${MODE:-}" = think ] || MODE=nothink
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG" "$M" "$F" "$MODEL" "$SRC" "$MODE" >> $D/inputs.tsv
  step "input $TAG: $MODEL, $SRC"
done <<< "$(awk -v t="$QT" '$1 == "#term" && $2 == t {print $3, $4, $5, $6}' "$Q")"
[ -s $D/inputs.tsv ] || { step "no #term lines for $T"; exit 1; }
step "measure $(wc -l < $D/inputs.tsv) models"
CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/analysis/term_probe.py --teacher $TEACHER --inputs $D/inputs.tsv --out $D/report.json \
  --samples $D/samples.jsonl $([ $T = PROBE-TERM3 ] && echo --no_teacher) $([[ $T == PROBE-TERM4 || $T == PROBE-TERM4-4B ]] && echo --no_teacher --term4) >> $LOG 2>&1; rc=$?
python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
cp $D/report.json $D/samples.jsonl $D/up/ 2> /dev/null
python $RUNBOOK/scripts/remote.py put $D/up runs/$T >> $LOG 2>&1 || { step "report upload failed (kept in $D)"; rc=1; }
mkdir -p $D/logs; cp $LOG $D/inputs.tsv $D/logs/ 2> /dev/null; tail -c 200000 $D/gen_eval.log > $D/logs/gen_eval.log 2> /dev/null
python $RUNBOOK/scripts/remote.py put $D/logs logs/$(python $RUNBOOK/scripts/control.py node)/term_probe/$T >> $LOG 2>&1 || true
S=$(python -c "import json, sys; r = json.load(open(sys.argv[1]))['models']
print(' '.join(f\"{k}:{v.get('measures', {}).get('p_stop_nl', {}).get('mean')}/{v.get('measures', {}).get('marker_frac', {}).get('mean')}\" if 'error' not in v else f'{k}:{v[\"error\"]}' for k, v in r.items()))" $D/report.json 2> /dev/null || echo "report missing")
python $RUNBOOK/scripts/control.py event $T term_probe_done "rc=$rc $S" > /dev/null 2>&1
step "DONE rc=$rc $S"
[ $rc = 0 ] && touch $D/.done
exit $rc
