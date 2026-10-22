#!/usr/bin/env bash
# PROBE-PROG (runbook [v2.5.9z43]; Q-058, the reviewer's pre-registration 10-22): the gate of E12b-P5-prog / -prog-up, the CPU lane.
# The two arms continue a run's step 25 for 25 more steps: E12b-P5-prog = the 4K recipe (E12b-P5's) from E12b-P5-8k s25 (shrink
# the window), E12b-P5-prog-up = the 8K recipe from E12b-P5-s44 s25 (grow it). Both parents ran data.seed 4242, 64 prompts a
# step, so a run started from s25's weights would meet steps 1-25's prompts again; the arms read a continuation file instead
# (code/train/skip_order.py), where the sampler's first batch is the parents' step 26. What is new is checked here, no GPU:
#  1. the unit tests of code/train/test_skip_order.py with the python the arms train in (torch and pyarrow: the sampler and
#     parquet tests run); any failure, or torch / pyarrow missing, is a FAIL (nothing counts as skipped)
#  2. the sampler model against both parents' rollout dumps: the predicted prompts of steps 1 and 26 of dapo_train.parquet are
#     the prompts of rollouts/1.jsonl and 26.jsonl (each parent's own copy here, else its public result repository's)
#  3. the continuation file data/dapo_train_skip1600_s4242.parquet: built (atomic), every row kept, its batch 1 = each
#     parent's step 26 and its batch 25 = each parent's step 50
#  0. first: the two starts (runs/<parent>/global_step_25/actor/huggingface) are here, the parent's local copy or this node's copy
#     of the public result repository's (adhoc_models/result with fetch_model.py's .fetch_ok: size and sha256), else WAIT_INPUTS
#     (waiting, no attempt). [ops-lead's A1] The gate pulls no start itself (4 GB would hold the one CPU slot for hours): the
#     queue's "#fetch result:runs/..." lines pull them in the background (run_tasks.sh fetch_pass, at most 32 connections)
# report.json: the checks, the build, the starts; to logs/<node>/probe_prog/ (private), an event, .done. Usage: bash probe_prog.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=PROBE-PROG; D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
PARENTS=${PROG_PARENTS:-E12b-P5-8k E12b-P5-s44}; SS=${PROG_START_STEP:-25}; TBS=64; SEED=4242
TRAIN=$OPD_ROOT/data/dapo_train.parquet; SKIPF=$OPD_ROOT/data/dapo_train_skip$((SS * TBS))_s$SEED.parquet
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_PROG $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k, f in (("checks", "checks.txt"), ("build", "build.txt"), ("starts", "starts.txt")):
    try:
        r[k] = [x for x in open(os.path.join(d, f)).read().splitlines() if x.strip()]
    except OSError:
        pass
try:
    r["unit_tests"] = open(os.path.join(d, "unit_tests.txt")).read().strip().splitlines()[-1]
except (OSError, IndexError):
    pass
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/unit_tests.txt $D/checks.txt $D/build.txt $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_prog >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "probe_prog_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
: > $D/checks.txt; : > $D/build.txt; : > $D/starts.txt
# 0. the starts (no download here)
for P in $PARENTS; do
  L=$OPD_ROOT/runs/$P/ckpt/global_step_$SS/actor; R=runs/$P/global_step_$SS/actor/huggingface
  if has_weights $L/huggingface_bf16 || has_weights $L/huggingface; then echo "$P s$SS local" >> $D/starts.txt
  elif [ -f $CKPT_ROOT/adhoc_models/result/$R/.fetch_ok ]; then echo "$P s$SS adhoc_models/result/$R" >> $D/starts.txt
  else echo "WAIT_INPUTS $T: start $P s$SS not in (no local copy, no adhoc_models/result/$R/.fetch_ok)"; exit 1; fi
done
# 1. unit tests with the python the arms train in
PY=${PPG_PY:-$OPD_ROOT/venv/bin/python}; [ -x "$PY" ] || PY=python3
SO=${PPG_SKIP_ORDER:-$RUNBOOK/code/train/skip_order.py}   # (the stub tests replace the CLI, not the module the unit tests import)
( cd $RUNBOOK/code/train && CUDA_VISIBLE_DEVICES= $PY - > $D/unit_tests.txt 2>&1 <<'PY'
import importlib, inspect, os, sys, traceback
sys.path.insert(0, ".")
n = fails = 0
mod = importlib.import_module("test_skip_order")
for name, f in inspect.getmembers(mod, inspect.isfunction):
    if name.startswith("test_") and f.__module__ == "test_skip_order":
        n += 1
        try:
            f()
        except Exception:
            fails += 1; print("FAILED", name); traceback.print_exc()
for m in ("torch", "pyarrow"):   # the sampler and parquet tests return early without them: here they must run
    try:
        importlib.import_module(m)
    except ImportError:
        if os.environ.get("PPG_REQUIRE_NODE_LIBS", "1") == "1":
            fails += 1; print(f"FAILED {m} not importable: its test did not run")
print(f"{n - fails} passed, {fails} failed, 0 skipped")
sys.exit(1 if fails or not n else 0)
PY
); rc=$?
step "unit tests: $(tail -1 $D/unit_tests.txt)"
[ $rc = 0 ] || finish FAIL "unit tests: $(tail -1 $D/unit_tests.txt)"
[ -f $TRAIN ] || finish FAIL "no data/dapo_train.parquet"
# the parents' rollout dumps: their own copy, else the public result repository's (about 7-12 MB a step)
roll() {   # $1 = parent, $2 = step -> the path, or nothing
  local L=$OPD_ROOT/runs/$1/rollouts/$2.jsonl R=runs/$1/rollouts/$2.jsonl
  [ -s $L ] && { echo $L; return; }
  python3 $RUNBOOK/scripts/fetch_model.py "$(python3 $RUNBOOK/scripts/remote.py repo public)" $R $CKPT_ROOT/adhoc_models/result --task $T >> $LOG 2>&1 \
    && [ -s $CKPT_ROOT/adhoc_models/result/$R ] && echo $CKPT_ROOT/adhoc_models/result/$R; }
for P in $PARENTS; do for s in 1 $((SS + 1)) $((2 * SS)); do
  r=$(roll $P $s); [ -n "$r" ] || finish FAIL "no rollouts/$s.jsonl of $P"; eval "R_${P//-/_}_$s=$r"
done; done
# 2. the sampler model: steps 1 and SS+1 of the training file are each parent's
for P in $PARENTS; do
  V=R_${P//-/_}; a=${V}_1; b=${V}_$((SS + 1))
  $PY $SO check --train $TRAIN --tokenizer $STUDENT_17B --pair 0:${!a} --pair $SS:${!b} >> $D/checks.txt 2>> $LOG \
    || finish FAIL "the sampler model does not give $P's steps 1 / $((SS + 1)): $(grep CHECK_FAIL $D/checks.txt | head -1 | cut -c1-300)"
done
step "sampler model: $(grep -c CHECK_OK $D/checks.txt) steps of $(echo $PARENTS | wc -w) parents as predicted"
# 3. the continuation file
$PY $SO build --train $TRAIN --tokenizer $STUDENT_17B --out $SKIPF --skip_steps $SS > $D/build.txt 2>> $LOG \
  || finish FAIL "continuation file: $(cut -c1-300 $D/build.txt)"
for P in $PARENTS; do
  V=R_${P//-/_}; b=${V}_$((SS + 1)); c=${V}_$((2 * SS))
  $PY $SO check --train $SKIPF --tokenizer $STUDENT_17B --pair 0:${!b} --pair $((SS - 1)):${!c} >> $D/checks.txt 2>> $LOG \
    || finish FAIL "the continuation file's batches 1 / $SS are not $P's steps $((SS + 1)) / $((2 * SS)): $(grep CHECK_FAIL $D/checks.txt | head -1 | cut -c1-300)"
done
step "continuation file $(basename $SKIPF): $(cut -c1-200 $D/build.txt)"
finish PASS "unit tests $(tail -1 $D/unit_tests.txt); $(grep -c CHECK_OK $D/checks.txt) rollout steps as predicted ($(echo $PARENTS | tr ' ' '/'): steps 1 / $((SS + 1)) of dapo_train, 1 / $SS of the continuation file = $((SS + 1)) / $((2 * SS))); $(basename $SKIPF) every row kept; starts here: $(cut -d' ' -f1-3 $D/starts.txt | tr '\n' ';')"
