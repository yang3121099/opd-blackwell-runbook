#!/usr/bin/env bash
# [v2.5.9z43] Stub tests of probe_prog.sh (PROBE-PROG, the CPU lane; no GPU, network or HF): the real unit tests of
# test_skip_order.py; skip_order.py's command line, fetch_model.py, remote.py and control.py stubbed. E12b-P5-8k's rollouts are
# local; E12b-P5-s44's rollouts are fetched. Without torch / pyarrow
# here the runner counts failures unless PPG_REQUIRE_NODE_LIBS=0 (on the node both are there and required). [ops-lead's A1] The gate
# pulls no start: E12b-P5-8k s25 is in adhoc_models/result (its .fetch_ok), E12b-P5-s44 s25 local; one missing = WAIT_INPUTS.
# Usage: bash scripts/test_probe_prog.sh -> PROBE_PROG_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/train $T/rb/code/analysis $T/root/data $T/ck $T/tok
  cp $SRC/scripts/probe_prog.sh $T/rb/scripts/
  cp $SRC/code/train/skip_order.py $SRC/code/train/test_skip_order.py $T/rb/code/train/; cp $SRC/code/analysis/diff_labels.py $T/rb/code/analysis/
  printf 'export OPD_ROOT=%s/root RUNBOOK=%s/rb CKPT_ROOT=%s/ck STUDENT_17B=%s/tok\nhas_weights() { ls "$1"/*.safetensors > /dev/null 2>&1; }\n' $T $T $T $T > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:3] == ["repo", "public"] and print("me/pub")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/fetch_model.py <<'PY'
import os, sys
repo, path, dest = sys.argv[1:4]
open(os.environ["CALLS"], "a").write(f"fetch {repo} {path} {dest} timeout={os.environ.get('FETCH_TIMEOUT', '')}\n")
if os.environ.get("FAKE_FETCH_FAIL") and os.environ["FAKE_FETCH_FAIL"] in path:
    sys.exit(1)
p = os.path.join(dest, path)
if path.endswith(".jsonl"):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write("{}\n")
else:
    os.makedirs(p, exist_ok=True); open(os.path.join(p, "model.safetensors"), "w").close()
PY
  cat > $T/so.py <<'PY'
import os, sys
a = sys.argv[1:]; open(os.environ["CALLS"], "a").write("skip_order " + " ".join(a) + "\n")
v = lambda k: a[a.index(k) + 1]
bad = os.environ.get("FAKE_BAD", "")   # "<train name part>:<batch>"
if a[0] == "build":
    open(v("--out"), "w").close()
    print("BUILD_FAIL {}" if os.environ.get("FAKE_BUILD_FAIL") else '{"rows": 17398, "fit": 17000, "skip": 1600}'.join(["BUILD_OK ", ""])); sys.exit(1 if os.environ.get("FAKE_BUILD_FAIL") else 0)
rc = 0
for i, x in enumerate(a):
    if x == "--pair":
        b, p = a[i + 1].split(":", 1); fail = bool(bad) and bad.split(":")[0] in os.path.basename(v("--train")) and bad.split(":")[1] == b
        print(("CHECK_FAIL " if fail else "CHECK_OK ") + f'{{"batch": {b}, "rollouts": "{os.path.basename(p)}"}}'); rc |= fail
sys.exit(rc)
PY
  : > $T/root/data/dapo_train.parquet
  mkdir -p $T/root/runs/E12b-P5-8k/rollouts $T/root/runs/E12b-P5-s44/ckpt/global_step_25/actor/huggingface
  for s in 1 26 50; do echo '{}' > $T/root/runs/E12b-P5-8k/rollouts/$s.jsonl; done
  : > $T/root/runs/E12b-P5-s44/ckpt/global_step_25/actor/huggingface/model.safetensors
  mkdir -p $T/ck/adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface; : > $T/ck/adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface/.fetch_ok
  export CALLS=$T/calls PPG_PY=python3 PPG_REQUIRE_NODE_LIBS=0 PPG_SKIP_ORDER=$T/so.py; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])).get(sys.argv[2], ''); print(v if isinstance(v, str) else '|'.join(v))" $T/root/runs/PROBE-PROG/report.json "$1"; }
R8=$'\n'
setup; bash $T/rb/scripts/probe_prog.sh > /dev/null 2>&1
ok "PASS: unit tests 0 failed 0 skipped, 8 rollout steps as predicted, the file built, starts here, upload, event, .done" '[ "$(res result)" = PASS ] && grep -q "passed, 0 failed, 0 skipped$" $T/root/runs/PROBE-PROG/unit_tests.txt && [ $(grep -c "^CHECK_OK" $T/root/runs/PROBE-PROG/checks.txt) = 8 ] && grep -q "^remote put $T/root/runs/PROBE-PROG/up logs/abcd1234/probe_prog$" $CALLS && grep -q "^control event PROBE-PROG probe_prog_pass " $CALLS && [ -f $T/root/runs/PROBE-PROG/.done ]'
ok "the sampler model: dapo_train steps 1 / 26 = batches 0 / 25, for both parents" 'grep -q "^skip_order check --train $T/root/data/dapo_train.parquet --tokenizer $T/tok --pair 0:$T/root/runs/E12b-P5-8k/rollouts/1.jsonl --pair 25:$T/root/runs/E12b-P5-8k/rollouts/26.jsonl$" $CALLS && grep -q "^skip_order check --train $T/root/data/dapo_train.parquet --tokenizer $T/tok --pair 0:$T/ck/adhoc_models/result/runs/E12b-P5-s44/rollouts/1.jsonl --pair 25:$T/ck/adhoc_models/result/runs/E12b-P5-s44/rollouts/26.jsonl$" $CALLS'
ok "the continuation file: dapo_train_skip1600_s4242.parquet, skip 25 steps; its batches 0 / 24 = steps 26 / 50" 'grep -q "^skip_order build --train $T/root/data/dapo_train.parquet --tokenizer $T/tok --out $T/root/data/dapo_train_skip1600_s4242.parquet --skip_steps 25$" $CALLS && grep -q "^skip_order check --train $T/root/data/dapo_train_skip1600_s4242.parquet --tokenizer $T/tok --pair 0:$T/root/runs/E12b-P5-8k/rollouts/26.jsonl --pair 24:$T/root/runs/E12b-P5-8k/rollouts/50.jsonl$" $CALLS && grep -q "^skip_order check --train $T/root/data/dapo_train_skip1600_s4242.parquet .* --pair 24:$T/ck/adhoc_models/result/runs/E12b-P5-s44/rollouts/50.jsonl$" $CALLS'
ok "rollouts: the local copy first (8k: none fetched), else the public repository's (s44: 1, 26, 50)" '! grep -q "^fetch me/pub runs/E12b-P5-8k/rollouts" $CALLS && [ $(grep -c "^fetch me/pub runs/E12b-P5-s44/rollouts/[0-9]*.jsonl $T/ck/adhoc_models/result " $CALLS) = 3 ]'
ok "starts: 8k s25 in adhoc_models/result (its .fetch_ok), s44 s25 local; the gate pulls neither; both in the report" '! grep -q "^fetch .*global_step_25" $CALLS && [ "$(res starts)" = "E12b-P5-8k s25 adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface|E12b-P5-s44 s25 local" ]'
ok "the reason names the steps and the file" '[[ "$(res reason)" == *"8 rollout steps as predicted (E12b-P5-8k/E12b-P5-s44: steps 1 / 26 of dapo_train, 1 / 25 of the continuation file = 26 / 50); dapo_train_skip1600_s4242.parquet every row kept; starts here: E12b-P5-8k s25 adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface;E12b-P5-s44 s25 local;"* ]]'
out=$(bash $T/rb/scripts/probe_prog.sh 2>&1); ok "a second run after .done does nothing" '[ "$out" = "TASK_ALREADY_DONE PROBE-PROG" ]'
rm -rf $T
check() { ok "$1" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"'"$2"'"* ]] && grep -q "^control event PROBE-PROG probe_prog_fail " $CALLS && [ -f $T/root/runs/PROBE-PROG/.done ]'; }
setup; PPG_REQUIRE_NODE_LIBS=1 bash $T/rb/scripts/probe_prog.sh > /dev/null 2>&1; check "no torch / pyarrow (required on the node) is FAIL, not a skip" "unit tests: "
ok "the missing libraries named, nothing checked or built" '{ python3 -c "import torch" 2> /dev/null || grep -q "^FAILED torch not importable" $T/root/runs/PROBE-PROG/unit_tests.txt; } && ! grep -q "^skip_order" $CALLS'; rm -rf $T
setup; FAKE_BAD=dapo_train.parquet:25 bash $T/rb/scripts/probe_prog.sh > /dev/null 2>&1; check "a parent step 26 not as predicted is FAIL, before any build" "the sampler model does not give E12b-P5-8k's steps 1 / 26"; ok "no build after a failed check" '! grep -q "^skip_order build" $CALLS'; rm -rf $T
setup; FAKE_BUILD_FAIL=1 bash $T/rb/scripts/probe_prog.sh > /dev/null 2>&1; check "a build that loses rows is FAIL" "continuation file: BUILD_FAIL"; rm -rf $T
setup; FAKE_BAD=skip1600:0 bash $T/rb/scripts/probe_prog.sh > /dev/null 2>&1; check "a continuation file whose batch 0 is not step 26 is FAIL, no start fetched" "the continuation file's batches 1 / 25 are not E12b-P5-8k's steps 26 / 50"; ok "no start fetched after it" '! grep -q "^fetch .*global_step_25" $CALLS'; rm -rf $T
setup; FAKE_FETCH_FAIL=E12b-P5-s44/rollouts/26 bash $T/rb/scripts/probe_prog.sh > /dev/null 2>&1; check "a rollout dump neither here nor fetched is FAIL" "no rollouts/26.jsonl of E12b-P5-s44"; rm -rf $T
setup; printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["repo", "public"] and print("me/pub"); sys.argv[1:2] == ["backoff"] and print(os.environ.get("FAKE_BACKOFF", "0"))\n' > $T/rb/scripts/remote.py
FAKE_BACKOFF=40 FAKE_FETCH_FAIL=E12b-P5-s44/rollouts/26 bash $T/rb/scripts/probe_prog.sh > $T/outbo 2>&1; rc=$?
ok "[v2.5.9z52] the same during the HF 429 backoff: WAIT_INPUTS (no attempt), exit 1, no report" '[ $rc = 1 ] && grep -q "^WAIT_INPUTS PROBE-PROG: rollouts/26.jsonl of E12b-P5-s44 not in (the HF 429 backoff lasts)$" $T/outbo && [ ! -e $T/root/runs/PROBE-PROG/report.json ]'; rm -rf $T
setup; rm $T/ck/adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface/.fetch_ok; out=$(bash $T/rb/scripts/probe_prog.sh 2>&1); rc=$?
ok "a start not in yet (no .fetch_ok) is WAIT_INPUTS: exit 1, no report, no .done, nothing run or pulled" '[ $rc = 1 ] && [[ "$out" == "WAIT_INPUTS PROBE-PROG: start E12b-P5-8k s25 not in (no local copy, no adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface/.fetch_ok)" ]] && [ ! -f $T/root/runs/PROBE-PROG/report.json ] && [ ! -f $T/root/runs/PROBE-PROG/.done ] && [ ! -s $CALLS ] && [ ! -e $T/root/runs/PROBE-PROG/unit_tests.txt ]'
mkdir -p $T/ck/adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface; : > $T/ck/adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface/model.safetensors
out=$(bash $T/rb/scripts/probe_prog.sh 2>&1); ok "weights without .fetch_ok (a pull under way) still wait" '[[ "$out" == "WAIT_INPUTS PROBE-PROG: start E12b-P5-8k s25 not in"* ]]'
: > $T/ck/adhoc_models/result/runs/E12b-P5-8k/global_step_25/actor/huggingface/.fetch_ok; bash $T/rb/scripts/probe_prog.sh > /dev/null 2>&1; ok "once it is in, the same run passes" '[ "$(res result)" = PASS ]'; rm -rf $T
setup; rm $T/root/data/dapo_train.parquet; bash $T/rb/scripts/probe_prog.sh > /dev/null 2>&1; check "no training data is FAIL" "no data/dapo_train.parquet"; rm -rf $T
[ $fails = 0 ] && echo "PROBE_PROG_TEST PASS ($n checks)" || echo "PROBE_PROG_TEST FAIL ($fails of $n)"
