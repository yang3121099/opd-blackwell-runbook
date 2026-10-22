#!/usr/bin/env bash
# [v2.5.9z46] Stub tests of probe_x5.sh (PROBE-X5; no GPU, network or HF): the real unit tests of test_opd_x5.py against a fake installed
# ref_input_utils.py (the P5 patch's block in it), x5_probe.py and a 3-step smoke stubbed; the smoke's x5.jsonl as it should be or not.
# Without torch here the runner counts one failure unless PX5_REQUIRE_TORCH=0 (on the node torch is there and required).
# Usage: bash scripts/test_probe_x5.sh -> PROBE_X5_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/models/Qwen2.5-Math-1.5B-Instruct $T/root/models/DeepSeek-R1-Distill-Qwen-1.5B $T/bin $T/py/verl/trainer/ppo
  cp $SRC/scripts/probe_x5.sh $T/rb/scripts/
  cp $SRC/code/patches/opd_x5.py $SRC/code/patches/x5_patch.py $SRC/code/patches/route_patch.py $SRC/code/patches/test_opd_x5.py $T/rb/code/patches/
  : > $T/root/models/Qwen2.5-Math-1.5B-Instruct/.fetch_ok; : > $T/root/models/DeepSeek-R1-Distill-Qwen-1.5B/.fetch_ok
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  : > $T/py/verl/__init__.py; : > $T/py/verl/trainer/__init__.py; : > $T/py/verl/trainer/ppo/__init__.py
  python3 - $T/py/verl/trainer/ppo/ref_input_utils.py $T/rb/code/patches <<'PY'
import sys; sys.path.insert(0, sys.argv[2]); import test_opd_x5 as TX
open(sys.argv[1], "w").write(TX.SNIPPET)   # the installed file's shape: the P5 patch's block, response_length, return batch
PY
  cat > $T/probe.py <<'PY'
import json, os, sys
a = sys.argv[1:]; open(os.environ["CALLS"], "a").write("x5_probe " + " ".join(a) + "\n")
if os.environ.get("FAKE_PROBE_FAIL") == a[0]:
    sys.exit(1)
out = a[a.index("--out") + 1]
if a[0] == "student":
    json.dump({"n": 64, "term_151645": 50, "term_151643": 2, "truncated": 12, "share_151645": 0.7812, "share_151643": 0.0312, "acc": 0.7},
              open(os.path.join(out, "student_summary.json"), "w"))
else:
    json.dump({"n": 64, "completion": 0.95, "accuracy": 0.8, "max_tokens": 16384, "p_stop_think": {"mean": 0.12, "median": 0.05, "n": 52},
               "p_stop_empty": {"mean": 0.7, "median": 0.8, "n": 52}, "x5e": True}, open(os.path.join(out, "teacher.json"), "w"))
PY
  # the smoke: three steps of 128 rows; FAKE_X5 = ok | masked | unmapped | tail2 | rows64
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\n${FAKE_X5_NOTE-x5: a student without a thinking mode (...)}\n" > $d/config.yaml
for i in 1 2 3; do echo "(TaskRunner pid=1) step:$i - timing_s/step:120" >> $d/train.log; done
python3 - $d/x5.jsonl <<'PY'
import json, os, sys
mode = os.environ.get("FAKE_X5", "ok"); rows = 64 if mode == "rows64" else 128
with open(sys.argv[1], "w") as f:
    for c in range(3):
        rec = {"call": c + 1, "rows": rows, "terminated": 100, "term_151645": 98, "term_151643": 2, "mapped": 99 if mode == "unmapped" else 100,
               "truncated": rows - 100, "empty": 0, "masked": {"151644": 1} if mode == "masked" else {},
               "prompt_tail": "1+1?<｜Assistant｜><think>\n" + ("<think>\n" if mode == "tail2" else ""), "empty_think": False}
        f.write(json.dumps(rec) + "\n")
PY
exit ${FAKE_RC:-0}
SH
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PX5_WAIT_STEP=0 PX5_PY=python3 PX5_REQUIRE_TORCH=0 PX5_PROBE=$T/probe.py PYTHONPATH=$T/py; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])).get(sys.argv[2], ''); print(v if isinstance(v, str) else json.dumps(v))" $T/root/runs/PROBE-X5/report.json "$1"; }
setup; bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1
ok "PASS: unit tests (the hooks into the installed file's shape), student / teacher measured, a 3-step smoke of E16-X5, upload, event, .done" '[ "$(res result)" = PASS ] && grep -q "passed, 0 failed, 0 skipped$" $T/root/runs/PROBE-X5/unit_tests.txt && grep -q "^run_arm 4 E16-X5 SMOKE=1 STEPS=3 PROBE_TASK=PROBE-X5-smoke$" $CALLS && grep -q "^remote put $T/root/runs/PROBE-X5/up logs/abcd1234/probe_x5$" $CALLS && grep -q "^control event PROBE-X5 probe_x5_pass " $CALLS && [ -f $T/root/runs/PROBE-X5/.done ]'
ok "the student first (its model, 64), then the teacher (its model), each with the probe's dir" '[ "$(grep "^x5_probe" $CALLS | head -1)" = "x5_probe student --model $T/root/models/Qwen2.5-Math-1.5B-Instruct --out $T/root/runs/PROBE-X5 --n 64" ] && [ "$(grep "^x5_probe" $CALLS | tail -1)" = "x5_probe teacher --model $T/root/models/DeepSeek-R1-Distill-Qwen-1.5B --out $T/root/runs/PROBE-X5 --n 64" ]'
ok "the report: the student, the teacher, the smoke totals, x5e from the teacher; the reason names the numbers" '[ "$(res x5e)" = true ] && [[ "$(res smoke)" == *"\"terminated\": 300"* ]] && [[ "$(res teacher)" == *"\"completion\": 0.95"* ]] && [[ "$(res reason)" == *"think 0.12, empty 0.7 (n 52; x5e True); smoke 384 rows, 300 terminated (294 / 6) all mapped, no special id inside"* ]]'
out=$(bash $T/rb/scripts/probe_x5.sh 4 2>&1); ok "a second run after .done does nothing" '[ "$out" = "TASK_ALREADY_DONE PROBE-X5" ]'; rm -rf $T
check() { ok "$1" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"'"$2"'"* ]] && grep -q "^control event PROBE-X5 probe_x5_fail " $CALLS'; }
setup; rm $T/root/models/DeepSeek-R1-Distill-Qwen-1.5B/.fetch_ok; out=$(bash $T/rb/scripts/probe_x5.sh 4 2>&1); rc=$?
ok "the teacher not in: WAIT_INPUTS, exit 1, nothing run, no report" '[ $rc = 1 ] && [ "$out" = "WAIT_INPUTS PROBE-X5: DeepSeek-R1-Distill-Qwen-1.5B not in (no .fetch_ok)" ] && [ ! -s $CALLS ] && [ ! -e $T/root/runs/PROBE-X5/report.json ]'; rm -rf $T
setup; FAKE_X5=masked bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1; check "a special id inside a response is FAIL" "special ids inside responses {'151644': 1} (expected none)"; rm -rf $T
setup; FAKE_X5=unmapped bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1; check "a terminated response not mapped is FAIL" "mapped 99 != terminated 100"; rm -rf $T
setup; FAKE_X5=tail2 bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1; check "two <think> at the teacher prompt's end is FAIL" "teacher prompt tail"; rm -rf $T
setup; FAKE_X5=rows64 bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1; check "rows != 128 is FAIL" "64 rows, expected 128"; rm -rf $T
setup; FAKE_X5_NOTE="" bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1; check "a smoke config without x5: is FAIL" "does not name x5"; rm -rf $T
setup; FAKE_PROBE_FAIL=teacher bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1; check "the teacher's measurements failing is FAIL, no smoke" "the teacher's measurements did not run"; ok "no smoke after it" '! grep -q "^run_arm" $CALLS'; rm -rf $T
setup; FAKE_RC=1 bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1; check "a smoke exiting 1 is FAIL" "smoke exited 1"; rm -rf $T
setup; PX5_REQUIRE_TORCH=1 bash $T/rb/scripts/probe_x5.sh 4 > /dev/null 2>&1
if python3 -c "import torch" 2> /dev/null; then ok "torch here: the unit tests pass" '[ "$(res result)" = PASS ]'; else check "no torch (required on the node) is FAIL, not a skip" "unit tests: "; ok "nothing measured after it" '! grep -q "^x5_probe" $CALLS'; fi; rm -rf $T
[ $fails = 0 ] && echo "PROBE_X5_TEST PASS ($n checks)" || echo "PROBE_X5_TEST FAIL ($fails of $n)"
