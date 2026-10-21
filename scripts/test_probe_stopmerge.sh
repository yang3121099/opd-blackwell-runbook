#!/usr/bin/env bash
# [v2.5.9l] Stub tests of probe_stopmerge.sh (no GPU, network or HF). Usage: bash scripts/test_probe_stopmerge.sh -> PROBE_STOPMERGE_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
MARK="# [opd-variants patch v2] [v2.5.9l] stop-merged OPD"
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/runs $T/bin $T/verl/trainer/ppo $T/verl/workers/actor; cd $T
  cp $SRC/scripts/probe_stopmerge.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  echo "x = 1" > $T/verl/trainer/ppo/ref_input_utils.py; echo "x = 1" > $T/verl/workers/actor/dp_actor.py
  # the smoke: config, train.log per step, stopmerge.jsonl (two reference micro-batches a step), the hooks in the installed files
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\nteacher: Qwen/Qwen3-8B (ref; think-template=1; param_offload=1)${FAKE_CFG-; stop-merged after the first box (opd_stopmerge.py)}\n" > $d/config.yaml
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
  echo "(TaskRunner pid=1) step:$i - critic/score/mean:${FAKE_SCORE:-0.55} - timing_s/step:300" >> $d/train.log
  for mb in 1 2; do echo '{"rows": 64, "rows_box": '${FAKE_BOX:-40}', "merged": '${FAKE_MERGED:-30}', "eos_total": 60, "eos_before_or_no_box": 30, "gain_mean": 2.5, "gain_min": '${FAKE_GMIN:-0.0}', "violations": '${FAKE_VIOL:-0}', "stop_mass_after_box_mean": 0.6}' >> $d/stopmerge.jsonl; done
done
if [ "${FAKE_NOHOOK:-0}" != 1 ]; then echo "$HOOKMARK" >> $FAKE_VD/trainer/ppo/ref_input_utils.py; for k in 1 2 3 4 5 6; do echo "$HOOKMARK" >> $FAKE_VD/workers/actor/dp_actor.py; done; fi
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  # the venv python: '-c' (find_spec) prints the fake verl package dir; '-' (the unit-test runner) prints a summary
  printf '#!/usr/bin/env bash\nif [ "$1" = -c ]; then echo $FAKE_VD; exit 0; fi\necho "py unit GOPD_VERL=$GOPD_VERL" >> $CALLS; cat > /dev/null\n[ "${FAKE_UNIT_RC:-0}" = 0 ] && { echo "8 passed, 0 failed"; exit 0; }; echo "7 passed, 1 failed"; exit 1\n' > $T/bin/fakepy; chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PSM_WAIT_STEP=0 PSM_PY=$T/bin/fakepy FAKE_VD=$T/verl HOOKMARK="$MARK"; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-STOPMERGE/report.json "$1"; }
# 1. PASS: unit tests against the installed verl, the 3-step smoke, the summary, report, upload, event
setup
bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1
ok "PASS" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-STOPMERGE/.done ]'
ok "unit tests against the installed verl; smoke = 3 steps of E10b-P5stopmerge" 'grep -q "^py unit GOPD_VERL=$T/verl$" $CALLS && grep -q "^run_arm 2 E10b-P5stopmerge SMOKE=1 STEPS=3 PROBE_TASK=PROBE-STOPMERGE-smoke$" $CALLS'
ok "summary: 62% with a box, 0.47 merges per response, gain 2.5, hooks 1 / 6" '[ "$(res "\" \".join(str(x) for x in (v[\"checks\"][\"stopmerge\"][\"rows_box_share\"], v[\"checks\"][\"stopmerge\"][\"merged_per_row\"], v[\"checks\"][\"stopmerge\"][\"gain_mean\"], v[\"checks\"][\"hooks\"][\"workers/actor/dp_actor.py\"]))")" = "0.625 0.4688 2.5 6" ] && [[ "$(res "v[\"reason\"]")" == *"62% of responses with a box"* ]]'
ok "report upload (with stopmerge.jsonl) and event" 'grep -q "^remote put $T/root/runs/PROBE-STOPMERGE/up logs/abcd1234/probe_stopmerge/PROBE-STOPMERGE" $CALLS && [ -f $T/root/runs/PROBE-STOPMERGE/up/stopmerge.jsonl ] && grep -q "^control event PROBE-STOPMERGE probe_stopmerge_pass" $CALLS'
cd $SRC; rm -rf $T
# 2. unit tests fail: FAIL before any smoke
setup; FAKE_UNIT_RC=1 bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1
ok "unit-test failure is FAIL, no smoke" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"1 failed"* ]] && ! grep -q "^run_arm" $CALLS'
cd $SRC; rm -rf $T
# 3. the smoke's own failure modes, one at a time
check() {   # $1 = label, $2 = the expected part of the reason; the environment of the call sets the fault
  ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'
}
setup; FAKE_VIOL=1 bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1; check "a merge at or before a box is FAIL" "merges at or before a box"; cd $SRC; rm -rf $T
setup; FAKE_MERGED=0 bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1; check "no merge at all is FAIL" "no student stop after a box was merged"; cd $SRC; rm -rf $T
setup; FAKE_BOX=0 FAKE_MERGED=0 bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1; check "no box found is FAIL" "no response with a complete box"; cd $SRC; rm -rf $T
setup; FAKE_GMIN=-0.3 bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1; check "a negative gain is FAIL" "below the teacher's own"; cd $SRC; rm -rf $T
setup; FAKE_NOHOOK=1 bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1; check "hooks missing in the installed files is FAIL" "the installed hooks"; cd $SRC; rm -rf $T
setup; FAKE_SCORE=1.4 bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1; check "a score outside [0, 1] is FAIL" "a score outside [0, 1]"; cd $SRC; rm -rf $T
setup; FAKE_CFG="" bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1; check "a config without the merge is FAIL" "does not name the stop merge"; cd $SRC; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_stopmerge.sh 2 > /dev/null 2>&1; check "OOM is FAIL" "smoke exited"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_STOPMERGE_TEST PASS ($n checks)" || echo "PROBE_STOPMERGE_TEST FAIL ($fails of $n)"
