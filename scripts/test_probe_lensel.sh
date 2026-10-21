#!/usr/bin/env bash
# [v2.5.8m] Stub tests of probe_lensel.sh (no GPU, network or HF). Usage: bash scripts/test_probe_lensel.sh -> PROBE_LENSEL_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
MARK="# [opd-variants patch v2] Q-030 length-selected OPD"
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/runs $T/bin; cd $T   # nothing the stubs write lands in the caller's directory
  cp $SRC/scripts/probe_lensel.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  echo "x = 1" > $T/rt.py
  # the smoke: config, train.log with lensel metrics per step, lensel.jsonl, and the hook written into the installed trainer
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\nrollouts: ${FAKE_PROMPTS:-32} prompts x 16 per step, length selection short: 4 kept per prompt before the log-probs (lensel.jsonl), mini-batch 8 prompts\n" > $d/config.yaml
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
  echo "(TaskRunner pid=1) step:$i - lensel/n_all:512.0 - lensel/n_selected:${FAKE_SEL:-128.0} - lensel/groups:32.0 - lensel/all_len_mean:3000.5 - lensel/sel_len_mean:${FAKE_SELLEN:-1200.25} - timing_s/step:300" >> $d/train.log
  for g in $(seq 32); do echo '{"step": '$i', "len": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], "kept": [0,1,2,3]}' >> $d/lensel.jsonl; done
done
[ "${FAKE_NOHOOK:-0}" = 1 ] || echo "$HOOKMARK" >> $FAKE_RT
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  # the venv python: '-c' (find_spec) prints the fake trainer path; '-' (the unit-test runner) prints a summary
  printf '#!/usr/bin/env bash\nif [ "$1" = -c ]; then echo $FAKE_RT; exit 0; fi\necho "py unit GOPD_RAY_TRAINER=$GOPD_RAY_TRAINER" >> $CALLS; cat > /dev/null\n[ "${FAKE_UNIT_RC:-0}" = 0 ] && { echo "9 passed, 0 failed"; exit 0; }; echo "8 passed, 1 failed"; exit 1\n' > $T/bin/fakepy; chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PLS_WAIT_STEP=0 PLS_PY=$T/bin/fakepy FAKE_RT=$T/rt.py HOOKMARK="$MARK"; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-LENSEL/report.json "$1"; }
# 1. PASS: unit tests against the installed trainer, the 2-step smoke of the short arm, report, upload, event
setup
bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "PASS" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-LENSEL/.done ]'
ok "unit tests run against the installed ray_trainer.py; smoke = 2 steps of E10b-P5lenS" 'grep -q "^py unit GOPD_RAY_TRAINER=$T/rt.py$" $CALLS && grep -q "^run_arm 4 E10b-P5lenS SMOKE=1 STEPS=2 PROBE_TASK=PROBE-LENSEL-smoke$" $CALLS'
ok "report upload and event" 'grep -q "^remote put $T/root/runs/PROBE-LENSEL/up logs/abcd1234/probe_lensel/PROBE-LENSEL" $CALLS && grep -q "^control event PROBE-LENSEL probe_lensel_pass" $CALLS'
rm -rf $T
# 2. unit tests fail: FAIL before any smoke
setup
FAKE_UNIT_RC=1 bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "unit-test failure is FAIL, no smoke" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"1 failed"* ]] && ! grep -q "^run_arm" $CALLS'
rm -rf $T
# 3. the selection did not reach the update (n_selected 512); 4. kept longer than all; 5. a step missing; 6. hook missing
setup
FAKE_SEL=512.0 bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "n_selected != 128 is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"n_selected"* ]]'
rm -rf $T
setup
FAKE_SELLEN=3500.0 bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "kept mean length above all is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"kept mean length"* ]]'
rm -rf $T
setup
FAKE_LOGSTEPS=1 bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "a missing step (and short jsonl) is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"expected 1..2"* && "$(res reason)" == *"lensel.jsonl: 32 rows"* ]]'
rm -rf $T
setup
FAKE_NOHOOK=1 bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "no hook in the installed trainer is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"holds the hook 0 times"* ]]'
rm -rf $T
# 7. wrong batch shape in the config; 8. OOM; 9. peak over the limit
setup
FAKE_PROMPTS=128 bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "config without 32 x 16 short is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"32 x 16"* ]]'
rm -rf $T
setup
FAKE_OOM=1 bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"exited"* || "$(res reason)" == *"out of memory"* ]]'
rm -rf $T
setup
FAKE_MEM=0 PLS_PEAK_MIB=0 bash $T/rb/scripts/probe_lensel.sh 4 > /dev/null 2>&1
ok "peak over the limit is FAIL" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"peak"* ]]'
rm -rf $T
cd $SRC
[ $fails = 0 ] && echo "PROBE_LENSEL_TEST PASS ($n checks)" || echo "PROBE_LENSEL_TEST FAIL ($fails of $n)"
