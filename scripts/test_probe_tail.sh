#!/usr/bin/env bash
# [v2.5.9n] Stub tests of probe_tail.sh (no GPU, network or HF). Usage: bash scripts/test_probe_tail.sh -> PROBE_TAIL_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
MARK="# [opd-variants patch v2] [v2.5.9n] tail-swap OPD"
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/runs $T/bin; cd $T
  cp $SRC/scripts/probe_tail.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  echo "x = 1" > $T/rt.py
  # the smoke: config, train.log with tail/* per step, tailswap.jsonl, the hook in the installed trainer
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf "task: $2\nteacher: Qwen/Qwen3-8B (ref; think-template=1; param_offload=1)${FAKE_CFG-; tail-swapped: the non-thinking reading after the first box (opd_tailswap.py)}\n" > $d/config.yaml
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
  echo "(TaskRunner pid=1) step:$i - critic/score/mean:${FAKE_SCORE:-0.55} - tail/rows:128.0 - tail/rows_box:${FAKE_BOX:-80.0} - tail/swapped:${FAKE_SW:-9000.0} - tail/swapped_frac:0.02 - tail/delta_mean:1.5 - tail/lp_think_finite:1.0 - tail/lp_plain_finite:${FAKE_FIN:-1.0} - tail/violations:${FAKE_VIOL:-0.0} - timing_s/step:400" >> $d/train.log
  echo '{"rows": 128, "swapped": 9000}' >> $d/tailswap.jsonl
done
[ "${FAKE_NOHOOK:-0}" = 1 ] || echo "$HOOKMARK" >> $FAKE_RT
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  printf '#!/usr/bin/env bash\nif [ "$1" = -c ]; then echo $FAKE_RT; exit 0; fi\necho "py unit GOPD_RAY_TRAINER=$GOPD_RAY_TRAINER" >> $CALLS; cat > /dev/null\n[ "${FAKE_UNIT_RC:-0}" = 0 ] && { echo "5 passed, 0 failed"; exit 0; }; echo "4 passed, 1 failed"; exit 1\n' > $T/bin/fakepy; chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PTL_WAIT_STEP=0 PTL_PY=$T/bin/fakepy FAKE_RT=$T/rt.py HOOKMARK="$MARK"; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-TAIL/report.json "$1"; }
setup
bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1
ok "PASS" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-TAIL/.done ]'
ok "unit tests against the installed trainer; smoke = 3 steps of E10b-P5tail" 'grep -q "^py unit GOPD_RAY_TRAINER=$T/rt.py$" $CALLS && grep -q "^run_arm 3 E10b-P5tail SMOKE=1 STEPS=3 PROBE_TASK=PROBE-TAIL-smoke$" $CALLS'
ok "summary: 62% with a box, 2% swapped, delta 1.5; report, tailswap.jsonl, event" '[ "$(res "\" \".join(str(x) for x in (v[\"checks\"][\"tail\"][\"rows_box_share\"], v[\"checks\"][\"tail\"][\"swapped_frac_mean\"], v[\"checks\"][\"tail\"][\"delta_mean\"]))")" = "0.625 0.02 1.5" ] && grep -q "^remote put $T/root/runs/PROBE-TAIL/up logs/abcd1234/probe_tail/PROBE-TAIL" $CALLS && [ -f $T/root/runs/PROBE-TAIL/up/tailswap.jsonl ] && grep -q "^control event PROBE-TAIL probe_tail_pass" $CALLS'
cd $SRC; rm -rf $T
setup; FAKE_UNIT_RC=1 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1
ok "unit-test failure is FAIL, no smoke" '[ "$(res "v[\"result\"]")" = FAIL ] && ! grep -q "^run_arm" $CALLS'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_FIN=0.0 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; check "a non-finite reading is FAIL" "a reading not finite"; cd $SRC; rm -rf $T
setup; FAKE_VIOL=2.0 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; check "a swap at or before a box is FAIL" "swaps at or before a box"; cd $SRC; rm -rf $T
setup; FAKE_SW=0.0 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; check "nothing swapped is FAIL" "no position was swapped"; cd $SRC; rm -rf $T
setup; FAKE_BOX=0.0 FAKE_SW=0.0 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; check "no box is FAIL" "no response with a complete box"; cd $SRC; rm -rf $T
setup; FAKE_NOHOOK=1 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; check "the hook missing in the installed trainer is FAIL" "holds the hook 0 times"; cd $SRC; rm -rf $T
setup; FAKE_LOGSTEPS=2 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; check "a missing step is FAIL" "expected 1..3"; cd $SRC; rm -rf $T
setup; FAKE_CFG="" bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; check "a config without the swap is FAIL" "does not name the tail swap"; cd $SRC; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; check "OOM is FAIL" "smoke exited"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_TAIL_TEST PASS ($n checks)" || echo "PROBE_TAIL_TEST FAIL ($fails of $n)"
