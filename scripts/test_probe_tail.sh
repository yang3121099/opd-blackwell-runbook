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
printf "task: $2\nteacher: Qwen/Qwen3-8B (ref; think-template=1; param_offload=1)${FAKE_CFG-; tail-swapped: the non-thinking reading after the first box (opd_tailswap.py)}${FAKE_CFG2:-}\n" > $d/config.yaml
for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
  echo "(TaskRunner pid=1) step:$i - critic/score/mean:${FAKE_SCORE:-0.55} - tail/rows:128.0 - tail/rows_box:${FAKE_BOX:-80.0} - tail/swapped:${FAKE_SW:-9000.0} - tail/swapped_frac:0.02 - tail/delta_mean:1.5 - tail/lp_think_finite:1.0 - tail/lp_plain_finite:${FAKE_FIN:-1.0} - tail/violations:${FAKE_VIOL:-0.0}${FAKE_BUDM:+ - $FAKE_BUDM} - timing_s/step:400" >> $d/train.log
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
# [v2.5.9z73] (Q-061) PROBE-TAIL-BUD: the same probe on E10b-P5bud2k with the budget's checks
BCFG="; budget: the switch at min(the first box, 2048 tokens)"
BM="tail/budget:2048.0 - tail/rows_switched:90.0 - tail/rows_by_box:70.0 - tail/rows_by_budget:20.0 - tail/switch_p10:300.0 - tail/switch_p50:1200.0 - tail/switch_p90:2048.0 - tail/switch_max:2048.0 - tail/budget_late:0.0"
resb() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-TAIL-BUD/report.json "$1"; }
bud() { PTL_PROBE=PROBE-TAIL-BUD PTL_ARM=E10b-P5bud2k PTL_BUD=2048 FAKE_CFG2="$BCFG" bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; }
checkb() { ok "$1" '[ "$(resb "v[\"result\"]")" = FAIL ] && [[ "$(resb "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_BUDM="$BM" bud
ok "[z73] PROBE-TAIL-BUD PASS: the smoke of E10b-P5bud2k, its own report and event; the switch summary" '[ "$(resb "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-TAIL-BUD/.done ] && [ ! -f $T/root/runs/PROBE-TAIL/.done ] && grep -q "^run_arm 3 E10b-P5bud2k SMOKE=1 STEPS=3 PROBE_TASK=PROBE-TAIL-BUD-smoke$" $CALLS && grep -q "^control event PROBE-TAIL-BUD probe_tail_pass" $CALLS && [ "$(resb "(v[\"checks\"][\"budget\"][\"rows_by_box\"], v[\"checks\"][\"budget\"][\"rows_by_budget\"])")" = "(210.0, 60.0)" ]'
cd $SRC; rm -rf $T
setup; FAKE_BUDM="${BM/rows_by_budget:20.0/rows_by_budget:0.0}" bud; checkb "[z73] the budget never binding is FAIL" "budget never bound"; cd $SRC; rm -rf $T
setup; FAKE_BUDM="${BM/budget_late:0.0/budget_late:3.0}" bud; checkb "[z73] a first swap beyond the budget is FAIL" "first swapped beyond the budget"; cd $SRC; rm -rf $T
setup; FAKE_BUDM="${BM/switch_max:2048.0/switch_max:2050.0}" bud; checkb "[z73] a switch position beyond the budget is FAIL" "a switch position beyond 2048"; cd $SRC; rm -rf $T
setup; FAKE_BUDM="${BM/rows_switched:90.0/rows_switched:91.0}" bud; checkb "[z73] switched rows not by box + by budget is FAIL" "switched rows 91.0"; cd $SRC; rm -rf $T
setup; FAKE_BUDM="${BM/budget:2048.0/budget:3072.0}" bud; checkb "[z73] the wrong budget is FAIL" "expected 2048"; cd $SRC; rm -rf $T
setup; FAKE_BUDM="$BM"; PTL_PROBE=PROBE-TAIL-BUD PTL_ARM=E10b-P5bud2k PTL_BUD=2048 bash $T/rb/scripts/probe_tail.sh 3 > /dev/null 2>&1; checkb "[z73] a config without the budget is FAIL" "does not name the budget 2048"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_TAIL_TEST PASS ($n checks)" || echo "PROBE_TAIL_TEST FAIL ($fails of $n)"
