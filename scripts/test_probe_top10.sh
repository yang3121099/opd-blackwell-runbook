#!/usr/bin/env bash
# Stub tests of v2.5.7h probe_top10.sh (no GPU, torch, network or HF). Usage: bash scripts/test_probe_top10.sh -> PROBE_TOP10_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/runs $T/root/venv-fa/bin
  cp $SRC/scripts/probe_top10.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  # the unit tests: two modules with plain test_ functions (the runner needs no pytest)
  printf 'def test_a():\n    pass\ndef test_b():\n    import os\n    assert os.environ.get("FAIL_UNIT") != "1"\n' > $T/rb/code/patches/test_opd_variants.py
  printf 'def test_c():\n    pass\n' > $T/rb/code/patches/test_variant_dispatch.py
  printf '#!/usr/bin/env bash\nexec python3 "$@"\n' > $T/root/venv-fa/bin/python; chmod +x $T/root/venv-fa/bin/python
  cat > $T/rb/scripts/run_arm.sh <<'X'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
for i in 1 2 3; do k=${KEPT:-0.1}; extra=""; [[ $2 == *rand* ]] && extra=" - opd/rand_scale_mean:2.5"
  echo "(TaskRunner pid=1) step:$i - opd/sel_kept_frac:$k - opd/sel_eos_frac:0.01 - opd/sel_disc_frac:0.2 - opd/eos_adv_mean:-0.3$extra" >> $d/train.log; done
exit 0
X
  export CALLS=$T/calls; : > $CALLS
}
res() { python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-TOP10/report.json "$1"; }
setup
bash $T/rb/scripts/probe_top10.sh 4 > /dev/null 2>&1
ok "PASS: unit tests, two 3-step smokes, 10% kept, token classes, matched scale" '[ "$(res "r[\"result\"]")" = PASS ] && [ "$(res "r[\"unit_tests\"][\"summary\"]")" = "3 passed, 0 failed" ] && grep -q "^run_arm 4 E10b-P5top10 SMOKE=1 STEPS=3 PROBE_TASK=PROBE-TOP10-P5top10" $CALLS && grep -q "^run_arm 4 E10b-P5rand10m SMOKE=1 STEPS=3 PROBE_TASK=PROBE-TOP10-P5rand10m" $CALLS && [ -f $T/root/runs/PROBE-TOP10/.done ]'
ok "report to the private logs, event" 'grep -q "^remote put $T/root/runs/PROBE-TOP10/up logs/abcd1234/probe_top10/PROBE-TOP10" $CALLS && grep -q "^control event PROBE-TOP10 probe_top10_done PASS" $CALLS'
rm -rf $T
setup
KEPT=0.2 FAIL_UNIT=1 bash $T/rb/scripts/probe_top10.sh 4 > /dev/null 2>&1
ok "FAIL: a unit test fails, 20% kept (the hook did not take the override)" '[ "$(res "r[\"result\"]")" = FAIL ] && res "r[\"reasons\"]" | grep -q "unit tests failed" && res "r[\"reasons\"]" | grep -q "P5top10: opd/sel_kept_frac \[0.2, 0.2, 0.2\]"'
rm -rf $T
[ $fails = 0 ] && echo "PROBE_TOP10_TEST PASS ($n checks)" || echo "PROBE_TOP10_TEST FAIL ($fails of $n)"
