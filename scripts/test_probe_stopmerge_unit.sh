#!/usr/bin/env bash
# [v2.5.9z6] Stub test of probe_stopmerge.sh's unit-only mode (PROBE-STOPMERGE-UNIT; no GPU, network or HF). G-OPD's two files come
# from GOPD_SRC (its verl dir, unpatched), installed into a fake package and patched in place with a .orig (the state the second
# run's smoke left on the node). Without torch (the Mac) the runner's torch check counts one failure, so the result is FAIL here;
# on the node the same line reads "N passed, 0 failed, 0 skipped".
#  - the counts line has a skipped count; the patched-installed case passes (no FAILED test line); no smoke is started;
#    the gate's own runs/PROBE-STOPMERGE is not touched; hooks.txt lists both files, the OPD_STOPMERGE guard and the 6 actor lines;
#  - GOPD_SRC patched without a .orig: the new test is SKIPPED (named, counted), not silently passed.
# Usage: GOPD_SRC=<G-OPD's verl dir> bash scripts/test_probe_stopmerge_unit.sh -> PSM_UNIT_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
[ -n "${GOPD_SRC:-}" ] && [ -f "$GOPD_SRC/workers/actor/dp_actor.py" ] || { echo "PSM_UNIT_TEST SKIP (GOPD_SRC not set)"; exit 0; }
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code $T/root/runs $T/site/verl/trainer/ppo $T/site/verl/workers/actor
  cp $SRC/scripts/probe_stopmerge.sh $T/rb/scripts/; ln -s $SRC/code/patches $T/rb/code/patches
  printf 'export OPD_ROOT=%s/root RUNBOOK=%s/rb\n' $T $T > $T/rb/scripts/env.sh
  printf 'import sys\nprint("node0") if sys.argv[1:2] == ["node"] else open("%s/calls", "a").write(" ".join(sys.argv[1:]) + "\\n")\n' $T > $T/rb/scripts/control.py
  printf 'import sys; sys.exit(0)\n' > $T/rb/scripts/remote.py
  : > $T/site/verl/__init__.py
  for rel in trainer/ppo/ref_input_utils.py workers/actor/dp_actor.py; do cp $GOPD_SRC/$rel $T/site/verl/$rel; done
  printf '#!/usr/bin/env bash\nPYTHONPATH=%s/site exec python3 "$@"\n' $T > $T/py; chmod +x $T/py
  ( cd $SRC/code/patches && python3 -c "
import stopmerge_patch as P
assert P.patch_file('$T/site/verl/trainer/ppo/ref_input_utils.py', P.patch_ref_text) and P.patch_file('$T/site/verl/workers/actor/dp_actor.py', P.patch_actor_text)" )
}
run() { ( PSM_UNIT_ONLY=1 PSM_PY=$T/py bash $T/rb/scripts/probe_stopmerge.sh cpu > $T/out 2>&1 ); }
U=runs/PROBE-STOPMERGE-UNIT
# 1. the installed files patched in place with .orig, GOPD_SRC unset in the probe's env -> the installed package (its .orig)
setup; ( unset GOPD_SRC; run )
ok "unit-only: report in runs/PROBE-STOPMERGE-UNIT, .done, the gate's runs/PROBE-STOPMERGE untouched" '[ -f $T/root/$U/report.json ] && [ -f $T/root/$U/.done ] && [ ! -e $T/root/runs/PROBE-STOPMERGE ]'
ok "the counts line has a skipped count: 0 skipped; only the torch check failed (no FAILED test line)" 'tail -1 $T/root/$U/unit_tests.txt | grep -q "passed, 1 failed, 0 skipped$" && ! grep -q "^FAILED test_" $T/root/$U/unit_tests.txt && grep -q "^FAILED torch not importable" $T/root/$U/unit_tests.txt'
ok "no smoke: run_arm not called, no smoke dir" '[ ! -e $T/root/runs/PROBE-STOPMERGE-UNIT-smoke ] && [ ! -e $T/root/$U/smoke.out ]'
ok "hooks.txt: both files, the OPD_STOPMERGE guard once, 6 actor hook lines" 'grep -q "^== trainer/ppo/ref_input_utils.py (the installed package): 1 hook lines" $T/root/$U/hooks.txt && grep -q "^== workers/actor/dp_actor.py (the installed package): 6 hook lines" $T/root/$U/hooks.txt && [ $(grep -c "environ.get(.OPD_STOPMERGE.) == .1.:" $T/root/$U/hooks.txt) = 1 ] && grep -q "^files: 2 distinct" $T/root/$U/hooks.txt'
ok "[v2.5.9z8] opd_box_end lines: the installed tree has 5 and exactly 1 write (the OPD_STOPMERGE branch in ref_input_utils.py), relative paths; the runbook's code walked too" 'grep -q "^-- verl: 5 lines, 1 writes$" $T/root/$U/hooks.txt && [ $(grep -c "^verl:.* WRITE " $T/root/$U/hooks.txt) = 1 ] && grep -q "^verl:trainer/ppo/ref_input_utils.py:[0-9]*: WRITE batch.batch\[.opd_box_end.\] = torch.tensor" $T/root/$U/hooks.txt && grep -q "^verl:workers/actor/dp_actor.py:[0-9]*: _sm_on, _sm_lp = \"opd_box_end\" in micro_batch.keys()" $T/root/$U/hooks.txt && grep -q "^-- opd_box_end in runbook/code$" $T/root/$U/hooks.txt && grep -q "^runbook/code:patches/stopmerge_patch.py:" $T/root/$U/hooks.txt && ! grep -q "^-- opd_box_end in verl-src" $T/root/$U/hooks.txt && ! grep -q "$T" $T/root/$U/hooks.txt'
ok "report: FAIL here (no torch), the reason carries the counts line" 'python3 -c "import json, sys; r = json.load(open(sys.argv[1])); assert r[\"result\"] == \"FAIL\" and \"0 skipped\" in r[\"reason\"], r" $T/root/$U/report.json'
cd $SRC; rm -rf $T
# 2. GOPD_SRC = a patched tree without .orig -> the new test is skipped, named and counted
setup; rm $T/site/verl/trainer/ppo/ref_input_utils.py.orig; ( GOPD_SRC=$T/site/verl run )
ok "a patched GOPD_SRC without .orig: SKIPPED test_real_file_check_on_patched_installed_files, 1 skipped" 'grep -q "^SKIPPED test_real_file_check_on_patched_installed_files" $T/root/$U/unit_tests.txt && tail -1 $T/root/$U/unit_tests.txt | grep -q ", 1 skipped$"'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PSM_UNIT_TEST PASS ($n checks)" || echo "PSM_UNIT_TEST FAIL ($fails of $n)"
