#!/usr/bin/env bash
# [v2.5.9z118] (research_program 3q) Stub tests of probe_fret.sh (no GPU, network or HF). Usage: bash scripts/test_probe_fret.sh -> PROBE_FRET_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/runs $T/bin; cd $T
  cp $SRC/scripts/probe_fret.sh $T/rb/scripts/; : > $T/ac.py
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; arm=$2
desc=""; case $arm in *topd*) desc="TOP-D's bounded reward log(a e^r + 1 - a), a 0.2";; esac
case $arm in *fret*) desc="${desc:+$desc, then }the suffix credit A_k = r_k + the later rewards' mean";; esac
[ -n "${FAKE_NOCFG:-}" ] && [ "$arm" = "$FAKE_NOCFG" ] && desc=""
printf "task: $arm\nteacher: Qwen/Qwen3-8B (ref)%s\n" "${desc:+; reward-shaped: $desc (opd_variants.shape, shape_patch.py)}" > $d/config.yaml
t=100; [ "$arm" != E10b-P5 ] && t=${FAKE_T:-105}
for i in $(seq $SMOKE_STEPS); do
  m="actor/grad_norm:${FAKE_GN:-1.5} - timing_s/step:$t - critic/score/mean:0.5"
  case $arm in *fret*) [ "${FAKE_NOFRET:-0}" = 1 ] || m="$m - shape/fret_abs_mean:0.8";; esac
  case $arm in *topd*) m="$m - shape/r_min:-9.1 - shape/r_max:3.0 - shape/rho_max:2.0085536923187668e+01 - shape/topd_min:${FAKE_TMIN:--0.2229} - shape/topd_max:${FAKE_TMAX:-1.5720}";; esac
  [ "$arm" = E10b-P5 ] && [ "${FAKE_P5SHAPE:-0}" = 1 ] && m="$m - shape/fret_abs_mean:0.8"
  echo "(TaskRunner pid=1) step:$i - $m" >> $d/train.log
done
[ "${FAKE_OOM:-}" = "$arm" ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  cat > $T/bin/fakepy <<'S3'
#!/usr/bin/env bash
if [ "$1" = -c ]; then echo $FAKE_AC; exit 0; fi
if [ "$1" = - ]; then echo "py unit GOPD_DP_ACTOR=$GOPD_DP_ACTOR" >> $CALLS; cat > /dev/null
  [ "${FAKE_UNIT_RC:-0}" = 0 ] && { echo "9 passed, 0 failed"; exit 0; }; echo "8 passed, 1 failed"; exit 1; fi
echo "py $*" >> $CALLS
case $1 in */shape_patch.py) [ "${FAKE_PATCH_RC:-0}" = 0 ] || exit 1
  [ "${FAKE_HOOKS:-1}" -ge 1 ] && for i in $(seq ${FAKE_HOOKS:-1}); do echo "# [opd-variants patch v2] [v2.5.9z118] fret / topd reward shaping" >> $GOPD_DP_ACTOR; done; echo SHAPE_HOOK_PATCHED;; esac; exit 0
S3
  chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PFR_WAIT_STEP=0 PFR_PY=$T/bin/fakepy FAKE_AC=$T/ac.py; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-FRET/report.json "$1"; }
setup; bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1
ok "PASS: unit tests on the installed file, the patch, four smokes in order, report and event" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-FRET/.done ] && grep -q "^py unit GOPD_DP_ACTOR=$T/ac.py$" $CALLS && grep -q "^py $T/rb/code/patches/shape_patch.py$" $CALLS && [ "$(grep "^run_arm" $CALLS | cut -d" " -f3 | tr "\n" " ")" = "E10b-P5 E10b-P5fret E10b-P5topd E10b-P5fret-topd " ] && grep -q "^run_arm 3 E10b-P5topd SMOKE=1 STEPS=3 PROBE_TASK=PROBE-FRET-topd$" $CALLS && grep -q "^control event PROBE-FRET probe_fret_pass" $CALLS && grep -q "^remote put $T/root/runs/PROBE-FRET/up logs/abcd1234/probe_fret/PROBE-FRET$" $CALLS && [[ "$(res "v[\"reason\"]")" == *"step time / P5 fret 1.05, topd 1.05, fret-topd 1.05"* ]]'
cd $SRC; rm -rf $T
setup; FAKE_UNIT_RC=1 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1
ok "a unit-test failure is FAIL, no smoke" '[ "$(res "v[\"result\"]")" = FAIL ] && ! grep -q "^run_arm" $CALLS'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_PATCH_RC=1 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "the patch failing is FAIL" "shape_patch.py failed"; cd $SRC; rm -rf $T
setup; FAKE_T=125 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "a slow arm is FAIL" "step time ratio fret / P5 1.25 above 1.2"; cd $SRC; rm -rf $T
setup; FAKE_GN=nan bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "a non-finite gradient is FAIL" "grad_norm nan not finite"; cd $SRC; rm -rf $T
setup; FAKE_TMIN=-0.3 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "a topd reward below log 0.8 is FAIL" "topd step 1: topd range [-0.3,"; cd $SRC; rm -rf $T
setup; FAKE_TMAX=2.5 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "a topd reward above log(0.2 rho_max + 0.8) is FAIL" "outside [log 0.8, log(0.2 * 20.0855 + 0.8)]"; cd $SRC; rm -rf $T
setup; FAKE_NOFRET=1 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "a fret arm without its metric is FAIL" "fret step 1: no shape/fret_abs_mean"; cd $SRC; rm -rf $T
setup; FAKE_P5SHAPE=1 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "shaping in the P5 reference is FAIL" "p5 step 1: shaping metrics"; cd $SRC; rm -rf $T
setup; FAKE_NOCFG=E10b-P5topd bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "a config without the shaping is FAIL" "the topd smoke's config does not name its shaping"; cd $SRC; rm -rf $T
setup; FAKE_OOM=E10b-P5fret-topd bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "an arm's smoke failing is FAIL" "smoke fret-topd exited 1"; cd $SRC; rm -rf $T
setup; FAKE_HOOKS=2 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "the hook twice in the installed file is FAIL" "holds the hook 2 times"; cd $SRC; rm -rf $T
setup; FAKE_MEM=5000 bash $T/rb/scripts/probe_fret.sh 3 > /dev/null 2>&1; check "a busy GPU is FAIL before the smokes" "gpu 3 not empty after 180 s"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_FRET_TEST PASS ($n checks)" || echo "PROBE_FRET_TEST FAIL ($fails of $n)"
