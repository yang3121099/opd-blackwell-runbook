#!/usr/bin/env bash
# [v2.5.9z76] (Q-065) Stub tests of probe_dense.sh (no GPU, network or HF). Usage: bash scripts/test_probe_dense.sh -> PROBE_DENSE_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
MARK="# [opd-variants patch v2] [v2.5.9z76] dense OPD"
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/runs $T/bin; cd $T
  cp $SRC/scripts/probe_dense.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  : > $T/ac.py; : > $T/wk.py
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
if [ $2 = ${FAKE_DENSE_ARM:-E10b-P5-dense} ]; then   # [v2.5.9z85] the 4B gate's arm too
  [ -z "${FAKE_CFG+x}" ] && FAKE_CFG="; dense ${FAKE_MODE:-rkl} over the teacher's top-64 (opd_dense.py)"
  printf "task: $2\nteacher: Qwen/Qwen3-8B (ref; think-template=1; param_offload=1)%s\n" "$FAKE_CFG" > $d/config.yaml
  for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
    echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.5 - actor/grad_norm:${FAKE_GN:-0.8} - dense/rkl:0.31 - dense/fkl:0.4 - dense/mass_topk:0.97 - dense/corr_sampled:${FAKE_CORR:-0.72} - dense/rkl_sampled:${FAKE_RS:-0.28} - timing_s/step:${FAKE_TD:-600}" >> $d/train.log
  done
  [ "${FAKE_NOHOOK:-0}" = 1 ] || { for k in 1 2 3 4 5 6 7 8; do echo "$HOOKMARK" >> $FAKE_AC; done; echo "$HOOKMARK" >> $FAKE_WK; }
else
  printf "task: $2\n" > $d/config.yaml
  for i in $(seq $SMOKE_STEPS); do echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.5 - actor/grad_norm:0.7 - timing_s/step:500" >> $d/train.log; done
fi
[ "${FAKE_OOM:-0}" = 1 ] && [ $2 = ${FAKE_DENSE_ARM:-E10b-P5-dense} ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  cat > $T/bin/fakepy <<'S3'
#!/usr/bin/env bash
if [ "$1" = -c ]; then case "$2" in *dp_actor*) echo $FAKE_AC;; *fsdp_workers*) echo $FAKE_WK;; esac; exit 0; fi
if [ "$1" = - ]; then echo "py unit GOPD_DP_ACTOR=$GOPD_DP_ACTOR GOPD_FSDP_WORKERS=$GOPD_FSDP_WORKERS" >> $CALLS; cat > /dev/null
  [ "${FAKE_UNIT_RC:-0}" = 0 ] && { echo "6 passed, 0 failed"; exit 0; }; echo "5 passed, 1 failed"; exit 1; fi
echo "py $*" >> $CALLS; exit ${FAKE_PATCH_RC:-0}
S3
  chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PDN_WAIT_STEP=0 PDN_PY=$T/bin/fakepy FAKE_AC=$T/ac.py FAKE_WK=$T/wk.py HOOKMARK="$MARK"; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-DENSE/report.json "$1"; }
setup; bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1
ok "PASS: unit tests on the installed files, the patch, smokes of P5 then dense on the same GPU; report, event" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-DENSE/.done ] && grep -q "^py unit GOPD_DP_ACTOR=$T/ac.py GOPD_FSDP_WORKERS=$T/wk.py$" $CALLS && grep -q "^py $T/rb/code/patches/dense_patch.py$" $CALLS && [ "$(grep "^run_arm" $CALLS | cut -d" " -f2-3 | tr "\n" " ")" = "5 E10b-P5 5 E10b-P5-dense " ] && grep -q "^control event PROBE-DENSE probe_dense_pass" $CALLS && [[ "$(res "v[\"reason\"]")" == *"dense / sampled r 0.720, mean ratio 1.11, step time x1.20 of P5, teacher top-K mass 0.970"* ]]'
cd $SRC; rm -rf $T
setup; FAKE_UNIT_RC=1 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1
ok "a unit-test failure is FAIL, no smoke" '[ "$(res "v[\"result\"]")" = FAIL ] && ! grep -q "^run_arm" $CALLS'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_CORR=0.2 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "a weak correlation is FAIL" "not above 0.3"; cd $SRC; rm -rf $T
setup; FAKE_RS=-0.3 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "a sign error (mean ratio < 0) is FAIL" "mean ratio -1.03"; cd $SRC; rm -rf $T
setup; FAKE_RS=0.1 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "a mean ratio of 3.1 is FAIL" "outside [0.5, 2.0]"; cd $SRC; rm -rf $T
setup; FAKE_TD=800 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "a slow step is FAIL" "step time ratio dense / P5 1.6"; cd $SRC; rm -rf $T
setup; FAKE_GN=nan bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "a non-finite gradient is FAIL" "grad_norm nan not finite"; cd $SRC; rm -rf $T
setup; FAKE_NOHOOK=1 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "the hooks missing in the installed files is FAIL" "hold the hooks (0, 0) times"; cd $SRC; rm -rf $T
setup; FAKE_CFG="" bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "a config without the dense loss is FAIL" "does not name the dense loss"; cd $SRC; rm -rf $T
setup; FAKE_LOGSTEPS=2 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "a missing step is FAIL" "expected 1..3"; cd $SRC; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "the dense smoke failing is FAIL" "smoke dense exited 1"; cd $SRC; rm -rf $T
setup; FAKE_PATCH_RC=1 bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1; check "the patch failing is FAIL" "dense_patch.py failed"; cd $SRC; rm -rf $T
# [v2.5.9z85] (Q-079) PROBE-DENSE-4B: PDN_PROBE / PDN_P5 / PDN_ARM / PDN_MODE: the 4B smokes (E10d-P5, E10d-P5-akl), its own report and event;
# a config naming another loss than PDN_MODE's is FAIL
r4() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-DENSE-4B/report.json "$1"; }
setup; FAKE_DENSE_ARM=E10d-P5-akl FAKE_MODE=akl PDN_PROBE=PROBE-DENSE-4B PDN_P5=E10d-P5 PDN_ARM=E10d-P5-akl PDN_MODE=akl bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1
ok "[v2.5.9z85] PROBE-DENSE-4B PASS: the 4B P5 and AKL smokes, its own report, event, .done; PROBE-DENSE untouched" '[ "$(r4 "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-DENSE-4B/.done ] && [ ! -e $T/root/runs/PROBE-DENSE ] && grep -q "^run_arm 5 E10d-P5 SMOKE=1 STEPS=3 PROBE_TASK=PROBE-DENSE-4B-p5$" $CALLS && grep -q "^run_arm 5 E10d-P5-akl SMOKE=1 STEPS=3 PROBE_TASK=PROBE-DENSE-4B-dense$" $CALLS && grep -q "^control event PROBE-DENSE-4B probe_dense_pass" $CALLS'
cd $SRC; rm -rf $T
setup; FAKE_DENSE_ARM=E10d-P5-akl PDN_PROBE=PROBE-DENSE-4B PDN_P5=E10d-P5 PDN_ARM=E10d-P5-akl PDN_MODE=akl bash $T/rb/scripts/probe_dense.sh 5 > /dev/null 2>&1
ok "[v2.5.9z85] PROBE-DENSE-4B: a config naming rkl, not akl, is FAIL" '[ "$(r4 "v[\"result\"]")" = FAIL ] && [[ "$(r4 "v[\"reason\"]")" == *"does not name the dense loss"* ]]'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_DENSE_TEST PASS ($n checks)" || echo "PROBE_DENSE_TEST FAIL ($fails of $n)"
