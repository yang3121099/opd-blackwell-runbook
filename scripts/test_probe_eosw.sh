#!/usr/bin/env bash
# [v2.5.9z102] (Q-087) Stub tests of probe_eosw.sh (no GPU, network or HF). Usage: bash scripts/test_probe_eosw.sh -> PROBE_EOSW_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
MARK="# [opd-variants patch v2] [v2.5.9z102] position-weighted OPD"
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/runs $T/bin; cd $T
  cp $SRC/scripts/probe_eosw.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  : > $T/ri.py; : > $T/ac.py
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
if [ $2 = E10b-P5-eosw ]; then
  [ -z "${FAKE_CFG+x}" ] && FAKE_CFG="; position-weighted: the 16 positions after the last complete box x64 (opd_eosw.py)"
  printf "task: $2\nteacher: Qwen/Qwen3-8B (ref; think-template=1; param_offload=1)%s\n" "$FAKE_CFG" > $d/config.yaml
  for i in $(seq ${FAKE_LOGSTEPS:-$SMOKE_STEPS}); do
    echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.5 - actor/grad_norm:${FAKE_GN:-0.8} - eosw/weighted_frac:${FAKE_FRAC:-0.004} - timing_s/step:${FAKE_TE:-520}" >> $d/train.log
  done
  [ "${FAKE_NOLOG:-0}" = 1 ] || for i in 1 2 3; do echo '{"rows": 128, "rows_box": 120, "weighted": 1900, "weighted_frac": 0.004}' >> $d/eosw.jsonl; done
  [ "${FAKE_NOHOOK:-0}" = 1 ] || { echo "$HOOKMARK" >> $FAKE_RI; echo "$HOOKMARK" >> $FAKE_AC; echo "$HOOKMARK" >> $FAKE_AC; }
else
  printf "task: $2\n" > $d/config.yaml
  for i in $(seq $SMOKE_STEPS); do echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.5 - actor/grad_norm:0.7 - timing_s/step:500" >> $d/train.log; done
fi
[ "${FAKE_OOM:-0}" = 1 ] && [ $2 = E10b-P5-eosw ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  cat > $T/bin/fakepy <<'S3'
#!/usr/bin/env bash
if [ "$1" = -c ]; then case "$2" in *ref_input_utils*) echo $FAKE_RI;; *dp_actor*) echo $FAKE_AC;; esac; exit 0; fi
if [ "$1" = - ]; then echo "py unit GOPD_REF_INPUT_UTILS=$GOPD_REF_INPUT_UTILS GOPD_DP_ACTOR=$GOPD_DP_ACTOR" >> $CALLS; cat > /dev/null
  [ "${FAKE_UNIT_RC:-0}" = 0 ] && { echo "9 passed, 0 failed"; exit 0; }; echo "8 passed, 1 failed"; exit 1; fi
echo "py $*" >> $CALLS; exit ${FAKE_PATCH_RC:-0}
S3
  chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PEW_WAIT_STEP=0 PEW_PY=$T/bin/fakepy FAKE_RI=$T/ri.py FAKE_AC=$T/ac.py HOOKMARK="$MARK"; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-EOSW/report.json "$1"; }
setup; bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1
ok "PASS: unit tests on the installed files, the patch, smokes of P5 then eosw on the same GPU; report, event" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-EOSW/.done ] && grep -q "^py unit GOPD_REF_INPUT_UTILS=$T/ri.py GOPD_DP_ACTOR=$T/ac.py$" $CALLS && grep -q "^py $T/rb/code/patches/eosw_patch.py$" $CALLS && [ "$(grep "^run_arm" $CALLS | cut -d" " -f2-3 | tr "\n" " ")" = "5 E10b-P5 5 E10b-P5-eosw " ] && grep -q "^control event PROBE-EOSW probe_eosw_pass" $CALLS && [[ "$(res "v[\"reason\"]")" == *"weighted share 0.0040 of the tokens, 360 rows with a box over 3 calls, step time x1.04 of P5"* ]]'
cd $SRC; rm -rf $T
setup; FAKE_UNIT_RC=1 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1
ok "a unit-test failure is FAIL, no smoke" '[ "$(res "v[\"result\"]")" = FAIL ] && ! grep -q "^run_arm" $CALLS'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_FRAC=0 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "no weighted positions is FAIL" "eosw/weighted_frac 0.0 not in (0, 0.05]"; cd $SRC; rm -rf $T
setup; FAKE_FRAC=0.3 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "a window far too wide is FAIL" "eosw/weighted_frac 0.3 not in"; cd $SRC; rm -rf $T
setup; FAKE_NOLOG=1 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "no eosw.jsonl is FAIL" "eosw.jsonl: 0 calls"; cd $SRC; rm -rf $T
setup; FAKE_TE=700 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "a slow step is FAIL" "step time ratio eosw / P5 1.4"; cd $SRC; rm -rf $T
setup; FAKE_GN=nan bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "a non-finite gradient is FAIL" "grad_norm nan not finite"; cd $SRC; rm -rf $T
setup; FAKE_NOHOOK=1 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "the hooks missing in the installed files is FAIL" "hold the hooks (0, 0) times"; cd $SRC; rm -rf $T
setup; FAKE_CFG="" bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "a config without the weighting is FAIL" "does not name the weighting"; cd $SRC; rm -rf $T
setup; FAKE_LOGSTEPS=2 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "a missing step is FAIL" "expected 1..3"; cd $SRC; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "the eosw smoke failing is FAIL" "smoke eosw exited 1"; cd $SRC; rm -rf $T
setup; FAKE_PATCH_RC=1 bash $T/rb/scripts/probe_eosw.sh 5 > /dev/null 2>&1; check "the patch failing is FAIL" "eosw_patch.py failed"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_EOSW_TEST PASS ($n checks)" || echo "PROBE_EOSW_TEST FAIL ($fails of $n)"
