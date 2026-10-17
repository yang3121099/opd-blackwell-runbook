#!/usr/bin/env bash
# Stub test of fa_probe.sh (runbook v2.5.3): no GPU, network or pip. Fake nvidia-smi, curl, sha256sum, venv python,
# run_arm.sh, remote.py and control.py record what the probe does. Cases: PASS; sha256 mismatch (no install, FAIL);
# a stack the pinned wheel does not cover (FAIL before any download); GPU busy for a while then free (waits, PASS);
# GPU busy for 180 s (FAIL, nothing copied).
# Usage: bash scripts/test_fa_probe.sh   -> prints FA_PROBE_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0
one() {   # $1 case name, $2 sha mode (good|bad), $3 torch version, $4 gpu used MiB
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/venv/bin $T/bin $T/root/runs
  cp $SRC/scripts/fa_probe.sh $SRC/scripts/fa_probe.py $T/rb/scripts/
  printf 'export OPD_ROOT=%s RUNBOOK=%s\n' "$T/root" "$T/rb" > $T/rb/scripts/env.sh
  cat > $T/rb/scripts/remote.py <<'PY'
import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\n")
PY
  cat > $T/rb/scripts/control.py <<'PY'
import sys, os
if sys.argv[1] == "node": print("testnode")
else: open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\n")
PY
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm VENV=$VENV SMOKE=$SMOKE PROBE_TASK=$PROBE_TASK args=$*" >> $CALLS
mkdir -p $OPD_ROOT/runs/$PROBE_TASK
echo "attention: flash_attention_2 (use_remove_padding=True)" > $OPD_ROOT/runs/$PROBE_TASK/config.yaml
for s in 1 2 3; do echo "(TaskRunner) step:$s - timing_s/gen:30.$s - timing_s/old_log_prob:5.$s - timing_s/ref:6.$s - timing_s/update_actor:20.$s - timing_s/step:62.$s" >> $OPD_ROOT/runs/$PROBE_TASK/train.log; done
SH
  # the venv python: versions / varlen answered here, report delegated to the real python3, pip records the install
  cat > $T/root/venv/bin/python <<SH
#!/usr/bin/env bash
case "\$*" in
  *"import sys; print(sys.prefix)"*) cd \$(dirname \$0)/.. && pwd;;
  *"import flash_attn; print(flash_attn.__file__)"*) [ -f \$(dirname \$0)/../.fa_installed ] && echo "\$(cd \$(dirname \$0)/.. && pwd)/lib/python3.10/site-packages/flash_attn/__init__.py" || exit 1;;
  *"from flash_attn.bert_padding import"*) [ -f \$(dirname \$0)/../.fa_installed ] && exit 0 || exit 1;;
  *"fa_probe.py versions"*) echo '{"python": "3.10.12", "torch": "$3", "torch_cuda": "12.8", "cxx11abi": "TRUE", "vllm": "0.10.2"}';;
  *"fa_probe.py varlen"*) echo '{"varlen_max_abs_diff": 0.004, "grad_finite": true, "capability": [10, 3]}';;
  *"fa_probe.py report"*) shift; exec python3 "\$@";;
  *"-m pip install --no-deps --no-index"*) touch \$(dirname \$0)/../.fa_installed; echo "pip \$*" >> \$CALLS;;
  *) exec python3 "\$@";;
esac
SH
  chmod +x $T/root/venv/bin/python
  RV=$(cd $T/root/venv && pwd -P); printf 'VIRTUAL_ENV="%s"\nexport PATH="%s/bin:$PATH"\n' "$RV" "$RV" > $T/root/venv/bin/activate   # a real venv names its own path
  cat > $T/bin/nvidia-smi <<SH
#!/usr/bin/env bash
n=\$(cat $T/nsmi 2>/dev/null || echo 0); echo \$((n + 1)) > $T/nsmi
case "$4" in busy3) [ \$n -lt 3 ] && echo 40000 || echo 0;; *) echo $4;; esac
SH
  cat > $T/bin/curl <<'SH'
#!/usr/bin/env bash
while [ $# -gt 0 ]; do [ "$1" = -o ] && { echo wheel > "$2"; echo "curl $*" >> $CALLS; }; shift; done
SH
  W=$(python3 $SRC/scripts/fa_probe.py wheel TRUE | cut -d' ' -f2)
  cat > $T/bin/sha256sum <<SH
#!/usr/bin/env bash
[ "$2" = good ] && echo "$W  \$1" || echo "0000000000000000000000000000000000000000000000000000000000000000  \$1"
SH
  chmod +x $T/bin/*
  export CALLS=$T/calls; : > $CALLS
  FA_WAIT_STEP=0 PATH=$T/bin:$PATH bash $T/rb/scripts/fa_probe.sh 3 > $T/out 2>&1
  R=$(python3 -c "import json; d=json.load(open('$T/root/runs/FA-PROBE/report.json')); print(d.get('result'), d.get('reason'), d.get('smoke_steps'), d.get('rmpad'))" 2>/dev/null)
  echo "$R" > $T/result
  check() { if eval "$2"; then :; else echo "  FAIL [$1] $3"; echo "  --- out:"; sed 's/^/    /' $T/out | tail -8; fails=$((fails + 1)); fi; }
  case $1 in
    pass) check pass '[[ "$R" == PASS*" 3 True" ]] && grep -q "pip " $CALLS && grep -q "run_arm VENV=$T/root/venv-fa SMOKE=1 PROBE_TASK=FA-PROBE-smoke args=3 E10b-P2" $CALLS && grep -q "remote put $T/root/runs/FA-PROBE logs/testnode/fa_probe" $CALLS && grep -q "control event FA-PROBE fa_probe_pass" $CALLS && [ -f $T/root/runs/FA-PROBE/.done ] && [ ! -e $T/root/runs/FA-PROBE/*.whl ]' "result=$R";;
    badsha) check badsha '[[ "$R" == FAIL\ sha256* ]] && ! grep -q "pip " $CALLS && ! grep -q run_arm $CALLS && [ -f $T/root/runs/FA-PROBE/.done ]' "result=$R";;
    stack) check stack '[[ "$R" == FAIL\ stack* ]] && ! grep -q "curl" $CALLS' "result=$R";;
    busythenfree) check busythenfree '[[ "$R" == PASS*" 3 True" ]]' "result=$R";;
    busy) check busy '[[ "$R" == FAIL\ gpu* ]] && [ ! -d $T/root/venv-fa ]' "result=$R";;
  esac
  [ -d $T/root/venv-fa ] && [ "$(readlink -f $T/root/venv-fa)" = "$(readlink -f $T/root/venv)" ] && { echo "  FAIL [$1] venv-fa is the venv in use"; fails=$((fails + 1)); }
  rm -rf $T
}
one pass good 2.8.0+cu128 0
one badsha bad 2.8.0+cu128 0
one stack good 2.9.0+cu130 0
one busythenfree good 2.8.0+cu128 busy3   # busy for 3 checks, then free: the probe waits and passes
one busy good 2.8.0+cu128 40000
[ $fails = 0 ] && echo "FA_PROBE_TEST PASS (5 cases)" || echo "FA_PROBE_TEST FAIL ($fails)"
