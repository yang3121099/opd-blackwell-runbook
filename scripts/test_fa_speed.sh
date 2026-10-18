#!/usr/bin/env bash
# Stub tests of v2.5.7c fa_speed.sh (no GPU, network or HF). Usage: bash scripts/test_fa_speed.sh -> FA_SPEED_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs/E12b-P0/rollouts $T/root/venv-fa/bin $T/bin
  cp $SRC/scripts/fa_speed.sh $SRC/scripts/fa_speed.py $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb STUDENT_17B=$T/m17 TEACHER=$T/m8 STUDENT_4B=$T/m4" > $T/rb/scripts/env.sh
  echo '{"input": "q", "output": "a"}' > $T/root/runs/E12b-P0/rollouts/1.jsonl
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/root/venv-fa/bin/python <<'SH'
#!/usr/bin/env bash
echo "pyfa $*" >> $CALLS
case "$*" in *"import flash_attn"*) exit 0;; *fa_speed.py\ lp*) echo "verl noise"; echo '{"max_abs_diff": 0.03, "mean_abs_diff": 0.0004, "p999_abs_diff": 0.01, "pass": true}';; esac
SH
  chmod +x $T/root/venv-fa/bin/python
  for s in run_arm.sh run_e12.sh; do cat > $T/rb/scripts/$s <<'SH'
echo "$(basename $0) $* VENV=${VENV:-}${E10D_FA:+ E10D_FA=$E10D_FA} FIXED=${FIXED_MICRO_BSZ:-} GC=${GC:-} E12B_GC=${E12B_GC:-} E12B_FA=${E12B_FA:-} STEPS=$SMOKE_STEPS TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d; : > $d/config.yaml; st=$([ "${VENV:-}" = fa ] && echo 100 || echo 130)
for i in 1 2 3; do echo "(TaskRunner pid=1) step:$i - timing_s/gen:30 - timing_s/update_actor:40 - timing_s/step:$st - response_length/mean:2000" >> $d/train.log; done
SH
  done
  printf '#!/usr/bin/env bash\necho 0\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH FS_WAIT_STEP=0; : > $CALLS
}
rep() { python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/$1/report.json "$2"; }
# 1. FA-SPEED-17: the log-prob checks, three smokes of E10b-P5 with the right settings, speed-up vs the first
setup
bash $T/rb/scripts/fa_speed.sh 4 FA-SPEED-17 > /dev/null 2>&1
ok "lp both models through venv-fa" 'grep -q "^pyfa .*fa_speed.py lp $T/m17 " $CALLS && grep -q "^pyfa .*fa_speed.py lp $T/m8 " $CALLS && [ "$(rep FA-SPEED-17 "r[\"lp_pass\"]")" = True ]'
ok "three smokes, settings" 'grep -q "^run_arm.sh 4 E10b-P5 VENV= FIXED=4 GC=0 .*TASK=FA-SPEED-17-sdpa_fix4" $CALLS && grep -q "^run_arm.sh 4 E10b-P5 VENV=fa E10D_FA=1 FIXED= GC=0 .*TASK=FA-SPEED-17-fa_tok" $CALLS && grep -q "^run_arm.sh 4 E10b-P5 VENV=fa E10D_FA=1 FIXED=4 GC=0 .*TASK=FA-SPEED-17-fa_fix4" $CALLS'
ok "speed-up vs sdpa" '[ "$(rep FA-SPEED-17 "r[\"variants\"][\"fa_tok\"][\"speedup_vs_sdpa_fix4\"][\"step\"]")" = 1.3 ] && [ "$(rep FA-SPEED-17 "r[\"smokes_ok\"]")" = True ]'
ok "report up, event, .done" 'grep -q "^remote put $T/root/runs/FA-SPEED-17/up logs/abcd1234/fa_speed/FA-SPEED-17" $CALLS && grep -q "^control event FA-SPEED-17 fa_speed_done" $CALLS && [ -f $T/root/runs/FA-SPEED-17/.done ]'
rm -rf $T
# 2. FA-SPEED-E12: E12b smokes with the E12b gates (GC check and flash-attn)
setup
bash $T/rb/scripts/fa_speed.sh 5 FA-SPEED-E12 > /dev/null 2>&1
ok "E12b smokes" 'grep -q "^run_e12.sh 5 E12b-P0 VENV= FIXED=4 GC=1 E12B_GC=0 E12B_FA=0 STEPS=3 TASK=FA-SPEED-E12-sdpa_gc1" $CALLS && grep -q "^run_e12.sh 5 E12b-P0 VENV= FIXED=4 GC=0 E12B_GC=1 E12B_FA=0 .*sdpa_gc0" $CALLS && grep -q "^run_e12.sh 5 E12b-P0 VENV=fa FIXED= GC=0 E12B_GC=1 E12B_FA=1 .*fa_gc0" $CALLS && ! grep -q "fa_speed.py lp" $CALLS'
rm -rf $T
# 3. FA-SPEED-4B fetches the 4B student when it is missing
setup
printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n"); os.makedirs(sys.argv[3], exist_ok=True)\n' > $T/rb/scripts/fetch_model.py
bash $T/rb/scripts/fa_speed.sh 6 FA-SPEED-4B > /dev/null 2>&1
ok "4B fetched, two smokes of E10d-P5" 'grep -q "^fetch Qwen/Qwen3-4B  $T/root/models/Qwen3-4B --task FA-SPEED-4B" $CALLS && [ "$(grep -c "^run_arm.sh 6 E10d-P5" $CALLS)" = 2 ]'
rm -rf $T
[ $fails = 0 ] && echo "FA_SPEED_TEST PASS ($n checks)" || echo "FA_SPEED_TEST FAIL ($fails of $n)"
