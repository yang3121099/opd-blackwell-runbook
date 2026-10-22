#!/usr/bin/env bash
# Stub tests of [v2.5.9z9] probe_t32.sh (no GPU, network or HF). Usage: bash scripts/test_probe_t32.sh -> PROBE_T32_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/root/models/Qwen3-32B $T/bin
  cp $SRC/scripts/probe_t32.sh $SRC/scripts/speed_probe.py $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  : > $T/root/models/Qwen3-32B/config.json; : > $T/root/models/Qwen3-32B/.fetch_ok
  # a fake smoke: train.log (3 steps), config.yaml (teacher line as run_arm writes it), route.jsonl (one call per step)
  cat > $T/rb/scripts/run_arm.sh <<'SH'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS TEACHER_32B=$TEACHER_32B PROBE_TASK=$PROBE_TASK OPTOFF=${T32_OPT_OFFLOAD:-}" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
printf 'task: %s\nstudent: /m/Qwen3-4B (no_think)\nteacher: %s (ref; think-template=%s; param_offload=1); the 32B teacher (Qwen3-32B) on its thinking template, offloaded\n' $PROBE_TASK ${FAKE_TEACHER:-$TEACHER_32B} ${FAKE_THINK:-1} > $d/config.yaml
for i in $(seq 1 $SMOKE_STEPS); do
  echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.5 - timing_s/gen:100 - timing_s/ref:90 - timing_s/update_actor:150 - timing_s/step:$([ $i = 1 ] && echo 600 || echo ${FAKE_STEP:-360}) - response_length/mean:3000 - perf/max_memory_allocated_gb:121.5 - perf/max_memory_reserved_gb:130.2" >> $d/train.log
  [ $i -le $(( SMOKE_STEPS - ${FAKE_FEWER_CALLS:-0} )) ] && echo "{\"call\": $i, \"mode\": \"count\", \"batch\": 128, \"rows\": ${FAKE_ROWS:-128}, \"think\": ${FAKE_ROWS:-128}, \"plain\": 0}" >> $d/route.jsonl
done
sleep ${FAKE_SLEEP:-0}; [ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }
[ "${FAKE_OOM_FIRST:-0}" = 1 ] && [ "${T32_OPT_OFFLOAD:-0}" != 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
SH
  printf '#!/usr/bin/env bash\ncase "$*" in *memory.total*) echo 274877;; *) echo ${FAKE_MEM:-150000};; esac\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PT32_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/PROBE-T32/report.json "$1"; }
why() { python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['reason'])" $T/root/runs/PROBE-T32/report.json; }
# 1. PASS: a 3-step smoke of E10d-P5-t32 with the 32B; 128 teacher rows a step on the thinking template; timing and projection
setup
FAKE_MEM=0 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "PASS, .done" '[ "$(res result)" = PASS ] && [ -f $T/root/runs/PROBE-T32/.done ]'
ok "3-step smoke of E10d-P5-t32 with the 32B on the share" 'grep -q "^run_arm 3 E10d-P5-t32 SMOKE=1 STEPS=3 TEACHER_32B=$T/root/models/Qwen3-32B PROBE_TASK=PROBE-T32-smoke" $CALLS'
ok "steps, ref timing, projection for 50 steps, teacher calls in the checks" '[ "$(res steps)" = 3 ] && [ "$(res projected_hours_50)" = 5.0 ] && python3 -c "import json, sys; r = json.load(open(sys.argv[1])); assert r[\"mean_s2plus\"][\"ref\"] == 90.0 and [c[\"rows\"] for c in r[\"checks\"][\"teacher_calls\"]] == [128] * 3, r" $T/root/runs/PROBE-T32/report.json'
ok "report to logs/<node>/probe_t32 with the smoke's config, event" 'grep -q "^remote put $T/root/runs/PROBE-T32/up logs/abcd1234/probe_t32" $CALLS && [ -f $T/root/runs/PROBE-T32/up/smoke_config.yaml ] && grep -q "^control event PROBE-T32 probe_t32_pass" $CALLS'
rm -rf $T
# 1b. [v2.5.9z11] a rerun: an earlier run's memory samples do not count (every probe resets mem.csv at its start)
setup; mkdir -p $T/root/runs/PROBE-T32; echo "1,999999" > $T/root/runs/PROBE-T32/mem.csv
FAKE_MEM=0 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "a rerun: PASS with this run's peak (an old mem.csv sample of 999999 MiB ignored)" '[ "$(res result)" = PASS ] && [ "$(res mem_peak_mib)" = 0.0 ]'
rm -rf $T
# 2. the 32B not in: WAIT_INPUTS, exit 1, no smoke, no .done (run_tasks keeps it waiting, no attempt)
setup; rm $T/root/models/Qwen3-32B/.fetch_ok
FAKE_MEM=0 bash $T/rb/scripts/probe_t32.sh 3 > $T/out 2>&1; rc=$?
ok "no .fetch_ok: WAIT_INPUTS, exit 1, no smoke, no .done" '[ $rc = 1 ] && grep -q "^WAIT_INPUTS PROBE-T32: Qwen3-32B not in" $T/out && ! grep -q "^run_arm" $CALLS && [ ! -f $T/root/runs/PROBE-T32/.done ]'
rm -rf $T
# 3. teacher rows not one per sample, or a missing reference call: FAIL
setup; FAKE_MEM=0 FAKE_ROWS=256 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "256 teacher rows for 128 samples: FAIL" '[ "$(res result)" = FAIL ] && why | grep -q "teacher rows are not 128 samples"'
rm -rf $T
setup; FAKE_MEM=0 FAKE_FEWER_CALLS=1 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "2 reference calls for 3 steps: FAIL" '[ "$(res result)" = FAIL ] && why | grep -q "route.jsonl has 2 reference calls, expected one per step (3)"'
rm -rf $T
# 4. the config without the 32B or without the thinking template: FAIL
setup; FAKE_MEM=0 FAKE_TEACHER=/m/Qwen3-8B bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "a config naming another teacher: FAIL" '[ "$(res result)" = FAIL ] && why | grep -q "does not name the 32B teacher"'
rm -rf $T
setup; FAKE_MEM=0 FAKE_THINK=0 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "think-template=0: FAIL" '[ "$(res result)" = FAIL ] && why | grep -q "on its thinking template"'
rm -rf $T
# 5. out of memory, and a peak at or above the limit: FAIL
setup; FAKE_MEM=0 FAKE_OOM=1 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "OOM is FAIL" '[ "$(res result)" = FAIL ] && grep -q "^control event PROBE-T32 probe_t32_fail" $CALLS'
rm -rf $T
setup; FAKE_MEM=500 FAKE_SLEEP=1 PT32_PEAK_MIB=100 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "peak over the limit is FAIL" '[ "$(res result)" = FAIL ] && why | grep -q "peak"'
rm -rf $T
# 6. [v2.5.9z40] the report's new fields; an out of memory once: a second smoke with the optimizer offloaded; projected > 20 h: HOLD
setup; FAKE_MEM=0 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "[v2.5.9z40] per step: seconds, rollout tok/s (3000 x 128 / 100 s), verl peaks; the rollout pool 0.4 x 268.4 GiB; no escalation" 'python3 -c "import json, sys; r = json.load(open(sys.argv[1])); p = r[\"per_step\"]; assert [x[\"step_s\"] for x in p] == [600.0, 360.0, 360.0] and p[1][\"rollout_tok_s\"] == 3840.0 and p[0][\"max_alloc_gib\"] == 121.5 and p[0][\"max_reserved_gib\"] == 130.2, p; assert r[\"rollout\"] == {\"gpu_memory_utilization\": 0.4, \"pool_gib\": 107.4, \"gpu_total_gib\": 268.4}, r[\"rollout\"]; assert r[\"escalation\"] is None and r[\"hold_limit_hours\"] == 20.0 and r[\"result\"] == \"PASS\"" $T/root/runs/PROBE-T32/report.json && [ "$(grep -c "^run_arm" $CALLS)" = 1 ] && grep -q "OPTOFF=0$" $CALLS'
rm -rf $T
setup; FAKE_MEM=0 FAKE_OOM_FIRST=1 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "[v2.5.9z40] an out of memory in the first smoke: a second one with T32_OPT_OFFLOAD=1, PASS, the escalation named" '[ "$(res result)" = PASS ] && [ "$(grep -c "^run_arm" $CALLS)" = 2 ] && grep -q "OPTOFF=1$" $CALLS && [[ "$(res escalation)" == *"optimizer offloaded"* ]] && [[ "$(res reason)" == *"optimizer offloaded after an out of memory"* ]]'
rm -rf $T
setup; FAKE_MEM=0 FAKE_STEP=1500 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "[v2.5.9z40] projected 20.8 h > 20: HOLD (not PASS), event probe_t32_hold, .done (the arm stays gated)" '[ "$(res result)" = HOLD ] && [[ "$(why)" == *"projected 20.8 h for 50 steps > 20 h"* ]] && grep -q "^control event PROBE-T32 probe_t32_hold" $CALLS && [ -f $T/root/runs/PROBE-T32/.done ]'
rm -rf $T
setup; FAKE_MEM=0 FAKE_STEP=1500 PT32_MAX_HOURS=30 bash $T/rb/scripts/probe_t32.sh 3 > /dev/null 2>&1
ok "[v2.5.9z40] PT32_MAX_HOURS=30 (the reviewer's yes): the same smoke PASSes" '[ "$(res result)" = PASS ]'
rm -rf $T
[ $fails = 0 ] && echo "PROBE_T32_TEST PASS ($n checks)" || echo "PROBE_T32_TEST FAIL ($fails of $n)"
