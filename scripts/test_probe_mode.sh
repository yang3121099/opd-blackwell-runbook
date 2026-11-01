#!/usr/bin/env bash
# [v2.5.9z108] (Q-083) Stub tests of probe_mode.sh (no GPU, network or HF). Usage: bash scripts/test_probe_mode.sh -> PROBE_MODE_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/root/runs $T/root/data $T/bin; cd $T
  cp $SRC/scripts/probe_mode.sh $T/rb/scripts/; : > $T/root/data/dapo_train.parquet; : > $T/ri.py
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/scripts/run_arm.sh <<'S2'
echo "run_arm $* SMOKE=$SMOKE STEPS=$SMOKE_STEPS PROBE_TASK=$PROBE_TASK" >> $CALLS
d=$OPD_ROOT/runs/$PROBE_TASK; mkdir -p $d
[ -z "${FAKE_CFG+x}" ] && FAKE_CFG="; mode token per prompt ([long] -> thinking, [short] -> non-thinking teacher; mix's coin)"
printf "task: $2\nteacher: Qwen/Qwen3-8B (ref)%s\n" "$FAKE_CFG" > $d/config.yaml
for i in $(seq $SMOKE_STEPS); do echo "(TaskRunner pid=1) step:$i - critic/score/mean:0.5" >> $d/train.log; done
python3 - $d/route.jsonl $SMOKE_STEPS "${FAKE_SPLIT:-1}" "${FAKE_LONG:-64}" "${FAKE_STRIPPED:-rows}" "${FAKE_BADTOK:-0}" <<'PY'
import json, sys
p, steps, split, longs, stripped, bad = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
with open(p, "w") as f:
    c = 0
    for s in range(steps):
        for part in range(split):   # split: the ranks' calls of one step
            rows = 128 // split; lo = longs // split; c += 1
            dec = [[f"k{s}{part}{i}", "[long]" if i < lo else "[short]", i < lo] for i in range(rows)]
            if bad: dec[0][2] = not dec[0][2]
            f.write(json.dumps({"call": c, "mode": "prefix", "batch": rows, "rows": rows, "think": lo, "plain": rows - lo, "unknown": 0,
                                "stripped": rows if stripped == "rows" else int(stripped), "decisions": dec}) + "\n")
PY
[ "${FAKE_OOM:-0}" = 1 ] && { echo "torch.OutOfMemoryError: CUDA out of memory" >> $d/train.log; exit 1; }; exit 0
S2
  cat > $T/bin/fakepy <<'S3'
#!/usr/bin/env bash
if [ "$1" = -c ]; then echo $FAKE_RI; exit 0; fi
if [ "$1" = - ]; then echo "py unit GOPD_REF_INPUT_UTILS=$GOPD_REF_INPUT_UTILS" >> $CALLS; cat > /dev/null
  [ "${FAKE_UNIT_RC:-0}" = 0 ] && { echo "19 passed, 0 failed"; exit 0; }; echo "18 passed, 1 failed"; exit 1; fi
echo "py $*" >> $CALLS
case $1 in
  */mode_data.py) [ "${FAKE_DATA_RC:-0}" = 0 ] || exit 1
    echo "{\"rows\": 17000, \"long\": ${FAKE_DATA_LONG:-8480}, \"short\": $((17000 - ${FAKE_DATA_LONG:-8480})), \"share\": $(python3 -c "print(round(${FAKE_DATA_LONG:-8480} / 17000, 4))"), \"seed\": $5, \"source\": \"dapo_train.parquet\"}" > $3.json
    echo "MODE_DATA rows 17000";;
  */mode_len.py) python3 - $2/len_by_token.json ${FAKE_LENSTEPS:-3} "${FAKE_LENTOK:-both}" <<'PY'
import json, sys
p, steps, tok = sys.argv[1], int(sys.argv[2]), sys.argv[3]
rows = [dict({"step": s + 1}, **({"long": {"n": 64, "mean_len": 900.0, "trunc": 0.0}} if tok in ("both", "long") else {}),
              **({"short": {"n": 64, "mean_len": 400.0, "trunc": 0.0}} if tok in ("both", "short") else {})) for s in range(steps)]
json.dump({"task": "x", "cap": 4096, "wall_long": None, "wall_short": None, "rows": rows}, open(p, "w"))
PY
    echo "MODE_LEN x";;
esac; exit 0
S3
  chmod +x $T/bin/fakepy
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PMD_WAIT_STEP=0 PMD_PY=$T/bin/fakepy FAKE_RI=$T/ri.py; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-MODE/report.json "$1"; }
setup; bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1
ok "PASS: unit tests on the installed file, the parquet, a smoke on both GPUs, report and event" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-MODE/.done ] && grep -q "^py unit GOPD_REF_INPUT_UTILS=$T/ri.py$" $CALLS && grep -q "^py $T/rb/scripts/mode_data.py $T/root/data/dapo_train.parquet $T/root/runs/PROBE-MODE/data/train_mode.parquet --seed 1234$" $CALLS && grep -q "^run_arm 4,5 E10b-P5mode SMOKE=1 STEPS=3 PROBE_TASK=PROBE-MODE-smoke$" $CALLS && grep -q "^control event PROBE-MODE probe_mode_pass" $CALLS && grep -q "^remote put $T/root/runs/PROBE-MODE/up logs/abcd1234/probe_mode$" $CALLS && [[ "$(res "v[\"reason\"]")" == *"long/short per call 64/64 64/64 64/64; stripped = rows; data share 0.4988"* ]]'
cd $SRC; rm -rf $T
setup; FAKE_SPLIT=2 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1
ok "two ranks' calls per step: the rows add up, PASS" '[ "$(res "v[\"result\"]")" = PASS ] && [ "$(res "len(v[\"mode_check\"][\"per_call\"])")" = 6 ]'
cd $SRC; rm -rf $T
setup; FAKE_UNIT_RC=1 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1
ok "a unit-test failure is FAIL, no smoke" '[ "$(res "v[\"result\"]")" = FAIL ] && ! grep -q "^run_arm" $CALLS'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_DATA_LONG=12000 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "a skewed coin in the parquet is FAIL" "share 0.7059"; cd $SRC; rm -rf $T
setup; FAKE_DATA_RC=1 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "mode_data.py failing is FAIL" "mode_data.py wrote no header"; cd $SRC; rm -rf $T
setup; FAKE_STRIPPED=0 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "the teacher seeing the token is FAIL" "the teacher saw 128 prompts with their token"; cd $SRC; rm -rf $T
setup; FAKE_BADTOK=1 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "a decision against its token is FAIL" "1 decisions not as their token"; cd $SRC; rm -rf $T
setup; FAKE_LONG=120 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "one token nearly everywhere is FAIL" "[long] share 120/128"; cd $SRC; rm -rf $T
setup; FAKE_CFG="" bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "a config without the mode token is FAIL" "does not name the mode token"; cd $SRC; rm -rf $T
setup; FAKE_OOM=1 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "the smoke failing is FAIL" "smoke exited 1"; cd $SRC; rm -rf $T
setup; FAKE_LENTOK=long bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "one token missing from the rollout dump is FAIL" "step 1: tokens ['long']"; cd $SRC; rm -rf $T
setup; FAKE_MEM=5000 PMD_WAIT_STEP=0 bash $T/rb/scripts/probe_mode.sh 4,5 > /dev/null 2>&1; check "a busy GPU is FAIL before the smoke" "gpu 4 not empty after 180 s"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_MODE_TEST PASS ($n checks)" || echo "PROBE_MODE_TEST FAIL ($fails of $n)"
