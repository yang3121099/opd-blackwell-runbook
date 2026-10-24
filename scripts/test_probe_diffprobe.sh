#!/usr/bin/env bash
# [v2.5.9z74] (Q-062) Stub tests of probe_diffprobe.sh (no GPU, network or HF; diff_probe.py faked). Usage: bash scripts/test_probe_diffprobe.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/analysis $T/root/runs/PROBE-DIFF17 $T/root/data $T/root/venv/bin $T/bin; cd $T
  cp $SRC/scripts/probe_diffprobe.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb STUDENT_17B=$T/m17" > $T/rb/scripts/env.sh; : > $T/root/venv/bin/activate
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/code/analysis/diff_probe.py <<'PY'
import json, os, sys
a = sys.argv; open(os.environ["CALLS"], "a").write("diff_probe " + " ".join(a[1:]) + "\n")
out = a[a.index("--out") + 1]
r = {"n_train": int(os.environ.get("FAKE_NTR", "6400")), "n_math500": 500, "layers": [{"cv_auc": 0.7, "test_auc": 0.72}, {"cv_auc": 0.81, "test_auc": 0.83}],
     "h1_layer": 1, "h1_auc": 0.83, "H1": True, "entropy_auc": 0.61, "confidence_auc": None if os.environ.get("FAKE_NOCONF") else 0.64, "H2": True,
     "H3": None if "--route" not in a else True, "h3": {"ratio_true": 2.1, "ratio_probe": 2.0} if "--route" in a else {}}
json.dump(r, open(os.path.join(out, "diff_probe.json"), "w")); sys.exit(int(os.environ.get("FAKE_RC", "0")))
PY
  printf '#!/usr/bin/env bash\necho 0\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  : > $T/root/runs/PROBE-DIFF17/labels_17.jsonl; : > $T/root/data/dapo_train.parquet
  mkdir -p $T/root/runs/EVAL-e10b-P5route; : > $T/root/runs/EVAL-e10b-P5route/e10b_P5route_s50_math500_T0.6_math500_no_think_t0.6.jsonl
  export CALLS=$T/calls PATH=$T/bin:$PATH PDP_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-DIFFPROBE/report.json "$1"; }
setup; bash $T/rb/scripts/probe_diffprobe.sh 2 > /dev/null 2>&1
ok "PASS with the route reading: H3 judged; report, public upload, private log, event" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-DIFFPROBE/.done ] && grep -q "^diff_probe --model $T/m17 --labels $T/root/runs/PROBE-DIFF17/labels_17.jsonl .* --route $T/root/runs/EVAL-e10b-P5route/" $CALLS && grep -q "^remote put $T/root/runs/PROBE-DIFFPROBE/up runs/PROBE-DIFFPROBE$" $CALLS && grep -q "^remote put .* logs/abcd1234/probe_diffprobe/PROBE-DIFFPROBE$" $CALLS && grep -q "^control event PROBE-DIFFPROBE probe_diffprobe_pass" $CALLS && [[ "$(res "v[\"reason\"]")" == *"layer 1 (best train CV) MATH-500 AUC 0.83 (H1 True)"*"H3 True"* ]]'
cd $SRC; rm -rf $T
setup; rm $T/root/runs/EVAL-e10b-P5route/*.jsonl; bash $T/rb/scripts/probe_diffprobe.sh 2 > /dev/null 2>&1
ok "no route reading: still PASS, H3 open, no --route" '[ "$(res "v[\"result\"]")" = PASS ] && ! grep -q -- "--route" $CALLS && [[ "$(res "v[\"reason\"]")" == *"H3 None"* ]]'
cd $SRC; rm -rf $T
setup; rm $T/root/runs/PROBE-DIFF17/labels_17.jsonl; out=$(bash $T/rb/scripts/probe_diffprobe.sh 2 2>&1); rc=$?
ok "no labels: WAIT_INPUTS, exit 1, no report" '[ $rc = 1 ] && [[ "$out" == *"WAIT_INPUTS PROBE-DIFFPROBE: runs/PROBE-DIFF17/labels_17.jsonl"* ]] && [ ! -f $T/root/runs/PROBE-DIFFPROBE/report.json ]'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_NTR=5000 bash $T/rb/scripts/probe_diffprobe.sh 2 > /dev/null 2>&1; check "too few train prompts matched is FAIL" "expected >= 6000"; cd $SRC; rm -rf $T
setup; FAKE_NOCONF=1 bash $T/rb/scripts/probe_diffprobe.sh 2 > /dev/null 2>&1; check "a proxy without an AUC is FAIL" "a proxy without an AUC"; cd $SRC; rm -rf $T
setup; FAKE_RC=1 bash $T/rb/scripts/probe_diffprobe.sh 2 > /dev/null 2>&1; check "diff_probe failing is FAIL" "diff_probe exited 1"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_DIFFPROBE_TEST PASS ($n checks)" || echo "PROBE_DIFFPROBE_TEST FAIL ($fails of $n)"
