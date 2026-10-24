#!/usr/bin/env bash
# [v2.5.9z82] (Q-072) Stub tests of probe_selfroute.sh (no GPU, network or HF; diff_probe.py faked). Usage: bash scripts/test_probe_selfroute.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/analysis $T/root/runs/PROBE-DIFF17 $T/root/runs/PROBE-DIFFPROBE/out $T/root/data $T/root/venv/bin $T/bin; cd $T
  cp $SRC/scripts/probe_selfroute.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb STUDENT_17B=$T/m17" > $T/rb/scripts/env.sh; : > $T/root/venv/bin/activate
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/code/analysis/diff_probe.py <<'PY'
import json, os, sys
a = sys.argv; open(os.environ["CALLS"], "a").write("diff_probe " + " ".join(a[1:]) + "\n")
out, emit = a[a.index("--out") + 1], a[a.index("--emit") + 1]
ntr = int(os.environ.get("FAKE_NTR", "6400")); rows = int(os.environ.get("FAKE_ROWS", str(ntr + 500)))
open(emit, "w").write("".join(json.dumps({"split": "train", "key": f"k{i}", "r": 0.6}) + "\n" for i in range(rows)))
tr = {"n": ntr, "probe_easy_share": 0.41, "true_easy_share": float(os.environ.get("FAKE_TRUE", "0.43")), "share_diff_pp": 2.0, "agreement": float(os.environ.get("FAKE_AGREE", "0.81"))}
ma = float(a[a.index("--min-agree") + 1]) if "--min-agree" in a else 0.75
tr["pass"] = abs(tr["probe_easy_share"] - tr["true_easy_share"]) * 100 < 5 and tr["agreement"] >= ma
r = {"layer": 14, "l2": 0.01, "n_train": ntr, "n_math500": 500, "threshold": 0.39, "train": tr, "math500": dict(tr, n=500), "auc_oof_train": 0.84, "auc_math500": 0.85,
     "lines": {"gate": f"train (out of fold): |easy share - true| < 5 pp and agreement >= {ma} (share-matched threshold; ...)"}}
json.dump(r, open(os.path.join(out, "selfroute.json"), "w")); sys.exit(int(os.environ.get("FAKE_RC", "0")))
PY
  printf '#!/usr/bin/env bash\necho 0\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  : > $T/root/runs/PROBE-DIFF17/labels_17.jsonl; : > $T/root/data/dapo_train.parquet; echo '{"h1_layer": 14}' > $T/root/runs/PROBE-DIFFPROBE/out/diff_probe.json
  export CALLS=$T/calls PATH=$T/bin:$PATH PSR_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-SELFROUTE/report.json "$1"; }
setup; bash $T/rb/scripts/probe_selfroute.sh 2 > /dev/null 2>&1
ok "PASS: --emit with PROBE-DIFFPROBE's report; labels kept; report, labels, selfroute.json up; private log; event" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-SELFROUTE/.done ] && grep -q "^diff_probe --model $T/m17 --labels $T/root/runs/PROBE-DIFF17/labels_17.jsonl --train $T/root/data/dapo_train.parquet --out $T/root/runs/PROBE-SELFROUTE/out --emit $T/root/runs/PROBE-SELFROUTE/labels_probe_17.jsonl --probe-report $T/root/runs/PROBE-DIFFPROBE/out/diff_probe.json --share-match --min-agree 0.74$" $CALLS && [ -s $T/root/runs/PROBE-SELFROUTE/labels_probe_17.jsonl ] && [ -f $T/root/runs/PROBE-SELFROUTE/up/labels_probe_17.jsonl ] && [ -f $T/root/runs/PROBE-SELFROUTE/up/selfroute.json ] && grep -q "^remote put $T/root/runs/PROBE-SELFROUTE/up runs/PROBE-SELFROUTE$" $CALLS && grep -q "^remote put $T/root/runs/PROBE-SELFROUTE/logs logs/abcd1234/probe_selfroute/PROBE-SELFROUTE$" $CALLS && grep -q "^control event PROBE-SELFROUTE probe_selfroute_pass layer 14" $CALLS && [[ "$(res "v[\"reason\"]")" == *"agreement 0.81"* ]] && [[ "$(res "v[\"reason\"]")" == *"threshold 0.39"* ]]'
cd $SRC; rm -rf $T
setup; rm $T/root/runs/PROBE-DIFFPROBE/out/diff_probe.json; out=$(bash $T/rb/scripts/probe_selfroute.sh 2 2>&1); rc=$?
ok "no PROBE-DIFFPROBE report: WAIT_INPUTS, exit 1, no report" '[ $rc = 1 ] && [[ "$out" == *"WAIT_INPUTS PROBE-SELFROUTE: runs/PROBE-DIFFPROBE/out/diff_probe.json"* ]] && [ ! -f $T/root/runs/PROBE-SELFROUTE/report.json ]'
cd $SRC; rm -rf $T
setup; rm $T/root/runs/PROBE-DIFF17/labels_17.jsonl; out=$(bash $T/rb/scripts/probe_selfroute.sh 2 2>&1); rc=$?
ok "no labels: WAIT_INPUTS, exit 1" '[ $rc = 1 ] && [[ "$out" == *"WAIT_INPUTS PROBE-SELFROUTE: runs/PROBE-DIFF17/labels_17.jsonl"* ]]'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_AGREE=0.7 bash $T/rb/scripts/probe_selfroute.sh 2 > /dev/null 2>&1; check "agreement below 0.74 is FAIL" "agreement 0.7 (line >= 0.74)"; cd $SRC; rm -rf $T
setup; FAKE_AGREE=0.745 bash $T/rb/scripts/probe_selfroute.sh 2 > /dev/null 2>&1; ok "[v2.5.9z82a] agreement 0.745 passes the 0.74 line (registered 0.75)" '[ "$(res "v[\"result\"]")" = PASS ]'; cd $SRC; rm -rf $T
setup; FAKE_TRUE=0.50 bash $T/rb/scripts/probe_selfroute.sh 2 > /dev/null 2>&1; check "easy share 9 pp off is FAIL" "vs true 0.5"; cd $SRC; rm -rf $T
setup; FAKE_NTR=5000 bash $T/rb/scripts/probe_selfroute.sh 2 > /dev/null 2>&1; check "too few train prompts is FAIL" "expected >= 6000"; cd $SRC; rm -rf $T
setup; FAKE_ROWS=100 bash $T/rb/scripts/probe_selfroute.sh 2 > /dev/null 2>&1; check "a short label file is FAIL" "has 100 rows, expected 6900"; cd $SRC; rm -rf $T
setup; FAKE_RC=1 bash $T/rb/scripts/probe_selfroute.sh 2 > /dev/null 2>&1; check "diff_probe failing is FAIL" "diff_probe --emit exited 1"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_SELFROUTE_TEST PASS ($n checks)" || echo "PROBE_SELFROUTE_TEST FAIL ($fails of $n)"
