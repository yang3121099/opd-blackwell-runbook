#!/usr/bin/env bash
# [v2.5.9x] Stub tests of probe_diff17.sh (no GPU, network or HF): a fake python stands for diff_labels.py.
# Usage: bash scripts/test_probe_diff17.sh -> PROBE_DIFF17_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/root/data $T/bin $T/venv/bin
  cp $SRC/scripts/probe_diff17.sh $T/rb/scripts/; : > $T/root/data/dapo_train.parquet; : > $T/venv/bin/activate
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb STUDENT_17B=/m/Qwen3-1.7B STUDENT_4B=/m/Qwen3-4B VENV_EVAL=$T/venv" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  # python = diff_labels.py: FAKE_N lines (default 6900), a report; or an exit code
  cat > $T/bin/python <<'SH'
#!/usr/bin/env bash
echo "python $* CUDA=$CUDA_VISIBLE_DEVICES" >> $CALLS
[ "${FAKE_RC:-0}" != 0 ] && { echo "RuntimeError: no source loaded for math500"; exit $FAKE_RC; }
while [ $# -gt 0 ]; do case $1 in --tag) TAG=$2;; --out) OUT=$2;; esac; shift; done
python3 - "$OUT" "$TAG" "${FAKE_N:-6900}" <<'PY'
import json, os, sys
out, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(os.path.join(out, f"labels_{tag}.jsonl"), "w") as f:
    for i in range(n):
        f.write(json.dumps({"split": "train" if i < 6400 else "math500", "pos": i, "key": f"{i:016x}", "k": i % 9, "n": 8, "r": (i % 9) / 8}) + "\n")
json.dump({"tag": tag, "train": {"easy_share": 0.44, "mean_r": 0.5}, "math500": {"easy_share": 0.71}}, open(os.path.join(out, f"report_{tag}.json"), "w"))
PY
SH
  chmod +x $T/bin/python
  printf '#!/usr/bin/env bash\necho ${FAKE_MEM:-0}\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  export CALLS=$T/calls PATH=$T/bin:$PATH PDF_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/$1/report.json "$2"; }
setup; bash $T/rb/scripts/probe_diff17.sh 6 > /dev/null 2>&1
ok "PASS: the 1.7B base on GPU 6, 50 steps x 128 + MATH-500 (6900 prompts), the labels public as runs/PROBE-DIFF17, event, .done" '[ "$(res PROBE-DIFF17 result)" = PASS ] && grep -q "^python $T/rb/code/analysis/diff_labels.py --model /m/Qwen3-1.7B --tag 17 --train $T/root/data/dapo_train.parquet --out $T/root/runs/PROBE-DIFF17 --steps 50 --tbs 128 --seed 42 --maxp 1024 --k 8 --max_tokens 4096 *CUDA=6$" $CALLS && grep -q "^remote put $T/root/runs/PROBE-DIFF17/up runs/PROBE-DIFF17$" $CALLS && grep -q "^control event PROBE-DIFF17 probe_diff17_pass" $CALLS && [ -f $T/root/runs/PROBE-DIFF17/.done ] && [[ "$(res PROBE-DIFF17 reason)" == "6900 prompts labelled; train easy share 0.44"* ]]'
bash $T/rb/scripts/probe_diff17.sh 6 4b > /dev/null 2>&1
ok "4b: PROBE-DIFF17-4B with the 4B base, labels_4b.jsonl next to labels_17.jsonl" '[ "$(res PROBE-DIFF17-4B result)" = PASS ] && grep -q "diff_labels.py --model /m/Qwen3-4B --tag 4b " $CALLS && [ -f $T/root/runs/PROBE-DIFF17/labels_4b.jsonl ] && [ -f $T/root/runs/PROBE-DIFF17/labels_17.jsonl ] && grep -q "^control event PROBE-DIFF17-4B probe_diff17_pass" $CALLS'
rm -rf $T
check() { ok "$1" '[ "$(res PROBE-DIFF17 result)" = FAIL ] && [[ "$(res PROBE-DIFF17 reason)" == *"'"$2"'"* ]]'; }
setup; FAKE_N=6899 bash $T/rb/scripts/probe_diff17.sh 6 > /dev/null 2>&1; check "one prompt short is FAIL (and no public upload)" "has 6899 prompts, expected 6900"; ok "nothing public after a FAIL" '! grep -q "^remote put .* runs/PROBE-DIFF17$" $CALLS'; rm -rf $T
setup; FAKE_RC=1 bash $T/rb/scripts/probe_diff17.sh 6 > /dev/null 2>&1; check "diff_labels failing is FAIL with its message" "no source loaded"; rm -rf $T
setup; rm $T/root/data/dapo_train.parquet; bash $T/rb/scripts/probe_diff17.sh 6 > /dev/null 2>&1; check "no training data is FAIL" "no data/dapo_train.parquet"; rm -rf $T
setup; DIFF_LIMIT=20 bash $T/rb/scripts/probe_diff17.sh 6 > /dev/null 2>&1; ok "DIFF_LIMIT (a smoke) passes --limit and expects 2 x limit" 'grep -q -- "--limit 20 " $CALLS || grep -q -- "--limit 20$" <(sed "s/ CUDA=6//" $CALLS)'; rm -rf $T
[ $fails = 0 ] && echo "PROBE_DIFF17_TEST PASS ($n checks)" || echo "PROBE_DIFF17_TEST FAIL ($fails of $n)"
