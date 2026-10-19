#!/usr/bin/env bash
# Stub tests of v2.5.7o strict_c1.sh (no GPU, network or HF; the real strict_contract.py). Usage: bash scripts/test_strict_c1.sh -> STRICT_C1_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/analysis $T/rb/tasks $T/root/runs/EVAL-x $T/root/venv/bin
cp $SRC/scripts/strict_c1.sh $T/rb/scripts/; cp $SRC/code/analysis/strict_contract.py $SRC/code/analysis/paired_bootstrap.py $T/rb/code/analysis/
mkdir -p $T/rb/code/eval; printf 'def grade(text, gold):\n    return text.endswith("{" + gold + "}"), None\n' > $T/rb/code/eval/gen_eval.py   # v2.5.7w: stub grader
: > $T/root/venv/bin/activate
printf 'export OPD_ROOT=%s RUNBOOK=%s CKPT_ROOT=%s\n' $T/root $T/rb $T/root > $T/rb/scripts/env.sh
printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
for s in a b; do
  python3 - $T/root/runs/EVAL-x/${s}_math500_no_think_t0.6 $s <<'PY'
import json, sys
p, s = sys.argv[1], sys.argv[2]
with open(p + ".jsonl", "w") as f:
    for i in range(4):
        f.write(json.dumps({"pid": f"p{i}", "sample": 0, "text": "\\boxed{1}" + (" \\boxed{2}" if s == "a" and i == 0 else ""), "correct": s == "a" or i < 2, "finish_reason": "stop"}) + "\n")
json.dump({"bench": "math500", "mode": "no_think", "temperature": 0.6, "max_tokens": 16384}, open(p + ".metrics.json", "w"))
PY
done
for s in c d; do   # v2.5.7zm: an AIME pair (not in the MATH500 scan)
  python3 - $T/root/runs/EVAL-x/${s}_aime24_no_think_t0.6 $s <<'PY'
import json, sys
p, s = sys.argv[1], sys.argv[2]
with open(p + ".jsonl", "w") as f:
    for i in range(4):
        f.write(json.dumps({"pid": f"q{i}", "sample": 0, "text": "\\boxed{7}", "correct": s == "c" or i < 1, "finish_reason": "stop"}) + "\n")
json.dump({"bench": "aime24", "mode": "no_think", "temperature": 0.6, "max_tokens": 32768}, open(p + ".metrics.json", "w"))
PY
done
cat > $T/rb/tasks/queue.txt <<'EOF2'
version: 2
#strict STRICT-C1 EVAL-x/a_math500_no_think_t0.6.jsonl EVAL-x/b_math500_no_think_t0.6.jsonl
#strict STRICT-C1 EVAL-x/c_aime24_no_think_t0.6.jsonl EVAL-x/d_aime24_no_think_t0.6.jsonl
#strict STRICT-C1 EVAL-x/c_aime24_no_think_t0.6.jsonl EVAL-x/b_math500_no_think_t0.6.jsonl
#strict STRICT-C1 ../etc/passwd EVAL-x/b_math500_no_think_t0.6.jsonl
#strict STRICT-OTHER EVAL-x/b_math500_no_think_t0.6.jsonl EVAL-x/a_math500_no_think_t0.6.jsonl
EOF2
export CALLS=$T/calls; : > $CALLS; export PATH=$(dirname $(command -v python3)):$PATH
python() { python3 "$@"; }; export -f python 2> /dev/null
bash $T/rb/scripts/strict_c1.sh 3 STRICT-C1 > $T/out 2>&1; rc=$?
D=$T/root/runs/STRICT-C1
ok "both readings judged, the pair from the queue (the bad one refused)" 'python3 -c "import json, sys; r = json.load(open(sys.argv[1])); a = {x[\"name\"]: x for x in r[\"readings\"]}; assert [k for k in sorted(a) if \"math500\" in k] == [\"EVAL-x/a_math500_no_think_t0.6.jsonl\", \"EVAL-x/b_math500_no_think_t0.6.jsonl\"]; assert a[\"EVAL-x/a_math500_no_think_t0.6.jsonl\"][\"acc_contract\"] == 75.0; assert r[\"pairs\"][0][\"lenient\"][\"diff\"] == 50.0" $D/report.json && grep -q "PAIR_REFUSED ../etc/passwd" $T/out'
ok "v2.5.7zm an AIME pair: each reading judged once (also when in two pairs), the pair computed; a pair with no common problems an error, not a crash" 'python3 -c "import json, sys; r = json.load(open(sys.argv[1])); n = [x[\"name\"] for x in r[\"readings\"]]; assert n.count(\"EVAL-x/c_aime24_no_think_t0.6.jsonl\") == 1 and n.count(\"EVAL-x/d_aime24_no_think_t0.6.jsonl\") == 1 and n.count(\"EVAL-x/b_math500_no_think_t0.6.jsonl\") == 1; p = [x for x in r[\"pairs\"] if x[\"b\"] == \"EVAL-x/d_aime24_no_think_t0.6.jsonl\"]; assert len(p) == 1 and \"error\" not in p[0] and p[0][\"lenient\"][\"diff\"] == 75.0; q = [x for x in r[\"pairs\"] if x[\"b\"] == \"EVAL-x/b_math500_no_think_t0.6.jsonl\" and x[\"a\"].startswith(\"EVAL-x/c_\")]; assert q[0][\"error\"] == \"no common problems\"" $D/report.json'
ok "path-free upload, private log, event, .done" 'grep -q "^remote put $D/up runs/STRICT-C1$" $CALLS && grep -q "^remote put $D/logs logs/abcd1234/strict_c1/STRICT-C1$" $CALLS && grep -q "^control event STRICT-C1 strict_c1_done rc=0 4 readings, mean diff" $CALLS && [ -f $D/up/table.txt ] && [ -f $D/.done ] && [ $rc = 0 ] && ! grep -q "$T" $D/up/report.json'
: > $CALLS; bash $T/rb/scripts/strict_c1.sh 3 STRICT-C1 > /dev/null 2>&1
ok "done: not run again" '! grep -q "event STRICT-C1" $CALLS'
rm -rf $T
[ $fails = 0 ] && echo "STRICT_C1_TEST PASS ($n checks)" || echo "STRICT_C1_TEST FAIL ($fails of $n)"
