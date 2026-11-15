#!/usr/bin/env bash
# [z162] Stub tests of run_queue_eval.sh's optional #eval fields and the #forced budgets field (no GPU, network or HF; gen_eval.py and
# forced_stop.py are fakes that record their argv as JSON). Checked:
# - us=p1 / us=p2 -> one --user_suffix argument with Q-098's registered text verbatim; seed=<n> -> --seed <n>; fields in any order;
# - a row without fields: no --user_suffix, no --seed, no --prefill (as before z162); pf= still -> --prefill;
# - a bad field (us=p3, seed=x, a field twice, an unknown key) -> REFUSED, nothing generated for that row;
# - #forced without budgets= -> the five budB levels (as before); with budgets=1024,...,16384 -> exactly that list; a bad list refused.
# Usage: bash scripts/test_queue_fields.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
P1="Think briefly: keep your reasoning short and stop as soon as you are confident in the answer."
P2="Be concise. Use as few reasoning steps as possible, then give the final answer."
T=$(mktemp -d); T=$(cd $T && pwd -P); mkdir -p $T/rb/scripts $T/rb/code/eval $T/rb/tasks $T/root/runs $T/root/venv/bin $T/root/models/m
cp $SRC/scripts/run_queue_eval.sh $SRC/scripts/control_rules.py $T/rb/scripts/; : > $T/root/venv/bin/activate; : > $T/root/models/m/model.safetensors
cat > $T/rb/scripts/env.sh <<EOF
export OPD_ROOT=$T/root RUNBOOK=$T/rb CKPT_ROOT=$T/root STUDENT_17B=$T/m17
has_weights() { [ -f "\$1/model.safetensors" ]; }
gpu_wait_free() { return 0; }
EOF
printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["repo"] and print("me/pub")\n' > $T/rb/scripts/remote.py
printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
cat > $T/rb/code/eval/gen_eval.py <<'PY'
import json, os, sys
a = sys.argv; g = lambda k: a[a.index(k) + 1]
open(os.environ["CALLS"], "a").write("gen_eval " + json.dumps(a[1:]) + "\n")
stem = os.path.join(g("--out"), f"{g('--tag')}_{g('--bench')}_{g('--mode')}_t{float(g('--temperature'))}")
open(stem + ".jsonl", "w").write(json.dumps({"pid": "math500-0", "sample": 0, "correct": True, "text": "x", "token_ids": [1]}) + "\n")
json.dump({"bench": g("--bench"), "mode": g("--mode"), "avg_acc": 50.0, "max_tokens": int(g("--max_tokens"))}, open(stem + ".metrics.json", "w"))
PY
cat > $T/rb/code/eval/forced_stop.py <<'PY'
import json, os, sys
a = sys.argv; g = lambda k: a[a.index(k) + 1]
open(os.environ["CALLS"], "a").write("forced_stop " + json.dumps(a[1:]) + "\n")
json.dump({"budgets": []}, open(g("--out"), "w")); open(g("--samples"), "w").write("")
PY
cat > $T/rb/tasks/queue.txt <<'EOF'
version: 2
#eval EVAL-f a_math500_T0.6 local:models/m math500 4 0.6 16384 think us=p1 seed=1000
#eval EVAL-f b_math500_T0.6 local:models/m math500 4 0.6 16384 think seed=2000 us=p2
#eval EVAL-f c_math500_T0.6 local:models/m math500 4 0.6 16384 no_think
#eval EVAL-f d_math500_T0.6 local:models/m math500 4 0.6 16384 no_think pf=32313 seed=3
#eval EVAL-g e_math500_T0.6 local:models/m math500 4 0.6 16384 think us=p3
#eval EVAL-g f_math500_T0.6 local:models/m math500 4 0.6 16384 think seed=x
#eval EVAL-g h_math500_T0.6 local:models/m math500 4 0.6 16384 think us=p1 us=p2
#eval EVAL-g i_math500_T0.6 local:models/m math500 4 0.6 16384 think temp=1
#forced EVAL-f a_math500_T0.6 local:models/m think budgets=1024,2048,3072,4096,6144,8192,16384
#forced EVAL-f c_math500_T0.6 local:models/m no_think
#forced EVAL-f b_math500_T0.6 local:models/m think budgets=1k,2k
EOF
export CALLS=$T/calls; : > $CALLS
bash $T/rb/scripts/run_queue_eval.sh 3 EVAL-f > $T/out_f 2>&1; rc_f=$?
bash $T/rb/scripts/run_queue_eval.sh 3 EVAL-g > $T/out_g 2>&1; rc_g=$?
args() { python3 - "$CALLS" "$1" "$2" <<'PY'
import json, sys
calls, kind, tag = sys.argv[1:4]
for l in open(calls):
    if l.startswith(kind + " "):
        a = json.loads(l[len(kind) + 1:])
        if tag in " ".join(a):
            print(json.dumps(a)); break
PY
}
A=$(args gen_eval a_math500_T0.6); B=$(args gen_eval b_math500_T0.6); Cc=$(args gen_eval c_math500_T0.6); D=$(args gen_eval d_math500_T0.6)
has() { python3 -c "import json, sys; a = json.loads(sys.argv[1]); k, v = sys.argv[2], sys.argv[3]; sys.exit(0 if k in a and a[a.index(k) + 1] == v else 1)" "$1" "$2" "$3"; }
hasnot() { python3 -c "import json, sys; sys.exit(1 if sys.argv[2] in json.loads(sys.argv[1]) else 0)" "$1" "$2"; }
ok "us=p1 seed=1000: one --user_suffix argument = P1 verbatim, --seed 1000" 'has "$A" --user_suffix "$P1" && has "$A" --seed 1000'
ok "fields in any order: seed=2000 us=p2" 'has "$B" --user_suffix "$P2" && has "$B" --seed 2000'
ok "no fields: no --user_suffix / --seed / --prefill (as before)" '[ -n "$Cc" ] && hasnot "$Cc" --user_suffix && hasnot "$Cc" --seed && hasnot "$Cc" --prefill'
ok "pf= with seed=: --prefill 32313 --seed 3, no suffix" 'has "$D" --prefill 32313 && has "$D" --seed 3 && hasnot "$D" --user_suffix'
ok "bad fields refused, nothing generated for them" 'for t in e f h i; do grep -q "^REFUSED ${t}_math500_T0.6: " $T/out_g || exit 1; done; ! grep -q "_math500_T0.6" <(grep "^gen_eval" $CALLS | grep -E "\"(e|f|h|i)_math500")'
FA=$(args forced_stop a_math500_T0.6); FC=$(args forced_stop c_math500_T0.6)
ok "#forced budgets=1024,...,16384: exactly that list" 'has "$FA" --budgets 1024,2048,3072,4096,6144,8192,16384 && has "$FA" --max_new 32 && has "$FA" --top_k 0'
ok "#forced without budgets=: the five budB levels (as before)" 'has "$FC" --budgets 2048,4096,8192,16384,32768'
ok "#forced bad budgets refused, no forced run for it" 'grep -q "^FORCED_REFUSED b_math500_T0.6: budgets must be" $T/out_f && [ -z "$(args forced_stop b_math500_T0.6)" ]'
rm -rf $T
[ $fails = 0 ] && echo "QUEUE_FIELDS_TEST PASS ($n checks)" || echo "QUEUE_FIELDS_TEST FAIL ($fails of $n)"
