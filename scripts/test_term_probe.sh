#!/usr/bin/env bash
# Stub tests of v2.5.7g term_probe.sh (no GPU, network or HF). Usage: bash scripts/test_term_probe.sh -> TERM_PROBE_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/eval $T/rb/code/analysis $T/rb/tasks $T/root/runs/E11b-P0 $T/root/venv/bin $T/m17 $T/m06
cp $SRC/scripts/term_probe.sh $SRC/scripts/control_rules.py $T/rb/scripts/; : > $T/root/venv/bin/activate; touch $T/m17/model.safetensors $T/m06/model.safetensors
cat > $T/rb/scripts/env.sh <<EOF
export OPD_ROOT=$T/root RUNBOOK=$T/rb CKPT_ROOT=$T/root STUDENT_06B=$T/m06 STUDENT_17B=$T/m17 TEACHER=$T/m8
has_weights() { [ -f "\$1/model.safetensors" ]; }
EOF
printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["repo"] and print("me/pub")\n' > $T/rb/scripts/remote.py
printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
cat > $T/rb/scripts/fetch_model.py <<'PY'
import sys, os
open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\n")
p = os.path.join(sys.argv[3], sys.argv[2])
if sys.argv[2].endswith(".jsonl"):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write("{}\n")
else:
    os.makedirs(p, exist_ok=True); open(os.path.join(p, "model.safetensors"), "w").write("w")
PY
cat > $T/rb/code/eval/gen_eval.py <<'PY'
import sys, os
a = sys.argv; open(os.environ["CALLS"], "a").write("gen_eval " + " ".join(a[1:]) + "\n")
open(os.path.join(a[a.index("--out") + 1], a[a.index("--tag") + 1] + "_math500_no_think_t0.6.jsonl"), "w").write("{}\n")
PY
cat > $T/rb/code/analysis/term_probe.py <<'PY'
import sys, os, json, shutil
a = sys.argv; open(os.environ["CALLS"], "a").write("term_probe " + " ".join(a[1:]) + "\n")
shutil.copy(a[a.index("--inputs") + 1], os.environ["CALLS"] + ".tsv")
json.dump({"models": {"17_P0": {"measures": {"p_stop_nl": {"mean": 0.4}, "marker_frac": {"mean": 0.1}}}, "17_P9": {"error": "model missing"}}}, open(a[a.index("--out") + 1], "w"))
open(a[a.index("--samples") + 1], "w").write("{}\n")
PY
: > $T/root/runs/E11b-P0/e11b_P0.jsonl
cat > $T/rb/tasks/queue.txt <<'EOF'
version: 2
#term PROBE-TERM 17_P0 Qwen/Qwen3-1.7B E11b-P0/e11b_P0.jsonl
#term PROBE-TERM 06_P2 yang31210999/opd-lab-ckpts/ckpts/E10-P2-0.6B-lr3e-6/global_step_150 archive:runs/e11/e11_P2_s150.jsonl
#term PROBE-TERM 06_P5noEOS yang31210999/opd-lab-ckpts/runs/E14-P5noEOS/global_step_100/actor/huggingface gen
#term PROBE-TERM 06_bad /etc/passwd gen
#term PROBE-TERM 17_P9 local:runs/E10b-P9/ckpt/global_step_150/actor/huggingface_bf16 E11b-P9/x.jsonl
#term PROBE-TERM 06_X Qwen/Qwen3-0.6B ../etc/x.jsonl
#term PROBE-OTHER 06_Y Qwen/Qwen3-0.6B gen
EOF
export CALLS=$T/calls; : > $CALLS
bash $T/rb/scripts/term_probe.sh 3 PROBE-TERM > $T/out 2>&1; rc=$?
D=$T/root/runs/PROBE-TERM
ok "inputs: node runs, archive (fetched), generated, missing, refused" 'grep -q "^17_P0	$T/m17	$T/root/runs/E11b-P0/e11b_P0.jsonl	Qwen/Qwen3-1.7B	node runs/E11b-P0/e11b_P0.jsonl	nothink$" $CALLS.tsv && grep -q "^06_P2	$T/root/adhoc_models/ckpts/E10-P2-0.6B-lr3e-6/global_step_150	$T/root/adhoc_models/runs/e11/e11_P2_s150.jsonl	.*	lab archive runs/e11/e11_P2_s150.jsonl	nothink$" $CALLS.tsv && grep -q "^06_P5noEOS	.*	$D/gen/term_06_P5noEOS_math500_no_think_t0.6.jsonl	.*	generated here" $CALLS.tsv && grep -q "^17_P9	-	-	local:runs/E10b-P9" $CALLS.tsv && grep -q "^06_X	$T/m06	-	" $CALLS.tsv && ! grep -q "06_bad\|06_Y" $CALLS.tsv && grep -q "REFUSED 06_bad" $T/out'
ok "generation with the EVAL-v1 settings, only for gen" '[ "$(grep -c "^gen_eval" $CALLS)" = 1 ] && grep -q "^gen_eval --model $T/root/adhoc_models/runs/E14-P5noEOS/global_step_100/actor/huggingface --bench math500 --mode no_think --n 4 --temperature 0.6 --top_p 0.95 --top_k 0 --max_tokens 16384 --chunk 50 --tag term_06_P5noEOS " $CALLS'
ok "fetches: the archive model and file" 'grep -q "^fetch yang31210999/opd-lab-ckpts runs/e11/e11_P2_s150.jsonl $T/root/adhoc_models --task PROBE-TERM" $CALLS && grep -q "^fetch yang31210999/opd-lab-ckpts ckpts/E10-P2-0.6B-lr3e-6/global_step_150 " $CALLS'
ok "measured with the teacher, path-free upload, private logs, event, .done" 'grep -q "^term_probe --teacher $T/m8 --inputs $D/inputs.tsv --out $D/report.json --samples $D/samples.jsonl" $CALLS && grep -q "^remote put $D/up runs/PROBE-TERM$" $CALLS && grep -q "^remote put $D/logs logs/abcd1234/term_probe/PROBE-TERM$" $CALLS && grep -q "^control event PROBE-TERM term_probe_done rc=0 17_P0:0.4/0.1 17_P9:model missing" $CALLS && [ -f $D/.done ] && [ $rc = 0 ] && [ -f $D/up/gen/term_06_P5noEOS_math500_no_think_t0.6.jsonl ] && [ ! -f $D/up/inputs.tsv ]'
: > $CALLS; bash $T/rb/scripts/term_probe.sh 3 PROBE-TERM > /dev/null 2>&1
ok "done: not run again" '! grep -q term_probe $CALLS'
# v2.5.7k PROBE-TERM2: the PROBE-TERM lines, PROBE-TERM's generations reused (no new gen_eval), its own output directory
: > $CALLS; bash $T/rb/scripts/term_probe.sh 3 PROBE-TERM2 > $T/out2 2>&1
D2=$T/root/runs/PROBE-TERM2
ok "PROBE-TERM2 reads the PROBE-TERM lines and reuses its E14 generations" '! grep -q "^gen_eval" $CALLS && grep -q "^06_P5noEOS	.*	$D/gen/term_06_P5noEOS_math500_no_think_t0.6.jsonl	" $CALLS.tsv && grep -q "^17_P0	" $CALLS.tsv && grep -q "^remote put $D2/up runs/PROBE-TERM2$" $CALLS && [ -f $D2/.done ]'
# v2.5.7m PROBE-TERM3: its own #term lines (a model on another model's generations), no teacher
printf '#term PROBE-TERM3 17_P5-on-P0text Qwen/Qwen3-1.7B E11b-P0/e11b_P0.jsonl\n' >> $T/rb/tasks/queue.txt
: > $CALLS; bash $T/rb/scripts/term_probe.sh 3 PROBE-TERM3 > $T/out3 2>&1
ok "PROBE-TERM3 reads its own lines, runs without the teacher" '[ "$(wc -l < $CALLS.tsv | tr -d " ")" = 1 ] && grep -q "^17_P5-on-P0text	$T/m17	$T/root/runs/E11b-P0/e11b_P0.jsonl	" $CALLS.tsv && grep -q "^term_probe .*--no_teacher" $CALLS && grep -q "^remote put $T/root/runs/PROBE-TERM3/up runs/PROBE-TERM3$" $CALLS'
# v2.5.7z PROBE-TERM4: its own lines, an optional 4th field `think` (6th tsv column), no teacher, --term4
printf '#term PROBE-TERM4 17_P0think Qwen/Qwen3-1.7B E11b-P0/e11b_P0.jsonl think\n#term PROBE-TERM4 17_P0 Qwen/Qwen3-1.7B E11b-P0/e11b_P0.jsonl\n' >> $T/rb/tasks/queue.txt
: > $CALLS; bash $T/rb/scripts/term_probe.sh 3 PROBE-TERM4 > $T/out4 2>&1
ok "PROBE-TERM4: own lines, thinking template flag, no teacher, --term4" '[ "$(wc -l < $CALLS.tsv | tr -d " ")" = 2 ] && grep -q "^17_P0think	$T/m17	.*	think$" $CALLS.tsv && grep -q "^17_P0	$T/m17	.*	nothink$" $CALLS.tsv && grep -q "^term_probe .*--no_teacher --term4" $CALLS && grep -q "^remote put $T/root/runs/PROBE-TERM4/up runs/PROBE-TERM4$" $CALLS'
rm -rf $T
[ $fails = 0 ] && echo "TERM_PROBE_TEST PASS ($n checks)" || echo "TERM_PROBE_TEST FAIL ($fails of $n)"
