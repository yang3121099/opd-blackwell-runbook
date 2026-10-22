#!/usr/bin/env bash
# [v2.5.9z24] Stub test of tok_check.sh / tok_check.py (TOKCHECK-X5; no GPU, network or HF): three fake model dirs (config,
# generation config, tokenizer config with a chat template, tokenizer.json, .fetch_ok, the pull's log line).
#  - a model not in yet: WAIT_INPUTS, exit 1, no .done;
#  - all in: vocab_size, tokenizer sizes, special-token ids, <think> in the teacher's template only; student vs teacher: the base
#    vocab the same, two added ids different (named); student vs base identical; each pull's rate and the combined rate over the
#    span, the seconds all three ran at once; no path in the report; report uploaded to logs/<node>/tokcheck_x5, an event, .done.
# Usage: bash scripts/test_tok_check.sh -> TOK_CHECK_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/runs $T/root/logs
cp $SRC/scripts/tok_check.sh $SRC/scripts/tok_check.py $T/rb/scripts/
echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
export CALLS=$T/calls; : > $CALLS
mk() {   # $1 name, $2 repo, $3 added tokens (json list), $4 template, $5 end UTC, $6 seconds, $7 bytes
  local d=$T/root/models/$1; mkdir -p $d
  printf '{"model_type": "qwen2", "vocab_size": 16, "bos_token_id": 4, "eos_token_id": 5, "tie_word_embeddings": true}' > $d/config.json
  printf '{"bos_token_id": 4, "eos_token_id": [5, 6]}' > $d/generation_config.json
  printf '{"bos_token": null, "eos_token": {"content": "%s"}, "pad_token": "<pad>", "chat_template": "%s"}' "$(python3 -c "import json, sys; print(json.loads(sys.argv[1])[0]['content'])" "$3")" "$4" > $d/tokenizer_config.json
  printf '{"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2, "<pad>": 3}}, "added_tokens": %s}' "$3" > $d/tokenizer.json
  printf '{"repo": "%s", "kind": "model", "prefix": "", "files": [["model.safetensors", "%s", "x"], ["config.json", "100", "y"]], "utc": "%s"}' $2 $7 $5 > $d/.fetch_ok
  printf 'noise\n{"ok": true, "bytes": %s, "seconds": %s, "mb_s": 1.0, "files": 2, "repo": "%s"}\n' $7 $6 $2 > $T/root/logs/fetch_$1.log
}
QA='[{"id": 4, "content": "<|endoftext|>"}, {"id": 5, "content": "<|im_end|>"}, {"id": 6, "content": "<|im_start|>"}]'
QB='[{"id": 4, "content": "<|endoftext|>"}, {"id": 5, "content": "<eos>"}, {"id": 6, "content": "<think>"}]'
mk Qwen2.5-Math-1.5B-Instruct Qwen/Qwen2.5-Math-1.5B-Instruct "$QA" "{{ messages }}<|im_start|>assistant" 2026-10-22T04:10:00Z 600 3000000000
mk DeepSeek-R1-Distill-Qwen-1.5B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B "$QB" "{{ messages }}<think>\\\\n" 2026-10-22T04:12:00Z 660 3500000000
R=$T/root/runs/TOKCHECK-X5/report.json
bash $T/rb/scripts/tok_check.sh > $T/out 2>&1; rc=$?
ok "base model not in: WAIT_INPUTS, exit 1, no .done" '[ $rc = 1 ] && grep -q "^WAIT_INPUTS TOKCHECK-X5: Qwen2.5-Math-1.5B not in" $T/out && [ ! -f $T/root/runs/TOKCHECK-X5/.done ]'
mk Qwen2.5-Math-1.5B Qwen/Qwen2.5-Math-1.5B "$QA" "{{ messages }}" 2026-10-22T04:11:00Z 570 3000000000
bash $T/rb/scripts/tok_check.sh > $T/out 2>&1; rc=$?
ok "all in: exit 0, .done, report" '[ $rc = 0 ] && [ -f $T/root/runs/TOKCHECK-X5/.done ] && [ -f $R ]'
q() { python3 -c "import json, sys; r = json.load(open(sys.argv[1])); print($1)" $R; }
ok "per model: repo, size, vocab_size, tokenizer sizes, eos (<|endoftext|> = 4) / pad ids" '[ "$(q "r[\"models\"][\"teacher\"][\"repo\"], r[\"models\"][\"teacher\"][\"bytes\"], r[\"models\"][\"teacher\"][\"config\"][\"vocab_size\"]")" = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 3500000100 16" ] && [ "$(q "r[\"models\"][\"student\"][\"tokenizer\"]")" = "{'"'"'base_vocab'"'"': 4, '"'"'added_tokens'"'"': 3, '"'"'distinct_ids'"'"': 7, '"'"'max_id_plus_1'"'"': 7}" ] && [ "$(q "r[\"models\"][\"student\"][\"special_token_ids\"], r[\"models\"][\"student\"][\"embedding_rows_minus_tokenizer\"]")" = "{'"'"'eos_token'"'"': 4, '"'"'pad_token'"'"': 3} 9" ]'
ok "<think> in the teacher's template only, with its id" '[ "$(q "[r[\"models\"][k][\"template_has_think\"] for k in (\"student\", \"teacher\", \"base\")], r[\"models\"][\"teacher\"][\"think_ids\"][\"<think>\"]")" = "[False, True, False] 6" ]'
ok "student vs teacher: base vocab the same, ids 5-6 different (named); student vs base identical" '[ "$(q "r[\"pairs\"][\"student_vs_teacher\"][\"different\"], r[\"pairs\"][\"student_vs_teacher\"][\"base_vocab_identical\"], r[\"pairs\"][\"student_vs_teacher\"][\"identical\"], r[\"pairs\"][\"student_vs_teacher\"][\"first_differences\"][0], r[\"pairs\"][\"student_vs_base\"][\"identical\"]")" = "2 True False [5, '"'"'<|im_end|>'"'"', '"'"'<eos>'"'"'] True" ]'
ok "pulls: each with start / end, combined bytes over the span (04:00:00-04:12:00), all three at once 04:01:30-04:10:00" '[ "$(q "r[\"models\"][\"teacher\"][\"pull\"][\"start\"], r[\"pull_combined\"][\"bytes\"], r[\"pull_combined\"][\"span_s\"], r[\"pull_combined\"][\"mb_s\"], r[\"pull_combined\"][\"all_at_once_s\"]")" = "2026-10-22T04:01:00Z 9500000000 720.0 13.19 510.0" ]'
ok "no path in the report" '! grep -q "$T" $R && ! grep -q "/root\|/models/" $R'
ok "report to logs/<node>/tokcheck_x5, an event tokcheck_done" 'grep -q "^remote put $T/root/runs/TOKCHECK-X5/up logs/abcd1234/tokcheck_x5$" $CALLS && grep -q "^control event TOKCHECK-X5 tokcheck_done " $CALLS'
bash $T/rb/scripts/tok_check.sh > $T/out 2>&1
ok "a second run: TASK_ALREADY_DONE" 'grep -q "^TASK_ALREADY_DONE TOKCHECK-X5" $T/out'
rm -rf $T
[ $fails = 0 ] && echo "TOK_CHECK_TEST PASS ($n checks)" || echo "TOK_CHECK_TEST FAIL ($fails of $n)"
