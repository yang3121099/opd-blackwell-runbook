#!/usr/bin/env bash
# [v2.5.9z59] Stub test of upload_sync.sh's x5alert (Fable 10-22): an E16 step whose responses carry special ids in more than
# X5_SPECIAL_ALERT (2%) of its rows is one event x5_special_rows; each step is looked at once; older lines without rows_special
# count their masked ids. Usage: bash scripts/test_x5_alert.sh -> X5_ALERT_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); RUN=$T/runs/E16-X5; TASK=E16-X5; RUNBOOK=$T/rb; DRY=0; mkdir -p $RUN $T/rb/scripts
printf 'import sys, os; open(os.environ["CALLS"], "a").write(" ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
export CALLS=$T/calls; : > $CALLS
eval "$(awk '/^x5alert\(\) \{/,/^\}$/' $SRC/scripts/upload_sync.sh)"
ok "the function was found" 'declare -F x5alert > /dev/null'
line() { echo "{\"call\": $1, \"rows\": 128, \"masked\": {\"151644\": $2}$([ -n "${3:-}" ] && echo ", \"rows_special\": $3")}" >> $RUN/x5.jsonl; }
line 1 1 1; line 2 3 3; line 3 9 2; line 4 4    # 0.8%, 2.3%, 1.6% (9 ids in 2 rows), 3.1% (an older line: 4 ids)
x5alert
ok "steps over 2% are events (call 2: 3 of 128; call 4: 4 ids without rows_special); 1 and 3 are not" '[ $(wc -l < $CALLS) = 2 ] && grep -q "^event E16-X5 x5_special_rows call 2: 3 of 128 rows (2.3% > 2%)" $CALLS && grep -q "^event E16-X5 x5_special_rows call 4: 4 of 128 rows" $CALLS'
x5alert
ok "each step looked at once" '[ $(wc -l < $CALLS) = 2 ] && [ "$(cat $RUN/.x5_seen)" = 4 ]'
line 5 6 6; X5_SPECIAL_ALERT=0.05 x5alert
ok "X5_SPECIAL_ALERT from the environment (5%: 6 of 128 = 4.7% is no event)" '[ $(wc -l < $CALLS) = 2 ]'
printf '{"call": 6, "rows": 128, "masked": {"151644": 5}, "rows_s' >> $RUN/x5.jsonl; x5alert   # [v2.5.9z61] ops-lead's repro: a half-written line
ok "[v2.5.9z61] a line still being written is not passed over" '[ "$(cat $RUN/.x5_seen)" = 5 ] && [ $(wc -l < $CALLS) = 2 ]'
printf 'pecial": 5}\n' >> $RUN/x5.jsonl; x5alert
ok "[v2.5.9z61] once complete it is read: 5 of 128 rows (3.9%) is an event" '[ "$(cat $RUN/.x5_seen)" = 6 ] && [ $(wc -l < $CALLS) = 3 ] && grep -q "call 6: 5 of 128 rows" $CALLS'
line 7 9 9; DRY=1 x5alert
ok "[v2.5.9z61] DRY: nothing read, no event" '[ "$(cat $RUN/.x5_seen)" = 6 ] && [ $(wc -l < $CALLS) = 3 ]'
ok "upload_sync.sh calls it for E16 tasks only" 'grep -q "^  case \$TASK in E16-\*) x5alert;; esac" $SRC/scripts/upload_sync.sh'
rm -rf $T
[ $fails = 0 ] && echo "X5_ALERT_TEST PASS ($n checks)" || echo "X5_ALERT_TEST FAIL ($fails of $n)"
