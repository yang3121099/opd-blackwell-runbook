#!/usr/bin/env bash
# [v2.5.9z52] Stub test of upload_sync.sh's per-task throttle (no HF, no torch): its due / upn / flush_due functions taken
# from the script and run against a stub `up`: the files due go up in one commit, at most once per RUNS_PUT_S (control.env
# first), the list is kept when the commit fails, and nothing is lost.
# Usage: bash scripts/test_upload_throttle.sh -> UPLOAD_THROTTLE_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); export OPD_ROOT=$T/root; RUN=$T/root/runs/E10b-X; TASK=E10b-X; DRY=0; mkdir -p $RUN
CALLS=$T/calls; : > $CALLS; FAIL_UP=0
up() { echo "up $(basename $1) $2 $(ls $1 | tr '\n' ' ')" >> $CALLS; [ $FAIL_UP = 0 ]; }
eval "$(awk '/^UPT=\$RUN\/\.last_commit/,/^  upn \$fl /' $SRC/scripts/upload_sync.sh)"
ok "the functions were found" 'declare -F due > /dev/null && declare -F upn > /dev/null && declare -F flush_due > /dev/null'
for f in train.log config.yaml wall.json len_by_step.csv; do echo $f > $RUN/$f; done
printf 'train.log\nconfig.yaml\nwall.json\n' >> $RUN/.up_due; echo wall.json >> $RUN/.up_due
flush_due > /dev/null
ok "the due files in one commit (each once)" '[ "$(cat $CALLS)" = "up .up_batch runs/E10b-X config.yaml train.log wall.json " ] && [ ! -e $RUN/.up_due ] && [ -f $RUN/.last_commit ]'
echo len_by_step.csv >> $RUN/.up_due; flush_due > /dev/null
ok "within RUNS_PUT_S (600 s): it waits, the list kept" '[ $(wc -l < $CALLS) = 1 ] && [ "$(cat $RUN/.up_due)" = len_by_step.csv ]'
touch -t $(date -v-11M +%Y%m%d%H%M 2> /dev/null || date -d "-11 min" +%Y%m%d%H%M) $RUN/.last_commit; FAIL_UP=1; flush_due > /dev/null
ok "a failed commit keeps the list" '[ $(wc -l < $CALLS) = 2 ] && [ "$(cat $RUN/.up_due)" = len_by_step.csv ]'
FAIL_UP=0; flush_due > /dev/null
ok "the next round sends it" '[ "$(tail -1 $CALLS)" = "up .up_batch runs/E10b-X len_by_step.csv " ] && [ ! -e $RUN/.up_due ]'
echo wall.json >> $RUN/.up_due; mkdir -p $T/root; echo "export RUNS_PUT_S=0" > $T/root/control.env; flush_due > /dev/null
ok "RUNS_PUT_S from control.env (0: no wait)" '[ "$(tail -1 $CALLS)" = "up .up_batch runs/E10b-X wall.json " ]'
upn train.log nothing.csv config.yaml
ok "upn: missing files left out, one commit" '[ "$(tail -1 $CALLS)" = "up .up_batch runs/E10b-X config.yaml train.log " ]'
rm -rf $T
[ $fails = 0 ] && echo "UPLOAD_THROTTLE_TEST PASS ($n checks)" || echo "UPLOAD_THROTTLE_TEST FAIL ($fails of $n)"
