#!/usr/bin/env bash
# [v2.5.9z55] Stub test of heartbeat.sh's status builder block (the lines from "[z55 builder begin]" to the minimal status): a
# builder that succeeds replaces status_<host>.json; one that fails or writes broken JSON removes the previous (stale but valid)
# file, so the minimal status naming the failure goes up instead, and its stderr is kept in logs/status_builder.err.
# Usage: bash scripts/test_status_builder.sh -> STATUS_BUILDER_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); export OPD_ROOT=$T/root CKPT_ROOT=$T/root RUNBOOK=$SRC; mkdir -p $OPD_ROOT/logs
H=zz55test$$; F=$OPD_ROOT/logs/status_$H.json
BLOCK=$(awk '/\[z55 builder begin\]/,/^  N=\$\(python/' $SRC/scripts/heartbeat.sh | grep -v '^  N=\$(python')
python() {   # the builder (python - <file>, its code on stdin) faked by FAKE; every other python call is the real one
  if [ "$1" = - ] && [ $# = 2 ]; then cat > /dev/null
    case $FAKE in ok) echo '{"utc": "new"}' > $2;; fail) echo "NameError: name 'x' is not defined" >&2; return 1;; badjson) echo '{' > $2;; esac
  else python3 "$@"; fi; }
run() { FAKE=$1; echo '{"utc": "stale 11:17"}' > $F; eval "$BLOCK"; }
run ok
ok "a builder that succeeds replaces the status file" '[ "$UP" = "$F" ] && grep -q "\"new\"" $F && [ ! -e $F.tmp ] && [ -z "$BERR" ]'
run fail
ok "a failing builder: the stale file goes, the minimal status names it, stderr kept" '[ ! -e $F ] && [ "$UP" != "$F" ] && grep -q "\"error\": \"status builder failed rc=1\"" $UP && python3 -c "import json, sys; json.load(open(sys.argv[1]))" $UP && grep -q "^rc=1$" $OPD_ROOT/logs/status_builder.err && grep -q "NameError" $OPD_ROOT/logs/status_builder.err'
ok "[v2.5.9z56] the builder's last stderr lines inside the minimal status" 'python3 -c "import json, sys; d = json.load(open(sys.argv[1])); assert d[\"builder_error\"][-1].startswith(\"NameError\") and d[\"disk_free_gb\"] > 0, d" $UP'
rm -f $UP
run badjson
ok "broken JSON (rc 0): the same" '[ ! -e $F ] && grep -q "status builder failed rc=0" $UP && grep -q "^rc=0$" $OPD_ROOT/logs/status_builder.err'
rm -f $UP
ok "the diag tails carry status_builder.err" 'grep -q "\"status_builder.err\"" $SRC/scripts/control.py'
rm -rf $T
[ $fails = 0 ] && echo "STATUS_BUILDER_TEST PASS ($n checks)" || echo "STATUS_BUILDER_TEST FAIL ($fails of $n)"
