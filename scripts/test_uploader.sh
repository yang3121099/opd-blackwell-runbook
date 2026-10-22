#!/usr/bin/env bash
# [v2.5.8] Stub tests of uploader.sh with the real control.py and a stub remote.py (no network). Usage: bash scripts/test_uploader.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/control $T/root/logs $T/remote $T/bin
cp $SRC/scripts/uploader.sh $SRC/scripts/control.py $SRC/scripts/control_rules.py $T/rb/scripts/
printf 'export OPD_ROOT=%s RUNBOOK=%s OPD_NODE=testnode REMOTE_PUT_TIMEOUT=2 UPLOADER_POLL=1 UPLOADER_BACKOFF=1 CONTROL_DIRECT=0 DIAG_TAILS_S=0\n' $T/root $T/rb > $T/rb/scripts/env.sh
cat > $T/rb/scripts/remote.py <<PY
import os, shutil, sys, time
src, dst = sys.argv[2], sys.argv[3]
if "hang" in dst: time.sleep(60)
d = os.path.join("$T/remote", dst); os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy(src, d)
PY
printf '#!/usr/bin/env bash\nexec python3 "$@"\n' > $T/bin/python; chmod +x $T/bin/python; export PATH=$T/bin:$PATH
(OPD_ROOT=$T/root RUNBOOK=$T/rb bash $T/rb/scripts/uploader.sh >> $T/root/logs/uploader.log 2>&1 < /dev/null &); sleep 2
q() { (cd $T/rb && OPD_ROOT=$T/root RUNBOOK=$T/rb OPD_NODE=testnode CONTROL_DIRECT=0 python3 scripts/control.py put_q "$1" "$2"); }   # [v2.5.9z53] the outbox's own path
echo s1 > $T/s; echo r > $T/r; q $T/s status/testnode.json; q $T/r status/testnode/replies/1.json; echo s2 > $T/s; q $T/s status/testnode.json
for _ in $(seq 10); do [ -f $T/remote/status/testnode.json ] && [ -f $T/remote/status/testnode/replies/1.json ] && break; sleep 1; done
ok "shipped, the status once with its newest content" '[ "$(cat $T/remote/status/testnode.json 2>/dev/null)" = s2 ] && [ -f $T/remote/status/testnode/replies/1.json ] && [ -z "$(ls $T/root/control/outbox)" ]'
echo h > $T/h; q $T/h status/testnode/hang.json; echo e > $T/e; q $T/e status/testnode/events/e.json; sleep 8
ok "a hung upload is cut (REMOTE_PUT_TIMEOUT) and stays queued; the loop goes on" 'grep -q "FAILED 1" $T/root/logs/uploader.log && ls $T/root/control/outbox | grep -q json && [ $(( $(date +%s) - $(cat $T/root/control/loop_uploader) )) -lt 10 ]'
(OPD_ROOT=$T/root RUNBOOK=$T/rb bash $T/rb/scripts/uploader.sh > $T/second.log 2>&1 < /dev/null)
ok "a second uploader exits at once" 'grep -q "already running" $T/second.log'
pkill -f "$T/rb/scripts/" 2> /dev/null; sleep 1; rm -rf $T   # only this test's processes (never a broad pattern)
[ $fails = 0 ] && echo "UPLOADER_TEST PASS ($n checks)" || echo "UPLOADER_TEST FAIL ($fails of $n)"
