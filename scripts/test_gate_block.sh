#!/usr/bin/env bash
# [v2.5.9z3] Stub tests of run_tasks.sh's gate blocking and input waiting (no GPU, network or HF; one DRY=1 dispatch pass each):
#  - a gate (PROBE-*) whose report says FAIL holds its dependents: blocked (.blocked, event), not started, not skipped, attempts unchanged;
#    rerun PASS -> they start, .blocked removed; a gate without a report / without "result" counts by .done (older probes);
#  - launch(): a task refused by its own gate check is blocked (no .failed); one that says WAIT_INPUTS is waiting (no .failed, not
#    launched again for WAIT_RETRY_MIN); a plain failure still counts an attempt.
# Usage: bash scripts/test_gate_block.sh -> GATE_BLOCK_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/tasks $T/root/runs $T/root/logs $T/root/control $T/bin
  cp $SRC/scripts/run_tasks.sh $T/rb/scripts/
  printf 'export OPD_ROOT=%s/root RUNBOOK=%s/rb CDIR=%s/root/control NGPU=2\n' $T $T $T > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
  printf 'import sys; sys.exit(1)\n' > $T/rb/scripts/remote.py   # nothing done remotely
  printf '#!/usr/bin/env bash\ncase "$*" in *compute-apps*) ;; *query-gpu=index*) printf "0\\n1\\n2\\n3\\n";; *) echo 0;; esac\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  mkdir -p $T/root/venv/bin; : > $T/root/venv/bin/activate
  printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; chmod +x $T/bin/setsid
  # GNU stat -c %Y (done_ uses it on the node; the Mac's stat differs)
  printf '#!/usr/bin/env bash\n[ "$1" = -c ] && { python3 -c "import os, sys; print(int(os.path.getmtime(sys.argv[1])))" "$3"; exit; }\nexec /usr/bin/stat "$@"\n' > $T/bin/stat; chmod +x $T/bin/stat
  cat > $T/rb/tasks/queue.txt <<'EOF'
version: 2
g1       open   PROBE-X PROBE-OLD A:PROBE-X B C:PROBE-OLD
EOF
  for d in PROBE-X PROBE-OLD; do mkdir -p $T/root/runs/$d; touch $T/root/runs/$d/.done; done
  echo '{"result": "FAIL", "reason": "unit tests: 7 passed, 1 failed"}' > $T/root/runs/PROBE-X/report.json
  echo '{"models": {}}' > $T/root/runs/PROBE-OLD/report.json   # an older probe: no result field
  export CALLS=$T/calls PATH=$T/bin:$PATH; : > $CALLS
}
dry() { ( cd $T && DRY=1 QUEUE_FILE=$T/rb/tasks/queue.txt bash $T/rb/scripts/run_tasks.sh 2>&1 ) > $T/dry.out; }
# 1. a failed gate holds A (blocked), B and C (gate without a result field) start
setup; dry; sleep 1
ok "A held by PROBE-X FAIL: not started, .blocked names PROBE-X, events blocked + gate_failed" '! grep -q "WOULD_START .* A " $T/dry.out && grep -q "^blocked_by: PROBE-X$" $T/root/runs/A/.blocked && grep -q "^control event A blocked by PROBE-X FAIL" $CALLS && grep -q "^control event PROBE-X gate_failed unit tests: 7 passed, 1 failed" $CALLS'
ok "A neither skipped nor failed (attempts unchanged)" '[ ! -f $T/root/runs/A/.skipped ] && [ ! -f $T/root/runs/A/.failed ]'
ok "B (no deps) and C (its gate has no result field: by .done, as before) start" 'grep -q "WOULD_START .* B " $T/dry.out && grep -q "WOULD_START .* C " $T/dry.out'
# 2. a second pass: gate_failed is not raised again for the same report
: > $CALLS; dry; sleep 1
ok "the same gate report: no second gate_failed or blocked event" '! grep -q "gate_failed" $CALLS && ! grep -q "A blocked" $CALLS'
# 3. the gate rerun passes: A starts, .blocked removed, event unblocked
echo '{"result": "PASS"}' > $T/root/runs/PROBE-X/report.json; : > $CALLS; dry; sleep 1
ok "PROBE-X PASS: A starts, .blocked removed, event unblocked" 'grep -q "WOULD_START .* A " $T/dry.out && [ ! -f $T/root/runs/A/.blocked ] && grep -q "^control event A unblocked" $CALLS'
cd $SRC; rm -rf $T
# 4. launch(): its own gate check -> blocked; WAIT_INPUTS -> waiting; another failure -> an attempt
setup
cat > $T/rb/scripts/task.sh <<'SH'
case $2 in X-gate) echo "PROBE-Y has not passed: X-gate not started"; exit 3;; X-wait) echo "WAIT_INPUTS X-wait: Geometry3K not in (no .fetch_ok)"; exit 1;; *) echo "RuntimeError: boom"; exit 1;; esac
SH
for c in gate wait other; do
  ( OPD_ROOT=$T/root RUNBOOK=$T/rb; attempts() { echo 0; }; eval "$(sed -n '/^launch() {/,/^}/p' $SRC/scripts/run_tasks.sh)"; launch 1 X-$c )
done
for _ in $(seq 30); do [ -f $T/root/runs/X-other/.failed ] && [ -f $T/root/runs/X-gate/.blocked ] && [ -f $T/root/runs/X-wait/.waiting ] && break; sleep 0.2; done; sleep 0.5
ok "own gate check: .blocked by PROBE-Y, no .failed, event blocked" '[ "$(sed -n "s/^blocked_by: //p" $T/root/runs/X-gate/.blocked)" = PROBE-Y ] && [ ! -f $T/root/runs/X-gate/.failed ] && grep -q "^control event X-gate blocked by PROBE-Y (its own gate check)" $CALLS'
ok "WAIT_INPUTS: .waiting with the reason, no .failed, event waiting" 'grep -q "^WAIT_INPUTS X-wait: Geometry3K not in" $T/root/runs/X-wait/.waiting && [ ! -f $T/root/runs/X-wait/.failed ] && grep -q "^control event X-wait waiting Geometry3K not in" $CALLS'
ok "another failure still counts an attempt" 'grep -q "^attempts: 1$" $T/root/runs/X-other/.failed && [ ! -f $T/root/runs/X-other/.blocked ]'
# 4b. (ops-lead's A1 condition) an old attempt's 'has not passed' / WAIT_INPUTS line in the log, this attempt a plain failure:
#     failed, attempts + 1 (launch truncates the log per attempt)
printf 'PROBE-Y has not passed: X-old not started\nWAIT_INPUTS X-old: x not in\n' > $T/root/logs/X-old.log
( OPD_ROOT=$T/root RUNBOOK=$T/rb; attempts() { echo 0; }; eval "$(sed -n '/^launch() {/,/^}/p' $SRC/scripts/run_tasks.sh)"; launch 1 X-old )
for _ in $(seq 30); do [ -f $T/root/runs/X-old/.failed ] && break; sleep 0.2; done; sleep 0.3
ok "[A1] an old attempt's gate / wait lines + a plain failure now: failed, attempts 1, not blocked or waiting" 'grep -q "^attempts: 1$" $T/root/runs/X-old/.failed && [ ! -f $T/root/runs/X-old/.blocked ] && [ ! -f $T/root/runs/X-old/.waiting ] && ! grep -q "has not passed" $T/root/logs/X-old.log'
# 5. the dispatcher keeps them: X-gate until PROBE-Y passes, X-wait for WAIT_RETRY_MIN
printf 'version: 2\ng1       open   X-gate X-wait\n' > $T/rb/tasks/queue.txt; dry
ok "X-gate (PROBE-Y without a report) and X-wait (just waiting) are not started" '! grep -q "WOULD_START .* X-gate " $T/dry.out && ! grep -q "WOULD_START .* X-wait " $T/dry.out'
mkdir -p $T/root/runs/PROBE-Y; echo '{"result": "PASS"}' > $T/root/runs/PROBE-Y/report.json; touch -t 202001010000 $T/root/runs/X-wait/.waiting; dry
ok "PROBE-Y PASS and the wait over: both start" 'grep -q "WOULD_START .* X-gate " $T/dry.out && grep -q "WOULD_START .* X-wait " $T/dry.out'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "GATE_BLOCK_TEST PASS ($n checks)" || echo "GATE_BLOCK_TEST FAIL ($fails of $n)"
