# Throttle test for run_tasks.sh v2.4 done_(): remote asked once, then not again within REMOTE_EVERY; a remote hit makes a
# local .done. Runs in a temp dir with a stub remote.py (no network, no GPU, kills nothing).
set -u; T=$(mktemp -d); mkdir -p $T/runbook/scripts $T/root/runs
cat > $T/runbook/scripts/remote.py <<'PY'
import sys, os
open(os.environ["CALLS"], "a").write(sys.argv[2] + "\n")
sys.exit(0 if sys.argv[2] == "taskB" else 1)
PY
mkdir -p $T/bin; printf "#!/bin/sh\nexec python3 \"\$@\"\n" > $T/bin/python; chmod +x $T/bin/python; export PATH=$T/bin:$PATH
export OPD_ROOT=$T/root RUNBOOK=$T/runbook CDIR=$T/root/control CALLS=$T/calls; touch $CALLS
eval "$(sed -n '/^done_() {/,/; }$/p' "$1")"
done_ taskA; done_ taskA; done_ taskA          # remote asked once
done_ taskB                                     # remote hit -> local .done
REMOTE_EVERY=0 done_ taskA                      # throttle window 0 -> asked again
a=$(grep -c '^taskA$' $CALLS); b=$(grep -c '^taskB$' $CALLS)
[ "$a" = 2 ] && [ "$b" = 1 ] && [ -f $T/root/runs/taskB/.done ] && [ ! -d $T/root/runs/taskA ] && echo "THROTTLE_TEST PASS (taskA asked $a, taskB $b, no runs/taskA dir)" || echo "THROTTLE_TEST FAIL a=$a b=$b"
rm -rf $T
