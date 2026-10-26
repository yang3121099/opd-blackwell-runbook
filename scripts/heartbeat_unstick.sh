#!/usr/bin/env bash
# [v2.5.9z88f] HEARTBEAT-UNSTICK (Fable / ops-lead 10-26, after CORPUS-RECORD's runaway sandboxes hung the heartbeat): the CPU-lane
# form of lab handoff heartbeat_restart.sh, so nobody has to log in to the node. Once no sandbox process is left (DRYRUN_SB /
# DRYRUN_REAL_PY in a /proc/<pid>/environ) it removes control/watchdog_heartbeat.gave_up, so the watchdog restarts the heartbeat
# itself with the loop's own environment; with sandbox processes left it changes nothing (WAIT). Nothing is started or signalled
# here. One look per run (rerun HEARTBEAT-UNSTICK for the next); runs/HEARTBEAT-UNSTICK/report.json. PROC_ROOT: tests only.
set -uo pipefail
source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/HEARTBEAT-UNSTICK; mkdir -p $D
python3 - "$OPD_ROOT/control" "$D" "${PROC_ROOT:-/proc}" <<'PY'
import json, os, sys, time
cd, d, proc = sys.argv[1], sys.argv[2], sys.argv[3]
def report(res, why, **k):
    json.dump(dict(result=res, reason=why, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **k), open(os.path.join(d, "report.json"), "w"), indent=1)
    print("HEARTBEAT_UNSTICK", res, why)
if not os.path.isdir(proc):
    report("FAIL", "no /proc on this machine"); sys.exit(1)
left = 0
for p in os.listdir(proc):
    if p.isdigit():
        try:
            e = b"\0" + open(os.path.join(proc, p, "environ"), "rb").read()
            left += (b"\0DRYRUN_SB=" in e or b"\0DRYRUN_REAL_PY=" in e)
        except OSError:
            pass
g = os.path.join(cd, "watchdog_heartbeat.gave_up")
try:
    age = int(time.time() - int(open(os.path.join(cd, "loop_heartbeat")).read().strip() or 0))
except (OSError, ValueError):
    age = -1
had = os.path.exists(g)
if left:
    report("WAIT", f"{left} sandbox processes still running: the gave_up marker stays", sandbox_procs=left, heartbeat_age_s=age, gave_up=had)
elif had:
    os.remove(g)
    report("PASS", "no sandbox process left: the gave_up marker removed (the watchdog restarts the heartbeat)", sandbox_procs=0, heartbeat_age_s=age, gave_up=True)
else:
    report("PASS", "no sandbox process left and no gave_up marker: nothing to do", sandbox_procs=0, heartbeat_age_s=age, gave_up=False)
PY
rc=$?
touch $D/.done
exit $rc
