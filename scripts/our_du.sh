#!/usr/bin/env bash
# [v2.5.9c] (reviewer 10-21; Runming: the work disk's space on the monitor page, with room for a disaster-recovery cache)
# Our own disk usage, measured off the heartbeat's path: du of $OPD_ROOT (our tree on the shared work disk; checkpoint
# directories there are links to $CKPT_ROOT and are not followed) and of $CKPT_ROOT, each niced and bounded by
# OUR_DU_TIMEOUT (900 s); written to control/our_du.json {utc, work_gb, ckpt_gb, secs, timed_out}, which the status and the
# pulse read. heartbeat.sh starts it when that file is older than OUR_DU_MIN (30) minutes and no copy runs.
set -uo pipefail
source "$(dirname "$0")/env.sh"
TO=""; command -v timeout > /dev/null && TO="timeout -k 30 ${OUR_DU_TIMEOUT:-900}"
t0=$(date +%s)
w=$(nice -n 19 $TO du -sk $OPD_ROOT 2> /dev/null | tail -1 | cut -f1); rw=$?
c=""; [ -d "${CKPT_ROOT:-}" ] && [ "$CKPT_ROOT" != "$OPD_ROOT" ] && c=$(nice -n 19 $TO du -sk $CKPT_ROOT 2> /dev/null | tail -1 | cut -f1)
mkdir -p $OPD_ROOT/control
python3 - "$OPD_ROOT/control/our_du.json" "${w:-}" "${c:-}" "$(( $(date +%s) - t0 ))" <<'PY'
import json, os, sys, time
out, w, c, secs = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
gb = lambda k: round(int(k) * 1024 / 1e9, 1) if k.isdigit() else None
d = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "work_gb": gb(w), "ckpt_gb": gb(c), "secs": secs,
     "timed_out": not w.isdigit()}
json.dump(d, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
PY
