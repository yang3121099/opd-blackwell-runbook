#!/usr/bin/env bash
# Every 10 minutes: write status/<hostname>.json (open stage, each task's state and current step, latest validation,
# truncation rate, GPU utilisation and memory, free disk, UTC timestamp) and upload it to the result repository.
# The project side only reads status/; it never logs in to this machine. Also uploads the first-arm comparison
# (status/compare_<hostname>.json) once, when E10b-P2 has reached step 10.
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
H=$(hostname -s); F=$OPD_ROOT/logs/status_$H.json
while true; do
  python - "$F" <<'PY'
import glob, json, os, re, subprocess, sys, time
root = os.environ["OPD_ROOT"]; runbook = os.environ["RUNBOOK"]
def last(pat, text):
    m = re.findall(pat, text); return float(m[-1]) if m else None
tasks = {}
for d in sorted(glob.glob(os.path.join(root, "runs", "E*"))):
    log = os.path.join(d, "train.log"); text = open(log, errors="ignore").read()[-400000:] if os.path.exists(log) else ""
    steps = re.findall(r"(?:^|[ )])step:(\d+) - ", text)
    tasks[os.path.basename(d)] = {
        "state": "done" if os.path.exists(os.path.join(d, ".done")) else ("running" if text else "queued"),
        "step": int(steps[-1]) if steps else 0,
        "val_math500_mean@2": last(r"val-core/math500/reward/mean@2:([0-9.]+)", text),
        "truncation": last(r"response_length/clip_ratio:([0-9.]+)", text),
        "step_time_s": last(r"timing_s/step:([0-9.]+)", text)}
gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True).stdout.strip().splitlines()
st = os.statvfs(root)
queue = open(os.path.join(runbook, "tasks", "queue.txt")).read() if os.path.exists(os.path.join(runbook, "tasks", "queue.txt")) else ""
ver = re.search(r"^version:\s*(\S+)", queue, re.M); opened = re.search(r"^open:\s*(\S+)", queue, re.M)
json.dump({"host": os.uname().nodename, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "queue_version": ver.group(1) if ver else None, "open_stage": opened.group(1) if opened else None,
           "tasks": tasks, "gpus": gpu, "disk_free_gb": round(st.f_bavail * st.f_frsize / 1e9, 1)}, open(sys.argv[1], "w"), indent=1)
PY
  python $RUNBOOK/scripts/remote.py put $F status/$H.json 2>> $OPD_ROOT/logs/heartbeat.err && echo "HEARTBEAT $(date -u +%T)"
  # first-arm comparison with the reference H100 run, uploaded once when E10b-P2 has reached step 10
  if [ ! -f $OPD_ROOT/logs/.compare_uploaded ] && python $RUNBOOK/scripts/compare_first_arm.py >/dev/null 2>&1; then
    python $RUNBOOK/scripts/remote.py put $OPD_ROOT/logs/compare_$H.json status/compare_$H.json && touch $OPD_ROOT/logs/.compare_uploaded && echo "COMPARE_UPLOADED $(date -u +%T)"
  fi
  sleep 600
done
