#!/usr/bin/env bash
# Every 10 minutes: write status/<hostname>.json (open stage, each task's state and current step, latest validation,
# truncation rate, GPU utilisation and memory, free disk, UTC timestamp) and upload it to the result repository.
# The project side only reads status/; it never logs in to this machine. Also uploads the first-arm comparison
# (status/compare_<hostname>.json) once, when E10b-P2 has reached step 10.
# v2.4 (ops-1): per GPU index/name/util/memory as fields; per task total steps, mean step time of the last 5 steps,
# ETA and the last 20 lines of logs/<task>.log; the control plane (node, applied_seq, recent results, control.env,
# last git pull, control_env_suspect); runbook commit and run_tasks sha. Log text is redacted (control.redact).
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
H=$(hostname -s); F=$OPD_ROOT/logs/status_$H.json
# software stack, read once: tells the project which install route this node is on
python -c "import json, torch, vllm; json.dump({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'vllm': vllm.__version__, 'arch_list': torch.cuda.get_arch_list(), 'device_capability': list(torch.cuda.get_device_capability())}, open('$OPD_ROOT/logs/stack.json', 'w'))" 2>/dev/null || echo '{}' > $OPD_ROOT/logs/stack.json
while true; do
  python - "$F" <<'PY'
import glob, json, os, re, subprocess, sys, time
root = os.environ["OPD_ROOT"]; runbook = os.environ["RUNBOOK"]
sys.path.insert(0, os.path.join(runbook, "scripts"))
from control import redact, status as control_status
def total_steps(d, text):
    cfg = open(os.path.join(d, "config.yaml"), errors="ignore").read() if os.path.exists(os.path.join(d, "config.yaml")) else ""
    m = re.search(r"^steps:\s*(\d+)", cfg, re.M) or re.search(r"total_training_steps[=:]\s*(\d+)", text)
    return int(m.group(1)) if m else None
def tail(path, n=20):
    try:
        return [redact(x) for x in open(path, errors="ignore").read()[-20000:].splitlines()[-n:]]
    except OSError:
        return []
def last(pat, text):
    m = re.findall(pat, text); return float(m[-1]) if m else None
tasks = {}
for d in sorted(x for x in glob.glob(os.path.join(root, "runs", "*")) if os.path.isdir(x) and not os.path.basename(x).startswith("smoke")):
    log = os.path.join(d, "train.log"); text = open(log, errors="ignore").read()[-400000:] if os.path.exists(log) else ""
    steps = re.findall(r"(?:^|[ )])step:(\d+) - ", text)
    failed = open(os.path.join(d, ".failed")).read() if os.path.exists(os.path.join(d, ".failed")) else ""
    att = re.search(r"attempts: (\d+)", failed)
    tasks[os.path.basename(d)] = {
        "state": "done" if os.path.exists(os.path.join(d, ".done")) else ("skipped" if os.path.exists(os.path.join(d, ".skipped")) else
                 ("failed" if failed and int(att.group(1)) >= 3 else ("retrying" if failed else ("running" if text else "queued")))),
        "attempts_failed": int(att.group(1)) if att else 0, "last_failure_tail": failed.splitlines()[-3:] if failed else [],
        "step": int(steps[-1]) if steps else 0,
        "val_math500": last(r"val-core/math500/reward/mean@\d+:([0-9.]+)", text),   # mean@2 (E10b) or mean@4 (E12b); v2.3
        "truncation": last(r"response_length/clip_ratio:([0-9.]+)", text),
        "step_time_s": last(r"timing_s/step:([0-9.]+)", text)}
    t = tasks[os.path.basename(d)]; times = [float(x) for x in re.findall(r"timing_s/step:([0-9.]+)", text)[-5:]]
    t["total_steps"] = total_steps(d, text); t["step_time_mean5_s"] = round(sum(times) / len(times), 1) if times else None
    t["eta_min"] = round((t["total_steps"] - t["step"]) * t["step_time_mean5_s"] / 60) if t["total_steps"] and times and t["state"] == "running" else None
    t["log_tail"] = tail(os.path.join(root, "logs", os.path.basename(d) + ".log"))
gpu = []
for line in subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip().splitlines():
    f = [x.strip() for x in line.split(",")]
    if len(f) == 5:
        gpu.append({"index": int(f[0]), "name": f[1], "util_pct": float(f[2]) if f[2].replace(".", "").isdigit() else None,
                    "mem_used_mib": float(f[3]), "mem_total_mib": float(f[4])})
git = lambda *a: subprocess.run(["git", "-C", runbook, *a], capture_output=True, text=True).stdout.strip()
st = os.statvfs(root)
queue = open(os.path.join(runbook, "tasks", "queue.txt")).read() if os.path.exists(os.path.join(runbook, "tasks", "queue.txt")) else ""
ver = re.search(r"^version:\s*(\S+)", queue, re.M)
stages = {m.group(1): m.group(2) for m in re.finditer(r"^(\S+)\s+(open|auto|locked)\s", queue, re.M)}   # v2.3: stage -> status
json.dump({"host": os.uname().nodename, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "queue_version": ver.group(1) if ver else None, "stages": stages,
           "stack": json.load(open(os.path.join(root, "logs", "stack.json"))) if os.path.exists(os.path.join(root, "logs", "stack.json")) else {},
           "tasks": tasks, "gpus": gpu, "disk_free_gb": round(st.f_bavail * st.f_frsize / 1e9, 1),
           # v2.4: alert below DISK_ALERT_GB (30): one 1.7B save needs ~11 GB for a moment (fp32 HF + bf16 copy), a SPOT
           # resume save ~28 GB
           "disk_alert": st.f_bavail * st.f_frsize / 1e9 < float(os.environ.get("DISK_ALERT_GB", "30")),
           # v2.4: the checkpoint disk ($CKPT_ROOT, e.g. /home on the B300 node) has its own reading and alert
           "ckpt_root": os.environ.get("CKPT_ROOT"),
           "ckpt_disk_free_gb": (lambda c: round(c.f_bavail * c.f_frsize / 1e9, 1))(os.statvfs(os.environ["CKPT_ROOT"])) if os.path.isdir(os.environ.get("CKPT_ROOT", "")) else None,
           "ckpt_disk_alert": (lambda c: c.f_bavail * c.f_frsize / 1e9 < float(os.environ.get("CKPT_ALERT_GB", "100")))(os.statvfs(os.environ["CKPT_ROOT"])) if os.path.isdir(os.environ.get("CKPT_ROOT", "")) else None,
           "disk_used_pct": round(100 * (1 - st.f_bavail / st.f_blocks), 1) if st.f_blocks else None,
           "runbook_head": git("rev-parse", "--short", "HEAD"), "runbook_version": "v2.4",
           "run_tasks_sha": open(os.path.join(root, "control", "run_tasks.sha")).read().strip() if os.path.exists(os.path.join(root, "control", "run_tasks.sha")) else None,
           "control": control_status()}, open(sys.argv[1], "w"), indent=1)
PY
  python $RUNBOOK/scripts/remote.py put $F status/$H.json 2>> $OPD_ROOT/logs/heartbeat.err && echo "HEARTBEAT $(date -u +%T)"
  # first-arm comparison with the reference H100 run, uploaded once when E10b-P2 has reached step 10
  if [ ! -f $OPD_ROOT/logs/.compare_uploaded ] && python $RUNBOOK/scripts/compare_first_arm.py >/dev/null 2>&1; then
    python $RUNBOOK/scripts/remote.py put $OPD_ROOT/logs/compare_$H.json status/compare_$H.json && touch $OPD_ROOT/logs/.compare_uploaded && echo "COMPARE_UPLOADED $(date -u +%T)"
  fi
  sleep 600
done
