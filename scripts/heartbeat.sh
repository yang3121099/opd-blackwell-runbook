#!/usr/bin/env bash
# v2.5.1: status/<node id>.json in the private repository (remote.py splits by destination once hf_repos.txt exists),
# every HEARTBEAT_SEC seconds (set HEARTBEAT_SEC=<60-3600>, default 600), and every 60 s a path-free pulse/<node id>.json
# in the public repository (control.py pulse: numbers, task names and states only, refused if any string fails the leak check).
# Every 10 minutes: write status/<hostname>.json (open stage, each task's state and current step, latest validation,
# truncation rate, GPU utilisation and memory, free disk, UTC timestamp) and upload it to the result repository.
# The project side only reads status/; it never logs in to this machine. Also uploads the first-arm comparison
# (status/compare_<hostname>.json) once, when E10b-P2 has reached step 10.
# v2.4 (ops-1): per GPU index/name/util/memory as fields; per task total steps, mean step time of the last 5 steps,
# ETA and the last 20 lines of logs/<task>.log; the control plane (node, applied_seq, recent results, control.env,
# last git pull, control_env_suspect); runbook commit and run_tasks sha. Log text is redacted (control.redact).
set -uo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/dr.sh"; source $OPD_ROOT/venv/bin/activate
H=$(hostname -s); F=$OPD_ROOT/logs/status_$H.json
# v2.5.7za (reviewer A1 12:3x UTC): every upload of this loop is bounded. On 2026-10-19 one HF upload hung at 12:16 UTC and
# froze the loop: no status, pulse, events or receipts, while run_tasks kept working. A timeout (HB_PUT_TIMEOUT, 180 s) is
# logged, sent as an event (itself bounded) and the loop goes on.
# [v2.5.8] -k 30: an upload that ignores TERM is KILLed 30 s later (drill B, 10-19 14:52 UTC: TERM alone left the loop
# hung until the watchdog); the timeout's exit is then 137 instead of 124
TO=""; command -v timeout > /dev/null && TO="timeout -k 30 ${HB_PUT_TIMEOUT:-180}"
hb_alive() { { date -u +%s > $OPD_ROOT/control/loop_heartbeat; } 2> /dev/null; }   # v2.5.7zb: liveness for the watchdog (local); [v2.5.8] its redirect error silenced too
hb_to() {
  $TO "$@"; local rc=$?; hb_alive
  if [ -n "$TO" ] && { [ $rc = 124 ] || [ $rc = 137 ]; }; then   # [v2.5.8] 137: KILLed after -k
    echo "UPLOAD_TIMEOUT $(date -u +%T) $(basename $2) $3 rc=$rc" >> $OPD_ROOT/logs/heartbeat.err
    $TO python $RUNBOOK/scripts/control.py event node heartbeat_upload_timeout "$(basename $2) $3" > /dev/null 2>&1
  fi
  return $rc; }
# software stack, read once: tells the project which install route this node is on
python -c "import json, torch, vllm; json.dump({'torch': torch.__version__, 'torch_cuda': torch.version.cuda, 'vllm': vllm.__version__, 'arch_list': torch.cuda.get_arch_list(), 'device_capability': list(torch.cuda.get_device_capability())}, open('$OPD_ROOT/logs/stack.json', 'w'))" 2>/dev/null || echo '{}' > $OPD_ROOT/logs/stack.json
# v2.5: run_tasks.sh restarts this loop when heartbeat.sh changes (it compares this sha with the current file)
mkdir -p $OPD_ROOT/control; sha1sum $RUNBOOK/scripts/heartbeat.sh | cut -c1-12 > $OPD_ROOT/control/heartbeat.sha; echo $$ > $OPD_ROOT/control/heartbeat.pid
# [v2.5.9c] our own usage (du) off the loop's path: our_du.sh in the background when control/our_du.json is older than
# OUR_DU_MIN (30) minutes and no copy runs
our_du_due() { [ -z "$(find $OPD_ROOT/control/our_du.json -mmin -${OUR_DU_MIN:-30} 2> /dev/null)" ] && ! pgrep -f "scripts/our_du\.s[h]" > /dev/null; }
while true; do
  our_du_due && setsid nohup bash $RUNBOOK/scripts/our_du.sh >> $OPD_ROOT/logs/our_du.log 2>&1 < /dev/null &
  hb_alive
  # [v2.5.9z55] (10-22: the private status stayed at 11:17:14 UTC for hours while the loop went round: a builder that fails left
  # the previous, still valid status_<host>.json in place, and its identical upload made no commit) the builder writes $F.tmp,
  # which replaces $F only when it is valid JSON; otherwise $F goes, the minimal status below names the failure, and the
  # builder's stderr is kept in logs/status_builder.err (diag tails, the GitHub heartbeat)
  # [z55 builder begin]
  rm -f $F.tmp; BERR=""
  python - "$F.tmp" 2> $OPD_ROOT/logs/status_builder.err.new <<'PY'
import glob, json, os, re, subprocess, sys, time
root = os.environ["OPD_ROOT"]; runbook = os.environ["RUNBOOK"]
sys.path.insert(0, os.path.join(runbook, "scripts"))
from control import redact, status as control_status, roots_report, dr_report
def jload(p, d=None):   # [v2.5.9z58] a torn or missing JSON input skips that field, never the whole status
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return d
def total_steps(d, text):
    cfg = open(os.path.join(d, "config.yaml"), errors="ignore").read() if os.path.exists(os.path.join(d, "config.yaml")) else ""
    m = re.search(r"^steps:\s*(\d+)", cfg, re.M) or re.search(r"total_training_steps[=:]\s*(\d+)", text)
    return int(m.group(1)) if m else None
def gpu_arg(s):   # [v2.5.9a] task.sh's GPU argument: '3' -> (3, [3]), '1,2' (a #gpus task) -> (1, [1, 2]), 'cpu' -> ('cpu', [])
    if s == "cpu":
        return ("cpu", [])
    if re.fullmatch(r"\d+(,\d+)*", s):
        g = [int(x) for x in s.split(",")]
        return (g[0], g)
    return None
def tail(path, n=20):
    try:
        return [redact(x) for x in open(path, errors="ignore").read()[-20000:].splitlines()[-n:]]
    except OSError:
        return []
def _hms(x):   # "12:34" / "1:02:03" -> seconds; "?" or empty -> None
    try:
        v = 0
        for part in x.split(":"):
            v = v * 60 + int(part)
        return v
    except (ValueError, AttributeError):
        return None
def progress(name, lines, queue):
    """[v2.5.9s] (ops-lead / Runming 10-21: the monitor page shows how far an evaluation is) a running evaluation's or probe's
    progress from its log tail (the 20 lines tail() reads, no further reading): the latest vLLM tqdm line (done, total,
    unit, rate per second, eta_s, the line itself) and, with several #eval rows, which one runs (part "k/n <tag>"); None
    when there is nothing to show."""
    p = {}
    for l in reversed(lines):
        l2 = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", l).strip()
        m = re.search(r"(\d+)/(\d+) \[([\d:]+)<([\d:?]*),\s*([0-9.]+)\s*(it/s|s/it)", l2)
        if m:
            done, total, r = int(m.group(1)), int(m.group(2)), float(m.group(5))
            rate = r if m.group(6) == "it/s" else (1 / r if r else 0.0)
            eta = _hms(m.group(4))
            if eta is None and rate:
                eta = (total - done) / rate
            p.update(done=done, total=total, unit="prompts" if "prompt" in l2.lower() else "it", rate=round(rate, 3),
                     eta_s=round(eta) if eta is not None else None, tqdm=l2[-200:])
            break
    tags = re.findall(r"^#eval " + re.escape(name) + r" (\S+)", queue, re.M)
    cur = next((x.split()[2] for x in reversed(lines) if x.startswith("EVAL_START ") and len(x.split()) > 2), None)
    if cur and cur in tags:
        p["part"] = f"{tags.index(cur) + 1}/{len(tags)} {cur}"
    if p:
        p["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return p or None
def last(pat, text):
    m = re.findall(pat, text); return float(m[-1]) if m else None
def live_tasks():
    """v2.5.5: {task: (gpu, start epoch)} of the running task.sh <gpu> <task> / run_adhoc_eval.sh <gpu> <seq> processes."""
    out = {}
    try:
        btime = int(next(l.split()[1] for l in open("/proc/stat") if l.startswith("btime ")))
        hz = os.sysconf("SC_CLK_TCK")
    except (OSError, StopIteration, ValueError):
        return out
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            a = [x.decode(errors="ignore") for x in open(f"/proc/{p}/cmdline", "rb").read().split(b"\0") if x]
            start = btime + int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[19]) / hz
        except (OSError, IndexError, ValueError):
            continue
        for i, x in enumerate(a[:-2]):
            if x.endswith(("/scripts/task.sh", "/scripts/run_adhoc_eval.sh")) and gpu_arg(a[i + 1]) is not None:
                name = a[i + 2] if x.endswith("task.sh") else f"adhoc_eval_{a[i + 2]}"
                if name not in out or start < out[name][1]:
                    g0, gl = gpu_arg(a[i + 1]); out[name] = (g0, start, gl)   # [v2.5.8] the CPU lane; [v2.5.9a] all its GPUs
    return out
def kind(name):
    return ("train" if re.match(r"E1(0[bcde]|2[bcd]|6)-", name) else "sft" if name.startswith("SFT-") else
            "eval" if re.match(r"E11b-|EVAL-|adhoc_eval_|E0[1-5]-", name) else "probe" if name in ("FA-PROBE", "SPEED-PROBE", "PROBE-4B", "PROBE-TERM", "PROBE-TERM2", "PROBE-TERM3", "PROBE-TERM4", "PROBE-TERM4-4B", "PROBE-TERM4-06B", "PROBE-TOP10", "STRICT-C1", "PROBE-T14", "PROBE-P5THINK", "PROBE-P5SELF", "PROBE-P5SELF-4B", "PROBE-P5SELF-06B", "PROBE-P2SELF", "PROBE-E12C", "PROBE-E12B-KL", "PROBE-E12B-STOP", "PROBE-E12B-STOP2", "PROBE-E12B-8K", "PROBE-E12B-16K", "PROBE-E12B-R3", "PROBE-X2507-X1", "PROBE-X2507-X2", "PROBE-X2507-X4", "PROBE-LENSEL", "PROBE-2GPU", "PROBE-TOOL", "PROBE-NET", "PROBE-NET2", "BUILD-VENV-VL", "PROBE-VLLM-VL", "READINGS-EXTRA", "PROBE-VLM", "PROBE-ROLLOUT-U60", "PROBE-ROLLOUT-U75", "PROBE-ROLLOUT-U85", "PROBE-STOPMERGE", "PROBE-WDIFF", "PROBE-TAIL", "PROBE-TAIL-BUD", "PROBE-DIFF17", "PROBE-DIFF17-4B", "PROBE-DIFFPROBE", "PROBE-PSTAR", "PROBE-DENSE", "PROBE-SELFROUTE", "PROBE-ROUTE", "PROBE-POLISH") or name.startswith("FA-SPEED-") else "other")
def results_pending():   # [v2.5.9z65] lines of weights_pending.txt whose destination is a result (runs/<task>[/...], not a weight)
    try:
        return sum(1 for l in open(os.path.join(root, "control", "weights_pending.txt")) if len(l.split("\t")) == 4
                   and l.split("\t")[1].startswith("runs/") and "/actor/huggingface" not in l.split("\t")[1])
    except OSError:
        return 0
def readings_held():   # [v2.5.9z69] control/readings_held.json from run_tasks.sh's last pass (fresh within 15 min): the held readings' count
    try:
        r = json.load(open(os.path.join(root, "control", "readings_held.json")))
        if not r.get("low") or time.time() - os.path.getmtime(os.path.join(root, "control", "readings_held.json")) > 900:
            return 0
        return len(r.get("held") or [])
    except (OSError, ValueError):
        return 0
live = live_tasks()
tasks = {}; stale = []
queue = open(os.path.join(runbook, "tasks", "queue.txt")).read() if os.path.exists(os.path.join(runbook, "tasks", "queue.txt")) else ""
queue_defs = "\n".join(l for l in queue.splitlines() if not l.lstrip().startswith("#"))   # [v2.5.9o] stage lines only: a name in a comment is no entry
def in_queue(name):   # [v2.5.9m] "name:dep" entries count too (a task waiting on a dependency was missed and called stale)
    return re.search(r"(^|\s)" + re.escape(name) + r"(\s|$|:)", queue_defs, re.M) is not None
pad_stages = {s for m in re.finditer(r"^#pad\s+(.*)$", queue, re.M) for s in m.group(1).split()}   # [v2.5.9m] padding stages
_gr = re.findall(r"^export GPU_RELEASE=([0-9]+)$", open(os.path.join(root, "control.env")).read(), re.M) if os.path.exists(os.path.join(root, "control.env")) else []
gpu_release_on = (_gr[-1] if _gr else "1") == "1"   # [v2.5.9z65a] setenv GPU_RELEASE=0: released GPUs are held as before
pad_tasks = {t.split(":")[0] for m in re.finditer(r"^(\S+)\s+(?:open|auto|locked)\s+(.*)$", queue, re.M) if m.group(1) in pad_stages for t in m.group(2).split()}
for d in sorted(x for x in glob.glob(os.path.join(root, "runs", "*")) if os.path.isdir(x) and not os.path.basename(x).startswith("smoke")):
    n = os.path.basename(d)   # [v2.5.8] a directory of a task no longer in the queue, not running and not done: counted, not listed
    if queue and not in_queue(n) and n not in live and not n.startswith("adhoc_eval_") and not os.path.exists(os.path.join(d, ".done")):
        stale.append(n); continue
    log = os.path.join(d, "train.log"); text = open(log, errors="ignore").read()[-400000:] if os.path.exists(log) else ""
    steps = re.findall(r"(?:^|[ )\]])step:(\d+) - ", text)   # v2.5.6d: verl's SFT trainer prints "...s/it]step:N - "
    failed = open(os.path.join(d, ".failed")).read() if os.path.exists(os.path.join(d, ".failed")) else ""
    att = re.search(r"attempts: (\d+)", failed)
    tasks[os.path.basename(d)] = {
        "state": "done" if os.path.exists(os.path.join(d, ".done")) else ("skipped" if os.path.exists(os.path.join(d, ".skipped")) else
                 ("failed" if failed and int(att.group(1)) >= 3 else ("retrying" if failed else ("running" if text and n in live else "queued")))),   # [v2.5.9o] a train.log alone (a killed run waiting for its GPU) is no running task
        "attempts_failed": int(att.group(1)) if att else 0, "last_failure_tail": failed.splitlines()[-3:] if failed else [],
        "step": int(steps[-1]) if steps else 0,
        "val_math500": last(r"val-core/math500/reward/mean@\d+:([0-9.]+)", text),   # mean@2 (E10b) or mean@4 (E12b); v2.3
        "truncation": last(r"response_length/clip_ratio:([0-9.]+)", text),
        "step_time_s": last(r"timing_s/step:([0-9.]+)", text) or last(r"train/time\(s\):([0-9.]+)", text)}   # RL / SFT
    t = tasks[os.path.basename(d)]; times = [float(x) for x in (re.findall(r"timing_s/step:([0-9.]+)", text) or re.findall(r"train/time\(s\):([0-9.]+)", text))[-5:]]
    t["total_steps"] = total_steps(d, text); t["step_time_mean5_s"] = round(sum(times) / len(times), 1) if times else None
    t["eta_min"] = round((t["total_steps"] - t["step"]) * t["step_time_mean5_s"] / 60) if t["total_steps"] and times and t["state"] == "running" else None
    t["log_tail"] = tail(os.path.join(root, "logs", os.path.basename(d) + ".log"))
    lv = live.get(os.path.basename(d))   # v2.5.5: the GPU a running task holds, its kind and start (no paths, no names)
    if lv and t["state"] == "queued":
        t["state"] = "running"            # FA-PROBE, SFT's teacher traces, evaluations: running without a train.log
    if not lv and os.path.exists(os.path.join(d, ".paused")) and t["state"] not in ("done", "skipped"):
        t["state"] = "paused"             # [v2.5.8] pause / kill / move-ckpt leave .paused; was shown as running or retrying
    if not lv and t["state"] in ("queued", "retrying") and os.path.exists(os.path.join(d, ".blocked")):   # [v2.5.9z3] held by a failed gate
        bm = re.search(r"blocked_by: (\S+)", open(os.path.join(d, ".blocked"), errors="ignore").read())
        t["state"] = "blocked"; t["blocked_by"] = bm.group(1) if bm else None
    elif not lv and t["state"] in ("queued", "retrying") and os.path.exists(os.path.join(d, ".waiting")):   # [v2.5.9z3] its inputs not in yet
        t["state"] = "waiting"
    t["gpu"] = lv[0] if lv else None; t["kind"] = kind(os.path.basename(d))
    t["gpus"] = lv[2] if lv else []   # [v2.5.9a] every GPU of the task (two for a #gpus 2 task); "gpu" stays the first
    t["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(lv[1])) if lv else None
    if t["state"] == "done" and os.path.exists(os.path.join(d, ".upload_pending")):   # [v2.5.9z65] done, its results still in the queue
        t["phase"] = "upload_pending"
    if lv and gpu_release_on and os.path.exists(os.path.join(d, ".gpu_released")):   # [v2.5.9z65a] its GPU work over, its results going up
        t["phase"] = "uploading"; t["released_gpu"] = t["gpu"]; t["gpu"] = None; t["gpus"] = []
    if t["state"] == "running" and t["kind"] in ("eval", "probe"):   # [v2.5.9s] never fails the heartbeat
        try:
            pr = progress(os.path.basename(d), t["log_tail"], queue)
            if pr:
                t["progress"] = pr
        except Exception:
            pass
gpu = []
for line in subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip().splitlines():
    f = [x.strip() for x in line.split(",")]
    if len(f) == 5:
        gpu.append({"index": int(f[0]), "name": f[1], "util_pct": float(f[2]) if f[2].replace(".", "").isdigit() else None,
                    "mem_used_mib": float(f[3]), "mem_total_mib": float(f[4])})
git = lambda *a: subprocess.run(["git", "-C", runbook, *a], capture_output=True, text=True).stdout.strip()
st = os.statvfs(root)
ver = re.search(r"^version:\s*(\S+)", queue, re.M)
for _n in pad_tasks & set(tasks):   # [v2.5.9m] tasks[].pad: a padding reading (sched-preempt, the monitor page, the patrol)
    tasks[_n]["pad"] = True
stages = {m.group(1): m.group(2) for m in re.finditer(r"^(\S+)\s+(open|auto|locked)\s", queue, re.M)}   # v2.3: stage -> status
json.dump({"host": os.uname().nodename, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "queue_version": ver.group(1) if ver else None, "stages": stages,
           "stack": jload(os.path.join(root, "logs", "stack.json"), {}),
           "tasks": tasks, "stale_task_dirs": stale, "gpus": gpu, "disk_free_gb": round(st.f_bavail * st.f_frsize / 1e9, 1),
           # v2.4: alert below DISK_ALERT_GB (30): one 1.7B save needs ~11 GB for a moment (fp32 HF + bf16 copy), a SPOT
           # resume save ~28 GB
           "disk_alert": st.f_bavail * st.f_frsize / 1e9 < float(os.environ.get("DISK_ALERT_GB", "30")),
           # v2.4: the checkpoint disk ($CKPT_ROOT, e.g. /home on the B300 node) has its own reading and alert
           "ckpt_root": os.environ.get("CKPT_ROOT"),
           "ckpt_disk_free_gb": (lambda c: round(c.f_bavail * c.f_frsize / 1e9, 1))(os.statvfs(os.environ["CKPT_ROOT"])) if os.path.isdir(os.environ.get("CKPT_ROOT", "")) else None,
           "ckpt_disk_alert": (lambda c: c.f_bavail * c.f_frsize / 1e9 < float(os.environ.get("CKPT_ALERT_GB", "100")))(os.statvfs(os.environ["CKPT_ROOT"])) if os.path.isdir(os.environ.get("CKPT_ROOT", "")) else None,
           "disk_used_pct": round(100 * (1 - st.f_bavail / st.f_blocks), 1) if st.f_blocks else None,
           # [v2.5.9c] the work disk's size and its exact free bytes (the GB reading stood still for hours on 10-21), the
           # checkpoint disk's size, and our own usage from our_du.sh (du, every 30 min)
           "disk_total_gb": round(st.f_blocks * st.f_frsize / 1e9, 1), "disk_free_bytes": st.f_bavail * st.f_frsize,
           "ckpt_disk_total_gb": (lambda c: round(c.f_blocks * c.f_frsize / 1e9, 1))(os.statvfs(os.environ["CKPT_ROOT"])) if os.path.isdir(os.environ.get("CKPT_ROOT", "")) else None,
           "our_du": jload(os.path.join(root, "control", "our_du.json")),
           "held": jload(os.path.join(root, "control", "held.json")),   # [v2.5.9c] GPUs held for a #gpus task
           "readings_held": readings_held(),   # [v2.5.9z69] readings not started for the work disk (READINGS_DISK_MIN_GB), 0 when the disk is fine
           "results_pending": results_pending(),   # [v2.5.9z65] result directories in weights_replay's queue (control/weights_pending.txt)
           "runbook_head": git("rev-parse", "--short", "HEAD"), "runbook_version": "v2.5.9",
           "run_tasks_sha": open(os.path.join(root, "control", "run_tasks.sha")).read().strip() if os.path.exists(os.path.join(root, "control", "run_tasks.sha")) else None,
           "ckpt_roots": roots_report(),   # v2.5: every candidate root: mount, free GB, writable, alert, task dirs and sizes
           "dr": dr_report(),   # v2.5.7: the disaster-recovery share: total, mirrored steps per task, kept models, cap
           "control": control_status(),
           # v2.5.5b: the last 200 uploads / downloads (private status only; the pulse does not carry them)
           "transfers": [json.loads(l) for l in open(os.path.join(root, "control", "transfers.jsonl")).read().splitlines()[-200:] if l.strip()]
                        if os.path.exists(os.path.join(root, "control", "transfers.jsonl")) else []}, open(sys.argv[1], "w"), indent=1)
PY
  brc=$?
  if [ $brc = 0 ] && python -c "import json, sys; json.load(open(sys.argv[1]))" $F.tmp 2> /dev/null; then
    mv -f $F.tmp $F
  else
    { date -u +%FT%TZ; echo "rc=$brc"; tail -c 4000 $OPD_ROOT/logs/status_builder.err.new; } > $OPD_ROOT/logs/status_builder.err 2> /dev/null
    rm -f $F $F.tmp; BERR="status builder failed rc=$brc"
  fi
  # [z55 builder end]
  # v2.5: never upload an empty or broken status file (B300 node, 2026-10-17 07:53: the disk was full and a 0-byte JSON
  # went up); a minimal status written to memory (/dev/shm) goes instead, with both disks' free space
  UP=$F
  if ! python -c "import json, sys; json.load(open(sys.argv[1]))" $F 2> /dev/null; then
    UP=/dev/shm/status_$H.json; [ -d /dev/shm ] || UP=/tmp/status_$H.json
    # [v2.5.9z56] the builder's last stderr lines (redacted) in the minimal status itself, so the project sees the failure at once
    MINPY=0
    [ -n "$BERR" ] && python - "$UP" "$H" "$BERR" "$(df -Pk $OPD_ROOT | awk 'NR==2{printf "%.1f", $4/1e6}')" "$(df -Pk ${CKPT_ROOT:-$OPD_ROOT} | awk 'NR==2{printf "%.1f", $4/1e6}')" 2> /dev/null <<'PY' && MINPY=1
import json, os, sys, time
sys.path.insert(0, os.path.join(os.environ["RUNBOOK"], "scripts"))
from control import redact
err = open(os.path.join(os.environ["OPD_ROOT"], "logs", "status_builder.err"), encoding="utf-8", errors="replace").read().strip().splitlines()
json.dump({"host": sys.argv[2], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error": sys.argv[3], "builder_error": [redact(x)[:300] for x in err[-6:]],
           "disk_free_gb": float(sys.argv[4]), "ckpt_disk_free_gb": float(sys.argv[5]), "runbook_version": "v2.5.9"}, open(sys.argv[1], "w"))
PY
    [ $MINPY = 1 ] || printf '{"host": "%s", "utc": "%s", "error": "%s", "disk_free_gb": %s, "ckpt_disk_free_gb": %s, "runbook_version": "v2.5.9"}\n' \
      "$H" "$(date -u +%FT%TZ)" "${BERR:-status JSON not written (disk full?)}" "$(df -Pk $OPD_ROOT | awk 'NR==2{printf "%.1f", $4/1e6}')" "$(df -Pk ${CKPT_ROOT:-$OPD_ROOT} | awk 'NR==2{printf "%.1f", $4/1e6}')" > $UP
  fi
  N=$(python $RUNBOOK/scripts/control.py node)
  # [v2.5.8] every upload below goes through the outbox (control.py put_q; procs / pulse / alerts / events inside
  # control.py too): this loop only writes local files while uploader.sh ships them; hb_to still bounds each call,
  # since put_q uploads directly while the uploader is stale
  hb_to python $RUNBOOK/scripts/control.py put_q $UP status/$N.json 2>> $OPD_ROOT/logs/heartbeat.err && echo "HEARTBEAT $(date -u +%T)"
  hb_to python $RUNBOOK/scripts/control.py pulse $UP > /dev/null 2>> $OPD_ROOT/logs/heartbeat.err   # v2.5.1: public pulse
  # v2.5 (D-004): the process table every round, and receipts/events again (a failed first upload is retried here)
  hb_to python $RUNBOOK/scripts/control.py procs --upload > /dev/null 2>> $OPD_ROOT/logs/heartbeat.err
  hb_to python $RUNBOOK/scripts/control.py alerts $UP > /dev/null 2>> $OPD_ROOT/logs/heartbeat.err   # disk / stalled / uploader events
  [ -n "${LOCAL_LRU_GB:-}" ] && python $RUNBOOK/scripts/control.py lru_clean >> $OPD_ROOT/logs/heartbeat.log 2>> $OPD_ROOT/logs/heartbeat.err   # [v2.5.8] off unless set
  for d in replies events; do [ -d $OPD_ROOT/control/$d ] && hb_to python $RUNBOOK/scripts/control.py put_q $OPD_ROOT/control/$d status/$N/$d > /dev/null 2>> $OPD_ROOT/logs/heartbeat.err; done
  # v2.5.7: the small control files also go to $(dr_root)/state/ (the share), so a rebuilt container finds them in one place
  if dr_on && [ -d "$(dirname "$(dr_root)")" ]; then
    DS=$(dr_root)/state; mkdir -p $DS
    for f in hf_repos.txt paths.env models.env control.env control/heartbeat_sec; do [ -f $OPD_ROOT/$f ] && cp -p $OPD_ROOT/$f $DS/$(basename $f); done
    cp -p $RUNBOOK/tasks/queue.txt $DS/queue.txt 2> /dev/null; git -C $RUNBOOK rev-parse HEAD > $DS/runbook_head 2> /dev/null; date -u +%FT%TZ > $DS/utc
  fi
  # first-arm comparison with the reference H100 run, uploaded once when E10b-P2 has reached step 10
  if [ ! -f $OPD_ROOT/logs/.compare_uploaded ] && python $RUNBOOK/scripts/compare_first_arm.py >/dev/null 2>&1; then
    hb_to python $RUNBOOK/scripts/control.py put_q $OPD_ROOT/logs/compare_$H.json status/compare_$N.json && touch $OPD_ROOT/logs/.compare_uploaded && echo "COMPARE_UPLOADED $(date -u +%T)"
  fi
  # v2.5: `heartbeat` (control.py) asks for the next round now: the flag is checked every 10 s
  # v2.5.1: the round lasts HEARTBEAT_SEC (control/heartbeat_sec, 60-3600, default 600); the pulse goes every 60 s
  HB=$(cat $OPD_ROOT/control/heartbeat_sec 2>/dev/null | tr -cd 0-9); HB=${HB:-600}; [ $HB -lt 60 ] && HB=60; [ $HB -gt 3600 ] && HB=3600
  for _i in $(seq $((HB / 10))); do
    [ -f $OPD_ROOT/control/heartbeat_now ] && { rm -f $OPD_ROOT/control/heartbeat_now; break; }
    sleep 10; hb_alive
    [ $((_i % 6)) = 0 ] && [ $_i -lt $((HB / 10)) ] && hb_to python $RUNBOOK/scripts/control.py pulse $UP > /dev/null 2>> $OPD_ROOT/logs/heartbeat.err
  done
done
