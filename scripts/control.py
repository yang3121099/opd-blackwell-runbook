"""Control plane of a remote node (runbook v2.4, ops-1): whitelisted commands from control/<node>.yaml in this runbook
repository, executed once each in seq order; the node never pushes to GitHub, results go to the heartbeat on HF.
  python control.py apply    after every git pull (run_tasks.sh): run every command with seq > applied_seq
  python control.py status   JSON for the heartbeat (node, applied_seq, recent results, control.env, last pull, ...)
  python control.py node     this node's id (OPD_NODE, else the first 8 hex of sha256(short hostname))
File format (append only; seq strictly increasing):
  node: 1a2b3c4d
  commands:
    - {seq: 1, cmd: "pause E12b-P5"}
Verbs: pause <task> | resume <task> | kill <task> | prio <task> <n> | restore <task> runs/<task>/global_step_<N> |
rerun_verify | setenv KEY=VAL | snapshot <path under $OPD_ROOT> | snapshot nvidia-smi. Nothing else, and never a shell:
arguments are split on whitespace and any shell metacharacter rejects the command. Results: $OPD_ROOT/control/log.jsonl.
"""
import calendar
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = os.environ.get("OPD_ROOT", os.path.expanduser("~/opd-run"))
RUNBOOK = os.environ.get("RUNBOOK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CDIR = os.path.join(ROOT, "control")
EXPECTED_ORIGIN = os.environ.get("OPD_CONTROL_ORIGIN", "github.com/yang3121099/opd-blackwell-runbook")
TASK_RE = re.compile(r"^[A-Za-z0-9._-]+$")
KEY_RE = re.compile(r"^(VLLM_[A-Z0-9_]+|TRITON_[A-Z0-9_]+|TORCHINDUCTOR_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|VLLM_UTIL|TEACHER_OFFLOAD"
                    r"|MAX_TOK_GPU|KEEP_LOCAL_HF|MAX_JOBS)$")
VAL_RE = re.compile(r"^[A-Za-z0-9_.,:/=+-]*$")
META = set(";&|`$<>\\\"'(){}*?!#~\n\r")
SECRET_NAME = re.compile(r"token|secret|credential|\.netrc|id_rsa", re.I)
SKIP_EXT = (".safetensors", ".pt", ".bin", ".parquet", ".pth", ".gguf")
MAX_FILE, MAX_TOTAL, MAX_FILES = 5 << 20, 20 << 20, 50
INIT_ERR = re.compile(r"vllm|triton|ptxas|ninja|EngineCore failed|no kernel image|CUDA error", re.I)
_REDACT = [(re.compile(r"hf_[A-Za-z0-9]{30,}"), "hf_***"),
           (re.compile(r"ms-[A-Za-z0-9-]{20,}"), "ms-***"),
           (re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=([A-Za-z0-9_+=-]{40,})"), r"\1=***")]


def redact(text):
    """The one redaction used for log tails, snapshots and control.env in the heartbeat (ops-1 amendment 1)."""
    for pat, rep in _REDACT:
        text = pat.sub(rep, text)
    return text


def node_id():
    return os.environ.get("OPD_NODE") or hashlib.sha256(socket.gethostname().split(".")[0].encode()).hexdigest()[:8]


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _p(*a):
    return os.path.join(CDIR, *a)


def _read(path, default=""):
    try:
        return open(path).read()
    except OSError:
        return default


def origin_ok():
    try:
        u = subprocess.run(["git", "-C", RUNBOOK, "remote", "get-url", "origin"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return False, ""
    norm = re.sub(r"^(https?://|git@)", "", u).replace("github.com:", "github.com/")
    norm = re.sub(r"\.git$", "", norm).rstrip("/")
    return norm == EXPECTED_ORIGIN, u


def running_pids(task=None):
    pat = f"task.sh [0-9]+ {re.escape(task)}$" if task else "task.sh [0-9]+ [A-Za-z0-9._-]+$"
    r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split() if x.strip().isdigit() and int(x) != os.getpid()]


def gpu_procs(g):
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader", "-i", str(g)],
                             capture_output=True, text=True, timeout=30).stdout
        return [x.strip() for x in out.splitlines() if x.strip()]
    except Exception:
        return ["nvidia-smi unavailable"]


def task_dir(t):
    d = os.path.join(ROOT, "runs", t); os.makedirs(d, exist_ok=True); return d


# ---------- verbs: each returns (status, message, extra dict) ----------
def v_pause(a, seq):
    t, = a; open(os.path.join(task_dir(t), ".paused"), "w").write(f"seq {seq} {utc()}\n")
    return "ok", f"{t} paused" + (" (still running; use kill to stop it)" if running_pids(t) else ""), {}


def v_resume(a, seq):
    t, = a
    for f in (".paused", ".killed"):
        try:
            os.remove(os.path.join(task_dir(t), f))
        except FileNotFoundError:
            pass
    return "ok", f"{t} resumed", {}


def _tree(pids):
    """The given processes and all their descendants (parent links in /proc), plus every process of their sessions.
    Collected before any signal, so Ray daemons that left the process group are still found while they are children."""
    children, sid = {}, {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        st = _read(f"/proc/{d}/stat")
        m = re.match(r"^\d+ \(.*\) \S (\d+) \d+ (\d+)", st)   # ppid, pgrp is skipped, session
        if m:
            children.setdefault(int(m.group(1)), []).append(int(d)); sid[int(d)] = int(m.group(2))
    out, todo = set(), list(pids)
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.add(p); todo += children.get(p, [])
    sessions = {sid[p] for p in pids if p in sid and sid[p] > 1} - {os.getsid(0)}   # never our own session (run_tasks)
    out |= {p for p, s_ in sid.items() if s_ in sessions}
    anc, p = set(), os.getpid()
    while p > 1:   # never ourselves or an ancestor
        anc.add(p)
        m = re.search(r"^PPid:\s+(\d+)", _read(f"/proc/{p}/status"), re.M)
        p = int(m.group(1)) if m else 1
    return out - anc - {1}


def _signal_all(pids, sig):
    for p in pids:
        try:
            os.kill(p, sig)
        except (ProcessLookupError, PermissionError):
            pass


def v_kill(a, seq):
    """Stops exactly one task: its task.sh, every descendant and its session (ops-1 amendment 2; never by GPU-wide
    patterns: a pattern match on /tmp/ray_gpu<G> killed another run's Ray session on 2026-10-16)."""
    t, = a; d = task_dir(t)
    open(os.path.join(d, ".paused"), "w").write(f"seq {seq} {utc()} (kill)\n")
    open(os.path.join(d, ".killed"), "w").write(f"seq {seq} {utc()}\n")   # run_tasks does not count this exit as a failure
    pids = running_pids(t)
    if not pids:
        return "ok", f"{t} was not running; paused", {}
    gpus = set()
    for pid in pids:
        m = re.search(r"task\.sh\s+(\d+)\s", _read(f"/proc/{pid}/cmdline").replace("\0", " ") + " ")
        if m:
            gpus.add(m.group(1))
    procs = _tree(pids)
    _signal_all(procs, signal.SIGTERM)
    deadline = time.time() + 30
    while time.time() < deadline and any(os.path.exists(f"/proc/{p}") and "zombie" not in _read(f"/proc/{p}/status").lower() for p in procs):
        time.sleep(1)
    alive = [p for p in procs if os.path.exists(f"/proc/{p}")]
    _signal_all(alive, signal.SIGKILL)
    time.sleep(5)
    clear = {g: not gpu_procs(g) for g in sorted(gpus)}
    return "ok", f"{t} killed and paused: {len(procs)} processes of its tree ({len(alive)} needed SIGKILL)", {"gpu_clear": clear}


def v_prio(a, seq):
    t, n = a
    if not re.fullmatch(r"\d{1,4}", n):
        return "rejected", "prio must be an integer 0-9999", {}
    open(os.path.join(task_dir(t), ".prio"), "w").write(n + "\n")
    return "ok", f"{t} prio {n}", {}


def _git_blob_sha1(path):
    h = hashlib.sha1(); h.update(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def v_restore(a, seq):
    t, path = a
    m = re.fullmatch(rf"runs/{re.escape(t)}/global_step_([0-9]+)", path)
    if not m:
        return "rejected", "path must be runs/<task>/global_step_<N>", {}
    if running_pids(t):
        return "rejected", f"{t} is running (kill it first)", {}
    if os.environ.get("UPLOAD_BACKEND", "hf") != "hf" or not os.environ.get("HF_REPO"):
        return "rejected", "restore needs the hf backend and HF_REPO", {}
    from huggingface_hub import HfApi, hf_hub_download
    repo = os.environ["HF_REPO"]; step = m.group(1)
    files = [f for f in HfApi().list_repo_tree(repo, path_in_repo=path, recursive=True, expand=True) if hasattr(f, "size")]
    if not files:
        return "failed", f"nothing under {path} in {repo}", {}
    tmp = _p(f"restore_{seq}"); shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    bad = []
    for f in files:
        local = hf_hub_download(repo, f.path, local_dir=tmp)
        if os.path.getsize(local) != f.size:
            bad.append(f"{f.path}: size"); continue
        lfs = getattr(f, "lfs", None)
        want = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        if want and _sha256(local) != want:
            bad.append(f"{f.path}: sha256")
        elif not want and f.blob_id and _git_blob_sha1(local) != f.blob_id:
            bad.append(f"{f.path}: git sha1")
    if bad:
        shutil.rmtree(tmp, ignore_errors=True)
        return "failed", f"checksum mismatch, nothing restored: {bad[:5]}", {}
    dest = os.path.join(task_dir(t), "ckpt", f"global_step_{step}")
    if os.path.exists(dest):
        os.rename(dest, f"{dest}.before_restore_{seq}")   # kept, never deleted
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(os.path.join(tmp, path), dest); shutil.rmtree(tmp, ignore_errors=True)
    resumable = any(n.startswith("optim_world_size_") for n in os.listdir(os.path.join(dest, "actor"))) if os.path.isdir(os.path.join(dest, "actor")) else False
    if resumable:
        open(os.path.join(task_dir(t), "ckpt", "latest_checkpointed_iteration.txt"), "w").write(step)
    return "ok", f"{len(files)} files verified and restored to {dest}" + (" (resume state: latest set)" if resumable else " (weights only)"), {}


def v_rerun_verify(a, seq):
    if running_pids():
        return "rejected", "tasks are running (pause/kill them first)", {}
    log = os.path.join(ROOT, "logs", f"verify_{seq}.log"); flag = _p("verify.running")
    sh = (f"echo $$ > {flag}; bash {RUNBOOK}/scripts/verify.sh > {log} 2>&1; "
          f"python {RUNBOOK}/scripts/remote.py put {log} logs/{node_id()}/requests/{seq}/verify.log; rm -f {flag}")
    subprocess.Popen(["setsid", "bash", "-c", sh], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "ok", f"verify started (log {log}); run_tasks starts nothing while it runs", {}


def _env_file():
    return os.path.join(ROOT, "control.env")


def read_env():
    out = {}
    for line in _read(_env_file()).splitlines():
        m = re.match(r"^export ([A-Za-z0-9_]+)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def v_setenv(a, seq):
    kv, = a
    if "=" not in kv:
        return "rejected", "setenv KEY=VAL", {}
    k, v = kv.split("=", 1)
    if not KEY_RE.match(k):
        return "rejected", f"key {k} not allowed", {}
    if not VAL_RE.match(v):
        return "rejected", "value has characters outside [A-Za-z0-9_.,:/=+-]", {}
    env = read_env()
    if v == "":
        env.pop(k, None)
    else:
        env[k] = v
    tmp = _env_file() + ".tmp"
    open(tmp, "w").write("# written by control.py (ops-1); sourced last by env.sh\n" + "".join(f"export {x}={y}\n" for x, y in sorted(env.items())))
    os.replace(tmp, _env_file())
    times = json.loads(_read(_p("setenv_times.json"), "{}") or "{}"); times[k] = time.time()
    json.dump(times, open(_p("setenv_times.json"), "w"))
    return "ok", f"{k} {'removed' if v == '' else 'set'}; applies to tasks started from now on", {}


def v_snapshot(a, seq):
    what, = a; out = _p(f"req_{seq}"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    manifest = {"seq": seq, "what": what, "included": [], "skipped": []}
    if what == "nvidia-smi":
        r = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=60)
        open(os.path.join(out, "nvidia-smi.txt"), "w").write(redact(r.stdout)); manifest["included"].append("nvidia-smi.txt")
    else:
        base = os.path.realpath(ROOT)
        target = os.path.realpath(what if os.path.isabs(what) else os.path.join(ROOT, what))
        if target != base and not target.startswith(base + os.sep):
            shutil.rmtree(out, ignore_errors=True)
            return "rejected", "path outside $OPD_ROOT", {}
        if not os.path.exists(target):
            shutil.rmtree(out, ignore_errors=True)
            return "rejected", "no such path", {}
        cands = [target] if os.path.isfile(target) else \
            [os.path.join(dp, f) for dp, _, fs in os.walk(target, followlinks=False) for f in sorted(fs)]
        total = 0
        for f in cands:
            rel = os.path.relpath(f, base); real = os.path.realpath(f)
            why = None
            if real != base and not real.startswith(base + os.sep):
                why = "link outside $OPD_ROOT"
            elif SECRET_NAME.search(os.path.basename(f)):
                why = "secret-like name"
            elif f.endswith(SKIP_EXT):
                why = "weights/data"
            elif os.path.getsize(real) > MAX_FILE:
                why = "larger than 5 MB"
            elif len(manifest["included"]) >= MAX_FILES or total + os.path.getsize(real) > MAX_TOTAL:
                why = "snapshot limit (50 files / 20 MB)"
            if why is None:
                try:
                    text = open(real, "rb").read().decode("utf-8")
                except UnicodeDecodeError:
                    why = "binary"
            if why:
                manifest["skipped"].append([rel, why]); continue
            dst = os.path.join(out, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "w").write(redact(text)); total += os.path.getsize(real); manifest["included"].append(rel)
    json.dump(manifest, open(os.path.join(out, "MANIFEST.json"), "w"), indent=1)
    dest = f"logs/{node_id()}/requests/{seq}"
    r = subprocess.run([sys.executable, os.path.join(RUNBOOK, "scripts", "remote.py"), "put", out, dest], capture_output=True, text=True)
    if r.returncode != 0:
        return "failed", f"upload to {dest} failed", {"manifest": manifest}
    return "ok", f"{len(manifest['included'])} files to {dest}, {len(manifest['skipped'])} skipped", {}


VERBS = {"pause": (v_pause, 1), "resume": (v_resume, 1), "kill": (v_kill, 1), "prio": (v_prio, 2),
         "restore": (v_restore, 2), "rerun_verify": (v_rerun_verify, 0), "setenv": (v_setenv, 1), "snapshot": (v_snapshot, 1)}
TASK_VERBS = {"pause", "resume", "kill", "prio", "restore"}


def run_one(cmd, seq):
    if not isinstance(cmd, str) or any(ch in META for ch in cmd):
        return "rejected", "not a string or contains shell metacharacters", {}
    parts = cmd.split()
    if not parts or parts[0] not in VERBS:
        return "rejected", f"unknown verb {parts[0] if parts else ''!r}", {}
    fn, n = VERBS[parts[0]]; args = parts[1:]
    if len(args) != n:
        return "rejected", f"{parts[0]} takes {n} argument(s)", {}
    if parts[0] in TASK_VERBS and not TASK_RE.match(args[0]):
        return "rejected", "bad task name", {}
    try:
        return fn(args, seq)
    except Exception as e:   # a failed command is recorded, never retried
        return "failed", f"{type(e).__name__}: {str(e)[:200]}", {}


def log(entry):
    with open(_p("log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def state(**kw):
    s = json.loads(_read(_p("state.json"), "{}") or "{}"); s.update(kw); json.dump(s, open(_p("state.json"), "w"))


def apply():
    os.makedirs(CDIR, exist_ok=True)
    ok, url = origin_ok()
    if not ok:
        state(origin_mismatch=url or "unknown", checked=utc()); print(f"CONTROL origin mismatch: {url!r}"); return
    state(origin_mismatch=None, checked=utc())
    path = os.path.join(RUNBOOK, "control", f"{node_id()}.yaml")
    if not os.path.exists(path):
        return
    import yaml
    try:
        doc = yaml.safe_load(open(path)) or {}
    except Exception as e:
        state(file_error=f"{type(e).__name__}"); return
    if str(doc.get("node")) != node_id():
        state(node_mismatch=str(doc.get("node"))); return
    state(node_mismatch=None, file_error=None)
    applied = int((_read(_p("applied_seq"), "0").strip() or 0)); prev = None
    seen = set(json.loads(_read(_p("state.json"), "{}") or "{}").get("rejected_entries", []))
    for i, c in enumerate(doc.get("commands") or []):
        seq = c.get("seq") if isinstance(c, dict) else None
        bad = None if isinstance(seq, int) and seq > 0 and (prev is None or seq > prev) else \
            ("seq not a positive integer" if not (isinstance(seq, int) and seq > 0) else "seq not increasing")
        if bad:   # never executed; reported once per position in the file
            key = f"{i}:{seq}"
            if key not in seen:
                seen.add(key); state(rejected_entries=sorted(seen))
                log({"seq": seq, "cmd": c.get("cmd") if isinstance(c, dict) else None, "status": "rejected", "message": bad, "utc": utc()})
            continue
        prev = seq
        if seq <= applied:
            continue
        status, msg, extra = run_one(c.get("cmd"), seq)
        log({"seq": seq, "cmd": c.get("cmd"), "status": status, "message": msg, "utc": utc(), **extra})
        print(f"CONTROL seq {seq} {c.get('cmd')!r}: {status}: {msg}")
        applied = seq; open(_p("applied_seq.tmp"), "w").write(str(seq)); os.replace(_p("applied_seq.tmp"), _p("applied_seq"))


def suspects():
    """setenv followed within 10 minutes by a task failure with a vLLM/Triton init error (report only)."""
    times = json.loads(_read(_p("setenv_times.json"), "{}") or "{}"); out = []
    for d in sorted(os.listdir(os.path.join(ROOT, "runs"))) if os.path.isdir(os.path.join(ROOT, "runs")) else []:
        failed = _read(os.path.join(ROOT, "runs", d, ".failed"))
        m = re.search(r"^last: (\S+)", failed, re.M)
        if not m:
            continue
        try:
            tf = calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            continue
        tail = failed + _read(os.path.join(ROOT, "logs", f"{d}.log"))[-20000:]
        keys = [k for k, t in times.items() if -5 <= tf - t <= 600]   # 5 s slack: the failure time has whole seconds
        if keys and INIT_ERR.search(tail):
            out.append({"task": d, "keys": keys, "failed_at": m.group(1)})
    return out


def status():
    lines = _read(_p("log.jsonl")).splitlines()[-5:]
    s = json.loads(_read(_p("state.json"), "{}") or "{}")
    return {"node": node_id(), "applied_seq": int((_read(_p("applied_seq"), "0").strip() or 0)),
            "recent": [json.loads(x) for x in lines if x.strip()], "origin_mismatch": s.get("origin_mismatch"),
            "node_mismatch": s.get("node_mismatch"), "file_error": s.get("file_error"),
            "control_env": {k: redact(f"{k}={v}").split("=", 1)[1] for k, v in read_env().items()},
            "last_pull": json.loads(_read(_p("last_pull.json"), "null") or "null"),
            "control_env_suspect": suspects(), "verify_running": os.path.exists(_p("verify.running")),
            "self_restart_refused": _read(_p("self_restart_refused")).strip() or None}


if __name__ == "__main__":
    os.makedirs(CDIR, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "apply":
        apply()
    elif cmd == "status":
        print(json.dumps(status()))
    elif cmd == "node":
        print(node_id())
    else:
        print(__doc__); sys.exit(2)
