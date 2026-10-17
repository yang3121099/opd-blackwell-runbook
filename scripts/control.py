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

v2.5 (D-004 / R-012): verbs from control_rules.VERBS (adds ps, logs, run, eval, update, upload, set, signal,
kill-pgid, cancel); every command gets a receipt status/<node>/replies/<seq>.json in the result repository (refusals
too); the pool's own start/retry/failed/done go to status/<node>/events/; status/<node>/procs.json lists our process
trees and every process on our GPUs; default expiry 2 h / 24 h by verb; reason required; rate limits per pull.
  python control.py procs [--upload] | event <task> <kind> [detail] | next_request | check_pull <before> <after>
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import control_rules as R  # noqa: E402

ROOT = os.environ.get("OPD_ROOT", os.path.expanduser("~/opd-run"))
RUNBOOK = os.environ.get("RUNBOOK", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CDIR = os.path.join(ROOT, "control")
EXPECTED_ORIGIN = os.environ.get("OPD_CONTROL_ORIGIN", "github.com/yang3121099/opd-blackwell-runbook")
TASK_RE = re.compile(r"^[A-Za-z0-9._-]+$")
KEY_RE = R.KEY_RE   # v2.5: shared with ctl.py
VAL_RE = re.compile(r"^[A-Za-z0-9_.,:/=+-]*$")
META = R.META
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
    repo = _public_repo()
    if os.environ.get("UPLOAD_BACKEND", "hf") != "hf" or not repo:
        return "rejected", "restore needs the hf backend and HF_REPO (or hf_repos.txt)", {}
    from huggingface_hub import HfApi, hf_hub_download
    step = m.group(1)
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


# ---------- v2.5: result repository, receipts, events ----------
def remote_put(src, dst):
    r = subprocess.run([sys.executable, os.path.join(RUNBOOK, "scripts", "remote.py"), "put", src, dst], capture_output=True, text=True)
    return r.returncode == 0


STATUS_NAME = {"ok": "done", "rejected": "refused"}   # receipts use D-004's names; log.jsonl keeps v2.4's


def receipt(seq, entry, status, detail, **extra):
    """status/<node>/replies/<seq>.json; written locally first, uploaded (the heartbeat re-uploads replies/ each round)."""
    os.makedirs(_p("replies"), exist_ok=True)
    path = _p("replies", f"{seq}.json")
    old = json.loads(_read(path, "{}") or "{}")
    r = {"seq": seq, "cmd": (entry or {}).get("cmd") or old.get("cmd"), "reason": (entry or {}).get("reason") or old.get("reason"), "node": node_id(),
         "runbook_head": _git_head(), "received_utc": old.get("received_utc") or utc(),
         "started_utc": old.get("started_utc") or (utc() if status in ("ok", "accepted", "failed") else None),
         "finished_utc": None if status == "accepted" else utc(), "status": STATUS_NAME.get(status, status),
         "detail": detail, "rules_version": R.RULES_VERSION}
    r.update({k: v for k, v in old.items() if k not in r and k != "status"}); r.update(extra)
    json.dump(r, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
    remote_put(path, f"status/{node_id()}/replies/{seq}.json")
    return r


def event(task, kind, detail=""):
    """The pool's own actions (start, retry, failed, done) in the receipt shape, under status/<node>/events/ (R-012)."""
    os.makedirs(_p("events"), exist_ok=True)
    name = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{task}_{kind}.json"
    e = {"event": kind, "task": task, "detail": redact(detail)[:2000], "node": node_id(), "utc": utc(),
         "runbook_head": _git_head(), "origin": "pool"}
    json.dump(e, open(_p("events", name), "w"), indent=1)
    remote_put(_p("events", name), f"status/{node_id()}/events/{name}")


def _git_head():
    try:
        return subprocess.run(["git", "-C", RUNBOOK, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


# ---------- v2.5: process table ----------
OUR_ROOTS = re.compile(r"scripts/(run_tasks|task|upload_sync|heartbeat|run_adhoc_eval)\.sh|scripts/control\.py|code/eval/gen_eval\.py|/tmp/ray_gpu")


def _proc(pid):
    """One process from /proc: None if it is gone."""
    st = _read(f"/proc/{pid}/stat")
    if not st:
        return None
    rest = st[st.rfind(")") + 2:].split()
    try:
        state, ppid, pgid, sid = rest[0], int(rest[1]), int(rest[2]), int(rest[3])
        ut, stt, start = int(rest[11]), int(rest[12]), int(rest[19])
    except (IndexError, ValueError):
        return None
    argv = _read(f"/proc/{pid}/cmdline").split("\0")
    cmd = " ".join(a for a in argv if a)
    stat_s = _read(f"/proc/{pid}/status")
    m = re.search(r"^Uid:\s+(\d+)", stat_s, re.M); uid = int(m.group(1)) if m else -1
    m = re.search(r"^VmRSS:\s+(\d+)", stat_s, re.M); rss = int(m.group(1)) // 1024 if m else 0
    hz = os.sysconf("SC_CLK_TCK")
    boot = int(re.search(r"^btime (\d+)", _read("/proc/stat"), re.M).group(1))
    t0 = boot + start / hz
    return {"pid": pid, "ppid": ppid, "pgid": pgid, "sid": sid, "uid": uid, "state": state,
            "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "cpu_pct": round(100 * (ut + stt) / hz / max(time.time() - t0, 1), 1), "rss_mb": rss,
            "argv_hash": hashlib.sha256("\0".join(argv).encode()).hexdigest()[:16], "_cmd": cmd}


def _gpu_apps():
    """pid -> (gpu index, used MiB) from nvidia-smi."""
    try:
        idx = {}
        for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.splitlines():
            i, u = [x.strip() for x in line.split(",")]; idx[u] = int(i)
        out = {}
        for line in subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=30).stdout.splitlines():
            p, u, mem = [x.strip() for x in line.split(",")]
            if p.isdigit():
                out[int(p)] = (idx.get(u), int(mem) if mem.isdigit() else None)
        return out
    except Exception:
        return {}


def proc_table():
    rows = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            r = _proc(int(d))
            if r:
                rows[r["pid"]] = r
    kids = {}
    for r in rows.values():
        kids.setdefault(r["ppid"], []).append(r["pid"])
    ours, todo = set(), [p for p, r in rows.items() if OUR_ROOTS.search(r["_cmd"])]
    while todo:
        p = todo.pop()
        if p not in ours:
            ours.add(p); todo += kids.get(p, [])
    gpus = _gpu_apps()

    def task_of(p):
        seen = 0
        while p in rows and seen < 64:
            m = re.search(r"scripts/task\.sh (\d+) ([A-Za-z0-9._-]+)", rows[p]["_cmd"]) or re.search(r"run_adhoc_eval\.sh (\d+) (\d+)", rows[p]["_cmd"])
            if m:
                return m.group(2) if "task.sh" in m.group(0) else f"adhoc_eval_{m.group(2)}"
            p, seen = rows[p]["ppid"], seen + 1
        return None
    out = []
    for p, r in sorted(rows.items()):
        if p not in ours and p not in gpus:
            continue
        row = {k: v for k, v in r.items() if k != "_cmd"}
        row.update(cmd=redact(r["_cmd"])[:200], ours=p in ours, task=task_of(p),
                   gpu=gpus.get(p, (None, None))[0], gpu_mem_mb=gpus.get(p, (None, None))[1])
        try:
            import pwd
            row["user"] = pwd.getpwuid(r["uid"]).pw_name
        except (KeyError, ImportError):
            row["user"] = str(r["uid"])
        out.append(row)
    return out


def publish_procs(upload=True):
    t = {"utc": utc(), "node": node_id(), "rows": proc_table()}
    json.dump(t, open(_p("procs.json.tmp"), "w"), indent=1); os.replace(_p("procs.json.tmp"), _p("procs.json"))
    if upload:
        remote_put(_p("procs.json"), f"status/{node_id()}/procs.json")
    return t


def v_ps(a, seq, flags=None, entry=None):
    t = publish_procs()
    return "ok", f"procs.json refreshed: {len(t['rows'])} rows", {"outputs": [f"status/{node_id()}/procs.json"]}


# ---------- v2.5: signal / kill-pgid (D-004 section 5, R-012) ----------
def _ancestors():
    out, p = set(), os.getpid()
    while p > 1:
        out.add(p)
        m = re.search(r"^PPid:\s+(\d+)", _read(f"/proc/{p}/status"), re.M)
        p = int(m.group(1)) if m else 1
    return out


def _published_row(pid):
    t = json.loads(_read(_p("procs.json"), "{}") or "{}")
    for r in t.get("rows", []):
        if r.get("pid") == pid:
            return r
    return None


def check_target(pid, sig, flags):
    """None if pid may receive sig, else the refusal reason."""
    row = _published_row(pid)
    if row is None:
        return "pid not in the latest published procs.json (run ps first)"
    now = _proc(pid)
    if now is None:
        return "process is gone"
    if now["start_utc"] != row["start_utc"] or now["argv_hash"] != row["argv_hash"]:
        return "pid recycled or command changed since procs.json was published"
    if pid in _ancestors():
        return "control.py, its run_tasks loop or their ancestors are never signalled (use update)"
    if R.NEVER_CMD.search(now["_cmd"]):
        return "protected system or session process"
    if now["uid"] != os.getuid():
        return "process of another user"
    if not row.get("ours"):
        if row.get("gpu") is None:
            return "not in our process trees and not on our GPUs"
        if not (flags.get("force") and flags.get("confirm_foreign")):
            return "process outside our trees: needs force and ctl.py --confirm-foreign"
    if sig in ("STOP", "CONT"):
        if R.TRAINING_CMD.search(now["_cmd"]) or (row.get("task") or "").startswith(("E10b", "E12b", "SFT")):
            return "STOP/CONT refused on training processes (use kill, pause or signal TERM)"
        if not (R.EVAL_CMD.search(now["_cmd"]) or (row.get("task") or "").startswith(("E11b", "adhoc_eval"))):
            return "STOP/CONT only on evaluation processes"
    return None


def _alive(pid):
    p = _proc(pid)
    return p is not None and p["state"] != "Z"


def v_signal(a, seq, flags=None, entry=None):
    flags = flags or {}; pid, sig = int(a[0]), a[1]
    why = check_target(pid, sig, flags)
    if why:
        return "rejected", why, {"target": _published_row(pid)}
    before = _published_row(pid); sent = []
    if sig == "KILL" and not flags.get("force"):   # TERM first, KILL only after 30 s (D-004 rule 4)
        os.kill(pid, signal.SIGTERM); sent.append("TERM")
        deadline = time.time() + 30
        while time.time() < deadline and _alive(pid):
            time.sleep(1)
        if _alive(pid):
            os.kill(pid, signal.SIGKILL); sent.append("KILL")
    else:
        os.kill(pid, getattr(signal, "SIG" + sig)); sent.append(sig)
    time.sleep(2); after = _proc(pid)
    publish_procs()
    return "ok", f"pid {pid}: sent {'+'.join(sent)}", {"before": before, "after": None if after is None else {k: v for k, v in after.items() if k != "_cmd"},
                                                      "force": bool(flags.get("force"))}


def v_killpgid(a, seq, flags=None, entry=None):
    flags = flags or {}; pgid = int(a[0])
    lead = _published_row(pgid)
    if lead is None or not lead.get("ours"):
        return "rejected", "group leader not one of our processes in the latest procs.json", {}
    now = _proc(pgid)
    if now is None or now["argv_hash"] != lead["argv_hash"]:
        return "rejected", "group leader gone or changed", {}
    if not re.search(r"scripts/task\.sh|code/eval/gen_eval\.py|run_adhoc_eval\.sh", now["_cmd"]):
        return "rejected", "only the group of a task session (task.sh, gen_eval, an ad-hoc eval) can be killed", {}
    members = [r["pid"] for r in proc_table() if r["pgid"] == pgid]
    if any(m in _ancestors() for m in members) or any(R.NEVER_CMD.search((_proc(m) or {}).get("_cmd", "")) for m in members):
        return "rejected", "group contains a protected process", {}
    if flags.get("force"):
        os.killpg(pgid, signal.SIGKILL); sent = "KILL"
    else:
        os.killpg(pgid, signal.SIGTERM); sent = "TERM"
        deadline = time.time() + 30
        while time.time() < deadline and any(_alive(m) for m in members):
            time.sleep(1)
        left = [m for m in members if _alive(m)]
        if left:
            _signal_all(left, signal.SIGKILL); sent += f"+KILL({len(left)})"
    time.sleep(2); publish_procs()
    return "ok", f"group {pgid}: {len(members)} processes, sent {sent}", {"members": members, "force": bool(flags.get("force"))}


# ---------- v2.5: logs, snapshot (no argument), set, run/eval requests, update, upload, cancel ----------
def v_logs(a, seq, flags=None, entry=None):
    t, n = a[0], int(a[1])
    src = os.path.join(ROOT, "logs", f"{t}.log")
    if not os.path.exists(src):
        return "rejected", f"no log for {t}", {}
    out = _p(f"req_{seq}"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    open(os.path.join(out, "log_tail.txt"), "w").write(redact("".join(_read(src).splitlines(True)[-n:])))
    dest = f"status/{node_id()}/replies/{seq}"
    return ("ok", f"last {n} lines of logs/{t}.log", {"outputs": [f"{dest}/log_tail.txt"]}) if remote_put(out, dest) else ("failed", "upload failed", {})


def v_snapshot0(seq):
    out = _p(f"req_{seq}"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    for name, cmd in (("nvidia-smi.txt", ["nvidia-smi"]), ("df.txt", ["df", "-h"])):
        try:
            open(os.path.join(out, name), "w").write(redact(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout))
        except Exception as e:
            open(os.path.join(out, name), "w").write(f"{type(e).__name__}\n")
    publish_procs(upload=True); shutil.copy(_p("procs.json"), os.path.join(out, "procs.json"))
    json.dump(roots_report(), open(os.path.join(out, "ckpt_roots.json"), "w"), indent=1)
    dest = f"status/{node_id()}/replies/{seq}"
    return ("ok", "nvidia-smi, df, procs.json and ckpt_roots.json", {"outputs": [dest]}) if remote_put(out, dest) else ("failed", "upload failed", {})


def v_set(a, seq, flags=None, entry=None):
    k, v = a[0].split("=", 1)
    if k in ("CKPT_ROOT", "CKPT_CANDIDATES+", "ARCHIVE_ROOT"):   # paths.env, checked against the candidate list
        st, msg = set_paths(k, v); return st, msg, {}
    if k in ("HF_REPO_PUBLIC", "HF_REPO_PRIVATE"):   # v2.5.1: $OPD_ROOT/hf_repos.txt, read by remote.py on every upload
        st, msg = set_repo(k, v); return st, msg, {}
    if k == "HEARTBEAT_SEC":   # v2.5.1: read by heartbeat.sh every round
        open(_p("heartbeat_sec.tmp"), "w").write(v + "\n"); os.replace(_p("heartbeat_sec.tmp"), _p("heartbeat_sec"))
        return "ok", f"heartbeat every {v} s from the next round (pulse every 60 s regardless)", {}
    return v_setenv(a, seq)


def _remote():
    sys.path.insert(0, os.path.join(RUNBOOK, "scripts"))
    import remote
    return remote


def _public_repo():
    try:
        return _remote().repos()[0]
    except Exception:   # a remote.py without repos() (v2.5 layout, tests): the single result repository
        return os.environ.get("HF_REPO")


def _split_mode():
    try:
        return bool(_remote().split_mode())
    except Exception:
        return False


def set_repo(k, v):
    """set HF_REPO_PUBLIC / HF_REPO_PRIVATE (v2.5.1, Runming 2026-10-17): written to $OPD_ROOT/hf_repos.txt. The repository
    must be reachable with this node's token, and the private one must be private. An empty value removes the key; with
    no key left the file goes and every upload uses HF_REPO again."""
    path = os.path.join(ROOT, "hf_repos.txt"); kind = "public" if k == "HF_REPO_PUBLIC" else "private"
    cur = {}
    for line in _read(path).splitlines():
        a, _, b = line.strip().partition("=")
        if a in ("public", "private") and b:
            cur[a] = b
    if v:
        try:
            from huggingface_hub import HfApi
            info = HfApi(endpoint=os.environ.get("HF_UPLOAD_ENDPOINT", "https://huggingface.co")).model_info(v)
        except Exception as e:
            return "rejected", f"{v} is not reachable with this node's token ({type(e).__name__})"
        if kind == "private" and not info.private:
            return "rejected", f"{v} is public; the private repository must be private"
        cur[kind] = v
    else:
        cur.pop(kind, None)
    if not cur:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return "ok", "hf_repos.txt removed: every upload goes to HF_REPO again"
    open(path + ".tmp", "w").write("".join(f"{a}={b}\n" for a, b in sorted(cur.items()))); os.replace(path + ".tmp", path)
    try:
        pub, priv = _remote().repos()
    except Exception:
        pub, priv = cur.get("public", os.environ.get("HF_REPO")), cur.get("private", os.environ.get("HF_REPO"))
    return "ok", f"public uploads (runs/ ckpts/ pulse/) -> {pub}; private (status/ logs/ replies events procs) -> {priv}; from the next upload"


# ---------- v2.5.1: the pulse (public, no paths / host names / user names) ----------
HOST_RE = re.compile(r"[A-Za-z0-9-]+\.[A-Za-z][A-Za-z0-9-]*\.[A-Za-z]{2,}|[A-Za-z][A-Za-z0-9-]*\.(?:com|net|org|cn|io|local|internal|lan)\b")


def _forbidden_tokens():
    """Words that must never reach a public file, read at run time from this machine (so no name is written into the
    public runbook): the user name, every part of the host name, every directory name of $HOME, $OPD_ROOT, $RUNBOOK,
    CKPT_ROOT, ARCHIVE_ROOT and the checkpoint candidates. Parts shorter than 5 characters are generic and skipped."""
    toks = set()

    def add(x):
        for t in re.split(r"[^A-Za-z0-9_]+", x or ""):
            if len(t) >= 5 and not t.isdigit():
                toks.add(t.lower())
    try:
        import getpass
        add(getpass.getuser())
    except Exception:
        pass
    for h in (socket.gethostname(), socket.getfqdn()):
        add(h)
    paths = [os.environ.get(x, "") for x in ("HOME", "OPD_ROOT", "RUNBOOK", "CKPT_ROOT", "ARCHIVE_ROOT")]
    try:
        paths += list(candidates())
    except Exception:
        pass
    for p in paths:
        for part in (p or "").split("/"):
            add(part)
    return toks


def leak_check(obj, toks=None):
    """The strings (keys and values) of obj that could carry a path, a host name or a user name."""
    toks = _forbidden_tokens() if toks is None else toks
    bad = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k); walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            words = {w.lower() for w in re.split(r"[^A-Za-z0-9_]+", x) if w}
            if any(c in x for c in "/\\@") or HOST_RE.search(x) or words & toks:
                bad.append(x)
    walk(obj)
    return bad


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def build_pulse(st, gpu_csv):
    """The public pulse from the (private) status JSON and `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,
    memory.total --format=csv,noheader,nounits`: numbers, task names, states and times only."""
    gpus = []
    for line in (gpu_csv or "").strip().splitlines():
        r = [x.strip() for x in line.split(",")]
        if len(r) == 4 and r[0].isdigit():
            f = lambda v: float(v) if re.fullmatch(r"[0-9]+(\.[0-9]+)?", v) else None
            gpus.append({"idx": int(r[0]), "util": f(r[1]), "mem_used": f(r[2]), "mem_total": f(r[3])})
    tasks = {}
    for name, v in (st.get("tasks") or {}).items():
        if not isinstance(v, dict) or not R.TASK_RE.match(name):
            continue
        state = v.get("state") if isinstance(v.get("state"), str) and re.fullmatch(r"[a-z_]+", v.get("state")) else None
        tasks[name] = {"state": state, "step": _num(v.get("step")), "total_steps": _num(v.get("total_steps")),
                       "val": _num(v.get("val_math500")), "trunc": _num(v.get("truncation")),
                       "s_per_step": _num(v.get("step_time_mean5_s")), "eta_min": _num(v.get("eta_min")),
                       "attempts_failed": _num(v.get("attempts_failed"))}
    ctl = st.get("control") if isinstance(st.get("control"), dict) else {}
    alerts = []
    for a in ctl.get("alerts") or []:
        if isinstance(a, str):
            code, _, arg = a.partition(":")
            if re.fullmatch(r"[a-z_]+", code):
                alerts.append(f"{code}:{arg}" if arg and R.TASK_RE.match(arg) and "/" not in arg else code)
    t = lambda x: x if isinstance(x, str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", x) else None
    head = st.get("runbook_head") if isinstance(st.get("runbook_head"), str) and re.fullmatch(r"[0-9a-f]{7,40}", st.get("runbook_head")) else None
    ver = st.get("runbook_version") if isinstance(st.get("runbook_version"), str) and re.fullmatch(r"v[0-9.]+", st.get("runbook_version")) else None
    return {"schema": 1, "utc": utc(), "node_id": node_id(), "status_utc": t(st.get("utc")), "runbook_head": head,
            "runbook_version": ver, "queue_version": _num(st.get("queue_version")) if not isinstance(st.get("queue_version"), str)
            else (st.get("queue_version") if re.fullmatch(r"[0-9]+", st.get("queue_version")) else None),
            "gpus": gpus, "tasks": tasks,
            "disks": [{"idx": 0, "free_gb": _num(st.get("disk_free_gb")), "alert": bool(st.get("disk_alert"))},
                      {"idx": 1, "free_gb": _num(st.get("ckpt_disk_free_gb")), "alert": bool(st.get("ckpt_disk_alert"))}],
            "alerts": sorted(set(alerts)),
            "control": {"applied_seq": _num(ctl.get("applied_seq")), "pending": _num(ctl.get("pending")),
                        "last_pull_utc": t(ctl.get("last_pull_utc"))}}


def pulse(status_path, upload=True):
    """Write control/pulse.json and, when the uploads are split (hf_repos.txt), upload it as pulse/<node>.json to the
    public repository. A pulse with any string that fails leak_check is not written; an event records the refusal
    (with the number of strings, never the strings)."""
    st = json.loads(_read(status_path, "{}") or "{}")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30).stdout
    except Exception:
        q = ""
    p = build_pulse(st, q)
    bad = leak_check(p)
    if bad:
        event("control", "pulse_refused", f"{len(bad)} string(s) failed the leak check")
        return None
    json.dump(p, open(_p("pulse.json.tmp"), "w")); os.replace(_p("pulse.json.tmp"), _p("pulse.json"))
    if upload and _split_mode():
        remote_put(_p("pulse.json"), f"pulse/{node_id()}.json")
    return p


def _request(kind, seq, body):
    os.makedirs(_p("requests"), exist_ok=True)
    json.dump(dict(body, kind=kind, seq=seq, utc=utc()), open(_p("requests", f"{seq}_{kind}.json"), "w"))


def v_run(a, seq, flags=None, entry=None):
    t = a[0]
    r = subprocess.run(["bash", "-c", f'T={t}; ARMS="P2|P3|P4lowlr|P5|P6|P6r|P2noEOS|Pstop|P6rand|P6top5|P4EOSpen"; '
                        'case $T in E10b-*) [[ $T =~ ^E10b-($ARMS)(-s(43|44))?$ ]];; E11b-*) [[ $T =~ ^E11b-(P0|$ARMS)$ ]];; '
                        'E12b-*) [[ $T =~ ^E12b-(P0|P1|$ARMS)$ ]];; SFT-P1|E03-MI-l0.[0357]|E03-MI-l1.0|E02-opsa17-c0.5|E02-opsa17-c1.5|E02-opsa4-c-1|'
                        'E02-opsa4-c0.5|E02-opsa4-c1.5) true;; *) false;; esac'])
    if r.returncode != 0:
        return "rejected", f"{t} is not accepted by task.sh", {}
    if running_pids(t):
        return "rejected", f"{t} is already running", {}
    if os.path.exists(os.path.join(ROOT, "runs", t, ".done")):
        return "rejected", f"{t} is already done", {}
    _request("run", seq, {"task": t})
    return "accepted", f"{t} starts on the next GPU that is free after the queue's own tasks (deps must be done)", {}


def v_eval(a, seq, flags=None, entry=None):
    _request("eval", seq, {"model": a[0], "bench": a[1], "n": int(a[2]) if len(a) > 2 else 4, "temp": a[3] if len(a) > 3 else "0.6"})
    return "accepted", "evaluation starts on the next GPU that is free after the queue's own tasks", {}


def next_request():
    """For run_tasks.sh, after its queue pass: the oldest open request, printed as 'run <task> <seq>' or 'eval <seq>'."""
    d = _p("requests")
    for f in sorted(os.listdir(d), key=lambda x: int(x.split("_")[0])) if os.path.isdir(d) else []:
        if not f.endswith(".json"):
            continue
        q = json.loads(_read(os.path.join(d, f), "{}") or "{}"); seq = q.get("seq")
        if os.path.exists(_p("cancelled", str(seq))):
            os.replace(os.path.join(d, f), os.path.join(d, f + ".cancelled")); continue
        os.replace(os.path.join(d, f), os.path.join(d, f + ".taken"))
        print(f"run {q['task']} {seq}" if q.get("kind") == "run" else f"eval {seq}"); return
    print("")


def check_pull(before, after):
    """R-012: a pull that changes control_rules.py is used only if control/CHANGES.md names its RULES_VERSION; else the
    clone is reset to `before` and the refusal is recorded (and reported in the heartbeat)."""
    if not before or before == after:
        return True
    ch = subprocess.run(["git", "-C", RUNBOOK, "diff", "--name-only", before, after, "--", "scripts/control_rules.py"], capture_output=True, text=True).stdout.strip()
    if not ch:
        return True
    m = re.search(r'^RULES_VERSION = "([^"]+)"', _read(os.path.join(RUNBOOK, "scripts", "control_rules.py")), re.M)
    ver = m.group(1) if m else "?"
    if re.search(rf"^\S*\s*rules {re.escape(ver)}\b", _read(os.path.join(RUNBOOK, "control", "CHANGES.md")), re.M):
        state(rules_refused=None); return True
    subprocess.run(["git", "-C", RUNBOOK, "reset", "-q", "--hard", before])
    state(rules_refused=f"{utc()} {before}->{after}: control_rules.py changed to {ver} without a line 'rules {ver}' in control/CHANGES.md")
    event("control", "rules_refused", f"{before}->{after} {ver}")
    return False


def v_update(a, seq, flags=None, entry=None):
    before = _git_head()
    r = subprocess.run(["git", "-C", RUNBOOK, "pull", "-q", "--ff-only"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return "failed", f"git pull: {r.stderr.strip()[:200]}", {"before": before}
    after = _git_head()
    if not check_pull(before, after):
        return "rejected", "pulled rules not signed off in control/CHANGES.md; reset to " + str(before), {"before": before, "after": after}
    return "ok", f"{before} -> {after}; run_tasks restarts itself if run_tasks.sh or env.sh changed", {"before": before, "after": after}


def _under_roots(path):
    roots = [os.path.realpath(ROOT)] + ([os.path.realpath(os.environ["CKPT_ROOT"])] if os.environ.get("CKPT_ROOT") else [])
    real = os.path.realpath(path if os.path.isabs(path) else os.path.join(ROOT, path))
    for b in roots:
        if real == b or real.startswith(b + os.sep):
            return real, b
    return None, None


def v_upload(a, seq, flags=None, entry=None):
    real, base = _under_roots(a[0])
    if not real or not os.path.exists(real):
        return "rejected", "path missing or outside $OPD_ROOT / $CKPT_ROOT", {}
    files = [real] if os.path.isfile(real) else [os.path.join(dp, f) for dp, _, fs in os.walk(real) for f in fs]
    if any(SECRET_NAME.search(os.path.basename(f)) for f in files):
        return "rejected", "contains a secret-like file name", {}
    size = sum(os.path.getsize(f) for f in files)
    if size > R.MAX_UPLOAD_BYTES:
        return "rejected", f"{size / 2**30:.1f} GB > 20 GB cap", {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_upload", str(seq), real, os.path.relpath(real, base)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", f"{size / 2**30:.2f} GB queued (one upload at a time; waits for a 120 s checkpoint quiet period)", {"bytes": size}


def do_upload(seq, real, rel):
    import fcntl
    lock = open(_p("upload.lock"), "w"); fcntl.flock(lock, fcntl.LOCK_EX)   # one upload at a time per node (R-012)
    os.makedirs(_p("jobs"), exist_ok=True); open(_p("jobs", f"{seq}.pid"), "w").write(str(os.getpid()))
    t_end = time.time() + 1800
    while time.time() < t_end:   # never while a task is writing a checkpoint
        newest = max([os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(ROOT, "runs"), followlinks=True)
                      if "/ckpt" in dp for f in fs] or [0])
        if time.time() - newest > 120:
            break
        time.sleep(30)
    t0 = time.time(); dest = f"nodes/{node_id()}/{rel}"
    ok = remote_put(real, dest)
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(real) for f in fs) if os.path.isdir(real) else os.path.getsize(real)
    receipt(seq, None, "ok" if ok else "failed", f"{'uploaded' if ok else 'upload failed'}: {dest}", outputs=[dest] if ok else [],
            bytes=size, seconds=round(time.time() - t0, 1))


def v_cancel(a, seq, flags=None, entry=None):
    target = int(a[0]); os.makedirs(_p("cancelled"), exist_ok=True)
    open(_p("cancelled", str(target)), "w").write(f"by seq {seq} {utc()}\n")
    pidf = _p("jobs", f"{target}.pid")
    if os.path.exists(pidf):
        try:
            os.killpg(os.getpgid(int(_read(pidf).strip())), signal.SIGTERM)
            receipt(target, None, "cancelled", f"stopped by seq {seq}")
            return "ok", f"running job of seq {target} stopped", {}
        except (ProcessLookupError, ValueError):
            pass
    return "ok", f"seq {target} will not run (or had already finished)", {}



# ---------- v2.5: heartbeat on demand, alerts (disk, stalled task, missing uploader) ----------
STALL_S = int(os.environ.get("STALL_MIN", "30")) * 60


def v_heartbeat(a, seq, flags=None, entry=None):
    open(_p("heartbeat_now"), "w").write(f"seq {seq} {utc()}\n")   # heartbeat.sh checks it every 10 s
    return "ok", "heartbeat requested: status/<host>.json, procs.json, receipts within ~10 s", {}


def alerts(status_path):
    """Called by heartbeat.sh every round: one event when a disk alert switches on, when a running task's log has not
    changed for STALL_MIN minutes (stalled; again only after it moved), and when a running task's upload_sync is gone.
    Returns the list of active alerts (also kept in state.json for status())."""
    try:
        st = json.load(open(status_path))
    except Exception:
        st = {}
    prev = set(json.loads(_read(_p("state.json"), "{}") or "{}").get("alerts", []))
    now = set()
    for k in ("disk_alert", "ckpt_disk_alert"):
        if st.get(k):
            now.add(k)
    for r in st.get("ckpt_roots") or []:   # one alert per candidate root (v2.5)
        if r.get("alert") and r.get("writable"):
            now.add(f"ckpt_root_alert:{r.get('base')}")
    ups = subprocess.run(["pgrep", "-af", "scripts/upload_syn[c].sh"], capture_output=True, text=True).stdout
    for pid in running_pids():
        m = re.search(r"task\.sh\s+\d+\s+(\S+)", _read(f"/proc/{pid}/cmdline").replace("\0", " ") + " ")
        if not m:
            continue
        t = m.group(1); lg = os.path.join(ROOT, "logs", f"{t}.log")
        if os.path.exists(lg) and time.time() - os.path.getmtime(lg) > STALL_S:
            now.add(f"stalled:{t}")
        if re.match(r"(E10b|E12b)-", t) and f"/runs/{t}" not in ups:
            now.add(f"upload_sync_missing:{t}")
    for a in sorted(now - prev):
        event(a.split(":", 1)[1] if ":" in a else "node", a.split(":", 1)[0], json.dumps({k: st.get(k) for k in ("disk_free_gb", "ckpt_disk_free_gb")}))
    for a in sorted(prev - now):
        event(a.split(":", 1)[1] if ":" in a else "node", a.split(":", 1)[0] + "_cleared")
    state(alerts=sorted(now))
    return sorted(now)


# ---------- v2.5: checkpoint roots (same tree under every candidate: <base>/opd-ckpt/<task>/ckpt/global_step_<N>/) ----------
def _paths_env():
    out = {}
    for line in _read(os.path.join(ROOT, "paths.env")).splitlines():
        m = re.match(r"^(?:export )?([A-Z_]+)=(.*)$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _write_paths_env(d):
    tmp = os.path.join(ROOT, "paths.env.tmp")
    open(tmp, "w").write("".join(f"{k}={v}\n" for k, v in sorted(d.items())))
    os.replace(tmp, os.path.join(ROOT, "paths.env"))


def candidates():
    """Candidate bases as env.sh computes them (its ckpt_candidates function), in order."""
    r = subprocess.run(["bash", "-c", f"source {RUNBOOK}/scripts/env.sh > /dev/null 2>&1; echo $CKPT_CANDIDATES"], capture_output=True, text=True)
    return [c for c in r.stdout.strip().split(":") if c]


def _as_base(v):
    v = v.rstrip("/")
    return v[: -len("/opd-ckpt")] if v.endswith("/opd-ckpt") else v


def set_paths(k, v):
    """set CKPT_ROOT / CKPT_CANDIDATES+ / ARCHIVE_ROOT; returns (status, message)."""
    pe = _paths_env()
    if k == "CKPT_CANDIDATES+":
        if not (os.path.isabs(v) and os.path.isdir(v) and os.access(v, os.W_OK)):
            return "rejected", "a new candidate must be an existing writable absolute directory"
        extra = [x for x in pe.get("CKPT_CANDIDATES_EXTRA", "").split(":") if x]
        if v.rstrip("/") not in extra:
            extra.append(v.rstrip("/"))
        pe["CKPT_CANDIDATES_EXTRA"] = ":".join(extra); _write_paths_env(pe)
        return "ok", f"candidate added: {v} (tree {v.rstrip('/')}/opd-ckpt/<task>/ckpt/...)"
    cands = candidates()
    if v == "" and k == "ARCHIVE_ROOT":
        pe.pop("ARCHIVE_ROOT", None); _write_paths_env(pe); return "ok", "no bf16 archive (default)"
    base = _as_base(v)
    if base not in cands:
        return "rejected", f"{v} is not a candidate; candidates: {cands} (add one with set CKPT_CANDIDATES+=<dir>)"
    if not os.access(base, os.W_OK):
        return "rejected", f"{base} is not writable"
    if k == "CKPT_ROOT":
        os.makedirs(os.path.join(base, "opd-ckpt"), exist_ok=True)
        pe["CKPT_ROOT"] = os.path.join(base, "opd-ckpt"); _write_paths_env(pe)
        return "ok", f"CKPT_ROOT={pe['CKPT_ROOT']} for tasks started from now on (running tasks keep their checkpoints)"
    pe["ARCHIVE_ROOT"] = base; _write_paths_env(pe)
    return "ok", f"bf16 archive copies go to {base}/opd-archive/<task>/global_step_<N>/"


def roots_report():
    """Per candidate: mount point, free GB, writable, alert, and the task directories under <base>/opd-ckpt with sizes."""
    cur = os.environ.get("CKPT_ROOT") or _paths_env().get("CKPT_ROOT")
    out = []
    for b in candidates():
        r = {"base": b, "ckpt_root": os.path.join(b, "opd-ckpt"), "current": os.path.join(b, "opd-ckpt") == cur}
        try:
            st = os.statvfs(b); r["free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
            r["alert"] = r["free_gb"] < float(os.environ.get("CKPT_ALERT_GB", "100"))
            r["mount"] = subprocess.run(["df", "-P", b], capture_output=True, text=True).stdout.splitlines()[-1].split()[-1]
        except (OSError, IndexError):
            r["free_gb"] = None; r["alert"] = True
        r["writable"] = os.access(b, os.W_OK)
        d = os.path.join(b, "opd-ckpt"); tasks = {}
        for t in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            sz = 0
            for dp, _, fs in os.walk(os.path.join(d, t)):
                for f in fs:
                    try:
                        sz += os.path.getsize(os.path.join(dp, f))
                    except OSError:
                        pass
            tasks[t] = round(sz / 1e9, 2)
        r["tasks_gb"] = tasks; out.append(r)
    return out


def v_move_ckpt(a, seq, flags=None, entry=None):
    t, target = a
    if running_pids(t):
        return "rejected", f"{t} is running (kill or wait for it first)", {}
    link = os.path.join(ROOT, "runs", t, "ckpt")
    if not os.path.exists(link):
        return "rejected", f"{t} has no checkpoints", {}
    base = _as_base(target)
    if base not in candidates():
        return "rejected", f"{target} is not a candidate", {}
    src = os.path.realpath(link); dst = os.path.join(base, "opd-ckpt", t, "ckpt")
    if src == os.path.realpath(dst):
        return "ok", f"{t} already under {base}", {}
    if os.path.exists(dst) and os.listdir(dst):
        return "rejected", f"{dst} exists and is not empty", {}
    open(os.path.join(ROOT, "runs", t, ".paused"), "w").write(f"seq {seq} {utc()} (move-ckpt)\n")
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_move", str(seq), t, src, dst],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", f"moving {src} -> {dst} (paused meanwhile; resume afterwards)", {}


def do_move(seq, t, src, dst):
    os.makedirs(_p("jobs"), exist_ok=True); open(_p("jobs", f"{seq}.pid"), "w").write(str(os.getpid()))
    t0 = time.time(); os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.rmdir(dst)   # empty (checked by the verb)
    shutil.copytree(src, dst, symlinks=True)
    size = lambda d: sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs if not os.path.islink(os.path.join(dp, f)))
    a, b = size(src), size(dst)
    if a != b:
        receipt(seq, None, "failed", f"copy size mismatch {a} != {b}; source kept, copy left in {dst}"); return
    link = os.path.join(ROOT, "runs", t, "ckpt")
    tmp = link + ".new"
    if os.path.lexists(tmp):
        os.remove(tmp)
    os.symlink(dst, tmp); os.replace(tmp, link)          # the link points to the new root before the old copy goes
    shutil.rmtree(src)
    receipt(seq, None, "ok", f"{t}: {a / 1e9:.1f} GB moved to {dst}; runs/{t}/ckpt re-pointed (task still paused)",
            bytes=a, seconds=round(time.time() - t0, 1))

def _v24(fn):
    return lambda a, seq, flags=None, entry=None: fn(a, seq)


VERBS = {"pause": _v24(v_pause), "resume": _v24(v_resume), "kill": _v24(v_kill), "prio": _v24(v_prio),
         "restore": _v24(v_restore), "rerun_verify": _v24(v_rerun_verify), "setenv": v_set, "set": v_set,
         "snapshot": lambda a, seq, flags=None, entry=None: v_snapshot0(seq) if not a else v_snapshot(a, seq),
         "ps": v_ps, "logs": v_logs, "run": v_run, "eval": v_eval, "update": v_update, "upload": v_upload,
         "signal": v_signal, "kill-pgid": v_killpgid, "cancel": v_cancel, "heartbeat": v_heartbeat, "move-ckpt": v_move_ckpt}
assert set(VERBS) == set(R.VERBS), set(VERBS) ^ set(R.VERBS)
TASK_VERBS = R.TASK_VERBS


def run_one(cmd, seq, flags=None, entry=None):
    entry = dict(entry or {"reason": "direct call"}, cmd=cmd)
    try:
        verb, args, f2 = R.parse(entry)
    except R.Reject as e:
        return "rejected", str(e), {}
    try:
        return VERBS[verb](args, seq, dict(f2, **(flags or {})), entry)
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
        state(last_apply_utc=utc(), pending=0); return
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
    n_cmd = n_sig = n_kpg = 0; cmds = doc.get("commands") or []
    for i, c in enumerate(cmds):
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
        if n_cmd >= R.MAX_PER_PULL:
            break   # the rest waits for the next pull, in order
        try:
            verb, args, flags = R.parse(c)
            deadline = R.expiry(c, verb)
        except R.Reject as e:
            status, msg, extra = "rejected", str(e), {}
        else:
            if os.path.exists(_p("cancelled", str(seq))):
                status, msg, extra = "cancelled", "cancelled before it ran", {}
            elif time.time() > deadline:
                status, msg, extra = "expired", f"expired {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(deadline)) if deadline else '(no utc / expires)'}", {}
            elif verb == "signal" and n_sig >= R.MAX_SIGNALS_PER_PULL:
                break
            elif verb == "kill-pgid" and not flags["force"] and n_kpg >= R.MAX_KILLPGID_NOFORCE_PER_PULL:
                break
            else:
                n_sig += verb == "signal"; n_kpg += verb == "kill-pgid" and not flags["force"]
                status, msg, extra = run_one(c.get("cmd"), seq, flags, c)
        n_cmd += 1
        extra = json.loads(json.dumps(extra, default=str))
        log({"seq": seq, "cmd": c.get("cmd"), "status": status, "message": msg, "utc": utc(), **extra})
        receipt(seq, c, status, msg, **extra)
        print(f"CONTROL seq {seq} {c.get('cmd')!r}: {status}: {msg}")
        applied = seq; open(_p("applied_seq.tmp"), "w").write(str(seq)); os.replace(_p("applied_seq.tmp"), _p("applied_seq"))
    pending = sum(1 for c in cmds if isinstance(c, dict) and isinstance(c.get("seq"), int) and c["seq"] > applied)
    state(last_apply_utc=utc(), pending=pending)


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
    lp = json.loads(_read(_p("last_pull.json"), "null") or "null")
    return {"node": node_id(), "control_version": R.RULES_VERSION, "applied_seq": int((_read(_p("applied_seq"), "0").strip() or 0)),
            "pending": s.get("pending"), "last_pull_utc": (lp or {}).get("utc"), "last_apply_utc": s.get("last_apply_utc"),
            "recent": [json.loads(x) for x in lines if x.strip()], "origin_mismatch": s.get("origin_mismatch"),
            "node_mismatch": s.get("node_mismatch"), "file_error": s.get("file_error"), "rules_refused": s.get("rules_refused"),
            "alerts": s.get("alerts", []),
            "control_env": {k: redact(f"{k}={v}").split("=", 1)[1] for k, v in read_env().items()},
            "last_pull": lp, "control_env_suspect": suspects(), "verify_running": os.path.exists(_p("verify.running")),
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
    elif cmd == "procs":
        t = publish_procs(upload="--upload" in sys.argv); print(f"{len(t['rows'])} rows")
    elif cmd == "event":
        event(sys.argv[2], sys.argv[3], " ".join(sys.argv[4:]))
    elif cmd == "next_request":
        next_request()
    elif cmd == "check_pull":
        sys.exit(0 if check_pull(sys.argv[2], sys.argv[3]) else 1)
    elif cmd == "do_upload":
        do_upload(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "do_move":
        do_move(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "roots":
        print(json.dumps(roots_report(), indent=1))
    elif cmd == "alerts":
        print(json.dumps(alerts(sys.argv[2])))
    elif cmd == "pulse":   # heartbeat.sh every 60 s (v2.5.1)
        r = pulse(sys.argv[2]); print("PULSE ok" if r else "PULSE refused")
    elif cmd == "rcpt":   # run_tasks / run_adhoc_eval: update a receipt (started, done, failed)
        receipt(int(sys.argv[2]), None, sys.argv[3], " ".join(sys.argv[4:]))
    else:
        print(__doc__); sys.exit(2)
