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


ORPHAN_CMD = re.compile(r"VLLM::EngineCore|VLLM::|multiprocessing\.spawn|^ray::")


def _gpu_used_mib(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out.splitlines()[0]) if out else None
    except Exception:
        return None


def _read_lossy(path):
    try:
        return open(path, "rb").read().decode("utf-8", "replace")
    except OSError:
        return ""


def _our_roots():
    """Real paths (with a trailing /) of OPD_ROOT and its venvs; venv may be a link to another disk (LINK_VENV)."""
    out = set()
    for p in (ROOT, os.path.join(ROOT, "venv"), os.path.join(ROOT, "venv-fa")):
        if os.path.exists(p):
            out.add(os.path.realpath(p).rstrip("/") + "/")
    return out


def reap(gpu, task=None, wait_s=120, proc_root="/proc", grace_s=10):
    """v2.5.4b: our orphaned GPU processes on <gpu>, after an evaluation or a task has ended. A process qualifies only if
    all three hold: its parent has died (PPid 1); it is a vLLM engine or worker (VLLM::EngineCore, multiprocessing.spawn)
    or a Ray worker (ray::...); its environment shows it was started by this node's tasks for this GPU (OPD_ROOT equal to
    ours, and CUDA_VISIBLE_DEVICES=<gpu> or RAY_TMPDIR=/tmp/ray_gpu<gpu>; v2.5.4c: where the process title has overwritten
    the environment, it maps a file under our OPD_ROOT or venv and nvidia-smi lists it on <gpu>). Such a process holds the GPU after gen_eval or
    a training exits (B300, 2026-10-17: ~250 GB each, the next vLLM engine could not start). TERM, KILL after grace_s; an
    event per process (pid, gpu, memory before); then wait until the GPU has <= 2 GB in use (at most wait_s). Returns the
    list of reaped pids."""
    found, roots, apps = [], None, None
    for d in os.listdir(proc_root):
        if not d.isdigit():
            continue
        pid = int(d)
        cmd = _read(os.path.join(proc_root, d, "cmdline")).replace("\0", " ").strip()
        if not cmd or not ORPHAN_CMD.search(cmd):
            continue
        m = re.search(r"^PPid:\s+(\d+)", _read(os.path.join(proc_root, d, "status")), re.M)
        if not m or int(m.group(1)) != 1:
            continue
        env = {}
        for kv in _read_lossy(os.path.join(proc_root, d, "environ")).split("\0"):
            k, _, v = kv.partition("=")
            if k:
                env[k] = v
        if "OPD_ROOT" in env:   # environment intact (SPT_NOENV=1, v2.5.4c): ours and this GPU by the environment
            if env["OPD_ROOT"] != ROOT:
                continue
            if env.get("CUDA_VISIBLE_DEVICES") != str(gpu) and env.get("RAY_TMPDIR") != f"/tmp/ray_gpu{gpu}":
                continue
        else:   # v2.5.4c: setproctitle (vLLM, Ray) wrote the title over the environment area (B300 2026-10-17: environ
            # read "VLLM::EngineCore"): ours if it maps a library from our venv / OPD_ROOT, this GPU by nvidia-smi's pid map
            if roots is None:
                roots = _our_roots()
            maps = _read_lossy(os.path.join(proc_root, d, "maps"))
            if not any(r in maps for r in roots):
                continue
            if apps is None:
                apps = _gpu_apps()
            if (apps.get(pid) or (None,))[0] != gpu:
                continue
        found.append((pid, cmd[:60]))
    before = _gpu_used_mib(gpu) if found else None
    for pid, _ in found:
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    t0 = time.time()
    while found and time.time() - t0 < grace_s and any(_alive(p) for p, _ in found):
        time.sleep(1)
    for pid, cmd in found:
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        event(task or f"gpu{gpu}", "orphan_reaped", f"pid={pid} gpu={gpu} gpu_mem_before_mib={before} cmd={cmd}")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        u = _gpu_used_mib(gpu)
        if u is None or u <= 2048:
            break
        time.sleep(5)
    return [p for p, _ in found]


def v_reap(a, seq, flags=None, entry=None):
    """reap <gpu> (v2.5.4b): our orphaned vLLM / Ray processes on that GPU, by the three conditions of reap(); nothing else."""
    g = int(a[0])
    n = len([l for l in subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines() if l.startswith("GPU ")])
    if n and g >= n:
        return "rejected", f"gpu {g} does not exist (this node has {n})", {}
    live = sorted({r.get("task") for r in proc_table() if r.get("gpu") == g and r.get("ours")
                   and re.match(r"(E10b-|E10c-|E10d-|E12b-|SFT-|E03-|E02-)", str(r.get("task") or ""))})
    if live:   # Ray workers are reaped only after their task ended (the design of reap); never next to a live training
        return "rejected", f"a training task is running on gpu {g} ({', '.join(live)}); reap runs after it ends", {}
    pids = reap(g, f"gpu{g}", wait_s=60)
    return "ok", (f"reaped {len(pids)} orphaned process(es) on gpu {a[0]}: {pids}" if pids else f"no orphaned process of ours on gpu {a[0]}"), {}


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
                       "attempts_failed": _num(v.get("attempts_failed")),
                       # v2.5.5: the GPU it holds, train / sft / eval / probe / other, its start (for grouping by GPU)
                       "gpu": v.get("gpu") if isinstance(v.get("gpu"), int) and 0 <= v.get("gpu") <= 15 else None,
                       "kind": v.get("kind") if v.get("kind") in ("train", "sft", "eval", "probe", "other") else None,
                       "started_utc": v.get("started_utc") if isinstance(v.get("started_utc"), str)
                       and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", v.get("started_utc")) else None}
    ctl = st.get("control") if isinstance(st.get("control"), dict) else {}
    alerts = []
    for a in ctl.get("alerts") or []:
        if isinstance(a, str):
            code, _, arg = a.partition(":")
            if re.fullmatch(r"[a-z_]+", code):
                alerts.append(f"{code}:{arg}" if arg and R.TASK_RE.match(arg) and "/" not in arg else code)
    # v2.5.2: a GPU close to full (default > 240 GB used, GPU_MEM_ALERT_MIB) is flagged: on 2026-10-17 E10b-P4lowlr's GPU
    # sat at ~250 GB for steps before its out-of-memory crash at step 97
    lim = float(os.environ.get("GPU_MEM_ALERT_MIB", 245760))
    evals, orphans = set(), set()   # v2.5.4b: evaluation GPUs are not flagged (their vLLM preallocates ~0.9 of the GPU)
    try:
        for r in json.loads(_read(_p("procs.json"), "{}") or "{}").get("rows", []):
            if r.get("gpu") is None:
                continue
            if re.match(r"(E11b-|EVAL-|adhoc_eval_|FA-PROBE|SPEED-PROBE|PROBE-4B|PROBE-TERM|PROBE-TOP10|FA-SPEED-)", str(r.get("task") or "")):
                evals.add(r["gpu"])
            if not r.get("ours") and (r.get("gpu_mem_mb") or 0) > 2048:   # a process outside our trees holding memory
                orphans.add(r["gpu"])
    except Exception:
        pass
    alerts += [f"gpu_orphan:{g}" for g in sorted(orphans)]
    alerts += [f"gpu_mem_high:{g['idx']}" for g in gpus if g["mem_used"] is not None and g["mem_used"] > lim and g["idx"] not in evals]
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
    r = subprocess.run(["bash", "-c", f'T={t}; ARMS="P2|P3|P4lowlr|P5|P6|P6r|P2noEOS|Pstop|P6rand|P6top5|P4EOSpen|P5top10|P5rand10m"; '
                        'case $T in E10b-*) [[ $T =~ ^E10b-($ARMS)(-s(43|44))?$ ]];; E10c-*) [[ $T =~ ^E10c-($ARMS)-(fix4|dyn16k|dyn6144)(-s43)?$ ]];; E10d-*) [[ $T =~ ^E10d-(P2|P5)$ ]];; E11b-*) [[ $T =~ ^E11b-(P0|($ARMS)(-s(43|44))?)$ ]];; '
                        'E12b-*) [[ $T =~ ^E12b-(P0|P1|$ARMS|P5-s43)$ ]];; EVAL-*) [[ $T =~ ^EVAL-[A-Za-z0-9._-]+$ ]];; SFT-P1|FA-PROBE|SPEED-PROBE|PROBE-4B|PROBE-TERM|PROBE-TERM2|PROBE-TERM3|PROBE-TOP10|FA-SPEED-17|FA-SPEED-4B|FA-SPEED-E12|FA-SPEED-TWIN|E03-MI-l0.[0357]|E03-MI-l1.0|E02-opsa17-c0.5|E02-opsa17-c1.5|E02-opsa4-c-1|'
                        'E02-opsa4-c0.5|E02-opsa4-c1.5) true;; *) false;; esac'])
    if r.returncode != 0:
        return "rejected", f"{t} is not accepted by task.sh", {}
    if running_pids(t):
        return "rejected", f"{t} is already running", {}
    if os.path.exists(os.path.join(ROOT, "runs", t, ".done")):
        return "rejected", f"{t} is already done", {}
    _request("run", seq, {"task": t})
    return "accepted", f"{t} starts on the next GPU that is free after the queue's own tasks (deps must be done)", {}


def v_rerun(a, seq, flags=None, entry=None):
    """rerun <task> (v2.5.5, v2.5.6b): a finished probe or queue evaluation (FA-PROBE, EVAL-*) runs once more, or any
    task that failed for good (3 attempts) or was skipped because a dependency failed gets a new start (B300 18:52:
    SFT-P1 failed 3 times on a missing patch, E12b-P1 was skipped). A finished training is never rerun. .done,
    report.json, .failed and .skipped are kept as *.prev_<seq>; the queue starts it on the next free GPU."""
    t = a[0]
    if running_pids(t):
        return "rejected", f"{t} is running", {}
    d = os.path.join(ROOT, "runs", t)
    done, failed = os.path.exists(os.path.join(d, ".done")), os.path.exists(os.path.join(d, ".failed"))
    skipped = os.path.exists(os.path.join(d, ".skipped"))
    if done and not re.fullmatch(r"FA-PROBE|SPEED-PROBE|PROBE-4B|PROBE-TERM[23]?|PROBE-TOP10|FA-SPEED-(17|4B|E12|TWIN)|EVAL-[A-Za-z0-9._-]+", t):
        return "rejected", f"{t} is done (only FA-PROBE, SPEED-PROBE and EVAL tasks are rerun after finishing)", {}
    if not (done or failed or skipped):
        return "rejected", f"{t} is neither done, failed nor skipped", {}
    kept = []
    for f in (".done", "report.json", ".failed", ".skipped"):
        if os.path.exists(os.path.join(d, f)):
            os.replace(os.path.join(d, f), os.path.join(d, f"{f}.prev_{seq}")); kept.append(f)
    return "ok", f"{t} will run again ({', '.join(kept)} kept as *.prev_{seq})", {}


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


def v_republish(a, seq, flags=None, entry=None):
    """republish <task> (v2.5.6): the text files this task already has in the public repository are uploaded again from
    this node's copies, through the scrubbing upload of v2.5.5c, over the versions that carried paths (2026-10-18). Only
    files already published and only text files (remote.TEXT_EXT); checkpoints and weights are never touched. v2.5.6k: a
    model directory's text files (tokenizer, configs) come from ckpt/global_step_N/actor/huggingface_bf16 or huggingface
    and go up unchanged (remote.scrub_model_file), repairing the vocabularies the scrub broke; this node's fetched copies
    of them are deleted, so the next evaluation fetches the repaired files."""
    t = a[0]
    if not os.path.isdir(os.path.join(ROOT, "runs", t)):
        return "rejected", f"no runs/{t} on this node", {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_republish", str(seq), t],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", f"runs/{t}: published text files are uploaded again, scrubbed", {}


def do_republish(seq, t):
    sys.path.insert(0, os.path.join(RUNBOOK, "scripts"))
    import remote
    from huggingface_hub import HfApi
    pub = remote.repos()[0]
    try:
        names = [f for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(pub) if f.startswith(f"runs/{t}/")]
    except Exception as e:
        receipt(seq, None, "failed", f"listing failed: {type(e).__name__}"); return
    stage = _p(f"req_{seq}"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    done, missing = [], []
    for f in names:
        rel = f[len(f"runs/{t}/"):]
        src = os.path.join(ROOT, "runs", t, rel)
        m = re.match(r"(global_step_\d+)/actor/huggingface/(.+)$", rel)
        if m and not os.path.isfile(src):   # v2.5.6k: a model directory's text files from the checkpoint (bf16 copy, else the HF
            for sub in ("huggingface_bf16", "huggingface"):   # directory, whose tokenizer / config stay after the weights go)
                q = os.path.join(ROOT, "runs", t, "ckpt", m.group(1), "actor", sub, m.group(2))
                if os.path.isfile(q):
                    src = q; break
        if "/ckpt/" in f"/{rel}" or rel.startswith(".cache/") or not remote.is_text(os.path.basename(rel), os.path.getsize(src) if os.path.isfile(src) else 0):
            continue
        if not os.path.isfile(src):
            missing.append(rel); continue
        os.makedirs(os.path.dirname(os.path.join(stage, rel)), exist_ok=True); shutil.copy2(src, os.path.join(stage, rel)); done.append(rel)
    ok = remote_put(stage, f"runs/{t}") if done else True
    shutil.rmtree(stage, ignore_errors=True)
    ck = os.environ.get("CKPT_ROOT") or _paths_env().get("CKPT_ROOT")
    if ok and ck:   # v2.5.6k: copies of this task's models fetched from the public repository (result:...) are fetched again
        for rel in done:
            m = re.match(r"(global_step_\d+/actor/huggingface)/", rel)
            if m:
                d = os.path.join(ck, "adhoc_models", "result", "runs", t, m.group(1))
                for x in (os.path.join(d, os.path.basename(rel)), os.path.join(d, ".fetch_ok")):
                    if os.path.isfile(x):
                        os.remove(x)
    receipt(seq, None, "ok" if ok else "failed", f"runs/{t}: {len(done)} text files {'republished scrubbed' if ok else 'not uploaded'}"
            + (f"; {len(missing)} not on this node" if missing else ""), files=done[:50], missing=missing[:50])
    event(t, "republished" if ok else "republish_failed", f"{len(done)} files")


OLD_PUBLIC = os.environ.get("OPD_OLD_PUBLIC_REPO", "yang31210999/opd-lab-b300")   # private since 2026-10-18 (history leak)


def v_publish(a, seq, flags=None, entry=None):
    """publish <task> (v2.5.6j): the files this task had in the old public repository (now private) go to the current
    public repository from this node's copies, through the scrubbing upload: text from runs/<task>/, weights
    (global_step_N/actor/huggingface/*) from the local bf16 copy ckpt/global_step_N/actor/huggingface_bf16/. Resume
    states and fp32 files are never uploaded; files missing on this node are listed in the receipt."""
    t = a[0]
    if not os.path.isdir(os.path.join(ROOT, "runs", t)):
        return "rejected", f"no runs/{t} on this node", {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_publish", str(seq), t],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", f"runs/{t}: old public files are re-published from this node", {}


def do_publish(seq, t):
    sys.path.insert(0, os.path.join(RUNBOOK, "scripts"))
    import remote
    from huggingface_hub import HfApi
    try:
        names = [f for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(OLD_PUBLIC) if f.startswith(f"runs/{t}/")]
    except Exception as e:
        receipt(seq, None, "failed", f"listing {OLD_PUBLIC} failed: {type(e).__name__}"); return
    stage = _p(f"req_{seq}"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    done, missing, nbytes = [], [], 0
    for f in names:
        rel = f[len(f"runs/{t}/"):]
        m = re.match(r"(global_step_\d+)/actor/huggingface/(.+)$", rel)
        src = os.path.join(ROOT, "runs", t, "ckpt", m.group(1), "actor", "huggingface_bf16", m.group(2)) if m else os.path.join(ROOT, "runs", t, rel)
        if "/ckpt/" in f"/{rel}" or rel.endswith((".pt", ".pth")) or not os.path.isfile(src):
            missing.append(rel); continue
        dst = os.path.join(stage, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            os.link(src, dst)          # weights: no copy on the same disk
        except OSError:
            shutil.copy2(src, dst)
        done.append(rel); nbytes += os.path.getsize(src)
    ok = remote_put(stage, f"runs/{t}") if done else True
    shutil.rmtree(stage, ignore_errors=True)
    receipt(seq, None, "ok" if ok else "failed", f"runs/{t}: {len(done)} files, {nbytes / 1e9:.2f} GB {'published' if ok else 'not uploaded'}"
            + (f"; {len(missing)} not on this node" if missing else ""), files=done[:60], missing=missing[:60])
    event(t, "published" if ok else "publish_failed", f"{len(done)} files {nbytes / 1e9:.2f} GB")


def v_leakscan(a, seq, flags=None, entry=None):
    """leakscan (v2.5.6j): scan the current public repository's non-weight files on this node (nothing is downloaded
    anywhere else) for this machine's user / host / path tokens, private or own IPv4 addresses (not in generation outputs)
    and HF tokens; the receipt carries counts per file only, never content."""
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_leakscan", str(seq)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", "public repository scan started on this node", {}


def leak_counts(text, toks, ips, generation, vocab=False):
    """Hits in one text: forbidden tokens, HF / ModelScope tokens, and (outside generation outputs) addresses. vocab
    (v2.5.6k): a model's vocabulary file, uploaded unchanged; its words are the base model's tokens, not counted."""
    sys.path.insert(0, os.path.join(RUNBOOK, "scripts"))
    import remote
    n = 0
    if toks and not vocab:
        n += len(re.findall(r"(?i)(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, sorted(toks, key=len, reverse=True))) + r")(?![A-Za-z0-9_])", text))
    n += len(re.findall(r"hf_[A-Za-z0-9]{30,}|ms-[A-Za-z0-9]{8}-[A-Za-z0-9-]{20,}", text))
    if not generation:
        n += len(remote.PRIVATE_IP.findall(text)) + sum(text.count(a) for a in ips)
    return n


def do_leakscan(seq):
    sys.path.insert(0, os.path.join(RUNBOOK, "scripts"))
    import tempfile
    import remote
    from huggingface_hub import HfApi, hf_hub_download
    pub = remote.repos()[0]
    try:
        info = HfApi().repo_info(pub)
        names = [s.rfilename for s in info.siblings if not s.rfilename.endswith((".safetensors", ".pt", ".bin", ".parquet", ".pth", ".gguf"))]
    except Exception as e:
        receipt(seq, None, "failed", f"listing failed: {type(e).__name__}"); return
    toks, ips, hits = _forbidden_tokens(), remote.own_ips(), {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in names:
            try:
                t = open(hf_hub_download(pub, f, revision=info.sha, cache_dir=tmp), errors="ignore").read()
            except Exception:
                hits[f] = -1; continue
            n = leak_counts(t, toks, ips, f.endswith((".jsonl", ".part")),
                            vocab=os.path.basename(f) in remote.VOCAB_FILES and remote.is_model_file("", f))
            if n:
                hits[f] = n
    real = {f: n for f, n in hits.items() if n > 0}
    receipt(seq, None, "ok", f"{pub} @ {info.sha[:12]}: {len(names)} text files scanned, {len(real)} with hits"
            + (f", {sum(1 for n in hits.values() if n < 0)} unreadable" if any(n < 0 for n in hits.values()) else ""),
            revision=info.sha, scanned=len(names), hits=dict(list(real.items())[:60]))
    event("node", "leakscan", f"{len(real)} files with hits")


def v_fa_build(a, seq, flags=None, entry=None):
    """fa-build (v2.5.6m, rules 2.5.10): scripts/fa_build.sh in the background, without a GPU (nice 19): flash-attn
    2.8.3.post2 built from its PyPI source into venv-fa (FA-PROBE's release wheel needs GLIBC_2.32). The receipt is
    written when the build ends; `rerun FA-PROBE` afterwards runs the probe's varlen and smoke steps."""
    if not os.path.isdir(os.path.join(ROOT, "venv-fa")):
        return "rejected", "no venv-fa on this node (FA-PROBE builds it)", {}
    if subprocess.run(["pgrep", "-f", "scripts/fa_build[.]sh"], capture_output=True).returncode == 0:
        return "rejected", "a build is already running", {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_fa_build", str(seq)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", "flash-attn source build started (no GPU, nice 19); the receipt follows when it ends", {}


def do_fa_build(seq):
    d = os.environ.get("FA_BUILD_DIR") or os.path.join(ROOT, "tmp", "fa_build"); os.makedirs(d, exist_ok=True)
    t0 = time.time()
    with open(os.path.join(d, "run.out"), "a") as out:
        rc = subprocess.run(["nice", "-n", "19", "bash", os.path.join(RUNBOOK, "scripts", "fa_build.sh")], stdin=subprocess.DEVNULL,
                            stdout=out, stderr=subprocess.STDOUT).returncode
    r = json.loads(_read(os.path.join(d, "result.json"), "{}") or "{}")
    ok = rc == 0 and r.get("result") == "PASS"
    receipt(seq, None, "ok" if ok else "failed", f"fa-build {r.get('result', 'no result')}: {r.get('reason', f'exit {rc}')}"[:600],
            seconds=round(time.time() - t0), **{k: r.get(k) for k in ("cuda_home", "wheel", "build_seconds", "after") if k in r})


def v_scrubdiff(a, seq, flags=None, entry=None):
    """scrubdiff [fix] (v2.5.6n, rules 2.5.11): every generation output in the public repository (runs/**/*.jsonl, .part)
    is compared line by line with this node's original; the receipt gives, per changed file, the changed lines, the
    replacements (<path> / <redacted> / <ip> / "~"), the sample ids and how many changed lines are sensitive (the
    original has a forbidden token, a path of this machine, or a path under $HOME), never any content. fix: files whose
    changed lines are all non-sensitive go up again unchanged, in one commit (reviewer 2026-10-18 05:2x UTC)."""
    if a and a[0] != "fix":
        return "rejected", "scrubdiff [fix]", {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_scrubdiff", str(seq)] + (["fix"] if a else []),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", "generation outputs of the public repository are compared with this node's originals" + ("; clean ones re-uploaded" if a else ""), {}


MARKS = ("<path>", "<redacted>", "<ip>", "~")


def scrub_changes(orig, pub, toks, roots):
    """Per-file comparison of an original text with its public copy: changed lines, replacement counts, sample ids,
    sensitive lines (see v_scrubdiff). No content is returned."""
    tokre = re.compile(r"(?i)(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, sorted(toks, key=len, reverse=True))) + r")(?![A-Za-z0-9_])") if toks else None

    def sensitive(line):
        if tokre and tokre.search(line):
            return True
        return any((p + "/") in line if name == "~" else p in line for p, name in roots)
    a, b = orig.splitlines(), pub.splitlines()
    out = {"lines": len(a), "changed": 0, "sensitive": 0, "kinds": {}, "ids": []}
    if len(a) != len(b):
        out["line_count_differs"] = [len(a), len(b)]
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            continue
        out["changed"] += 1; out["sensitive"] += sensitive(x)
        for m in MARKS:
            d = y.count(m) - x.count(m)
            if d > 0:
                out["kinds"][m] = out["kinds"].get(m, 0) + d
        try:
            r = json.loads(x)
            sid = [r[k] for k in ("pid", "sample") if k in r] or [r[k] for k in ("uid", "index", "idx") if k in r] or [f"line {i + 1}"]
        except (ValueError, TypeError, AttributeError):
            sid = [f"line {i + 1}"]
        if len(out["ids"]) < 20:
            out["ids"].append(sid[0] if len(sid) == 1 else sid)
    return out


def do_scrubdiff(seq, fix=False):
    sys.path.insert(0, os.path.join(RUNBOOK, "scripts"))
    import tempfile
    import remote
    from huggingface_hub import HfApi, hf_hub_download
    pub = remote.repos()[0]
    try:
        info = HfApi().repo_info(pub)
        names = [s.rfilename for s in info.siblings if s.rfilename.startswith("runs/") and s.rfilename.endswith((".jsonl", ".part"))]
    except Exception as e:
        receipt(seq, None, "failed", f"listing failed: {type(e).__name__}"); return
    toks, roots = _forbidden_tokens(), remote.scrub_roots()
    changed, missing, errors, clean = {}, [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for f in names:
            local = os.path.join(ROOT, f)
            if not os.path.isfile(local):
                missing.append(f); continue
            try:
                p = open(hf_hub_download(pub, f, revision=info.sha, cache_dir=tmp), encoding="utf-8", errors="replace").read()
            except Exception:
                errors.append(f); continue
            o = open(local, encoding="utf-8", errors="replace").read()
            if p == o:
                continue
            c = scrub_changes(o, p, toks, roots); changed[f] = c
            if c["changed"] and not c["sensitive"] and "line_count_differs" not in c and not leak_counts(o, toks, set(), True):
                clean.append(f)
    up = False
    if fix and clean:
        try:
            remote.put_public_unchanged([(os.path.join(ROOT, f), f) for f in clean], f"scrubdiff fix seq {seq}: {len(clean)} generation files unchanged")
            up = True
        except Exception as e:
            errors.append(f"upload {type(e).__name__}")
    tot = {m: sum(c["kinds"].get(m, 0) for c in changed.values()) for m in MARKS}
    receipt(seq, None, "ok" if not errors else "failed",
            f"{pub} @ {info.sha[:12]}: {len(names)} generation files, {len(changed)} differ from this node's originals "
            f"({sum(c['changed'] for c in changed.values())} lines, {sum(c['sensitive'] for c in changed.values())} sensitive); "
            f"{len(clean)} clean" + (f", re-uploaded unchanged" if up else "") + (f"; {len(missing)} not on this node" if missing else ""),
            revision=info.sha, totals=tot, files=dict(list(changed.items())[:60]), clean=clean[:100], missing=missing[:50], errors=errors[:20])
    event("node", "scrubdiff", f"{len(changed)} differ, {len(clean)} clean" + (" re-uploaded" if up else ""))


# ---------- v2.5.7: disaster recovery and our own disk space (Runming 2026-10-18 via the reviewer, A1 06:4x UTC) ----------
def dr_root():
    """As scripts/dr.sh: DR_ROOT, else the directory above $OPD_ROOT, /opd-dr (on the network share with $OPD_ROOT)."""
    return os.environ.get("DR_ROOT") or os.path.join(os.path.dirname(ROOT.rstrip("/")), "opd-dr")


def _tree_bytes(p):
    n = 0
    for dp, ds, fs in os.walk(p, followlinks=False):
        for f in fs:
            try:
                n += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return n


def dr_report():
    """For the heartbeat: DR_ROOT's total, each mirrored task's step and size, the kept models, the cap."""
    d = dr_root(); out = {"exists": os.path.isdir(d), "cap_gb": float(os.environ.get("DR_CAP_GB", "900")), "tasks": {}, "keep": {}}
    if not out["exists"]:
        return out
    tot = 0
    for sub, key in (("dr", "tasks"), ("keep", "keep")):
        base = os.path.join(d, sub)
        for t in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            b = _tree_bytes(os.path.join(base, t)); tot += b
            r = {"gb": round(b / 1e9, 2)}
            if sub == "dr":
                r["step"] = (_read(os.path.join(base, t, "latest")) or "").strip() or None
            out[key][t] = r
    out["total_gb"] = round((tot + _tree_bytes(os.path.join(d, "state"))) / 1e9, 2)
    out["over_cap"] = out["total_gb"] >= out["cap_gb"]
    return out


def our_prefixes():
    """The roots du / rm-ours name symbolically (never as paths: control files are public): opd_root, dr, and old_base,
    the directory the venv in use really lives in when that is outside $OPD_ROOT (the base before the 10-17 move)."""
    root = os.path.realpath(ROOT); out = {"opd_root": root, "dr": os.path.realpath(dr_root())}
    v = os.path.realpath(os.path.join(ROOT, "venv"))
    if not (v + "/").startswith(root + "/") and os.path.isdir(v):
        out["old_base"] = os.path.dirname(v)
    return out


def _protected():
    """Paths rm-ours never deletes, nor anything that contains them: both venvs' real directories, $OPD_ROOT, the
    runbook clone, CKPT_ROOT, the models of models.env / the environment."""
    ps = [os.path.join(ROOT, "venv"), os.path.join(ROOT, "venv-fa"), ROOT, RUNBOOK, os.environ.get("CKPT_ROOT") or _paths_env().get("CKPT_ROOT")]
    for k in ("STUDENT_06B", "STUDENT_17B", "TEACHER"):
        ps.append(os.environ.get(k))
    for line in (_read(os.path.join(ROOT, "models.env")) or "").splitlines():
        if "=" in line:
            ps.append(line.split("=", 1)[1].strip())
    return [os.path.realpath(p) for p in ps if p and os.path.isabs(p) and os.path.exists(p)]


def resolve_ours(prefix, rel, deleting=False):
    """(absolute target, None) or (None, refusal)."""
    pre = our_prefixes()
    if prefix not in pre:
        return None, f"unknown prefix {prefix} (known here: {', '.join(sorted(pre))})"
    if deleting and prefix not in ("dr", "old_base"):
        return None, "rm-ours deletes under dr or old_base only"
    base = pre[prefix]; rel = (rel or "").strip("/")
    if rel.startswith("/") or ".." in rel.split("/") or (deleting and rel in ("", ".")):
        return None, "relative path below the prefix required (no .., not the prefix itself)"
    t = os.path.realpath(os.path.join(base, rel)) if rel else base
    if t != base and not t.startswith(base + "/"):
        return None, "path leaves the prefix (symlink?)"
    if not os.path.exists(t):
        return None, "no such path"
    if deleting:
        if t == base:
            return None, "not the prefix itself"
        for p in _protected():
            if t == p or p.startswith(t + "/") or t.startswith(p + "/"):
                return None, "protected (a venv, $OPD_ROOT, the runbook, CKPT_ROOT or a model in use is at or below / above it)"
        if prefix == "dr":
            parts = os.path.relpath(t, base).split("/")
            if len(parts) >= 2 and parts[0] == "dr" and running_pids(parts[1]):
                return None, f"{parts[1]} is running (its mirror is in use)"
        if subprocess.run(["pgrep", "-f", t], capture_output=True).returncode == 0:
            return None, "a running process names this path"
    return t, None


def du_children(t):
    """{child: GB} one level below t (symlinks not followed) and the total."""
    kids = {}
    if os.path.isdir(t) and not os.path.islink(t):
        for c in sorted(os.listdir(t)):
            p = os.path.join(t, c)
            kids[c] = round((os.lstat(p).st_size if os.path.islink(p) or not os.path.isdir(p) else _tree_bytes(p)) / 1e9, 3)
        tot = sum(kids.values())
    else:
        tot = round(os.lstat(t).st_size / 1e9, 3)
    return kids, round(tot, 3)


def v_du(a, seq, flags=None, entry=None):
    """du <opd_root|dr|old_base> [relative path] (v2.5.7, read-only): sizes one level below, in the background."""
    t, why = resolve_ours(a[0], a[1] if len(a) > 1 else "")
    if why:
        return "rejected", why, {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_du", str(seq), a[0], a[1] if len(a) > 1 else ""],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", f"sizing {a[0]}/{a[1] if len(a) > 1 else ''} (read-only)", {}


def do_du(seq, prefix, rel):
    t0 = time.time(); t, why = resolve_ours(prefix, rel)
    if why:
        receipt(seq, None, "failed", why); return
    kids, tot = du_children(t)
    st = os.statvfs(t)
    receipt(seq, None, "ok", f"{prefix}/{rel}: {tot:.1f} GB in {len(kids)} entries", total_gb=tot, children_gb=kids,
            fs_free_gb=round(st.f_bavail * st.f_frsize / 1e9, 1), seconds=round(time.time() - t0, 1))


def v_rm_ours(a, seq, flags=None, entry=None):
    """rm-ours <dr|old_base> <relative path> (v2.5.7): our own directory or file under one of the two prefixes, never a
    venv, $OPD_ROOT, the runbook, CKPT_ROOT, a model in use, a running task's mirror or a path a running process names;
    sized first, then deleted, in the background; the receipt gives the GB freed. Each deletion under old_base is shown
    to the reviewer and Runming (du first) before it is sent."""
    t, why = resolve_ours(a[0], a[1], deleting=True)
    if why:
        return "rejected", why, {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_rm_ours", str(seq), a[0], a[1]],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", f"{a[0]}/{a[1]} is sized, then deleted", {}


def do_rm_ours(seq, prefix, rel):
    t0 = time.time(); t, why = resolve_ours(prefix, rel, deleting=True)   # checked again: things may have changed
    if why:
        receipt(seq, None, "failed", why); return
    kids, tot = du_children(t)
    errs = []
    if os.path.isdir(t) and not os.path.islink(t):
        shutil.rmtree(t, onerror=lambda f, p, e: errs.append(type(e[1]).__name__))
    else:
        try:
            os.unlink(t)
        except OSError as e:
            errs.append(type(e).__name__)
    gone = not os.path.lexists(t)
    receipt(seq, None, "ok" if gone and not errs else "failed", f"{prefix}/{rel}: {tot:.1f} GB {'freed' if gone else 'not fully deleted'}",
            freed_gb=tot if gone else None, children_gb=kids, errors=errs[:20], seconds=round(time.time() - t0, 1))
    event("node", "rm_ours", f"{prefix}/{rel} {tot:.1f} GB {'freed' if gone else 'partly deleted'}")


def v_dr_keep(a, seq, flags=None, entry=None):
    """dr-keep <task> (v2.5.7): a finished task's final model goes to $(dr_root)/keep/<task>/ on the network share (for
    SFT-P1: global_step_<final_step>/huggingface, the start of E12b-P1); refused while it runs or over DR_CAP_GB."""
    t = a[0]
    if running_pids(t):
        return "rejected", f"{t} is running", {}
    if not os.path.isdir(os.path.join(ROOT, "runs", t, "ckpt")):
        return "rejected", f"{t} has no checkpoints on this node", {}
    r = dr_report()
    if r.get("over_cap"):
        return "rejected", f"DR_ROOT holds {r.get('total_gb')} GB >= {r['cap_gb']} GB", {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_dr_keep", str(seq), t],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", f"{t}: final model copied to the share", {}


def do_dr_keep(seq, t):
    t0 = time.time(); ck = os.path.join(ROOT, "runs", t, "ckpt")
    fs = (_read(os.path.join(ROOT, "runs", t, "final_step")) or "").strip()
    steps = sorted((int(m.group(1)), d) for d in os.listdir(ck) for m in [re.match(r"global_step_(\d+)$", d)] if m)
    step = next((d for n, d in steps if str(n) == fs), steps[-1][1] if steps else None)
    src = None
    for sub in ("huggingface", os.path.join("actor", "huggingface_bf16"), os.path.join("actor", "huggingface")):
        p = os.path.join(ck, step, sub) if step else ""
        if step and os.path.isdir(p) and any(f.endswith(".safetensors") for f in os.listdir(p)):
            src = (p, sub); break
    if not src:
        receipt(seq, None, "failed", f"{t}: no final model with weights under ckpt/"); return
    dst = os.path.join(dr_root(), "keep", t, "ckpt", step, src[1])
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.rmtree(dst + ".tmp", ignore_errors=True)
    rc = subprocess.run(["nice", "-n", "19", "cp", "-a", src[0], dst + ".tmp"]).returncode
    if rc == 0:
        shutil.rmtree(dst, ignore_errors=True); os.rename(dst + ".tmp", dst)
        for f in ("final_step", "config.yaml"):
            if os.path.isfile(os.path.join(ROOT, "runs", t, f)):
                shutil.copy2(os.path.join(ROOT, "runs", t, f), os.path.join(dr_root(), "keep", t, f))
    gb = round(_tree_bytes(dst) / 1e9, 2) if rc == 0 else None
    receipt(seq, None, "ok" if rc == 0 else "failed", f"{t}: {step}/{src[1]} {'kept on the share' if rc == 0 else 'copy failed'}" + (f", {gb} GB" if gb else ""),
            kept_gb=gb, seconds=round(time.time() - t0, 1))
    event(t, "dr_kept" if rc == 0 else "dr_keep_failed", f"{step} {gb} GB")


def _script_verb(name, script, out_json):
    """v2.5.7a: a verb that runs one script in the background (nice 19) and writes the receipt from its JSON result."""
    def v(a, seq, flags=None, entry=None):
        if subprocess.run(["pgrep", "-f", f"scripts/{script}"], capture_output=True).returncode == 0:
            return "rejected", f"{script} is already running", {}
        subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_script", str(seq), script, out_json],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "accepted", f"{name} started (no GPU, nice 19); the receipt follows when it ends", {}
    return v


def do_script(seq, script, out_json):
    t0 = time.time(); log = os.path.join(ROOT, "logs", script.replace(".sh", ".out"))
    with open(log, "a") as out:
        rc = subprocess.run(["nice", "-n", "19", "bash", os.path.join(RUNBOOK, "scripts", script)], stdin=subprocess.DEVNULL,
                            stdout=out, stderr=subprocess.STDOUT).returncode
    r = json.loads(_read(os.path.join(ROOT, "logs", out_json), "{}") or "{}")
    ok = rc == 0 and (r.get("result") == "PASS" or r.get("pass") in (True, "true"))
    detail = r.get("reason") or (f"{r.get('seconds_total')} s, steps {json.dumps(r.get('steps'))}" if "steps" in r else f"exit {rc}")
    receipt(seq, None, "ok" if ok else "failed", f"{script}: {detail}"[:900], seconds=round(time.time() - t0), result=r)
    event("node", script.replace(".sh", ""), ("pass " if ok else "fail ") + str(detail)[:300])


v_persist_env = _script_verb("persist-env", "persist_env.sh", "persist_env.json")
v_rebuild_drill = _script_verb("rebuild-drill", "rebuild_drill.sh", "rebuild_drill.json")


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
         "ps": v_ps, "logs": v_logs, "run": v_run, "rerun": v_rerun, "republish": v_republish, "publish": v_publish, "leakscan": v_leakscan, "fa-build": v_fa_build, "scrubdiff": v_scrubdiff, "du": v_du, "rm-ours": v_rm_ours, "dr-keep": v_dr_keep, "persist-env": v_persist_env, "rebuild-drill": v_rebuild_drill, "eval": v_eval, "update": v_update, "upload": v_upload,
         "signal": v_signal, "kill-pgid": v_killpgid, "cancel": v_cancel, "heartbeat": v_heartbeat, "move-ckpt": v_move_ckpt, "reap": v_reap}
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
    elif cmd == "do_publish":
        do_publish(int(sys.argv[2]), sys.argv[3])
    elif cmd == "do_leakscan":
        do_leakscan(int(sys.argv[2]))
    elif cmd == "do_fa_build":
        do_fa_build(int(sys.argv[2]))
    elif cmd == "do_scrubdiff":
        do_scrubdiff(int(sys.argv[2]), fix=sys.argv[3:4] == ["fix"])
    elif cmd == "do_du":
        do_du(int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif cmd == "do_rm_ours":
        do_rm_ours(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "do_dr_keep":
        do_dr_keep(int(sys.argv[2]), sys.argv[3])
    elif cmd == "do_script":
        do_script(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "dr":
        print(json.dumps(dr_report(), indent=1))
    elif cmd == "do_republish":
        do_republish(int(sys.argv[2]), sys.argv[3])
    elif cmd == "do_move":
        do_move(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "roots":
        print(json.dumps(roots_report(), indent=1))
    elif cmd == "alerts":
        print(json.dumps(alerts(sys.argv[2])))
    elif cmd == "reap":   # v2.5.4b: reap <gpu> [task]: our orphaned vLLM / Ray processes on that GPU
        print("REAPED", " ".join(map(str, reap(int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None, wait_s=300))) or "none")   # v2.5.6e: up to 5 min for the GPU to empty
    elif cmd == "pulse":   # heartbeat.sh every 60 s (v2.5.1)
        r = pulse(sys.argv[2]); print("PULSE ok" if r else "PULSE refused")
    elif cmd == "rcpt":   # run_tasks / run_adhoc_eval: update a receipt (started, done, failed)
        receipt(int(sys.argv[2]), None, sys.argv[3], " ".join(sys.argv[4:]))
    else:
        print(__doc__); sys.exit(2)
