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
import glob
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
    pat = f"task.sh ([0-9]+(,[0-9]+)*|cpu) {re.escape(task)}$" if task else "task.sh ([0-9]+(,[0-9]+)*|cpu) [A-Za-z0-9._-]+$"   # [v2.5.8] cpu lane; [v2.5.9a] a,b
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
        m = re.search(r"task\.sh\s+(\d+(?:,\d+)*)\s", _read(f"/proc/{pid}/cmdline").replace("\0", " ") + " ")
        if m:
            gpus.update(m.group(1).split(","))   # [v2.5.9a] every GPU of a #gpus task
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
    times = _jread("setenv_times.json", {}); times[k] = time.time()
    _jwrite("setenv_times.json", times)
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
def remote_put(src, dst, kind=None):
    """v2.5.7zb: bounded (REMOTE_PUT_TIMEOUT, 180 s): a hung upload inside `apply` froze run_tasks for 94 min on
    2026-10-19 (12:17-13:50 UTC); on a timeout the child is killed, the call returns False and is logged. [v2.5.9z52] kind
    "public": to the public repository whatever the class (the pulse's mirror); a list of pairs: one commit (remote.py putmany)."""
    if isinstance(src, list):
        lf = _p("outbox_batch.json"); json.dump(src, open(lf, "w"))
        args = ["putmany", lf]; dst = f"batch of {len(src)}"
    else:
        args = ["put", src, dst] + ([kind] if kind else [])
    try:
        r = subprocess.run([sys.executable, os.path.join(RUNBOOK, "scripts", "remote.py"), *args], capture_output=True, text=True,
                           timeout=int(os.environ.get("REMOTE_PUT_TIMEOUT", "180")))
    except subprocess.TimeoutExpired:
        try:
            with open(os.path.join(ROOT, "logs", "control_timeouts.log"), "a") as f:
                f.write(f"{utc()} remote_put timeout {dst}\n")
        except OSError:
            pass
        return False
    return r.returncode == 0


# ---------- [v2.5.8] outbox + uploader (design 4b.2 with the reviewer's A1 additions a-d, 2026-10-19 13:3x UTC) ----------
# The control plane's small uploads (receipts, events, procs.json, logs / snapshot replies, the status and the pulse) go
# into a local outbox; uploader.sh ships it with the bounded remote_put, so a hung upload stops only the uploader, which
# the watchdog restarts, never dispatching or the heartbeat. An item is one pending (src, dst) pair, named
# <enqueue time ns>__<hash of dst>.json; the content is read when it ships. (a) Coalescing: enqueueing a dst that is
# already waiting replaces the older item, which moves to the end, so the status, pulse and procs.json go up once, with
# the newest content, and a receipt goes after the outputs enqueued before it. Items ship in time order; a failed upload
# ends the pass (the next pass retries it first); after 3 failures in a row the item moves to the end. Without a fresh
# uploader (control/loop_uploader, OUTBOX_STALE_S 600 s) or with OUTBOX=0, queue_put uploads directly, as before.
def _outbox():
    return _p("outbox")


def enqueue(src, dst, kind=None):
    d = _outbox(); os.makedirs(d, exist_ok=True); k = hashlib.sha256((dst + (f"|{kind}" if kind else "")).encode()).hexdigest()[:16]
    for n in os.listdir(d):
        if n.endswith(f"__{k}.json"):
            try:
                os.remove(os.path.join(d, n))
            except FileNotFoundError:
                pass
    name = f"{time.time_ns():020d}__{k}.json"
    json.dump({"src": os.path.abspath(src), "dst": dst, "utc": utc(), "fails": 0, **({"kind": kind} if kind else {})}, open(os.path.join(d, "." + name), "w"))
    os.replace(os.path.join(d, "." + name), os.path.join(d, name))
    return True


def uploader_alive():
    t = (_read(_p("loop_uploader")) or "").strip()
    return t.isdigit() and time.time() - int(t) < int(os.environ.get("OUTBOX_STALE_S", "600"))


# [v2.5.9z53] (Fable 10-22 14:1x UTC, ops-lead A1: after z51 / z52 the status and the receipts still did not reach the private
# repository while the direct runs/ uploads did) the status, the private pulse and the receipts (and reply payloads) go up
# directly, as runs/ does, bounded by CONTROL_PUT_TIMEOUT (60) s, never through the outbox; the z52 throttles stay (status and
# pulse every 5 min); a failure is logged, one event per failing episode, and the next round writes again. CONTROL_DIRECT=0:
# through the outbox as before.
DIRECT_RE = re.compile(r"status/[^/]+\.json|pulse/[^/]+\.json|status/[^/]+/replies/[^/]+")


def _direct_put(src, dst):
    old = os.environ.get("REMOTE_PUT_TIMEOUT")
    os.environ["REMOTE_PUT_TIMEOUT"] = _cfg("CONTROL_PUT_TIMEOUT", "60")
    try:
        ok = remote_put(src, dst)
    finally:
        os.environ.pop("REMOTE_PUT_TIMEOUT") if old is None else os.environ.__setitem__("REMOTE_PUT_TIMEOUT", old)
    cls = "receipts" if "/replies/" in dst else dst.split("/")[0] + ("" if dst.startswith("pulse/") else "_json")
    st = _jread("direct_fail.json", {}); n = int(st.get(cls) or 0)
    if ok and n:
        st[cls] = 0; event("node", "control_direct_recovered", f"{cls} after {n} failed puts")
    elif not ok:
        st[cls] = n + 1
        try:
            with open(os.path.join(ROOT, "logs", "control_timeouts.log"), "a") as f:
                f.write(f"{utc()} direct put failed {dst}\n")
        except OSError:
            pass
        if n == 0:
            event("node", "control_direct_failed", f"{cls}: {dst}")
    if (ok and n) or not ok:
        tmp = _p(f"direct_fail.json.{os.getpid()}.tmp"); json.dump(st, open(tmp, "w")); os.replace(tmp, _p("direct_fail.json"))
    return ok


def queue_put(src, dst, kind=None):
    direct = kind is None and DIRECT_RE.fullmatch(dst.lstrip("/")) and _cfg("CONTROL_DIRECT", "1") == "1"   # [v2.5.9z53]
    if not direct and os.environ.get("OUTBOX", "1") == "1" and uploader_alive():
        return enqueue(src, dst, kind)
    k = _throttle({"dst": dst, "kind": kind})[0]   # [v2.5.9z52] the direct upload keeps the throttles of the latest-state items
    if k and k != "events":                         # (an event skipped here would be lost: those go up as before)
        if not _due({"dst": dst, "kind": kind}):
            return True
        _stamp([k])
    if direct:
        return _direct_put(src, dst)
    return remote_put(src, dst, kind) if kind else remote_put(src, dst)


CONTROL_PREFIXES = ("status/", "pulse/")


def _ship_order(d):
    """[v2.5.9z51] (10-22: the status, the pulse and the receipts stopped from 11:17 UTC while events still went up) the order of
    a pass: the control plane's items (status/: the status, receipts, events, procs, replies; pulse/) before everything else,
    each group in time order. The coalescing re-enqueue moves the status and the pulse to the end every minute, so with a slow
    or failing backlog ahead of them (each item up to REMOTE_PUT_TIMEOUT, a pass at most OUTBOX_PASS_S) they never shipped."""
    names = sorted(n for n in os.listdir(d) if n.endswith(".json") and not n.startswith("."))
    def prio(n):
        try:
            return 0 if str(json.load(open(os.path.join(d, n))).get("dst", "")).lstrip("/").startswith(CONTROL_PREFIXES) else 1
        except (OSError, ValueError):
            return 1
    return sorted(names, key=lambda n: (prio(n), n))


# [v2.5.9z52] (ops-lead / Fable 10-22 21:0x Beijing: the HF account answered 429 from 12:38 UTC, and the status, the pulse and
# the receipts stopped going up) throttles: each latest-state item (the status, the pulse, procs.json, the heartbeat's re-upload
# of replies/ and events/) ships at most once per interval, counted from its last try, with the newest content (the outbox
# coalesces); the events wait for one commit every EVENTS_PUT_S; receipts and other items are not held. While remote.py's 429
# backoff lasts the pass tries nothing. Intervals from control.env (setenv), else the environment, else these; 0: no throttle.
THROTTLE = (("status", r"status/[^/]+\.json", "STATUS_PUT_S", 300), ("procs", r"status/[^/]+/procs\.json", "PROCS_PUT_S", 600),
            ("events", r"status/[^/]+/events/[^/]+", "EVENTS_PUT_S", 300), ("resync", r"status/[^/]+/(?:events|replies)", "RESYNC_PUT_S", 3600),
            ("pulse", r"pulse/[^/]+\.json", "PULSE_PUT_S", 300))


def _throttle(it):
    """(stamp key, interval s) of an outbox item, or (None, 0): one key for all events, one per destination otherwise; the
    pulse's public mirror (kind public) every PULSE_MIRROR_S (1800)."""
    dst = str(it.get("dst", "")).lstrip("/")
    if it.get("kind") == "public":
        return "mirror:" + dst, int(_cfg("PULSE_MIRROR_S", "1800"))
    for name, pat, knob, default in THROTTLE:
        if re.fullmatch(pat, dst):
            return (name if name == "events" else f"{name}:{dst}"), int(_cfg(knob, str(default)))
    return None, 0


def _stamps():
    try:
        s = _jread("put_stamps.json", {})
        return s if isinstance(s, dict) else {}
    except ValueError:
        return {}


def _stamp(keys):
    s = _stamps(); now = time.time()
    for k in keys:
        s[k] = now
    tmp = _p(f"put_stamps.json.{os.getpid()}.tmp"); json.dump(s, open(tmp, "w")); os.replace(tmp, _p("put_stamps.json"))


def _due(it, stamps=None):
    k, iv = _throttle(it)
    return not k or iv <= 0 or time.time() - float((_stamps() if stamps is None else stamps).get(k, 0)) >= iv


def _backoff_left():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import remote as _rm_mod
        return _rm_mod.backoff_left()
    except Exception:
        return 0


def _ship_events(d, names):
    """[v2.5.9z52] the events waiting in the outbox, in one commit once EVENTS_PUT_S has passed since the last try (one event:
    an ordinary put): (shipped, dropped, failed)."""
    ev, dropped = [], 0
    for name in names:
        p = os.path.join(d, name)
        try:
            it = json.load(open(p))
        except (OSError, ValueError):
            continue
        if _throttle(it)[0] != "events":
            continue
        if not os.path.isfile(it.get("src", "")):
            _rm(p); dropped += 1; continue
        ev.append((p, it))
    if not ev or not _due(ev[0][1]):
        return 0, dropped, 0
    _stamp(["events"])
    ok = remote_put(ev[0][1]["src"], ev[0][1]["dst"]) if len(ev) == 1 else remote_put([[it["src"], it["dst"]] for _, it in ev], None)
    open(_p("loop_uploader"), "w").write(str(int(time.time())))
    for p, it in ev:
        if ok:
            _rm(p)
        elif os.path.exists(p):
            it["fails"] = it.get("fails", 0) + 1; json.dump(it, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
    return (len(ev), dropped, 0) if ok else (0, dropped, len(ev))


def diag_tails(force=False):
    """[v2.5.9z53] (Fable 10-22) while the status does not reach the project: every DIAG_TAILS_S (1800) s, from the uploader's pass,
    the last 200 lines of the control plane's logs (redacted) and the outbox's head (destination, tries, age) go directly to
    logs/<node>/diag/ in the private repository, bounded by CONTROL_PUT_TIMEOUT."""
    try:
        iv = int(_cfg("DIAG_TAILS_S", "1800"))
    except ValueError:
        iv = 1800
    last = _read(_p("diag_tails.t")).strip()
    if iv <= 0 or (not force and last.isdigit() and time.time() - int(last) < iv):
        return None
    open(_p("diag_tails.t"), "w").write(str(int(time.time())))
    out = _p("diag"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    errs = []
    for f in ("heartbeat.err", "heartbeat.log", "uploader.log", "uploader.err", "control.log", "control_timeouts.log", "run_tasks.log", "watchdog.log",
              "gh_heartbeat.log", "gh_heartbeat.err", "status_builder.err"):   # [v2.5.9z55] the status builder's last failure
        p = os.path.join(ROOT, "logs", f)
        try:   # [v2.5.9z56] UTF-8 whatever the locale (progress bars in the logs); one file's failure skips that file only
            if os.path.exists(p):
                open(os.path.join(out, f + ".tail"), "w", encoding="utf-8", errors="replace").write(redact(_tail_lines(p, 200)))
        except Exception as e:
            errs.append(f"{f}: {type(e).__name__}")
    d = _outbox(); now = time.time(); head = []
    for n in _ship_order(d)[:40] if os.path.isdir(d) else []:
        try:
            it = json.load(open(os.path.join(d, n)))
        except (OSError, ValueError):
            continue
        head.append({"dst": it.get("dst"), "kind": it.get("kind"), "fails": it.get("fails"), "age_s": int(now - int(n[:20]) / 1e9) if n[:20].isdigit() else None,
                     "src_exists": os.path.exists(it.get("src", "")), "src_bytes": os.path.getsize(it["src"]) if os.path.isfile(it.get("src", "")) else None})
    st = _state_read()
    json.dump({"utc": utc(), "applied_seq": _read(_p("applied_seq")).strip(), "pending": st.get("pending"), "alerts": st.get("alerts"),
               "outbox_items": len([n for n in os.listdir(d) if n.endswith(".json")]) if os.path.isdir(d) else 0, "outbox_head": head,
               "put_stamps": _stamps(), "direct_fail": _jread("direct_fail.json", {}), "hf_backoff_s": int(_backoff_left()),
               "loops_age_s": {k: (lambda t: int(now - int(t)) if t.isdigit() else None)(_read(_p(f"loop_{k}")).strip()) for k in ("run_tasks", "heartbeat", "uploader")},
               "tail_errors": errs},
              open(os.path.join(out, "control_state.json.tmp"), "w"), indent=1)
    os.replace(os.path.join(out, "control_state.json.tmp"), os.path.join(out, "control_state.json"))
    return _direct_put(out, f"logs/{node_id()}/diag")


def ship_outbox(budget_s=None):
    """One pass of uploader.sh: (shipped, dropped, failed). A missing src is dropped; the pass ends at the first failure
    or after OUTBOX_PASS_S (420 s); control/loop_uploader is refreshed after every item."""
    budget_s = budget_s or int(os.environ.get("OUTBOX_PASS_S", "420"))
    d = _outbox(); os.makedirs(d, exist_ok=True); t0 = time.time(); shipped = dropped = failed = 0
    if _backoff_left() > 0:   # [v2.5.9z52] the 429 backoff: nothing tried, nothing counted
        return shipped, dropped, failed
    try:
        diag_tails()   # [v2.5.9z53] every DIAG_TAILS_S
    except Exception as e:
        print(f"DIAG_TAILS_FAILED {type(e).__name__}", file=sys.stderr)
        try:   # [v2.5.9z56] (10-22: no diag reached the project and nothing said why) the failure as an event, at most every DIAG_TAILS_S
            event("node", "diag_tails_failed", f"{type(e).__name__}: {redact(str(e))[:300]}")
        except Exception:
            pass
    names = _ship_order(d)
    shipped, dropped, failed = _ship_events(d, names)   # [v2.5.9z52] first, as the control plane's oldest items were
    stamps = _stamps()
    for name in names:   # [v2.5.9z51] the control plane's items first
        if time.time() - t0 > budget_s or _backoff_left() > 0:
            break
        p = os.path.join(d, name)
        try:
            it = json.load(open(p))
        except (OSError, ValueError):
            continue   # replaced while we listed, or an event shipped above
        if not os.path.exists(it.get("src", "")):
            _rm(p); dropped += 1; continue
        k = _throttle(it)[0]
        if k == "events" or not _due(it, stamps):   # [v2.5.9z52] an event goes with the others; a throttled item waits
            continue
        if k:
            _stamp([k])
        ok = remote_put(it["src"], it["dst"], "public") if it.get("kind") == "public" else remote_put(it["src"], it["dst"])
        open(_p("loop_uploader"), "w").write(str(int(time.time())))
        if ok and it.get("kind") == "public":   # [v2.5.9z52] the pulse's field public_mirror_utc
            open(_p("public_mirror_utc"), "w").write(utc() + "\n")
        if ok:
            _rm(p); shipped += 1; continue
        failed += 1; it["fails"] = it.get("fails", 0) + 1
        if os.path.exists(p):
            if it["fails"] >= 3:   # moved to the end, so one item that never goes up cannot hold back the others
                _rm(p); enqueue(it["src"], it["dst"], it.get("kind"))
            else:
                json.dump(it, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)
        # [v2.5.9z50] (the reviewer / Runming 10-22: no head-of-line blocking) a failed item stays for the next pass and the
        # pass goes on with the others (was: the pass ended here, and a failing public item held back status and receipts)
        continue
    return shipped, dropped, failed


def _rm(p):
    try:
        os.remove(p)
    except FileNotFoundError:
        pass


STATUS_NAME = {"ok": "done", "rejected": "refused"}   # receipts use D-004's names; log.jsonl keeps v2.4's


def receipt(seq, entry, status, detail, upload=True, **extra):
    """status/<node>/replies/<seq>.json; written locally first, uploaded (the heartbeat re-uploads replies/ each round).
    v2.5.7zb: apply writes it with upload=False, advances applied_seq, and only then uploads (bounded), so a command
    is never run twice because its receipt's upload hung or timed out."""
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
    if upload:
        queue_put(path, f"status/{node_id()}/replies/{seq}.json")   # [v2.5.8] through the outbox
    return r


def event(task, kind, detail=""):
    """The pool's own actions (start, retry, failed, done) in the receipt shape, under status/<node>/events/ (R-012)."""
    os.makedirs(_p("events"), exist_ok=True)
    safe = lambda x: re.sub(r"[^A-Za-z0-9._-]", "_", str(x))   # v2.5.8: never a path in a file name
    name = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{safe(task)}_{safe(kind)}.json"
    e = {"event": kind, "task": task, "detail": redact(detail)[:2000], "node": node_id(), "utc": utc(),
         "runbook_head": _git_head(), "origin": "pool"}
    json.dump(e, open(_p("events", name), "w"), indent=1)
    queue_put(_p("events", name), f"status/{node_id()}/events/{name}")   # [v2.5.8]


def _git_head():
    try:
        return subprocess.run(["git", "-C", RUNBOOK, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return None


# ---------- v2.5: process table ----------
# v2.5.7zf (from v2.5.8 item 9): also remote.py / fetch_model.py / watchdog.sh / run_queue_eval.sh: an upload or fetch left
# behind by a killed loop or wrapper (drill B; the prefetch of EVAL-e12d-P0 whose wrapper was TERMed at 16:01 UTC, 10-19:
# their parent is init, and the orphaned prefetch still holds back every other prefetch) is listed in procs.json and
# can be signalled
# [v2.5.8] also uploader.sh
OUR_ROOTS = re.compile(r"scripts/(run_tasks|task|upload_sync|heartbeat|run_adhoc_eval|watchdog|run_queue_eval|uploader)\.sh|scripts/(control|remote|fetch_model)\.py"
                       r"|scripts/gh_heartbeat\.sh"   # [v2.5.9z54]
                       r"|scripts/weights_replay\.(?:sh|py)"   # [v2.5.9z62]
                       r"|code/eval/gen_eval\.py|/tmp/ray_gpu")


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
            m = re.search(r"scripts/task\.sh (\d+(?:,\d+)*) ([A-Za-z0-9._-]+)", rows[p]["_cmd"]) or re.search(r"run_adhoc_eval\.sh (\d+) (\d+)", rows[p]["_cmd"])
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
        queue_put(_p("procs.json"), f"status/{node_id()}/procs.json")   # [v2.5.8]
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
    t = _jread("procs.json", {})
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
    if R.protected(now["_cmd"]):   # [v2.5.8] our loops by their own program
        return "protected system or session process"
    if now["uid"] != os.getuid():
        return "process of another user"
    if not row.get("ours"):
        if row.get("gpu") is None:
            return "not in our process trees and not on our GPUs"
        if not (flags.get("force") and flags.get("confirm_foreign")):
            return "process outside our trees: needs force and ctl.py --confirm-foreign"
    if sig in ("STOP", "CONT"):
        if R.TRAINING_CMD.search(now["_cmd"]) or (row.get("task") or "").startswith(("E10b", "E10c", "E10d", "E12b", "E12c", "E12d", "SFT")):   # [v2.5.8a] E10c / E12c
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
    if any(m in _ancestors() for m in members) or any(R.protected((_proc(m) or {}).get("_cmd", "")) for m in members):   # [v2.5.8]
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
def _tail_lines(path, n, max_bytes=4 << 20):
    """[v2.5.9z53] the last n lines of a file from at most its last max_bytes (seq 482 `logs heartbeat 60`, 10-22: the whole
    heartbeat.log was read on the loaded shared disk; apply may have outlived APPLY_TIMEOUT on it every round)."""
    with open(path, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - max_bytes)); data = f.read().decode("utf-8", "replace")
    return "".join(data.splitlines(True)[-n:])


def v_logs(a, seq, flags=None, entry=None):
    t, n = a[0], int(a[1])
    src = os.path.join(ROOT, "logs", f"{t}.log")
    if not os.path.exists(src):
        return "rejected", f"no log for {t}", {}
    out = _p(f"req_{seq}"); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    open(os.path.join(out, "log_tail.txt"), "w").write(redact(_tail_lines(src, n)))   # [v2.5.9z53] was: the whole file read
    dest = f"status/{node_id()}/replies/{seq}"
    return ("ok", f"last {n} lines of logs/{t}.log", {"outputs": [f"{dest}/log_tail.txt"]}) if queue_put(out, dest) else ("failed", "upload failed", {})


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
    return ("ok", "nvidia-smi, df, procs.json and ckpt_roots.json", {"outputs": [dest]}) if queue_put(out, dest) else ("failed", "upload failed", {})


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
            rt = env.get("RAY_TMPDIR", "")   # [v2.5.9a] a #gpus task: CUDA_VISIBLE_DEVICES=a,b, RAY_TMPDIR=/tmp/ray_gpu<a>_<b>
            if str(gpu) not in env.get("CUDA_VISIBLE_DEVICES", "").split(",") and not (
                    rt.startswith("/tmp/ray_gpu") and str(gpu) in rt[len("/tmp/ray_gpu"):].split("_")):
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
                   and re.match(r"(E10b-|E10c-|E10d-|E12b-|E12c-|E12d-|SFT-|E03-|E02-)", str(r.get("task") or ""))})
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


def _progress(p):
    """[v2.5.9s] the public form of a task's progress: numbers as they are; the strings without "/" ("900/2000" -> "900 of
    2000", "it/s" -> "it per s", [v2.5.9t] "toks/s" / "tokens/s" -> "... per s"; no general x/y rewrite, which would let a path
    through), at most 200 characters, and dropped when leak_check still finds anything in them."""
    if not isinstance(p, dict):
        return None
    out = {k: _num(p.get(k)) for k in ("done", "total", "rate", "eta_s") if _num(p.get(k)) is not None}
    if isinstance(p.get("unit"), str) and re.fullmatch(r"[a-z]{1,12}", p["unit"]):
        out["unit"] = p["unit"]
    if isinstance(p.get("utc"), str) and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", p["utc"]):
        out["utc"] = p["utc"]
    for k in ("part", "tqdm"):
        s = p.get(k)
        if isinstance(s, str):
            s = s.replace("toks/s", "toks per s").replace("tokens/s", "tokens per s")   # [v2.5.9t] vLLM's speed suffix; only these two
            s = re.sub(r"(\d+)/(\d+)", r"\1 of \2", s).replace("it/s", "it per s").replace("s/it", "s per it")[:200]
            if not leak_check(s):
                out[k] = s
    return out or None


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) else None


ASSET_READY, ASSET_PULLING = "已就位", "拉取中"


def _pulse_assets():
    """(assets, dropped) for the pulse: assets_view() with every entry checked by leak_check on its own (a failing entry is left
    out and counted), or (None, None) when assets_view() raises - the pulse goes on; one event assets_view_failed when it starts
    failing (state.json assets_failed), assets_view_ok when it works again."""
    failed = bool(_state_read().get("assets_failed"))
    try:
        a = assets_view()
    except Exception as e:
        if not failed:
            state(assets_failed=True)
            try:
                event("node", "assets_view_failed", type(e).__name__)
            except Exception:
                pass
        return None, None
    if failed:
        state(assets_failed=False)
        try:
            event("node", "assets_view_ok")
        except Exception:
            pass
    keep = {k: v for k, v in a.items() if not leak_check({k: v})}
    return keep, len(a) - len(keep)


def assets_view():
    """[v2.5.9z13] (ops-lead / Runming's monitor page 10-22) the pulse's assets: {<owner>__<name>: {"state": "已就位" | "拉取中",
    "at": "<UTC> <size>"}} for models/, data/hf/ and data/git/. Ready = a .fetch_ok: a model's or dataset's records its repo, time and
    files (the size is their sum); a git copy's holds the commit (its time is the file's, its size the copy's, cached by that mtime in
    assets_cache.json). Pulling = no .fetch_ok, and partial files (.incomplete / .part) in it or a fetch process naming it. Names are
    repo ids with "/" as "__" (the pulse's leak check refuses "/"), never a path."""
    cache = _jread("assets_cache.json", {}); out = {}
    try:
        procs = subprocess.run(["pgrep", "-af", "fetch_model[.]py|git_fetch[.]sh"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        procs = ""
    fmt = lambda t, b: f"{t} {b / 1e9:.2f} GB" if b is not None else t
    for sub in ("models", os.path.join("data", "hf"), os.path.join("data", "git")):
        base = os.path.join(ROOT, sub)
        for n in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            d = os.path.join(base, n)
            if not os.path.isdir(d) or not re.fullmatch(r"[A-Za-z0-9._-]+", n):
                continue
            ok = os.path.join(d, ".fetch_ok")
            if os.path.isfile(ok):
                try:
                    j = json.load(open(ok))
                except ValueError:
                    j = None
                if isinstance(j, dict):
                    name = str(j.get("repo") or n).replace("/", "__")
                    size = sum(int(f[1]) for f in j.get("files") or [] if isinstance(f, (list, tuple)) and len(f) > 1 and str(f[1]).isdigit())
                    at = fmt(str(j.get("utc") or ""), size or None)
                else:   # a git copy: the commit
                    name, m = n, os.path.getmtime(ok); key = f"{sub}/{n}"
                    if (cache.get(key) or {}).get("mtime") != m:
                        cache[key] = {"mtime": m, "bytes": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(d) for f in fs
                                                                if not os.path.islink(os.path.join(dp, f)))}
                    at = fmt(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(m)), cache[key]["bytes"])
                if re.fullmatch(r"[A-Za-z0-9._-]+", name):
                    out[name] = {"state": ASSET_READY, "at": at}
            elif d in procs or any(f.endswith((".incomplete", ".part")) for _, _, fs in os.walk(d) for f in fs):
                out[n] = {"state": ASSET_PULLING, "at": None}
    try:
        _jwrite("assets_cache.json", cache)
    except OSError:
        pass
    return out


def build_pulse(st, gpu_csv):
    """The public pulse from the (private) status JSON and `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,
    memory.total --format=csv,noheader,nounits`: numbers, task names, states and times only."""
    gpus = []
    for line in (gpu_csv or "").strip().splitlines():
        r = [x.strip() for x in line.split(",")]
        if len(r) == 4 and r[0].isdigit():
            f = lambda v: float(v) if re.fullmatch(r"[0-9]+(\.[0-9]+)?", v) else None
            gpus.append({"idx": int(r[0]), "util": f(r[1]), "mem_used": f(r[2]), "mem_total": f(r[3])})
    held = st.get("held") if isinstance(st.get("held"), dict) else {}   # [v2.5.9c] a free GPU held for a #gpus task
    if R.TASK_RE.match(str(held.get("task") or "")) and isinstance(held.get("gpus"), list):
        for g in gpus:
            if g["idx"] in held["gpus"]:
                g["held_for"] = held["task"]
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
                       "gpus": [g for g in v.get("gpus") or [] if isinstance(g, int) and 0 <= g <= 15],   # [v2.5.9c] all of a #gpus task's
                       "kind": v.get("kind") if v.get("kind") in ("train", "sft", "eval", "probe", "other") else None,
                       "started_utc": v.get("started_utc") if isinstance(v.get("started_utc"), str)
                       and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", v.get("started_utc")) else None}
        if isinstance(v.get("blocked_by"), str) and R.TASK_RE.match(v["blocked_by"]):   # [v2.5.9z3] the gate holding a blocked task
            tasks[name]["blocked_by"] = v["blocked_by"]
        if v.get("phase") in ("uploading", "upload_pending"):   # [v2.5.9z65a] GPU released, uploading; [v2.5.9z65] done, results queued
            tasks[name]["phase"] = v["phase"]
        if v.get("pad") is True:   # [v2.5.9q] a padding reading (#pad stages, v2.5.9m): only when true, like the monitor page reads it
            tasks[name]["pad"] = True
        pr = _progress(v.get("progress"))   # [v2.5.9s] a running evaluation's progress for the monitor page
        if pr:
            tasks[name]["progress"] = pr
    ctl = st.get("control") if isinstance(st.get("control"), dict) else {}
    rp = st.get("results_pending")   # [v2.5.9z65] result directories still in the results queue
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
        for r in _jread("procs.json", {}).get("rows", []):
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
    du = st.get("our_du") if isinstance(st.get("our_du"), dict) else {}   # [v2.5.9c]
    assets, dropped = _pulse_assets()   # [v2.5.9z13] a failure or a bad entry never stops the pulse
    head = st.get("runbook_head") if isinstance(st.get("runbook_head"), str) and re.fullmatch(r"[0-9a-f]{7,40}", st.get("runbook_head")) else None
    ver = st.get("runbook_version") if isinstance(st.get("runbook_version"), str) and re.fullmatch(r"v[0-9.]+", st.get("runbook_version")) else None
    return {"schema": 1, "utc": utc(), "node_id": node_id(), "status_utc": t(st.get("utc")), "runbook_head": head,
            "runbook_version": ver, "queue_version": _num(st.get("queue_version")) if not isinstance(st.get("queue_version"), str)
            else (st.get("queue_version") if re.fullmatch(r"[0-9]+", st.get("queue_version")) else None),
            "gpus": gpus, "tasks": tasks, "results_pending": int(rp) if isinstance(rp, int) and rp >= 0 else None,   # [v2.5.9z65]
            "readings_held": int(st["readings_held"]) if isinstance(st.get("readings_held"), int) and st["readings_held"] >= 0 else None,   # [v2.5.9z69]
            "ctrl_on_ckpt": os.path.islink(os.path.join(ROOT, "control")),   # [v2.5.9z71] control/ and logs/ on the checkpoint disk
            "tasks_guard": (_read(os.path.join(ROOT, "control", "tasks_guard")).strip() or None),   # [v2.5.9z71] 'work' below 300 GB
            "disks": [{"idx": 0, "free_gb": _num(st.get("disk_free_gb")), "alert": bool(st.get("disk_alert")),   # [v2.5.9c] + total, ours
                       "total_gb": _num(st.get("disk_total_gb")), "free_bytes": _num(st.get("disk_free_bytes")),
                       "our_gb": _num(du.get("work_gb")), "our_gb_utc": t(du.get("utc"))},
                      {"idx": 1, "free_gb": _num(st.get("ckpt_disk_free_gb")), "alert": bool(st.get("ckpt_disk_alert")),
                       "total_gb": _num(st.get("ckpt_disk_total_gb")), "our_gb": _num(du.get("ckpt_gb")), "our_gb_utc": t(du.get("utc"))}],
            "alerts": sorted(set(alerts)),
            "assets": assets, "assets_dropped": dropped,   # [v2.5.9z13]
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
    try:   # [v2.5.9z50] where the node's files go now; the public mirror's last time (none until the mirror exists)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import remote as _rm_mod
        p["source_policy"] = "private_first" if _rm_mod.public_paused() else "split"
    except Exception:
        p["source_policy"] = None
    p["public_mirror_utc"] = _read(_p("public_mirror_utc"), "").strip() or None
    bad = leak_check(p)
    if bad:
        event("control", "pulse_refused", f"{len(bad)} string(s) failed the leak check")
        return None
    json.dump(p, open(_p("pulse.json.tmp"), "w")); os.replace(_p("pulse.json.tmp"), _p("pulse.json"))
    if upload and _split_mode():
        queue_put(_p("pulse.json"), f"pulse/{node_id()}.json")   # [v2.5.8]
        if p.get("source_policy") == "private_first" and _cfg("PULSE_PUBLIC_MIRROR", "0") == "1":   # [v2.5.9z52] while private first:
            queue_put(_p("pulse.json"), f"pulse/{node_id()}.json", "public")   # a copy to the public repository every PULSE_MIRROR_S
    return p


def _request(kind, seq, body):
    os.makedirs(_p("requests"), exist_ok=True)
    json.dump(dict(body, kind=kind, seq=seq, utc=utc()), open(_p("requests", f"{seq}_{kind}.json"), "w"))


def v_run(a, seq, flags=None, entry=None):
    t = a[0]
    r = subprocess.run(["bash", "-c", f'T={t}; ARMS="P2|P3|P4lowlr|P5|P6|P6r|P2noEOS|Pstop|P6rand|P6top5|P4EOSpen|P5top10|P5rand10m"; '
                        'case $T in E12b-P5-prog|E12b-P5-prog-up|PROBE-PROG) exit 0;; esac; '   # [v2.5.9z43] (Q-058)
                        'case $T in E10b-P5route-inv|PROBE-ROUTE-INV) exit 0;; esac; '   # [v2.5.9z45] (Q-063)
                        'case $T in E16-X5|E16-X5e|PROBE-X5) exit 0;; esac; '   # [v2.5.9z46] (Q-057)
                        'case $T in E10b-P5mix60) exit 0;; esac; '   # [v2.5.9z47]
                        'case $T in E12d-P5-8k) exit 0;; esac; '   # [v2.5.9z60] (Q-066)
                        'case $T in E12d-P5-16k) exit 0;; esac; '   # [v2.5.9z67] (Q-069)
                        'case $T in E10b-P5-8k|E10b-P5smerge-8k|PROBE-P5-8K|PROBE-SMERGE-8K) exit 0;; esac; '   # [v2.5.9z66] (Q-059)
                        'case $T in E10c-P5tail) exit 0;; esac; '   # [v2.5.9z70] (Q-055)
                        'case $T in E10e-P5self|PROBE-8B) exit 0;; esac; '   # [v2.5.9z68] (Q-068)
                        'case $T in E10b-P5bud2k|E10b-P5bud3k|PROBE-TAIL-BUD) exit 0;; esac; '   # [v2.5.9z73] (Q-061)
                        'case $T in PROBE-DIFFPROBE) exit 0;; esac; '   # [v2.5.9z74] (Q-062)
                        'case $T in PROBE-PSTAR) exit 0;; esac; '   # [v2.5.9z75] (Q-064)
                        'case $T in E10b-P5-dense|E10b-P5-akl|PROBE-DENSE) exit 0;; esac; '   # [v2.5.9z76] (Q-065)
                        'case $T in E12d-P5-8k-s43) exit 0;; esac; '   # [v2.5.9z77] (Q-071; E10d-P5self-s43 by the E10d family regex)
                        'case $T in E10b-P5tail-s43|E12b-P5-8k-s44) true;; E10b-*) [[ $T =~ ^E10b-(($ARMS)(-s(43|44))?|(P2|P5)-t14|P5think|P5self|P5self-s43|P2self|P5len[SLR]|P5stopmerge|P5tail|P5route|P5mix|P5mix75|P5mix90)$ ]];; E10c-*) [[ $T =~ ^E10c-(($ARMS)-(fix4|dyn16k|dyn6144)(-s43)?|P5-polish|P5self)$ ]];; E10d-*) [[ $T =~ ^E10d-((P2|P5)(-s43)?|P5self(-s43)?|X1|X2|X4|P5-t32)$ ]];; E11b-*) [[ $T =~ ^E11b-(P0|($ARMS)(-s(43|44))?|(P2|P5)-t14)$ ]];; '
                        'E12b-*) [[ $T =~ ^E12b-(P0|P1|$ARMS|P5-s4[34]|P2-s43|P5self|P5self-s43|P5-kl|P5-stop|P5-stop2|P5-8k|P5-16k|P5-r3|P5-16k-kl)$ ]];; E12c-*) [[ $T =~ ^E12c-(P0|P5)$ ]];; E12d-*) [[ $T =~ ^E12d-(P0|P5)$ ]];; EVAL-*) [[ $T =~ ^EVAL-[A-Za-z0-9._-]+$ ]];; SFT-P1|FA-PROBE|SPEED-PROBE|PROBE-4B|PROBE-TERM|PROBE-TERM2|PROBE-TERM3|PROBE-TERM4|PROBE-TERM4-4B|PROBE-TERM4-06B|PROBE-TOP10|STRICT-C1|PROBE-T14|PROBE-T32|PROBE-P5THINK|PROBE-P5SELF|PROBE-P5SELF-4B|PROBE-P5SELF-06B|PROBE-P2SELF|PROBE-E12C|PROBE-E12B-KL|PROBE-E12B-STOP|PROBE-E12B-STOP2|PROBE-E12B-8K|PROBE-E12B-16K|PROBE-E12B-R3|PROBE-X2507-X1|PROBE-X2507-X2|PROBE-X2507-X4|PROBE-LENSEL|PROBE-2GPU|PROBE-TOOL|PROBE-NET|PROBE-NET2|BUILD-VENV-VL|PROBE-VLLM-VL|READINGS-EXTRA|PROBE-VLM|E15-VLM-P5|PROBE-ROLLOUT-U60|PROBE-ROLLOUT-U75|PROBE-ROLLOUT-U85|PROBE-STOPMERGE|PROBE-STOPMERGE-UNIT|PROBE-WDIFF|TOKCHECK-X5|VENV-VL-NVRTC|PROBE-MIXP|PROBE-TAIL|PROBE-TAIL-BUD|PROBE-DIFF17|PROBE-DIFF17-4B|PROBE-DIFFPROBE|PROBE-PSTAR|PROBE-DENSE|PROBE-ROUTE|PROBE-POLISH|FA-SPEED-17|FA-SPEED-4B|FA-SPEED-E12|FA-SPEED-TWIN|E03-MI-l0.[0357]|E03-MI-l1.0|E02-opsa17-c0.5|E02-opsa17-c1.5|E02-opsa4-c-1|'
                        'E02-opsa4-c0.5|E02-opsa4-c1.5) true;; *) false;; esac'])
    if r.returncode != 0:
        return "rejected", f"{t} is not accepted by task.sh", {}
    if running_pids(t):
        return "rejected", f"{t} is already running", {}
    if os.path.exists(os.path.join(ROOT, "runs", t, ".done")):
        return "rejected", f"{t} is already done", {}
    _request("run", seq, {"task": t})
    return "accepted", f"{t} starts on the next GPU that is free after the queue's own tasks (deps must be done)", {}


def _gate_failed(t, d):
    """[v2.5.9z79] a gate (PROBE-*) whose report.json says result FAIL."""
    if not t.startswith("PROBE-"):
        return False
    try:
        return json.load(open(os.path.join(d, "report.json"))).get("result") == "FAIL"
    except (OSError, ValueError, AttributeError):
        return False


def v_rerun(a, seq, flags=None, entry=None):
    """rerun <task> (v2.5.5, v2.5.6b): a finished probe or queue evaluation (FA-PROBE, EVAL-*) runs once more, or any
    task that failed for good (3 attempts) or was skipped because a dependency failed gets a new start (B300 18:52:
    SFT-P1 failed 3 times on a missing patch, E12b-P1 was skipped). A finished training is never rerun. .done,
    report.json, .failed and .skipped are kept as *.prev_<seq>; the queue starts it on the next free GPU.
    [v2.5.9z79] (ops-lead 10-24 03:14 UTC: seq 515 `rerun PROBE-8B` was rejected) a finished gate (PROBE-*) whose report.json says
    FAIL is rerun too, listed or not."""
    t = a[0]
    if running_pids(t):
        return "rejected", f"{t} is running", {}
    d = os.path.join(ROOT, "runs", t)
    done, failed = os.path.exists(os.path.join(d, ".done")), os.path.exists(os.path.join(d, ".failed"))
    skipped = os.path.exists(os.path.join(d, ".skipped"))
    if done and not _gate_failed(t, d) and not re.fullmatch(r"FA-PROBE|SPEED-PROBE|PROBE-4B|PROBE-TERM(?:[234]|4-4B|4-06B)?|PROBE-TOP10|STRICT-C1|PROBE-T14|PROBE-T32|PROBE-P5THINK|PROBE-P5SELF|PROBE-P5SELF-4B|PROBE-P5SELF-06B|PROBE-P2SELF|PROBE-E12C|PROBE-E12B-KL|PROBE-E12B-STOP|PROBE-E12B-STOP2|PROBE-E12B-8K|PROBE-E12B-16K|PROBE-E12B-R3|PROBE-X2507-X1|PROBE-X2507-X2|PROBE-X2507-X4|PROBE-LENSEL|PROBE-2GPU|PROBE-TOOL|PROBE-NET2?|BUILD-VENV-VL|PROBE-VLLM-VL|READINGS-EXTRA|PROBE-VLM|PROBE-ROLLOUT-U(60|75|85)|PROBE-STOPMERGE(?:-UNIT)?|PROBE-WDIFF|TOKCHECK-X5|VENV-VL-NVRTC|PROBE-MIXP|PROBE-PROG|PROBE-ROUTE-INV|PROBE-X5|PROBE-TAIL|PROBE-TAIL-BUD|PROBE-DIFF17|PROBE-DIFF17-4B|PROBE-DIFFPROBE|PROBE-PSTAR|PROBE-DENSE|PROBE-ROUTE|PROBE-POLISH|FA-SPEED-(17|4B|E12|TWIN)|EVAL-[A-Za-z0-9._-]+", t):
        return "rejected", f"{t} is done (only FA-PROBE, SPEED-PROBE, EVAL tasks and failed PROBE-* gates are rerun after finishing)", {}
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
    d = dr_root(); out = {"exists": os.path.isdir(d), "cap_gb": float(os.environ.get("DR_CAP_GB", "1500")), "tasks": {}, "keep": {}}
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
    if os.environ.get("CKPT_ROOT") and os.path.isdir(os.environ["CKPT_ROOT"]):   # v2.5.7y: du ckpt (rules allow it for du only)
        out["ckpt"] = os.path.realpath(os.environ["CKPT_ROOT"])
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


# ---------- [v2.5.9z72] (ops-lead 10-23 18:05 UTC, item 1; Runming 10-24 02:0x Beijing: everything of ours to HF as the cloud backup) ----------
INV_TRAINING = re.compile(r"(E0[2-9]|E1[0-9][a-e]?|SFT)-")
INV_WEIGHT = re.compile(r"^ckpt/(global_step_\d+)/actor/(huggingface(?:_bf16)?)/([^/]+\.safetensors)$")
INV_HF = re.compile(r"^ckpt/(global_step_\d+)/actor/huggingface(?:_bf16)?/(.+)$")   # the HF folder's config / tokenizer files
INV_SUMMARY = re.compile(r"(\.metrics\.json|(^|/)report\.json|(^|/)summary[^/]*\.json|(^|/)paired[^/]*\.json|\.md|\.csv)$")


def _repo_runs(kind):
    """{path: size} of every file under runs/ in one repository ('public', 'private', or 'old': the old public one, private
    since 10-18): one recursive listing."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import remote as _rm_mod
    api, repo = _rm_mod.hf("private" if kind == "old" else kind)
    repo = OLD_PUBLIC if kind == "old" else repo
    try:
        return {f.path: f.size for f in api.list_repo_tree(repo, path_in_repo="runs", recursive=True) if type(f).__name__ == "RepoFile"}
    except Exception as e:
        if _rm_mod.is_429(e):
            _rm_mod.note_429()
        raise


def inventory_scan(remote, root=None, ckpt=None, busy=None, unit=1e9):
    """[v2.5.9z72] what of ours is on this node only (read-only). remote: {repository: {path: size}}.
    Weights: each runs/<task>/ckpt/global_step_<N> with HF weights (huggingface_bf16, else huggingface) is in the cloud when one
    repository holds every one of its safetensors at runs/<task>/global_step_<N>/actor/huggingface/<file> at the same size, else
    queued when that destination is in control/weights_pending.txt, else local only. Every other file of runs/<task> is in the
    cloud when a repository holds runs/<task>/<path> at the same size; it counts as a log (*.log, *.err, *.out, *.log.gz), a
    reading (any task that is not a training) or other (a training's rollouts, configs, ...). The resume state (the rest of ckpt/)
    stays local by the checkpoint policy and is counted apart, as is logs/ (only tails go up). A done reading is missing when
    none of its summary files (*.metrics.json, report.json, summary*.json, paired*.json, *.md, *.csv) is in the cloud. Each
    inode is counted once (the holds are hard links). Returns the report."""
    root = root or ROOT
    ckpt = ckpt if ckpt is not None else os.environ.get("CKPT_ROOT", "")
    busy = busy if busy is not None else (lambda t: bool(running_pids(t)))
    runs = os.path.join(root, "runs")
    queued = set()
    for line in _read(os.path.join(root, "control", "weights_pending.txt")).splitlines():
        f = line.split("\t")
        if len(f) >= 2:
            queued.add(f[1].strip("/"))
    seen = set()

    def once(st):
        k = (st.st_dev, st.st_ino)
        if k in seen:
            return False
        seen.add(k); return True

    def up(path, size):
        return next((k for k, r in remote.items() if r.get(path) == size), None)

    gb = {c: {"local_only": 0, "queued": 0, "in_cloud": 0} for c in ("weights", "readings", "logs", "other")}
    gb["weights"]["second_copy"] = 0
    gb["resume_state_local"] = 0
    weights, missing, by_task = [], [], {}
    for t in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        d = os.path.join(runs, t)
        if t.startswith(".") or not os.path.isdir(d):
            continue
        training = bool(INV_TRAINING.match(t)); steps = {}; summ = []; done = os.path.exists(os.path.join(d, ".done"))
        for dp, dns, fs in os.walk(d, followlinks=True):
            for f in fs:
                p = os.path.join(dp, f); rel = os.path.relpath(p, d)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                m = INV_WEIGHT.match(rel)
                if m:
                    steps.setdefault(m.group(1), {}).setdefault(m.group(2), {})[m.group(3)] = (st.st_size, st)
                    continue
                if not once(st):
                    continue
                m = INV_HF.match(rel)
                if m:   # a weight folder's small files: with the weights, at the upload's path
                    gb["weights"]["in_cloud" if up(f"runs/{t}/{m.group(1)}/actor/huggingface/{m.group(2)}", st.st_size) else "local_only"] += st.st_size
                    continue
                if rel.startswith("ckpt/"):
                    gb["resume_state_local"] += st.st_size; continue
                cat = "logs" if f.endswith((".log", ".err", ".out", ".log.gz")) else ("other" if training else "readings")
                k = up(f"runs/{t}/{rel}", st.st_size)
                gb[cat]["in_cloud" if k else "local_only"] += st.st_size
                if not k and cat != "logs":
                    by_task[t] = by_task.get(t, 0) + st.st_size
                if cat == "readings" and INV_SUMMARY.search(rel):
                    summ.append(bool(k))
        if done and not training and summ and not any(summ):
            missing.append(t)
        final = _read(os.path.join(d, "final_step")).strip()
        for gs, folders in sorted(steps.items(), key=lambda x: int(x[0].split("_")[-1])):
            dst = f"runs/{t}/{gs}/actor/huggingface"
            where = None
            for name in ("huggingface_bf16", "huggingface"):
                fl = folders.get(name)
                if fl:
                    where = next((k for k in remote if all(remote[k].get(f"{dst}/{f}") == sz for f, (sz, _) in fl.items())), None)
                    if where:
                        break
            fl = folders.get("huggingface_bf16") or folders.get("huggingface")
            size = sum(sz for sz, st in fl.values() if once(st))
            for name, other in folders.items():   # an fp32 copy beside the bf16 one: local, counted apart
                if other is not fl:
                    gb["weights"]["second_copy"] += sum(sz for sz, st in other.values() if once(st))
            state = f"cloud:{where}" if where else ("queued" if dst in queued else "local")
            gb["weights"]["in_cloud" if where else ("queued" if dst in queued else "local_only")] += size
            weights.append({"task": t, "step": int(gs.split("_")[-1]), "gb": round(size / 1e9, 2), "state": state,
                            "final": final == gs.split("_")[-1], "running": busy(t),
                            "folder": "huggingface_bf16" if folders.get("huggingface_bf16") else "huggingface"})
    hold = os.path.join(ckpt, "weights_pending") if ckpt else ""
    for dp, dns, fs in os.walk(hold) if hold and os.path.isdir(hold) else []:   # holds whose run directory is gone
        dns[:] = [x for x in dns if not x.startswith(".stage_")]
        for f in fs:
            try:
                st = os.stat(os.path.join(dp, f))
            except OSError:
                continue
            if f.endswith(".safetensors") and once(st):
                gb["weights"]["queued"] += st.st_size
    logs = os.path.join(root, "logs")
    for dp, _, fs in os.walk(logs) if os.path.isdir(logs) else []:
        for f in fs:
            try:
                st = os.stat(os.path.join(dp, f))
            except OSError:
                continue
            if once(st):
                gb["logs"]["local_only"] += st.st_size
    out = {"utc": utc(), "repos": {k: len(v) for k, v in remote.items()},
           "gb": {c: ({k: round(v / unit, 1) for k, v in x.items()} if isinstance(x, dict) else round(x / unit, 1)) for c, x in gb.items()},
           "weights_local": [w for w in weights if w["state"] == "local"], "weights_queued": [w for w in weights if w["state"] == "queued"],
           "weights_in_cloud": len([w for w in weights if w["state"].startswith("cloud")]), "readings_missing": missing,
           "local_only_by_task_gb": dict(sorted(((t, round(b / 1e9, 2)) for t, b in by_task.items()), key=lambda x: -x[1])[:40])}
    return out


def inventory_pend(rep, root=None):
    """[v2.5.9z72] (ops-lead: what is missing goes to weights_pending.txt or the results queue) every local-only weight step of a
    task that is not running into the weights queue (remote.py defer_weight: hard links under $CKPT_ROOT/weights_pending, one
    line in weights_pending.txt), every done reading with no summary in the cloud into the results queue (remote.py
    pend_result, unless it already waits there). Returns (weights, readings) added."""
    root = root or ROOT
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import remote as _rm_mod
    nw = nr = 0
    for w in rep["weights_local"]:
        if w["running"]:
            continue
        src = os.path.join(root, "runs", w["task"], "ckpt", f"global_step_{w['step']}", "actor", w["folder"])
        try:
            _rm_mod.defer_weight(src, f"runs/{w['task']}/global_step_{w['step']}/actor/huggingface"); nw += 1
        except Exception as e:
            print(f"INVENTORY_PEND_FAILED {w['task']} s{w['step']}: {type(e).__name__}", file=sys.stderr)
    for t in rep["readings_missing"]:
        d = os.path.join(root, "runs", t)
        if os.path.exists(os.path.join(d, ".upload_pending")):
            continue
        try:
            _rm_mod.pend_result(d, f"runs/{t}"); nr += 1
        except Exception as e:
            print(f"INVENTORY_PEND_FAILED {t}: {type(e).__name__}", file=sys.stderr)
    return nw, nr


def v_inventory(a, seq, flags=None, entry=None):
    """inventory [pend] ([v2.5.9z72], read-only unless pend): in the background, one recursive listing of runs/ in each of the
    three repositories against this node's runs/, the holds and logs/; the receipt gives the local-only GB by weights /
    readings / logs / other; the full report to logs/<node>/inventory/ (private). pend: then queues what is missing."""
    if _backoff_left() > 0:
        return "rejected", "the HF API backs off (429): later", {}
    subprocess.Popen(["setsid", sys.executable, os.path.abspath(__file__), "do_inventory", str(seq)] + list(a[:1]),
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "accepted", "inventory started (read-only" + ("; then the missing items queued)" if a else ")"), {}


def do_inventory(seq, pend=False):
    t0 = time.time()
    try:
        remote = {k: _repo_runs(k) for k in ("public", "private", "old")}
    except Exception as e:
        receipt(seq, None, "failed", f"listing failed: {type(e).__name__}"); return
    rep = inventory_scan(remote)
    if pend:
        rep["pended_weights"], rep["pended_readings"] = inventory_pend(rep)
    rep["seconds"] = round(time.time() - t0, 1)
    d = _p("inventory"); os.makedirs(d, exist_ok=True)
    json.dump(rep, open(os.path.join(d, "inventory.json"), "w"), indent=1)
    ok = remote_put(d, f"logs/{node_id()}/inventory")
    g = rep["gb"]
    receipt(seq, None, "ok", f"local only: weights {g['weights']['local_only']} GB (+{g['weights']['queued']} GB queued), readings "
            f"{g['readings']['local_only']} GB ({len(rep['readings_missing'])} done readings with no summary in the cloud), logs "
            f"{g['logs']['local_only']} GB, other {g['other']['local_only']} GB; resume state {g['resume_state_local']} GB (local by policy)"
            + (f"; queued now: {rep['pended_weights']} weight steps, {rep['pended_readings']} readings" if pend else "")
            + ("" if ok else "; the report upload failed (kept in control/inventory)"),
            gb=g, weights_local=[f"{w['task']} s{w['step']} {w['gb']} GB" for w in rep["weights_local"]][:60],
            readings_missing=rep["readings_missing"][:60], seconds=rep["seconds"])


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


def _dr_keep(t, step=None):
    """Copy one step's model of task t to $(dr_root)/keep/<t>/ckpt/<step>/: (ok, message, GB). step None = the final step
    (runs/<t>/final_step, else the newest). The bf16 copy is preferred over the fp32 HF save; SFT keeps huggingface/."""
    ck = os.path.join(ROOT, "runs", t, "ckpt")
    if step is None:
        fs = (_read(os.path.join(ROOT, "runs", t, "final_step")) or "").strip()
        steps = sorted((int(m.group(1)), d) for d in os.listdir(ck) for m in [re.match(r"global_step_(\d+)$", d)] if m)
        step = next((d for n, d in steps if str(n) == fs), steps[-1][1] if steps else None)
    src = None
    for sub in ("huggingface", os.path.join("actor", "huggingface_bf16"), os.path.join("actor", "huggingface")):
        p = os.path.join(ck, step, sub) if step else ""
        if step and os.path.isdir(p) and any(f.endswith(".safetensors") for f in os.listdir(p)):
            src = (p, sub); break
    if not src:
        return False, f"{t}: no {step or 'final'} model with weights under ckpt/", None
    dst = os.path.join(dr_root(), "keep", t, "ckpt", step, src[1])
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.rmtree(dst + ".tmp", ignore_errors=True)
    rc = subprocess.run(["nice", "-n", "19", "cp", "-a", src[0], dst + ".tmp"]).returncode
    if rc == 0:
        shutil.rmtree(dst, ignore_errors=True); os.rename(dst + ".tmp", dst)
        for f in ("final_step", "config.yaml"):
            if os.path.isfile(os.path.join(ROOT, "runs", t, f)):
                shutil.copy2(os.path.join(ROOT, "runs", t, f), os.path.join(dr_root(), "keep", t, f))
    gb = round(_tree_bytes(dst) / 1e9, 2) if rc == 0 else None
    return rc == 0, f"{t}: {step}/{src[1]} {'kept on the share' if rc == 0 else 'copy failed'}" + (f", {gb} GB" if gb else ""), gb


def do_dr_keep(seq, t):
    t0 = time.time(); ok, msg, gb = _dr_keep(t)
    receipt(seq, None, "ok" if ok else "failed", msg, kept_gb=gb, seconds=round(time.time() - t0, 1))
    event(t, "dr_kept" if ok else "dr_keep_failed", msg)


def dr_keep_auto(t):
    """[v2.5.8] (reviewer 08:2x UTC 10-19: DR first, the B300 may be released mid-week) called by upload_sync.sh when a
    task's uploads are done: the final step and every uploaded step (UPLOAD_WEIGHTS) are kept on the share, one event
    each; a step over DR_CAP_GB is skipped with an event dr_cap. DR=0 turns it off."""
    if os.environ.get("DR", "1") != "1":
        return 0
    ck = os.path.join(ROOT, "runs", t, "ckpt")
    if not os.path.isdir(ck):
        return 0
    up = [d for d in os.listdir(ck) if re.match(r"global_step_\d+$", d) and os.path.exists(os.path.join(ck, d, ".uploaded"))
          and not os.path.exists(os.path.join(ck, d, ".not_uploaded"))]
    n = 0
    for step in [None] + sorted(up, key=lambda d: int(d.rsplit("_", 1)[1])):
        if step is not None and os.path.isdir(os.path.join(dr_root(), "keep", t, "ckpt", step)):
            continue   # the final step, already kept above
        r = dr_report()
        if r.get("over_cap"):
            event(t, "dr_cap", f"keep of {step or 'final'} skipped: DR_ROOT {r.get('total_gb')} GB >= {r['cap_gb']} GB"); break
        ok, msg, gb = _dr_keep(t, step)
        event(t, "dr_kept" if ok else "dr_keep_failed", msg); n += ok
    return n


def lru_clean(limit_gb=None):
    """[v2.5.8] (Runming via the reviewer, 10-19 08:5x UTC: local weights kept, the local disk up to 5 TB) with
    LOCAL_LRU_GB set (setenv; unset = off, the default): while the HF weights of our checkpoints (runs/<task>/ckpt, on
    CKPT_ROOT or the old local layout) hold more than LOCAL_LRU_GB, delete the oldest eligible copy, one event
    lru_deleted each. Eligible: a step whose weights went to the result repository (.uploaded, no .not_uploaded) of a
    task that is not running and is not its final step (runs/<task>/final_step, else its newest step). Order: an fp32
    save that has a local bf16 copy, then an fp32 save, then a bf16 copy; oldest first in each. Never a final step,
    a resume state (*.pt) or anything outside actor/huggingface*. Returns the GB freed."""
    limit_gb = limit_gb if limit_gb is not None else os.environ.get("LOCAL_LRU_GB")
    if not limit_gb:
        return 0.0
    limit = float(limit_gb) * 1e9; seen, cands, total = set(), [], 0
    size = lambda d: sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".safetensors")) if os.path.isdir(d) else 0
    for t in sorted(os.listdir(os.path.join(ROOT, "runs"))) if os.path.isdir(os.path.join(ROOT, "runs")) else []:
        ck = os.path.realpath(os.path.join(ROOT, "runs", t, "ckpt"))
        if ck in seen or not os.path.isdir(ck):
            continue
        seen.add(ck)
        steps = sorted(int(m.group(1)) for d in os.listdir(ck) for m in [re.match(r"global_step_(\d+)$", d)] if m)
        fs = (_read(os.path.join(ROOT, "runs", t, "final_step")) or "").strip()
        final = int(fs) if fs.isdigit() else (steps[-1] if steps else None)
        busy = bool(running_pids(t))
        for n in steps:
            sd = os.path.join(ck, f"global_step_{n}"); fp, bf = os.path.join(sd, "actor", "huggingface"), os.path.join(sd, "actor", "huggingface_bf16")
            a, b = size(fp), size(bf); total += a + b
            if busy or n == final or not os.path.exists(os.path.join(sd, ".uploaded")) or os.path.exists(os.path.join(sd, ".not_uploaded")):
                continue
            if a:
                cands.append((0 if b else 1, os.path.getmtime(fp), t, n, fp, a))
            if b:
                cands.append((2, os.path.getmtime(bf), t, n, bf, b))
    freed = 0
    for tier, _, t, n, d, b in sorted(cands):
        if total <= limit:
            break
        for f in os.listdir(d):
            if f.endswith(".safetensors"):
                os.remove(os.path.join(d, f))
        total -= b; freed += b
        event(t, "lru_deleted", f"global_step_{n}/actor/{os.path.basename(d)} {b / 1e9:.1f} GB (our weights {total / 1e9:.0f} GB, limit {limit / 1e9:.0f} GB)")
    return round(freed / 1e9, 2)


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
    prev = set(_state_read().get("alerts", []))
    now = set()
    for k in ("disk_alert", "ckpt_disk_alert"):
        if st.get(k):
            now.add(k)
    for i, r in enumerate(st.get("ckpt_roots") or []):   # one alert per candidate root (v2.5)
        if r.get("alert") and r.get("writable"):
            # v2.5.8: a symbolic target (root<i>, its mount's free GB in the detail), never the path: the path in the
            # event file name made every round raise FileNotFoundError from 2026-10-17 09:31 UTC on, and the alert,
            # with every alert after it, never went out
            now.add(f"ckpt_root_alert:root{i}")
    ups = subprocess.run(["pgrep", "-af", "scripts/upload_syn[c].sh"], capture_output=True, text=True).stdout
    for pid in running_pids():
        m = re.search(r"task\.sh\s+\d+(?:,\d+)*\s+(\S+)", _read(f"/proc/{pid}/cmdline").replace("\0", " ") + " ")
        if not m:
            continue
        t = m.group(1); lg = os.path.join(ROOT, "logs", f"{t}.log")
        if os.path.exists(lg) and time.time() - os.path.getmtime(lg) > STALL_S:
            now.add(f"stalled:{t}")
        if re.match(r"(E10b|E10c|E10d|E12b|E12c|E12d)-", t) and f"/runs/{t}" not in ups:   # [v2.5.8a] E10c / E12c
            now.add(f"upload_sync_missing:{t}")
    rsv = reserve_low()   # [v2.5.9z12] fewer tasks ready than free GPUs + 2, for 10 minutes
    if rsv:
        now.add("reserve_low")
    roots = {f"root{i}": {"free_gb": r.get("free_gb"), "current": r.get("current")} for i, r in enumerate(st.get("ckpt_roots") or [])}
    state(alerts=sorted(now))   # v2.5.8: saved first, so a failing event can never re-fire every round
    for a in sorted(now - prev):
        tgt = a.split(":", 1)[1] if ":" in a else "node"
        detail = {k: st.get(k) for k in ("disk_free_gb", "ckpt_disk_free_gb")}
        if a == "reserve_low":
            detail = {k: rsv.get(k) for k in ("ready", "free_gpus", "need", "ready_tasks", "held_by_disk")}
        if tgt in roots:
            detail["root"] = roots[tgt]
        try:
            event(tgt, a.split(":", 1)[0], json.dumps(detail))
        except Exception as e:   # v2.5.8: one failing event never stops the others
            print(f"ALERT_EVENT_FAILED {a}: {type(e).__name__}", file=sys.stderr)
    for a in sorted(prev - now):
        try:
            event(a.split(":", 1)[1] if ":" in a else "node", a.split(":", 1)[0] + "_cleared")
        except Exception as e:
            print(f"ALERT_EVENT_FAILED {a}: {type(e).__name__}", file=sys.stderr)
    return sorted(now)


# ---------- [v2.5.9z12] (ops-lead / the reviewer 10-22, the offline-readiness list) queue reruns, the disk guard, the reserve ----------
AT_WINDOW_MIN = int(os.environ.get("AT_WINDOW_MIN", "120"))


def _auto_rows(qfile):
    """'#at <UTC HH:MM> rerun <task>' and '#after <task> rerun <task2>' lines of the queue file."""
    at, after = [], []
    for line in _read(qfile).splitlines():
        f = line.split()
        if len(f) == 4 and f[0] == "#at" and f[2] == "rerun" and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", f[1]) and R.TASK_RE.match(f[3]):
            at.append((f[1], f[3]))
        elif len(f) == 4 and f[0] == "#after" and f[2] == "rerun" and R.TASK_RE.match(f[1]) and R.TASK_RE.match(f[3]):
            after.append((f[1], f[3]))
    return at, after


def _auto_rerun(kind, tag, t):
    """One rerun from a queue row: the rerun allowlist of a control command (control_rules), then v_rerun (seq = tag)."""
    try:
        R.parse({"cmd": f"rerun {t}", "reason": f"queue #{kind} row"})
    except R.Reject as e:
        return "refused", str(e)
    st, msg, _ = v_rerun([t], tag)
    return st, msg


def auto_reruns(qfile, now=None):
    """run_tasks.sh every pass, before it dispatches.
    #at HH:MM rerun <task>: once a UTC day, within AT_WINDOW_MIN (120) minutes after HH:MM (a node down at the time skips
    that day); while the task runs it is tried again later inside the window; a task still queued (neither done, failed nor
    skipped) needs nothing that day.
    #after <task> rerun <probe>: when <task>'s .done is newer than <probe>'s (a reading landed after the probe last finished)
    and neither runs. The probe's rerun moves its .done away and its next .done is newer, so nothing repeats; a refusal is
    not repeated for the same .done of <task> (auto_rerun/after_<task>__<probe> holds its mtime).
    Each rerun or refusal: an event (at_rerun_<status> / after_rerun_<status>) and a line AUTO_RERUN on stdout."""
    now = time.time() if now is None else now
    at, after = _auto_rows(qfile)
    if not (at or after):
        return []
    os.makedirs(_p("auto_rerun"), exist_ok=True)
    day, out = time.strftime("%Y%m%d", time.gmtime(now)), []
    for hhmm, t in at:
        h, m = map(int, hhmm.split(":"))
        start = calendar.timegm(time.gmtime(now)[:3] + (h, m, 0))
        mark = _p("auto_rerun", f"at_{hhmm.replace(':', '')}_{t}")
        if not start <= now < start + AT_WINDOW_MIN * 60 or _read(mark).strip() == day:
            continue
        st, msg = _auto_rerun("at", f"at{day}T{hhmm.replace(':', '')}", t)
        if st == "rejected" and msg.endswith("is running"):
            continue
        open(mark, "w").write(day + "\n")
        out.append((t, f"at_rerun_{st}", f"#at {hhmm} rerun {t}: {msg}"))
    for src, t in after:
        ds, dt = os.path.join(ROOT, "runs", src, ".done"), os.path.join(ROOT, "runs", t, ".done")
        if not (os.path.exists(ds) and os.path.exists(dt)) or os.path.getmtime(ds) <= os.path.getmtime(dt):
            continue
        mark = _p("auto_rerun", f"after_{src}__{t}")
        if _read(mark).strip() == str(os.path.getmtime(ds)) or running_pids(src) or running_pids(t):
            continue
        st, msg = _auto_rerun("after", f"after{time.strftime('%Y%m%dT%H%M%S', time.gmtime(now))}", t)
        open(mark, "w").write(str(os.path.getmtime(ds)) + "\n")
        out.append((t, f"after_rerun_{st}", f"#after {src} rerun {t}: {msg}"))
    for t, kind, msg in out:
        print(f"AUTO_RERUN {t} {kind} {msg}")
        try:
            event(t, kind, msg)
        except Exception as e:
            print(f"AUTO_RERUN_EVENT_FAILED {t}: {type(e).__name__}", file=sys.stderr)
    return out


def _cfg(name, default):
    """[v2.5.9z48] a setting from $OPD_ROOT/control.env (setenv; read on every call, so a running loop sees it), else the
    environment, else the default."""
    try:
        for line in open(os.path.join(ROOT, "control.env")):
            k, _, x = line.strip().removeprefix("export ").partition("=")
            if k == name and x.strip().strip('"'):
                return x.strip().strip('"')
    except OSError:
        pass
    return os.environ.get(name) or default


def disk_low():
    """The disks below the heartbeat's alert lines: 'work' (OPD_ROOT, DISK_ALERT_GB) and / or 'ckpt' (CKPT_ROOT,
    CKPT_ALERT_GB 100), space separated; '' when both are above. run_tasks.sh holds new trainings while it is not empty.
    [v2.5.9z48] (ops-lead / the reviewer 10-22 19:0x Beijing: the shared work disk lost ~4.8 TB to others in 30 min) the work
    line 1000 GB (was 30), both lines read from control.env first (`setenv DISK_ALERT_GB=` / `CKPT_ALERT_GB=` apply at once)."""
    low = []
    for name, path, var, dflt in (("work", ROOT, "DISK_ALERT_GB", "1000"), ("ckpt", os.environ.get("CKPT_ROOT", ""), "CKPT_ALERT_GB", "100")):
        if path and os.path.isdir(path):
            s = os.statvfs(path)
            if s.f_bavail * s.f_frsize / 1e9 < float(_cfg(var, dflt)):
                low.append(name)
    return " ".join(low)


# [v2.5.9z48] (the reviewer 10-22 19:0x Beijing via ops-lead; the 10-19 control-plane hang) the control plane's small writes
# leave the work disk before it fills: below CTRL_FALLBACK_LOW_GB (200) free, $OPD_ROOT/control (heartbeat, pulse, events,
# receipts, the upload outbox) and $OPD_ROOT/logs move to the checkpoint disk and their old paths become symlinks to it, so
# every writer goes on unchanged; above CTRL_FALLBACK_HIGH_GB (400) they move back (the gap: no flapping). Below DISK_FULL_GB
# (1) one event disk_write_fail per episode: running tasks' writes there may fail, and nothing is stopped.
# [v2.5.9z71] (ops-lead 10-23 18:05 UTC, Runming's disk policy of 10-24 02:0x Beijing: the shared work disk is ours to arrange)
# a planned switch: with CTRL_ON_CKPT (1; control.env first, read on every call) control/ and logs/ live on the checkpoint disk
# whatever the free space; setenv CTRL_ON_CKPT=0 is the rollback (back on the work disk; z48's two lines apply again). z48's
# swap was two renames with a moment between them where control/ did not exist, and the loops' open log files stayed on the
# old disk. Now each switch: copies the directory into a fresh $CKPT_ROOT/opd_fallback/<name>.<utc> (twice: the bulk, then
# what changed meanwhile; each file through a temp file, so a copy cut short never stands as a finished one), swaps the
# directory and a ready symlink in ONE renameat2(RENAME_EXCHANGE) (control/ exists at every moment), copies once more what
# was written in between, and turns each file that a live process still holds open in the old directory (a running task's
# logs/<task>.log) into a symlink to that file, so readers see what the writer writes. The loops' own logs and locks are
# left to the loops: run_tasks.sh reopens its log and restarts the others (control/ctrl_reopen). Where the work disk has no
# atomic exchange nothing is switched (event ctrl_switch_refused). The old directory stays as <name>.pre_fallback_<utc>
# and every copy on the checkpoint disk stays: nothing is removed (the live links point into the old directory).
FALLBACK_DIRS = ("control", "logs")
LOOPS = ("run_tasks", "heartbeat", "watchdog", "uploader", "gh_heartbeat", "weights_replay")   # [v2.5.9z71] the resident loops


def _free_gb(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def _sync_tree(src, dst):
    """Copy the files of src that dst lacks or has older (small files); nothing removed. The number copied."""
    n = 0
    for dirpath, _, files in os.walk(src):
        d = os.path.join(dst, os.path.relpath(dirpath, src))
        os.makedirs(d, exist_ok=True)
        for f in files:
            s, t = os.path.join(dirpath, f), os.path.join(d, f)
            try:
                if not os.path.exists(t) or os.path.getmtime(s) > os.path.getmtime(t):
                    shutil.copy2(s, t); n += 1
            except OSError:
                pass
    return n


def _exchange(a, b):
    """[v2.5.9z71] swap the two paths in one step: Linux renameat2(RENAME_EXCHANGE) (glibc's, else the raw system call),
    macOS renamex_np(RENAME_SWAP) (the tests); OSError when the system or the filesystem has none."""
    import ctypes
    import errno
    import platform
    libc = ctypes.CDLL(None, use_errno=True)
    fa, fb = os.fsencode(a), os.fsencode(b)
    if sys.platform.startswith("linux"):
        try:
            f = libc.renameat2
            f.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            r = f(-100, fa, -100, fb, 2)   # AT_FDCWD, RENAME_EXCHANGE
        except AttributeError:   # glibc older than 2.28
            nr = {"x86_64": 316, "aarch64": 276}.get(platform.machine())
            if nr is None:
                raise OSError(errno.ENOSYS, f"no renameat2 number for {platform.machine()}")
            libc.syscall.restype = ctypes.c_long
            r = libc.syscall(ctypes.c_long(nr), ctypes.c_int(-100), ctypes.c_char_p(fa), ctypes.c_int(-100), ctypes.c_char_p(fb), ctypes.c_uint(2))
    elif sys.platform == "darwin":
        f = libc.renamex_np
        f.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        r = f(fa, fb, 2)   # RENAME_SWAP
    else:
        raise OSError(errno.ENOSYS, f"no atomic exchange on {sys.platform}")
    if r != 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e), a)


def _exchange_selftest(root):
    """[v2.5.9z71] one exchange of a scratch directory and a scratch symlink in root (the work disk): OSError when it fails."""
    d, ln = os.path.join(root, f".z71_xtest_d.{os.getpid()}"), os.path.join(root, f".z71_xtest_l.{os.getpid()}")
    os.makedirs(d, exist_ok=True)
    if os.path.lexists(ln):
        os.unlink(ln)
    os.symlink(d, ln)
    try:
        _exchange(ln, d)
        if not (os.path.islink(d) and os.path.isdir(ln) and not os.path.islink(ln)):
            raise OSError(5, "the exchange did not swap the two entries")
    finally:
        for p in (d, ln):
            if os.path.islink(p):
                os.unlink(p)
            elif os.path.isdir(p):
                os.rmdir(p)


# [v2.5.9z71b] (the node 10-23 19:01 UTC: renameat2(RENAME_EXCHANGE) -> EINVAL on the shared work disk, so z71 refused) the errors
# that mean "this filesystem has no exchange": the swap then falls back to two renames back to back (_put_at)
NO_EXCHANGE = (22, 38, 95, 45)   # EINVAL, ENOSYS, EOPNOTSUPP / ENOTSUP (Linux 95; macOS 45)


def _put_at(new, p, old, stamp, exchange=True):
    """[v2.5.9z71b] new (a ready symlink or directory) to p, and what was at p to old. With exchange: one renameat2
    exchange, then old's rename (p never missing). Without (the filesystem has none): rename(p, old) and rename(new, p) back to
    back: p is missing for the time between the two calls (returned, in ns, for the event); a directory a writer created at p
    inside that gap (mkdir -p / makedirs) is moved to <p>.gap_<utc> and the second rename tried once more; if it still fails the
    old directory goes back to p (nothing switched; what the stray directory held is copied into it) and the error is raised.
    Returns (gap_ns, stray or None)."""
    if exchange:
        _exchange(new, p); os.rename(new, old); return 0, None
    t0 = time.perf_counter_ns()
    os.rename(p, old)
    try:
        os.rename(new, p); return time.perf_counter_ns() - t0, None
    except OSError:
        pass
    stray = None
    if os.path.isdir(p) and not os.path.islink(p):
        stray = f"{p}.gap_{stamp}"; os.rename(p, stray)
    try:
        os.rename(new, p); return time.perf_counter_ns() - t0, stray
    except OSError:
        if not os.path.lexists(p):
            os.rename(old, p)
            if stray:
                _copy_tree(stray, p)
        raise


def _copy_tree(src, dst):
    """[v2.5.9z71] copy what dst lacks or has older (by mtime: a file written at dst after the swap is never overwritten
    by the older one): each file through dst's own temp file and one rename; symlinks as symlinks; a symlink already at
    dst (a live link) left alone; nothing removed. The number copied."""
    n = 0
    for dirpath, dirs, files in os.walk(src):
        d = os.path.join(dst, os.path.relpath(dirpath, src))
        os.makedirs(d, exist_ok=True)
        for f in files + [x for x in dirs if os.path.islink(os.path.join(dirpath, x))]:
            s, t = os.path.join(dirpath, f), os.path.join(d, f)
            if f.endswith(".z71tmp"):
                continue
            tmp = f"{t}.z71tmp"
            try:
                if os.path.islink(t):
                    continue
                if os.path.islink(s):
                    if not os.path.lexists(t):
                        os.symlink(os.readlink(s), tmp); os.replace(tmp, t); n += 1
                    continue
                if os.path.exists(t) and os.stat(s).st_mtime <= os.stat(t).st_mtime:
                    continue
                shutil.copy2(s, tmp); os.replace(tmp, t); n += 1
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    return n


def _open_files(prefix):
    """[v2.5.9z71] the files under prefix (a real path) that a live process holds open (Linux /proc/<pid>/fd; an empty set
    elsewhere)."""
    out, pre = set(), prefix.rstrip("/") + "/"
    for pid in os.listdir("/proc") if os.path.isdir("/proc/self/fd") else []:
        if not pid.isdigit() or int(pid) == os.getpid():   # not this check's own log (logs/control.log)
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                t = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if t.startswith(pre) and not t.endswith(" (deleted)"):
                out.add(t)
    return out


SHARED_LOGS = ("control.log", "git_fetch.log")   # [v2.5.9z71] appended by many short commands, each opening the path anew


def _loop_file(name):
    """[v2.5.9z71] a resident loop's own log / error log / lock / sha file (the loop reopens it when it restarts) or a log
    many short commands append to (the next one opens the new path): never turned into a live link."""
    return name.endswith(".lock") or name in SHARED_LOGS or any(name.startswith(x + ".") for x in LOOPS)


def _link_live(old, new, held):
    """[v2.5.9z71] each file of old (a real path) in held, the loops' own files aside, becomes at new a symlink to the file
    in old (one rename over the copy): a running task writes on through its open fd and readers of the new path read it.
    The number linked."""
    n = 0
    for f in sorted(held):
        rel = os.path.relpath(f, old)
        if rel.startswith("..") or _loop_file(os.path.basename(f)) or not os.path.isfile(f):
            continue
        t = os.path.join(new, rel)
        if os.path.islink(t):
            continue
        os.makedirs(os.path.dirname(t), exist_ok=True)
        tmp = f"{t}.z71tmp"
        if os.path.lexists(tmp):
            os.unlink(tmp)
        os.symlink(f, tmp); os.replace(tmp, t); n += 1
    return n


def _clear(p, stamp):
    """[v2.5.9z71] a leftover at a switch's temp name: a symlink is removed, a directory (a switch cut short after its swap)
    kept as <p>.left_<utc>."""
    if os.path.islink(p):
        os.unlink(p)
    elif os.path.exists(p):
        os.rename(p, f"{p}.left_{stamp}")


def _switch_on(root, fb, n, stamp, held_fn, exchange=True, gaps=None):
    p, q = os.path.join(root, n), os.path.join(fb, f"{n}.{stamp}")
    os.makedirs(p, exist_ok=True)
    _copy_tree(p, q); _copy_tree(p, q)                                  # the bulk, then what changed during it
    tmp = p + ".z71_link"; _clear(tmp, stamp); os.symlink(q, tmp)
    old = f"{p}.pre_fallback_{stamp}"
    try:
        gap, stray = _put_at(tmp, p, old, stamp, exchange)               # p: the symlink; old: the old directory
    except OSError:
        if os.path.islink(tmp):
            os.unlink(tmp)
        raise
    if gaps is not None:
        gaps.append(gap)
    _copy_tree(old, q)                                                   # what was written between the copy and the swap
    if stray:
        _copy_tree(stray, q)                                             # [v2.5.9z71b] and what a writer put into the gap
    return _link_live(os.path.realpath(old), q, held_fn(os.path.realpath(old)))


def _switch_off(root, n, stamp, held_fn, exchange=True, gaps=None):
    p = os.path.join(root, n); q = os.path.realpath(p)
    back = p + ".z71_back"; _clear(back, stamp); os.makedirs(back)
    _copy_tree(q, back); _copy_tree(q, back)
    off = p + ".z71_off"; _clear(off, stamp)
    try:
        gap, stray = _put_at(back, p, off, stamp, exchange)              # p: a directory again; off: the symlink
    except OSError:
        if os.path.isdir(back) and not os.path.islink(back):
            os.rename(back, f"{back}.left_{stamp}")
        raise
    if gaps is not None:
        gaps.append(gap)
    if os.path.islink(off):
        os.unlink(off)
    _copy_tree(q, p)
    if stray:
        _copy_tree(stray, p)
    return _link_live(q, p, held_fn(q))                                  # q stays on the checkpoint disk


def disk_fallback(root=None, ckpt=None, free_fn=None, emit=True, stamp=None, check_dev=True, blocking=True, held_fn=None):
    """One check (run_tasks.sh every round, in the background): 'on' / 'off' / 'switched_on' / 'switched_off', 'refused'
    (no atomic exchange on the work disk), 'busy' (blocking=False and another check holds the lock), or 'n/a' (no
    checkpoint disk apart from the work disk). [v2.5.9z71] on while CTRL_ON_CKPT (1) is 1, or while z48's lines say so."""
    import fcntl
    root = root or ROOT
    ckpt = ckpt if ckpt is not None else os.environ.get("CKPT_ROOT", "")
    if not ckpt or not os.path.isdir(ckpt) or (check_dev and os.stat(ckpt).st_dev == os.stat(root).st_dev):
        return "n/a"
    free = (free_fn or _free_gb)(root)
    low, high = float(_cfg("CTRL_FALLBACK_LOW_GB", "200")), float(_cfg("CTRL_FALLBACK_HIGH_GB", "400"))
    planned = _cfg("CTRL_ON_CKPT", "1") == "1"
    fb = os.path.join(ckpt, "opd_fallback"); os.makedirs(fb, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    held_fn = held_fn or _open_files
    linked, why, emit_refused, gaps, exchange = 0, "", False, [], True
    with open(os.path.join(fb, ".lock"), "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "busy"
        links = {n: os.path.islink(os.path.join(root, n)) for n in FALLBACK_DIRS}
        active = links["control"]
        want = planned or free < low or (active and free <= high)
        todo = [n for n in FALLBACK_DIRS if links[n] != want]   # a switch cut short is finished by the next check
        state = "on" if active else "off"
        refused_mark = os.path.join(fb, "ctrl_refused_reported")
        if todo:
            try:
                try:
                    _exchange_selftest(root); exchange = True
                except OSError as e:   # [v2.5.9z71b] no exchange on this filesystem: two renames back to back
                    if e.errno not in NO_EXCHANGE:
                        raise
                    exchange = False
                for n in todo:
                    linked += (_switch_on(root, fb, n, stamp, held_fn, exchange, gaps) if want else _switch_off(root, n, stamp, held_fn, exchange, gaps))
                state = "switched_on" if want else "switched_off"
                if os.path.exists(refused_mark):
                    os.unlink(refused_mark)
            except OSError as e:
                state, why = "refused", f"{e.strerror or 'no reason given'} ({type(e).__name__} errno {e.errno})"   # never the path
                if os.path.exists(refused_mark):
                    emit_refused = False
                else:
                    open(refused_mark, "w").write(stamp); emit_refused = True
            if any(os.path.islink(os.path.join(root, n)) != links[n] for n in FALLBACK_DIRS):
                cd = os.path.join(root, "control")   # the loops reopen their files at the new paths (run_tasks.sh)
                open(os.path.join(cd, "ctrl_reopen"), "w").write(stamp + "\n")
                for x in LOOPS:
                    if x not in ("run_tasks", "weights_replay"):   # run_tasks reopens its log; the replay restarts between two uploads
                        try:
                            os.unlink(os.path.join(cd, x + ".sha"))
                        except OSError:
                            pass
                open(os.path.join(cd, "ctrl_reopen_wr"), "w").write(stamp + "\n")
        mark = os.path.join(fb, "disk_full_reported")
        full = free < float(_cfg("DISK_FULL_GB", "1"))
        if full and not os.path.exists(mark):
            open(mark, "w").write(stamp)
            if emit:
                event("node", "disk_write_fail", f"work disk {free:.1f} GB free: running tasks' writes there (train.log, rollouts, "
                      "readings) may fail; nothing is stopped")
        elif not full and free > 10 and os.path.exists(mark):
            os.unlink(mark)
    if emit and state.startswith("switched"):
        on = state == "switched_on"
        kind = "ctrl_switch_on" if on and planned else ("disk_fallback_on" if on else "disk_fallback_off")   # the rollback: _off too
        how = "one exchange each" if exchange else f"two renames each (no exchange on the work disk), gaps {', '.join(f'{g / 1000:.0f}' for g in gaps)} us"
        event("node", kind, f"work disk {free:.1f} GB free, CTRL_ON_CKPT={int(planned)} (z48 lines {low:.0f} / {high:.0f} GB): "
              f"control/ and logs/ {'on the checkpoint disk' if on else 'back on the work disk'}; {how}; {linked} open files linked")
    elif emit and state == "refused" and emit_refused:   # once per episode
        event("node", "ctrl_switch_refused", f"control/ and logs/ stay where they are: {why}")
    return state


def tasks_disk_low():
    """[v2.5.9z71] (ops-lead 10-23 18:05 UTC) 'work' while the work disk has less than TASKS_DISK_MIN_GB (300) free, 'work
    critical' below TASKS_DISK_CRIT_GB (100), else ''; both lines from control.env first, read on every call. run_tasks.sh
    then starts no new task of any kind (trainings, readings, probes, the CPU lane, ad-hoc runs) and no new fetch; the
    running ones go on, nothing is stopped (below 100 GB ops-lead decides about their upload processes)."""
    free = _free_gb(ROOT)
    if free < float(_cfg("TASKS_DISK_CRIT_GB", "100")):
        return "work critical"
    return "work" if free < float(_cfg("TASKS_DISK_MIN_GB", "300")) else ""


def _markers_only(p):
    """[v2.5.9z71c] a run directory place_run may move: no subdirectory, at most RUNS_MOVE_MAX_MB (64) in all."""
    files = []
    for e in os.scandir(p):
        if e.is_dir(follow_symlinks=False):
            return False
        files.append(e)
    return sum(e.stat(follow_symlinks=False).st_size for e in files) <= float(_cfg("RUNS_MOVE_MAX_MB", "64")) * 1e6


def runs_ckpt(root=None, ckpt=None, check_dev=True):
    """[v2.5.9z71c] '1' while place_run puts new run directories on the checkpoint disk (RUNS_ON_CKPT=1 and a checkpoint disk
    apart from the work disk), else '0'."""
    root = root or ROOT
    ckpt = ckpt if ckpt is not None else os.environ.get("CKPT_ROOT", "")
    if _cfg("RUNS_ON_CKPT", "1") != "1" or not ckpt or not os.path.isdir(ckpt):
        return "0"
    return "0" if check_dev and os.stat(ckpt).st_dev == os.stat(root).st_dev else "1"


def run_placeable(t, root=None):
    """[v2.5.9z71c] run_tasks.sh, for a task whose runs/<task> is a real directory: True when place_run will move it to the checkpoint
    disk (marker files only), False when it stays on the work disk (an earlier attempt's outputs)."""
    p = os.path.join(root or ROOT, "runs", t)
    return os.path.islink(p) or not os.path.lexists(p) or (os.path.isdir(p) and _markers_only(p))


def place_run(t, root=None, ckpt=None, check_dev=True, stamp=None, busy=None):
    """[v2.5.9z71] (ops-lead 10-23 18:05 UTC, item 2) before a task starts (run_tasks.sh launch): its run directory
    runs/<task> becomes a symlink to a fresh $CKPT_ROOT/runs_live/<task>.<utc>, so what it writes (train.log, rollouts,
    gen_eval.log, the readings' jsonl, markers) lands on the checkpoint disk; readers, pairing and uploads go through the
    same path. A runs/<task> that is already a symlink stays; a real one holding only small files (markers of earlier
    passes: no subdirectory, < RUNS_MOVE_MAX_MB 64) is copied over and swapped in one exchange (kept as
    $OPD_ROOT/runs_pre_link/<task>.<utc>); a bigger one (an earlier attempt's outputs) stays where it is, as does any run
    of a task that is running. RUNS_ON_CKPT=0 (control.env) turns it off. Returns what it did."""
    root = root or ROOT
    ckpt = ckpt if ckpt is not None else os.environ.get("CKPT_ROOT", "")
    if _cfg("RUNS_ON_CKPT", "1") != "1":
        return "off"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", t) or not ckpt or not os.path.isdir(ckpt) or \
            (check_dev and os.stat(ckpt).st_dev == os.stat(root).st_dev):
        return "n/a"
    p = os.path.join(root, "runs", t)
    if os.path.islink(p):
        return "linked"
    if (busy if busy is not None else bool(running_pids(t))):
        return "running"
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    q = os.path.join(ckpt, "runs_live", f"{t}.{stamp}")
    tmp = os.path.join(root, "runs", f".{t}.z71_link")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if not os.path.lexists(p):
        os.makedirs(q, exist_ok=True); _clear(tmp, stamp); os.symlink(q, tmp)
        try:
            os.rename(tmp, p)
        except OSError:   # created meanwhile: the next start looks again
            os.unlink(tmp); return "raced"
        return "linked_new"
    if not os.path.isdir(p) or not _markers_only(p):
        return "kept"
    os.makedirs(q, exist_ok=True); _copy_tree(p, q)
    _clear(tmp, stamp); os.symlink(q, tmp)
    pre = os.path.join(root, "runs_pre_link"); os.makedirs(pre, exist_ok=True)
    try:
        try:
            _put_at(tmp, p, os.path.join(pre, f"{t}.{stamp}"), stamp, exchange=True)
        except OSError as e:   # [v2.5.9z71b] no exchange: two renames (nothing writes a run directory of a task not started)
            if e.errno not in NO_EXCHANGE:
                raise
            _put_at(tmp, p, os.path.join(pre, f"{t}.{stamp}"), stamp, exchange=False)
    except OSError:
        if os.path.islink(tmp):
            os.unlink(tmp)
        return "kept"
    _copy_tree(os.path.join(pre, f"{t}.{stamp}"), q)
    return "linked_moved"


RESERVE_ALERT_MIN = int(os.environ.get("RESERVE_ALERT_MIN", "10"))


def readings_disk_low():
    """[v2.5.9z69] (Fable / ops-lead 10-23: on the shared work disk the readings keep 1 TB, the trainings 300 GB) 'work' while the
    work disk ($OPD_ROOT) has less than READINGS_DISK_MIN_GB (1000; control.env first, read on every call) free, else ''.
    run_tasks.sh then starts no new reading (EVAL-*); the running ones go on, nothing is stopped."""
    s = os.statvfs(ROOT)
    return "work" if s.f_bavail * s.f_frsize / 1e9 < float(_cfg("READINGS_DISK_MIN_GB", "1000")) else ""


def reserve_note(ready, free, tasks, held, now=None):
    """run_tasks.sh after each dispatch pass: the GPU tasks ready but not started (every check passed, no GPU free for
    them), the GPUs still free, and the trainings the disk guard held, to reserve.json; low_since = since when ready <
    free + RESERVE_MIN (2) has held without a break (None while it does not)."""
    now = time.time() if now is None else now
    prev = _jread("reserve.json", {})
    need = free + int(os.environ.get("RESERVE_MIN", "2"))
    low = ready < need
    r = {"utc": utc(), "t": now, "ready": ready, "free_gpus": free, "need": need, "ready_tasks": tasks[:20], "held_by_disk": held[:20],
         "low_since": (prev.get("low_since") or now) if low else None}
    _jwrite("reserve.json", r)
    return r


def reserve_low(now=None):
    """The reserve alert: reserve.json fresh (< 15 min) and low for RESERVE_ALERT_MIN (10) minutes or more."""
    now = time.time() if now is None else now
    r = _jread("reserve.json", {})
    if not r or now - (r.get("t") or 0) > 900 or not r.get("low_since"):
        return None
    return r if now - r["low_since"] >= RESERVE_ALERT_MIN * 60 else None


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
         "ps": v_ps, "logs": v_logs, "run": v_run, "rerun": v_rerun, "republish": v_republish, "publish": v_publish, "leakscan": v_leakscan, "fa-build": v_fa_build, "scrubdiff": v_scrubdiff, "du": v_du, "inventory": v_inventory, "rm-ours": v_rm_ours, "dr-keep": v_dr_keep, "persist-env": v_persist_env, "rebuild-drill": v_rebuild_drill, "eval": v_eval, "update": v_update, "upload": v_upload,
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


def _jwrite(name, obj):
    """[v2.5.9z58] a control-plane JSON file written atomically: the writer's own tmp file, then os.replace (two writers with
    open(..., "w") interleaved and left state.json unreadable on 10-22)."""
    tmp = _p(f"{name}.{os.getpid()}.tmp"); json.dump(obj, open(tmp, "w")); os.replace(tmp, _p(name))


def _top_objects(raw):
    """Every complete top-level JSON object in raw, in order: one starts at the file's start or right after a closing brace
    (so a nested object inside a torn tail is not taken for a whole one)."""
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        j = raw.find("{", i)
        if j < 0:
            return out
        before = raw[:j].rstrip()
        if before and not before.endswith("}"):
            i = j + 1; continue
        try:
            o, end = dec.raw_decode(raw, j)
        except ValueError:
            i = j + 1; continue
        if isinstance(o, dict):
            out.append(o)
        i = end


def _merge_objects(objs):
    m = {}
    for o in objs:
        m.update(o)
    return m


def _jread(name, default=None, pick=_merge_objects):
    """[v2.5.9z58] a control-plane JSON file read tolerantly. Unreadable: the complete objects in it are recovered (pick: the
    file's own rule; by default merged in order), the bad file is copied to <name>.bad.<utc> (at most 5 copies), the file is
    rewritten clean and one event state_json_repaired names it."""
    raw = _read(_p(name), "") or ""
    if not raw.strip():
        return default
    try:
        return json.loads(raw)
    except ValueError:
        pass
    objs = _top_objects(raw)
    obj = pick(objs) if objs else default
    try:
        if len(glob.glob(_p(f"{name}.bad.*"))) < 5:
            open(_p(f"{name}.bad.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"), "w").write(raw)
        if obj is not None:
            _jwrite(name, obj)
    except OSError:
        pass
    try:
        event("node", "state_json_repaired", f"{name}: {len(raw)} bytes unreadable, {len(objs)} complete object(s) recovered")
    except Exception:
        pass
    return obj


def _pick_state(objs):
    """[ops-lead's A1 condition] state.json: never backwards - the object written last by apply (the latest last_apply_utc, then
    the most keys) is the base; rejected_entries is the union of all, so a refused entry is not refused (and reported) again;
    an applied_seq in any of them, or in control/applied_seq, gives the largest. (The commands themselves are guarded by
    control/applied_seq, written atomically by apply and never by state().)"""
    base = dict(max(objs, key=lambda o: (str(o.get("last_apply_utc") or ""), len(o))))
    rej = sorted({x for o in objs for x in (o.get("rejected_entries") or [])})
    if rej:
        base["rejected_entries"] = rej
    seqs = [int(o["applied_seq"]) for o in objs if str(o.get("applied_seq", "")).isdigit()]
    if seqs:
        f = _read(_p("applied_seq"), "0").strip()
        base["applied_seq"] = max(seqs + ([int(f)] if f.isdigit() else []))
    return base


def _state_read():
    """[v2.5.9z58] control/state.json, read tolerantly (10-22 ~11:17 UTC it held one object plus extra data, "Extra data: line 1
    column 124", and every reader raised: the heartbeat's status builder (no status for 4 h), apply (no command after seq 481 ran,
    no receipt), diag_tails; state() could never rewrite it since it reads first)."""
    s = _jread("state.json", {}, pick=_pick_state)
    return s if isinstance(s, dict) else {}


def state(**kw):
    s = _state_read(); s.update(kw); _jwrite("state.json", s)


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
    seen = set(_state_read().get("rejected_entries", []))
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
        receipt(seq, c, status, msg, upload=False, **extra)   # v2.5.7zb: local first ...
        print(f"CONTROL seq {seq} {c.get('cmd')!r}: {status}: {msg}", flush=True)
        applied = seq; open(_p("applied_seq.tmp"), "w").write(str(seq)); os.replace(_p("applied_seq.tmp"), _p("applied_seq"))
        queue_put(_p("replies", f"{seq}.json"), f"status/{node_id()}/replies/{seq}.json")   # ... then the bounded upload ([v2.5.8] the outbox)
    pending = sum(1 for c in cmds if isinstance(c, dict) and isinstance(c.get("seq"), int) and c["seq"] > applied)
    state(last_apply_utc=utc(), pending=pending)


def suspects():
    """setenv followed within 10 minutes by a task failure with a vLLM/Triton init error (report only)."""
    times = _jread("setenv_times.json", {}); out = []
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


# ---------- [v2.5.9z54] the GitHub fallback heartbeat (Runming 10-22 21:5x Beijing via Fable; ops-lead A1) ----------
# While the HF account answered 429 the status, the pulse and the receipts stopped reaching the project for hours. A second,
# independent path: gh_heartbeat.sh calls gh_round every GH_HB_POLL (10) s; every GH_HB_SEC (600) s, and at once after an urgent
# event (a failure, a failed gate or probe, a disk line, idle GPUs, a stall; at most every GH_HB_URGENT_MIN_S, 60 s), one file
# hb/<node>.json (overwritten) is committed and pushed to a private GitHub repository (GH_HB_REMOTE: an SSH alias set up on this
# node: GH_HB_URL, by default HTTPS with git's credential store reading GH_CRED, which Runming wrote; an ssh URL (the alias
# github-hb) works too). A failed push backs off 60 s doubling to 900 s and never touches the HF path. Before the first push
# `git ls-remote` checks access (event gh_hb_auth_failed once). The whole file passes the pulse's leak_check before it is written:
# a string that fails it becomes <withheld> (event gh_hb_withheld at most hourly), so even a repository made public by mistake
# leaks nothing.
GH_URGENT = {"failed", "gate_failed", "disk_alert", "ckpt_disk_alert", "ckpt_root_alert", "disk_write_fail", "disk_fallback_on",
             "ctrl_switch_refused", "tasks_guard_critical",   # [v2.5.9z71]
             "reserve_low", "stalled", "upload_sync_missing", "watchdog_gave_up", "watchdog_stuck"}
GH_URL = "https://github.com/yang3121099/opd-b300-heartbeat.git"   # (Fable 14:4x UTC) HTTPS through the node's proxy, as the runbook's pulls
GH_CRED = "~/.config/opd/gh_hb_credentials"   # written by Runming (a token for this repository only); git reads it, this code never does


def _gh_clean(x, toks):
    """A string for the private GitHub file: redacted as the private status' log tails, and withheld when it still names this
    machine's user, host or directories."""
    x = redact(str(x))
    return "<withheld>" if {w.lower() for w in re.split(r"[^A-Za-z0-9_]+", x) if w} & toks else x


def gh_snapshot(n_replies=15, n_events=20):
    """The one file: the path-free pulse (the status' summary: tasks, GPUs, disks), the latest receipts and events (redacted),
    the control plane's state, the loops' ages, the outbox and the HF backoff."""
    toks = _forbidden_tokens()
    def recent(d, n, key):
        try:
            names = sorted((x for x in os.listdir(_p(d)) if x.endswith(".json")), key=key)[-n:]
        except OSError:
            return []
        out = []
        for x in names:
            try:
                out.append(json.load(open(_p(d, x))))
            except (OSError, ValueError):
                pass
        return out
    seqkey = lambda x: int(x.split(".")[0]) if x.split(".")[0].isdigit() else -1
    reps = [{"seq": r.get("seq"), "cmd": _gh_clean(r.get("cmd") or "", toks)[:200], "status": r.get("status"), "finished_utc": r.get("finished_utc"),
             "detail": _gh_clean(r.get("detail") or "", toks)[:300]} for r in recent("replies", n_replies, seqkey)]
    evs = [{"utc": e.get("utc"), "task": _gh_clean(e.get("task") or "", toks), "event": e.get("event"), "detail": _gh_clean(e.get("detail") or "", toks)[:200]}
           for e in recent("events", n_events, lambda x: x)]
    now = time.time()
    age = lambda f: (lambda t: int(now - int(t)) if t.isdigit() else None)(_read(_p(f)).strip())
    ob = [n for n in os.listdir(_outbox()) if n.endswith(".json") and not n.startswith(".")] if os.path.isdir(_outbox()) else []
    s = _state_read()
    sf = sorted(glob.glob(os.path.join(ROOT, "logs", "status_*.json")), key=os.path.getmtime)[-1:]   # heartbeat.sh's status file
    try:
        sfu = json.loads(_read(sf[0], "{}") or "{}").get("utc") if sf else None
    except ValueError:
        sfu = "unreadable"
    return {"node": node_id(), "utc": utc(), "runbook_head": _git_head(), "rules": R.RULES_VERSION,
            "status_file": {"age_s": int(now - os.path.getmtime(sf[0])) if sf else None, "utc_inside": sfu,   # [v2.5.9z55] builder failures
                            "builder_err": [_gh_clean(x, toks)[:200] for x in _read(os.path.join(ROOT, "logs", "status_builder.err")).strip().splitlines()[-4:]]},
            "control": {"applied_seq": int((_read(_p("applied_seq"), "0").strip() or 0)), "pending": s.get("pending"),
                        "last_apply_utc": s.get("last_apply_utc"), "alerts": s.get("alerts", [])},
            "loops_age_s": {k: age(f"loop_{k}") for k in ("run_tasks", "heartbeat", "uploader", "gh_heartbeat")},
            "outbox": {"items": len(ob), "oldest_age_s": int(now - int(min(ob)[:20]) / 1e9) if ob and min(ob)[:20].isdigit() else None},
            "hf_backoff_s": int(_backoff_left()), "hf_429_last": (_read(_p("hf_429.log")).strip().splitlines() or [None])[-1],
            "pulse": _jread("pulse.json", {}), "replies": reps, "events": evs}


def _gh_withhold(obj):
    """(obj with every string that fails leak_check replaced by <withheld>, how many): keys stay (they are this code's own)."""
    bad = set(leak_check(obj)); n = [0]
    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str) and x in bad:
            n[0] += 1
            return "<withheld>"
        return x
    return walk(obj), n[0]


def _gh_state(**kw):
    st = _jread("gh_hb.json", {})
    if kw:
        st.update(kw); tmp = _p(f"gh_hb.json.{os.getpid()}.tmp"); json.dump(st, open(tmp, "w")); os.replace(tmp, _p("gh_hb.json"))
    return st


def _gh_urgent(st):
    """(urgent, newest event name): an event of a GH_URGENT kind, or one ending in _fail (a probe), newer than the last push."""
    try:
        names = sorted(x for x in os.listdir(_p("events")) if x.endswith(".json"))
    except OSError:
        return False, st.get("seen")
    seen = st.get("seen") or ""
    new = [x for x in names if x > seen] if seen else []   # the first round only takes the baseline
    hit = False
    for x in new[-200:]:
        try:
            k = json.load(open(_p("events", x))).get("event") or ""
        except (OSError, ValueError):
            continue
        hit = hit or k in GH_URGENT or k.endswith("_fail")
    return hit, (names[-1] if names else seen)


def gh_round(now=None):
    """One round of gh_heartbeat.sh: off | idle | backoff | pushed (time|urgent) | failed (<step>) ."""
    if _cfg("GH_HB", "1") != "1":
        return "off"
    now = time.time() if now is None else now
    st = _gh_state()
    try:
        sec = min(3600, max(60, int(_cfg("GH_HB_SEC", "600"))))
    except ValueError:
        sec = 600
    urgent, newest = _gh_urgent(st)
    if urgent:
        _gh_state(pending_urgent=True)
    due = (now - float(st.get("last_try", 0)) >= sec or int(st.get("fails") or 0) > 0   # a failed try: again when its backoff ends
           or ((urgent or st.get("pending_urgent")) and now - float(st.get("last_try", 0)) >= int(_cfg("GH_HB_URGENT_MIN_S", "60"))))
    if not due:
        _gh_state(seen=newest)
        return "idle"
    if now < float(st.get("next_try", 0)):
        _gh_state(seen=newest)
        return "backoff"
    why = "urgent" if (urgent or st.get("pending_urgent")) and now - float(st.get("last_try", 0)) < sec else "time"
    d = _p("gh_hb"); url = _cfg("GH_HB_URL", "") or os.environ.get("GH_HB_REMOTE") or GH_URL; to = int(os.environ.get("GH_HB_GIT_TIMEOUT", "60"))
    env = dict(os.environ, GIT_SSH_COMMAND=os.environ.get("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2"),
               GIT_TERMINAL_PROMPT="0")
    cred = os.path.expanduser(os.environ.get("GH_HB_CRED") or GH_CRED)
    cfg = ["-c", "credential.helper=", "-c", f"credential.helper=store --file={cred}"] if url.startswith("https://") and os.path.exists(cred) else []
    git = lambda *a: subprocess.run(["git", *cfg, "-C", d, *a], capture_output=True, text=True, timeout=to, env=env)
    step = "auth"
    try:
        if not int(st.get("pushes") or 0) and not st.get("auth_ok"):   # before the first push: can this node reach the repository?
            r = subprocess.run(["git", *cfg, "ls-remote", url], capture_output=True, text=True, timeout=to, env=env)
            if r.returncode:
                if not st.get("auth_failed"):
                    event("node", "gh_hb_auth_failed", _gh_clean(r.stderr.strip()[-160:], _forbidden_tokens()))
                _gh_state(auth_failed=True)
                raise RuntimeError("git ls-remote failed")
            _gh_state(auth_ok=True, auth_failed=False)
        step = "clone"
        if not os.path.isdir(os.path.join(d, ".git")):   # (Runming's setup may have cloned it already: used as it is)
            shutil.rmtree(d, ignore_errors=True)
            r = subprocess.run(["git", *cfg, "clone", "-q", "--depth", "1", url, d], capture_output=True, text=True, timeout=to, env=env)
            if r.returncode:
                raise RuntimeError(r.stderr.strip()[-200:])
        step = "write"
        os.makedirs(os.path.join(d, "hb"), exist_ok=True); f = os.path.join(d, "hb", f"{node_id()}.json")
        snap = gh_snapshot(); snap.update(push_n=int(st.get("pushes") or 0) + 1, why=why, gh_fails_before=int(st.get("fails") or 0))
        snap, n_withheld = _gh_withhold(snap)   # [ops-lead's A1 condition] the pulse's leak_check over the whole file
        if n_withheld and now - float(st.get("withheld_event", 0)) >= 3600:
            event("node", "gh_hb_withheld", f"{n_withheld} string(s)"); _gh_state(withheld_event=now)
        json.dump(snap, open(f + ".tmp", "w"), indent=1); os.replace(f + ".tmp", f)
        step = "commit"
        git("add", f"hb/{node_id()}.json")
        if git("diff", "--cached", "--quiet").returncode:   # nothing new (the same second): push what is there
            r = git("-c", "user.name=opd-node", "-c", "user.email=opd-node@users.noreply.github.com", "commit", "-q", "-m", f"hb {utc()} {why}")
            if r.returncode:
                raise RuntimeError(r.stderr.strip()[-200:])
        step = "push"
        r = git("push", "-q", "origin", "HEAD:refs/heads/main")
        if r.returncode:
            if git("fetch", "-q", "--depth", "1", "origin", "main").returncode == 0:   # someone else wrote: ours on top next time
                git("reset", "-q", "--soft", "FETCH_HEAD")
            raise RuntimeError(r.stderr.strip()[-200:])
    except Exception as e:
        b = min(900, max(60, 2 * int(st.get("backoff_s") or 30)))
        _gh_state(last_try=now, backoff_s=b, next_try=now + b, fails=int(st.get("fails") or 0) + 1, seen=newest,
                  last_error=f"{step}: {type(e).__name__} {_gh_clean(str(e), _forbidden_tokens())[:160]}")
        return f"failed ({step}, next try in {b} s)"
    _gh_state(last_try=now, last_push=now, last_ok_utc=utc(), backoff_s=0, next_try=0, fails=0, seen=newest, pending_urgent=False, last_error=None,
              pushes=int(st.get("pushes") or 0) + 1)
    return f"pushed ({why})"


def status():
    lines = _read(_p("log.jsonl")).splitlines()[-5:]
    s = _state_read()
    lp = _jread("last_pull.json", None)
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
    elif cmd == "do_inventory":   # [v2.5.9z72] v_inventory's background half
        do_inventory(int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "pend")
    elif cmd == "do_du":
        do_du(int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif cmd == "do_rm_ours":
        do_rm_ours(int(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "disk_fallback":   # [v2.5.9z48] run_tasks.sh every round; [v2.5.9z71] in the background, never waiting on the lock
        r = disk_fallback(blocking=False)
        if r.startswith("switched"):   # [v2.5.9z71] this line to the new logs/control.log, not the old file this process opened
            sys.stdout = open(os.path.join(ROOT, "logs", "control.log"), "a")
        print("DISK_FALLBACK", utc(), r)
    elif cmd == "lru_clean":   # [v2.5.8] heartbeat.sh every round; nothing unless LOCAL_LRU_GB is set
        g = lru_clean(); print(f"LRU_FREED {g} GB") if g else None
    elif cmd == "diag_tails":   # [v2.5.9z53] at once (the uploader's pass sends them every DIAG_TAILS_S)
        print("DIAG_TAILS", diag_tails(force=True))
    elif cmd == "gh_round":   # [v2.5.9z54] gh_heartbeat.sh, one round
        print(gh_round())
    elif cmd == "ship_outbox":   # [v2.5.8] uploader.sh, one pass
        a, b, c = ship_outbox(); print(f"SHIPPED {a} DROPPED {b} FAILED {c} LEFT {len([n for n in os.listdir(_outbox()) if n.endswith('.json') and not n.startswith('.')])}")
    elif cmd == "put_q":   # [v2.5.8] heartbeat.sh: put_q <src> <dst> through the outbox (directly without a live uploader)
        sys.exit(0 if queue_put(sys.argv[2], sys.argv[3]) else 1)
    elif cmd == "dr_keep_auto":   # [v2.5.8] upload_sync.sh at UPLOAD_DONE
        print(f"DR_KEPT_AUTO {sys.argv[2]} {dr_keep_auto(sys.argv[2])}")
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
    elif cmd == "auto_reruns":   # [v2.5.9z12] run_tasks.sh every pass: the queue's #at / #after rerun lines
        auto_reruns(sys.argv[2])
    elif cmd == "disk_low":   # [v2.5.9z12] run_tasks.sh every pass: the disk guard of new trainings
        print(disk_low())
    elif cmd == "readings_disk_low":   # [v2.5.9z69] run_tasks.sh every pass: the disk guard of new readings
        print(readings_disk_low())
    elif cmd == "tasks_disk_low":   # [v2.5.9z71] run_tasks.sh every pass: no new task of any kind below TASKS_DISK_MIN_GB
        print(tasks_disk_low())
    elif cmd == "runs_ckpt":   # [v2.5.9z71c] run_tasks.sh every pass: do new run directories go to the checkpoint disk (1 / 0)
        print(runs_ckpt())
    elif cmd == "run_placeable":   # [v2.5.9z71c] run_placeable <task>: exit 0 when its run directory will be on the checkpoint disk
        sys.exit(0 if run_placeable(sys.argv[2]) else 1)
    elif cmd == "place_run":   # [v2.5.9z71] run_tasks.sh launch: place_run <task> -> the run directory on the checkpoint disk
        print("PLACE_RUN", sys.argv[2], place_run(sys.argv[2]))
    elif cmd == "reserve_note":   # [v2.5.9z12] run_tasks.sh after a pass: reserve_note <ready> <free> "<tasks>" "<held>"
        reserve_note(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(), sys.argv[5].split() if len(sys.argv) > 5 else [])
    elif cmd == "rcpt":   # run_tasks / run_adhoc_eval: update a receipt (started, done, failed)
        receipt(int(sys.argv[2]), None, sys.argv[3], " ".join(sys.argv[4:]))
    else:
        print(__doc__); sys.exit(2)
