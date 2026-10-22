"""weights_replay.py (runbook v2.5.9z62; ops-lead A1 10-23): the weights held on the checkpoint disk while uploads were private
first (control/weights_pending.txt, written by remote.py's defer_weight) go to the PUBLIC result repository under the path they
always had (runs/<task>/global_step_<N>/actor/huggingface/<file>), one item at a time, from weights_replay.sh's own loop (not the
heartbeat, not the control plane's outbox).
- Files: the weight directory's own files only (*.safetensors, the index, config / generation config, tokenizer files); no logs.
  remote.py put ... public scrubs the JSON configs' path values as every public upload (no absolute path goes up).
- Order: each arm's final step and every step a queue reading uses first (result:runs/<task>/global_step_<N>), older first.
- Bounds: one upload at a time; its timeout from its size at WEIGHTS_REPLAY_MBPS (0.5 MB/s) x 1.5 + 600 s, at most
  WEIGHTS_REPLAY_MAX_S (21600 s); nothing while remote.py's 429 backoff lasts or while a control-plane outbox item is older than
  REPLAY_OUTBOX_MAX_AGE (600 s): the status and the receipts go first.
- Done: only when one get_paths_info on the public repository shows every weight file with its local size; then the line leaves
  weights_pending.txt (under the lock defer_weight appends with) and goes to weights_replayed.txt. The held copy is NOT deleted:
  the *.safetensors are hard-linked into a .stage_ directory, every other file copied (nothing the upload does reaches the hold).
- [ops-lead's A1 conditions] a failed item cools down 30 min x 2^(n-1) (at most 6 h) and the next one goes meanwhile; its third
  failure in a row is one weights_replay_stuck event. A hold whose real path is not under $CKPT_ROOT/weights_pending/ is refused.
- Events: one weights_replayed event per batch (every WEIGHTS_REPLAY_BATCH items, and when nothing is left), not per file.
- [v2.5.9z64] (ops-lead's ruling 10-22 20:45 UTC) a held RESULT directory (z50's is_weight held any directory with a file over
  100 MB: six readings) goes first and is never paused: to the PRIVATE repository as an ordinary result (remote.py put, no public),
  a log over 5 MB as its last 5 MB gzipped (<name>.tail.gz), timeout from its size at RESULT_MBPS; struck once get_paths_info on the
  private repository shows every staged file at its size; one results_replayed event each.
Usage: python weights_replay.py once -> REPLAY_IDLE | REPLAY_OFF | REPLAY_WAIT <why> | REPLAY_OK <dst> | REPLAY_FAIL <dst> <why>"""
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import remote as RM  # noqa: E402

KEEP = re.compile(r".*\.safetensors|model\.safetensors\.index\.json|config\.json|generation_config\.json|tokenizer\.json|"
                  r"tokenizer_config\.json|special_tokens_map\.json|added_tokens\.json|vocab\.json|merges\.txt|chat_template\.jinja|.*\.model")


def root():
    return os.environ.get("OPD_ROOT", "")


def cpath(*a):
    return os.path.join(root(), "control", *a)


def cfg(name, default):
    return RM._cfg(name, default)


def pending():
    """[(utc, dst, hold, size)] from weights_pending.txt (malformed lines skipped)."""
    out = []
    try:
        for l in open(cpath("weights_pending.txt")):
            f = l.rstrip("\n").split("\t")
            if len(f) == 4 and f[3].isdigit():
                out.append((f[0], f[1], f[2], int(f[3])))
    except OSError:
        pass
    return out


def wanted_steps(queue_text):
    """{(task, step)} that a queue reading uses (result:runs/<task>/global_step_<N>/actor/huggingface)."""
    return {(t, int(s)) for t, s in re.findall(r"result:runs/([A-Za-z0-9._-]+)/global_step_(\d+)/actor/huggingface", queue_text)}


def order(items, queue_text):
    """Each task's final step (its largest pending one) and the steps readings use first, then the rest; older first."""
    want, last = wanted_steps(queue_text), {}
    key = lambda d: re.match(r"runs/([^/]+)/global_step_(\d+)/", d.lstrip("/") + "/")
    done = []
    try:
        done = [l.split("\t")[1] for l in open(cpath("weights_replayed.txt")) if l.count("\t") >= 1]
    except OSError:
        pass
    for dst in [it[1] for it in items] + done:   # the final step: the largest held or already replayed
        m = key(dst)
        if m:
            last[m.group(1)] = max(last.get(m.group(1), 0), int(m.group(2)))
    def rank(it):
        m = key(it[1])
        first = bool(m) and ((m.group(1), int(m.group(2))) in want or last.get(m.group(1)) == int(m.group(2)))
        return (0 if first else 1, it[0])
    return sorted(items, key=rank)


def weight_files(hold):
    return sorted(f for f in os.listdir(hold) if os.path.isfile(os.path.join(hold, f)) and KEEP.fullmatch(f)) if os.path.isdir(hold) else []


def control_backlog():
    """The age (s) of the oldest control-plane item in the outbox (status/, pulse/), 0 if none."""
    d, now, old = cpath("outbox"), time.time(), 0
    try:
        for n in os.listdir(d):
            if not n.endswith(".json") or not n[:20].isdigit():
                continue
            try:
                dst = json.load(open(os.path.join(d, n))).get("dst", "")
            except (OSError, ValueError):
                continue
            if dst.startswith(("status/", "pulse/")):
                old = max(old, now - int(n[:20]) / 1e9)
    except OSError:
        pass
    return old


def verify(dst, files, hold):
    """True when the public repository has every file under dst with its local size (one get_paths_info call); the JSON / text
    files only need to be there (the public upload scrubs path values, which can change their size)."""
    api, repo = RM.hf("public")
    info = {x.path: getattr(x, "size", None) for x in api.get_paths_info(repo, [f"{dst}/{f}" for f in files])}
    for f in files:
        p = f"{dst}/{f}"
        if p not in info:
            return False, f"{f} missing"
        if f.endswith(".safetensors") and info[p] != os.path.getsize(os.path.join(hold, f)):
            return False, f"{f} size {info[p]} != {os.path.getsize(os.path.join(hold, f))}"
    return True, ""


def strike(dst, nbytes, secs):
    """The line of dst leaves weights_pending.txt (under defer_weight's lock) and goes to weights_replayed.txt."""
    with open(cpath("weights_pending.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        lines = open(cpath("weights_pending.txt")).read().splitlines(True) if os.path.exists(cpath("weights_pending.txt")) else []
        keep = [l for l in lines if l.split("\t")[1:2] != [dst]]
        tmp = cpath(f"weights_pending.txt.{os.getpid()}.tmp"); open(tmp, "w").writelines(keep); os.replace(tmp, cpath("weights_pending.txt"))
        with open(cpath("weights_replayed.txt"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{dst}\t{nbytes}\t{secs}\n")


def state():
    try:
        st = json.load(open(cpath("weights_replay.json")))
    except (OSError, ValueError):
        st = {}
    st.setdefault("items", 0); st.setdefault("bytes", 0); st.setdefault("fails", {})
    return st


def save(st):
    tmp = cpath(f"weights_replay.json.{os.getpid()}.tmp"); json.dump(st, open(tmp, "w")); os.replace(tmp, cpath("weights_replay.json"))


def event(kind, detail):
    subprocess.run([sys.executable, os.path.join(HERE, "control.py"), "event", "node", kind, detail],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def cooling(st, dst, now):
    """Seconds this destination still cools down after its last failure: 30 min x 2^(n-1), at most 6 h."""
    f = st["fails"].get(dst)
    if not f:
        return 0
    wait = min(float(cfg("WEIGHTS_REPLAY_COOL_S", "1800")) * 2 ** (int(f["n"]) - 1), float(cfg("WEIGHTS_REPLAY_COOL_MAX_S", "21600")))
    return max(0.0, float(f["last"]) + wait - now)


def failed(dst, why):
    st = state(); f = st["fails"].get(dst, {"n": 0}); f = {"n": int(f["n"]) + 1, "last": time.time(), "why": why[:200]}
    st["fails"][dst] = f; save(st)
    if f["n"] == 3:
        event("weights_replay_stuck", f"{dst}: 3 failed tries in a row, the last: {why[:200]}")
    return f"REPLAY_FAIL {dst} {why} (left in the list; try {f['n']}, cooling down)"


def batch_event(done_now, left, dst=None):
    """One weights_replayed event per WEIGHTS_REPLAY_BATCH (5) items, and when nothing is left."""
    st = state()
    if dst:
        st["fails"].pop(dst, None)
    if done_now:
        st["items"] += 1; st["bytes"] += done_now
    if st["items"] and (st["items"] >= int(cfg("WEIGHTS_REPLAY_BATCH", "5")) or not left):
        event("weights_replayed", f"{st['items']} item(s), {st['bytes'] / 1e9:.1f} GB to the public repository; {left} left")
        st["items"], st["bytes"] = 0, 0
    save(st)


WEIGHT_DST = re.compile(r"runs/[A-Za-z0-9._-]+/global_step_\d+/actor/huggingface")
RESULT_DST = re.compile(r"runs/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
LOG_TAIL = 5 * 10 ** 6
RESULT_MBPS = 0.08   # [v2.5.9z64a] was 0.4: one connection gets 0.09-0.13 MB/s through the proxy (10-22: 268 MB at 129 kB/s)


def stage_result(hold, stage):
    """The held result directory into stage: every file hard-linked (else copied) except a *.log over 5 MB, which becomes
    <name>.tail.gz (its last 5 MB, gzipped); returns [(relative path, bytes)]."""
    import gzip
    out = []
    for r, _, fs in os.walk(hold):
        for f in fs:
            src = os.path.join(r, f); rel = os.path.relpath(src, hold); t = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(t), exist_ok=True)
            if f.endswith(".log") and os.path.getsize(src) > LOG_TAIL:
                with open(src, "rb") as fh:
                    fh.seek(os.path.getsize(src) - LOG_TAIL); data = fh.read()
                t += ".tail.gz"; rel += ".tail.gz"
                with gzip.open(t, "wb") as g:
                    g.write(data)
            else:
                try:
                    os.link(src, t)
                except OSError:
                    shutil.copy2(src, t)
            out.append((rel.replace(os.sep, "/"), os.path.getsize(t)))
    return out


def verify_private(dst, staged):
    api, repo = RM.hf("private")
    info = {}
    for i in range(0, len(staged), 50):
        for x in api.get_paths_info(repo, [f"{dst}/{rel}" for rel, _ in staged[i:i + 50]]):
            info[x.path] = getattr(x, "size", None)
    for rel, size in staged:
        if info.get(f"{dst}/{rel}") != size:
            return False, f"{rel}: {info.get(f'{dst}/{rel}')} != {size}"
    return True, ""


def results_once(items):
    """[v2.5.9z64] the oldest held result directory to the private repository (see the docstring); None if none is due."""
    st, now = state(), time.time()
    todo = sorted((it for it in items if not WEIGHT_DST.fullmatch(it[1].lstrip("/")) and RESULT_DST.fullmatch(it[1].lstrip("/"))
                   and cooling(st, "result:" + it[1].lstrip("/"), now) <= 0), key=lambda it: it[0])   # its own key: z62's refusals of it do not count
    if not todo:
        return None
    utc, dst, hold, size = todo[0]
    dst = dst.lstrip("/")
    base = os.path.realpath(os.path.join(os.environ.get("CKPT_ROOT", "/nonexistent"), "weights_pending"))
    if not os.path.realpath(hold).startswith(base + os.sep) or not os.path.isdir(hold):
        return failed("result:" + dst, "the hold path is not a directory under $CKPT_ROOT/weights_pending")
    stage = os.path.join(os.path.dirname(hold.rstrip("/")), ".stage_" + hashlib.sha256(dst.encode()).hexdigest()[:12])
    shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    try:
        staged = stage_result(hold, stage)
        nbytes = sum(b for _, b in staged)
        to = int(min(float(cfg("WEIGHTS_REPLAY_MAX_S", "21600")), nbytes / 1e6 / RESULT_MBPS * 1.5 + 600))
        t0 = time.time()
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, "remote.py"), "put", stage, dst], capture_output=True, text=True,
                               timeout=to, env=dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0"))
            rc, why = r.returncode, (r.stderr.strip().splitlines() or [""])[-1][:160]
        except subprocess.TimeoutExpired:
            rc, why = 1, f"timed out after {to} s"
        secs = round(time.time() - t0, 1)
        if rc:
            return failed("result:" + dst, why)
        try:
            ok, why = verify_private(dst, staged)
        except Exception as e:
            ok, why = False, f"verify: {type(e).__name__}"
        if not ok:
            return failed("result:" + dst, f"not verified: {why}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    strike(dst, nbytes, secs)
    st = state(); st["fails"].pop("result:" + dst, None); st["fails"].pop(dst, None); save(st)
    event("results_replayed", f"{dst}: {len(staged)} file(s), {nbytes / 1e6:.0f} MB to the private repository in {secs} s")
    return f"REPLAY_OK {dst} (result) {nbytes / 1e6:.0f} MB in {secs} s"


def once():
    items = pending()
    if not items:
        return "REPLAY_IDLE"
    if RM.backoff_left() > 0:
        return f"REPLAY_WAIT the 429 backoff ({int(RM.backoff_left())} s)"
    age = control_backlog()
    if age > float(cfg("REPLAY_OUTBOX_MAX_AGE", "600")):
        return f"REPLAY_WAIT a control-plane item waits {int(age)} s in the outbox"
    r = results_once(items)   # [v2.5.9z64] held results first, whatever WEIGHTS_REPLAY says
    if r:
        return r
    if cfg("WEIGHTS_REPLAY", "1") != "1":
        return "REPLAY_OFF"
    items = [it for it in items if WEIGHT_DST.fullmatch(it[1].lstrip("/"))]
    if not items:
        return "REPLAY_IDLE"
    q = ""
    try:
        q = open(os.path.join(os.environ.get("RUNBOOK", os.path.dirname(HERE)), "tasks", "queue.txt")).read()
    except OSError:
        pass
    st, now = state(), time.time()
    todo = [it for it in order(items, q) if cooling(st, it[1].lstrip("/"), now) <= 0]
    if not todo:
        return f"REPLAY_WAIT all {len(items)} item(s) cooling down after failures"
    utc, dst, hold, size = todo[0]
    dst = dst.lstrip("/")
    if not re.fullmatch(r"runs/[A-Za-z0-9._-]+/global_step_\d+/actor/huggingface", dst):
        return failed(dst, "not a weight destination")
    base = os.path.realpath(os.path.join(os.environ.get("CKPT_ROOT", "/nonexistent"), "weights_pending"))
    if not os.path.realpath(hold).startswith(base + os.sep):   # [ops-lead's A1 condition] only what defer_weight held
        return failed(dst, "the hold path is not under $CKPT_ROOT/weights_pending")
    files = weight_files(hold)
    if not any(f.endswith(".safetensors") for f in files):
        return failed(dst, "no weights held at the hold path")
    stage = os.path.join(os.path.dirname(hold.rstrip("/")), ".stage_" + hashlib.sha256(dst.encode()).hexdigest()[:12])
    shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    for f in files:   # the weights hard-linked (else copied); every other file copied, so nothing reaches the held files
        if f.endswith(".safetensors"):
            try:
                os.link(os.path.join(hold, f), os.path.join(stage, f)); continue
            except OSError:
                pass
        shutil.copy2(os.path.join(hold, f), os.path.join(stage, f))
    nbytes = sum(os.path.getsize(os.path.join(hold, f)) for f in files)
    to = int(min(float(cfg("WEIGHTS_REPLAY_MAX_S", "21600")), nbytes / 1e6 / float(cfg("WEIGHTS_REPLAY_MBPS", "0.5")) * 1.5 + 600))
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "remote.py"), "put", stage, dst, "public"], capture_output=True, text=True,
                           timeout=to, env=dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0"))
        rc, why = r.returncode, (r.stderr.strip().splitlines() or [""])[-1][:160]
    except subprocess.TimeoutExpired:
        rc, why = 1, f"timed out after {to} s"
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    secs = round(time.time() - t0, 1)
    if rc:
        return failed(dst, why)
    try:
        ok, why = verify(dst, files, hold)
    except Exception as e:
        ok, why = False, f"verify: {type(e).__name__}"
    if not ok:
        return failed(dst, f"not verified: {why}")
    strike(dst, nbytes, secs)
    batch_event(nbytes, len(pending()), dst)
    return f"REPLAY_OK {dst} {nbytes / 1e9:.2f} GB in {secs} s"


if __name__ == "__main__":
    print(once() if sys.argv[1:] == ["once"] else __doc__)
