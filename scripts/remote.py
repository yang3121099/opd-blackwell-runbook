"""Result-repository backend for the remote node: Hugging Face (HF_TOKEN, HF_REPO) or ModelScope (MS_TOKEN, MS_REPO),
chosen by UPLOAD_BACKEND. Credentials are read from the environment only and never printed or written to disk.
  python remote.py check                 -> exit 0 if the backend is usable (token valid, repo reachable)
  python remote.py put <local> <remote>  -> upload a file or a folder to <remote> in the repo
  python remote.py done <task>           -> exit 0 if runs/<task>/.done exists in the repo (HF only; ModelScope: no)
  python remote.py repo public|private   -> print the repository that class of files goes to
  python remote.py put <local> <remote> public -> [v2.5.9z52] to the public repository whatever the class (the pulse's mirror)
  python remote.py putmany <list.json>   -> [v2.5.9z52] [[local file, remote], ...] in one commit to the private repository
  python remote.py backoff               -> [v2.5.9z52] seconds left of the 429 backoff (0: none)
The ModelScope path is not tested yet: check upload.log for UP_OK after the first checkpoint.

v2.5.1 (two repositories, Runming 2026-10-17): $OPD_ROOT/hf_repos.txt (written by `set HF_REPO_PUBLIC=` /
`set HF_REPO_PRIVATE=`, lines `public=<owner/name>` and `private=<owner/name>`) splits the uploads by destination:
runs/, ckpts/ and pulse/ (weights, metrics, generations, the path-free pulse) go to the public repository, everything
else (status/, logs/, replies, events, procs.json: files that can carry paths, host names or user names) to the private
one. Without the file both are HF_REPO, as in v2.5. Every call reads the file again, so loops that are already running
switch on their next upload. A destination path component equal to this machine's host name (full or short) becomes the
node id, so no file is named after the host. Private-class files are refused when the private repository is public.
"""
import hashlib
import os
import re
import time
import json
import socket
import sys

BACKEND = os.environ.get("UPLOAD_BACKEND", "hf")
PUBLIC_PREFIXES = ("runs/", "ckpts/", "pulse/")


def node_id():
    return os.environ.get("OPD_NODE") or hashlib.sha256(socket.gethostname().split(".")[0].encode()).hexdigest()[:8]


def repos():
    """(public, private) repository names: $OPD_ROOT/hf_repos.txt, else HF_REPO for both."""
    pub = priv = os.environ.get("HF_REPO")
    path = os.path.join(os.environ.get("OPD_ROOT", ""), "hf_repos.txt")
    try:
        for line in open(path):
            k, _, v = line.strip().partition("=")
            if k == "public" and v:
                pub = v
            elif k == "private" and v:
                priv = v
    except OSError:
        pass
    return pub, priv


def split_mode():
    return os.path.exists(os.path.join(os.environ.get("OPD_ROOT", ""), "hf_repos.txt"))


def _cfg(name, default):
    """[v2.5.9z50] a setting from $OPD_ROOT/control.env (setenv; read on every call), else the environment, else the default."""
    try:
        for line in open(os.path.join(os.environ.get("OPD_ROOT", ""), "control.env")):
            k, _, x = line.strip().removeprefix("export ").partition("=")
            if k == name and x.strip().strip('"'):
                return x.strip().strip('"')
    except OSError:
        pass
    return os.environ.get(name) or default


def public_paused():
    """[v2.5.9z50] (Runming 10-22 20:2x Beijing via the reviewer: private first; the node lost the public repository from
    11:17 UTC while the private one and GitHub worked) PUBLIC_UPLOADS=0, the default: every upload goes to the private
    repository under the same path (the pulse as pulse/<node>.json, results as runs/...), and fetches try the private
    repository before the public one. `setenv PUBLIC_UPLOADS=1` restores the split."""
    return _cfg("PUBLIC_UPLOADS", "0") != "1"


def klass(dst):
    if public_paused():   # [v2.5.9z50]
        return "private"
    return "public" if dst.lstrip("/").startswith(PUBLIC_PREFIXES) else "private"


def fetch_repos():
    """[v2.5.9z50] the repositories results are read from, in order, comma separated (fetch_model.py tries each): the private
    one first while public uploads are paused (new results land there; older ones are still only public)."""
    pub, priv = repos()
    if public_paused() and priv and priv != pub:
        return f"{priv},{pub}" if pub else priv
    return pub


def rename(dst):
    """Destination with every path component that is this host's name (full or short, or name.<ext>) replaced by the node id."""
    names = {socket.gethostname(), socket.gethostname().split(".")[0]}
    try:
        names.add(socket.getfqdn())
    except OSError:
        pass
    names = {n for n in names if n and n != "localhost"}
    out = []
    for part in dst.split("/"):
        for n in sorted(names, key=len, reverse=True):
            if part == n:
                part = node_id(); break
            if part.startswith(n + ".") or part.endswith("_" + n + ".json") or part.startswith(n + "_"):
                part = part.replace(n, node_id()); break
        out.append(part)
    return "/".join(out)


def hf(kind="public"):
    from huggingface_hub import HfApi
    # the result repository is always reached at huggingface.co (v2.2): HF_ENDPOINT (e.g. the read-only hf-mirror.com)
    # only speeds up downloads of public models and data and must not redirect uploads; HF_UPLOAD_ENDPOINT overrides
    pub, priv = repos()
    return (HfApi(endpoint=os.environ.get("HF_UPLOAD_ENDPOINT", "https://huggingface.co"), token=os.environ["HF_TOKEN"]),
            pub if kind == "public" else priv)


def ms():
    from modelscope.hub.api import HubApi
    api = HubApi(); api.login(os.environ["MS_TOKEN"])
    return api, os.environ["MS_REPO"]


def private_ok(api, repo):
    """In split mode the private repository must really be private; the answer is cached per repository name."""
    flag = os.path.join(os.environ.get("OPD_ROOT", ""), "control", ".private_ok_" + hashlib.sha256(repo.encode()).hexdigest()[:12])
    if os.path.exists(flag):
        return True
    ok = bool(api.model_info(repo).private)
    if ok:
        os.makedirs(os.path.dirname(flag), exist_ok=True); open(flag, "w").write(repo + "\n")
    return ok


def check():
    if BACKEND == "hf":
        api, repo = hf("public"); api.whoami()
        # create the result repository if it does not exist yet (public by default; REQUIRE_PRIVATE=1 creates it private
        # and refuses a public one). A token limited to one repository may not be allowed to create it: then the project
        # lead creates it on the website first.
        need_private = os.environ.get("REQUIRE_PRIVATE", "0") == "1"
        api.create_repo(repo, repo_type="model", private=need_private, exist_ok=True)
        if need_private:
            assert api.model_info(repo).private, "REQUIRE_PRIVATE=1 but the result repository is public"
        if split_mode():
            _, priv = hf("private")
            api.create_repo(priv, repo_type="model", private=True, exist_ok=True)
            assert private_ok(api, priv), "the private repository (hf_repos.txt) is public"
    else:
        api, repo = ms(); api.get_model(repo)


def record_transfer(rec):
    """v2.5.5b: one JSON line per upload / download in $OPD_ROOT/control/transfers.jsonl (the private status carries the
    last 200; MB/s only for >= 100 MB taking > 5 s, since upload_folder skips files the Hub already has)."""
    try:
        b, s = rec.get("bytes") or 0, rec.get("seconds") or 0
        rec["mb_s"] = round(b / 1e6 / s, 1) if b >= 100e6 and s > 5 else None
        d = os.path.join(os.environ.get("OPD_ROOT", os.path.expanduser("~/opd-run")), "control"); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "transfers.jsonl"), "a") as f:
            f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec}) + "\n")
    except OSError:
        pass


def _drill_fault(dst=""):
    """v2.5.7zb fault injection for the control-plane drill, only with DRILL=1 and REMOTE_FAULT set (setenv; read from
    the environment or, so the running heartbeat sees it without a restart, from $OPD_ROOT/control.env): the next put
    hangs once (marker control/drill_<fault>.done). hang_once can be cut by the caller's timeout; hang_once_hard also
    ignores SIGTERM, so only the watchdog (process-group KILL) ends it. v2.5.7zc: only the heartbeat's own status upload
    (dst status/<node>.json) is hit, never a receipt, an event or a weight upload."""
    if not re.fullmatch(r"status/[^/]+\.json", dst or ""):
        return
    env = dict(os.environ)
    try:
        for line in open(os.path.join(os.environ.get("OPD_ROOT", ""), "control.env")):
            k, _, v = line.strip().removeprefix("export ").partition("=")
            if k in ("DRILL", "REMOTE_FAULT"):
                env[k] = v.strip().strip('"')
    except OSError:
        pass
    fault = env.get("REMOTE_FAULT", "")
    if env.get("DRILL") != "1" or fault not in ("hang_once", "hang_once_hard"):
        return
    marker = os.path.join(os.environ.get("OPD_ROOT", "."), "control", f"drill_{fault}.done")
    if os.path.exists(marker):
        return
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    if fault == "hang_once_hard":
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(7200)


# [v2.5.9z52] (ops-lead / Fable 10-22 21:0x Beijing: the HF account answered 429 Too Many Requests from 12:38 UTC; status,
# pulse and receipts stopped going up) one backoff for every HF call of this node: after a 429 nothing is tried for 60 s, then
# 120, 240, 480, 900 s (the cap) after each further 429; a call that succeeds ends it. control/hf_backoff.json holds it, so the
# heartbeat, the uploader, upload_sync.sh and fetch_model.py share it; a call during the wait fails at once without a request.
BACKOFF_START_S, BACKOFF_MAX_S = 60, 900


class Backoff(Exception):
    pass


def _backoff_path():
    return os.path.join(os.environ.get("OPD_ROOT", ""), "control", "hf_backoff.json")


def _backoff_state():
    try:
        b = json.load(open(_backoff_path()))
        return b if isinstance(b, dict) else {}
    except (OSError, ValueError):
        return {}


def backoff_left():
    try:
        return max(0.0, float(_backoff_state().get("until", 0)) - time.time())
    except (TypeError, ValueError):
        return 0.0


def is_429(e):
    r = getattr(e, "response", None)
    return getattr(r, "status_code", None) == 429 or "Too Many Requests" in str(e)[:500]


def note_429():
    """The next wait: twice the last one (60 s first), at most BACKOFF_MAX_S; returns it."""
    try:
        n = int(_backoff_state().get("level", 0)) + 1
    except (TypeError, ValueError):
        n = 1
    wait = min(BACKOFF_START_S * 2 ** (n - 1), BACKOFF_MAX_S)
    p = _backoff_path(); os.makedirs(os.path.dirname(p), exist_ok=True); tmp = f"{p}.{os.getpid()}.tmp"
    json.dump({"level": n, "wait_s": wait, "until": time.time() + wait, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(tmp, "w"))
    os.replace(tmp, p)
    try:   # one line per 429 (control/hf_429.log: utc, level, wait): the count and the first time, for the reviewer's table
        with open(os.path.join(os.path.dirname(p), "hf_429.log"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{n}\t{wait}\n")
    except OSError:
        pass
    return wait


def note_ok():
    try:
        os.remove(_backoff_path())
    except FileNotFoundError:
        pass


def _gate():
    if backoff_left() > 0:
        raise Backoff()


WEIGHT_MB = 100   # [v2.5.9z50] ops-lead's A1 condition: weights stay out of the private repository while public uploads are paused


def is_weight(src, dst):
    """ckpts/ or any single file over WEIGHT_MB (a weight directory, a large generation file)."""
    if dst.lstrip("/").startswith("ckpts/"):
        return True
    files = [os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs] if os.path.isdir(src) else [src]
    return any(os.path.isfile(f) and os.path.getsize(f) > WEIGHT_MB * 1e6 for f in files)


def defer_weight(src, dst):
    """[v2.5.9z50] a public-class weight upload while public uploads are paused: nothing goes to the private repository (its
    quota is the status' and the receipts' too). The files are held under $CKPT_ROOT/weights_pending/<destination> (hard links
    on the same disk, else copies) BEFORE the put reports success, because upload_sync.sh deletes its local bf16 copy (KEEP_BF16=0)
    and the HF save after a successful put; one line in control/weights_pending.txt (utc, destination, the held copy, bytes) for
    the public mirror or PUBLIC_UPLOADS=1 to send later. Without a checkpoint disk the put fails, so the caller keeps its files."""
    import shutil
    ck = os.environ.get("CKPT_ROOT", "")
    if not ck or not os.path.isdir(ck):
        raise RuntimeError("no checkpoint disk to hold a deferred weight upload")
    hold = os.path.join(ck, "weights_pending", dst.lstrip("/"))
    files = [os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs] if os.path.isdir(src) else [src]
    size = 0
    for f in files:
        if not os.path.isfile(f):
            continue
        t = os.path.join(hold, os.path.relpath(f, src)) if os.path.isdir(src) else os.path.join(hold, os.path.basename(f))
        os.makedirs(os.path.dirname(t), exist_ok=True)
        if os.path.lexists(t):
            os.unlink(t)
        try:
            os.link(f, t)
        except OSError:
            shutil.copy2(f, t)
        size += os.path.getsize(f)
    d = os.path.join(os.environ.get("OPD_ROOT", ""), "control"); os.makedirs(d, exist_ok=True)
    import fcntl
    with open(os.path.join(d, "weights_pending.lock"), "w") as lk:   # [v2.5.9z62] weights_replay.py rewrites the list under this lock
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(os.path.join(d, "weights_pending.txt"), "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\t{dst}\t{hold}\t{size}\n")
    return size


def put(src, dst, kind=None):
    _drill_fault(dst)   # v2.5.7zb: a no-op unless DRILL=1 and REMOTE_FAULT are set (zc: the status upload only)
    if kind != "public" and public_paused() and rename(dst).lstrip("/").startswith(PUBLIC_PREFIXES) and is_weight(src, dst):   # [v2.5.9z50]
        record_transfer({"dir": "up", "dst": rename(dst), "deferred": "weights_pending", "bytes": defer_weight(src, rename(dst)), "ok": True})
        return
    _gate()   # [v2.5.9z52] (a held weight above needs no request)
    t0, ok = time.time(), False
    try:
        _put(src, dst, kind) if kind else _put(src, dst); ok = True
    except Exception as e:
        if is_429(e):
            note_429()
        raise
    finally:
        files = [os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs] if os.path.isdir(src) else [src]
        record_transfer({"dir": "up", "dst": rename(dst), "files": len(files),
                         "bytes": sum(os.path.getsize(f) for f in files if os.path.isfile(f)), "seconds": round(time.time() - t0, 1), "ok": ok})
    note_ok()


def put_many(listfile):
    """[v2.5.9z52] the events of an outbox pass in one commit (was one commit each): [[local file, destination], ...] to the
    private repository; files that are gone are left out."""
    pairs = [(s, d) for s, d in json.load(open(listfile)) if os.path.isfile(s)]
    if BACKEND != "hf":
        for s, d in pairs:
            put(s, d)
        return
    from huggingface_hub import CommitOperationAdd
    _gate()
    t0, ok = time.time(), False
    try:
        api, repo = hf("private")
        if split_mode() and not private_ok(api, repo):
            raise PermissionError("private-class file refused: the private repository is public")
        if pairs:
            api.create_commit(repo_id=repo, repo_type="model", commit_message=f"{len(pairs)} files: {rename(pairs[0][1])} ...",
                              operations=[CommitOperationAdd(path_in_repo=rename(d), path_or_fileobj=s) for s, d in pairs])
        ok = True
    except Exception as e:
        if is_429(e):
            note_429()
        raise
    finally:
        record_transfer({"dir": "up", "dst": f"batch of {len(pairs)}: {rename(pairs[0][1]) if pairs else '-'}", "files": len(pairs),
                         "bytes": sum(os.path.getsize(s) for s, _ in pairs), "seconds": round(time.time() - t0, 1), "ok": ok})
    note_ok()


TEXT_EXT = (".log", ".json", ".jsonl", ".txt", ".yaml", ".yml", ".csv", ".md", ".out", ".err", ".sh", ".py", ".cfg", ".toml", ".meta", ".part")
IGNORE = (".git/", ".cache/huggingface/")


def is_text(name, size):
    """Scrubbed as text: a known text extension, or a small file whose name has no other extension (train_log-like
    names and markers such as .failed / .inputs_ok, whose only dot is the leading one; v2.5.6a)."""
    n = name.lower()
    return n.endswith(TEXT_EXT) or (size < (1 << 20) and "." not in n.lstrip("."))


def scrub_roots():
    """(absolute path, placeholder) pairs, longest first: OPD_ROOT, CKPT_ROOT, RUNBOOK, ARCHIVE_ROOT, HOME and their real
    paths, and the directory the venv really lives in (LINK_VENV / FROM_OPD_ROOT: another disk)."""
    m = {}

    def add(p, name):
        if p and os.path.isabs(p):
            for q in (p.rstrip("/"), os.path.realpath(p).rstrip("/")):
                if q and q != "/" and q not in m:
                    m[q] = name
    for k, name in (("OPD_ROOT", "$OPD_ROOT"), ("CKPT_ROOT", "$CKPT_ROOT"), ("RUNBOOK", "$RUNBOOK"), ("ARCHIVE_ROOT", "$ARCHIVE_ROOT"),
                    ("FROM_OPD_ROOT", "$OPD_ROOT")):
        add(os.environ.get(k), name)
    v = os.path.join(os.environ.get("OPD_ROOT", ""), "venv")
    if os.path.exists(v):
        add(os.path.dirname(os.path.realpath(v)), "$OPD_ROOT")
    add(os.environ.get("HOME"), "~")
    return sorted(m.items(), key=lambda kv: -len(kv[0]))


PRIVATE_IP = re.compile(r"(?<![0-9.])(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])|169\.254)"
                        r"\.\d{1,3}\.\d{1,3}(?![0-9]|\.\d)")


def own_ips():
    """This machine's IPv4 addresses (v2.5.6g: Ray / vLLM print them in train.log)."""
    out = set()
    try:
        for fam, _, _, _, addr in socket.getaddrinfo(socket.gethostname(), None):
            if fam == socket.AF_INET:
                out.add(addr[0])
    except OSError:
        pass
    return {a for a in out if not a.startswith("127.")}


def scrub_text(t, roots, toks, ips=None, gen=False):
    """ips: addresses to replace by <ip> in addition to private / internal ranges; None = no address scrub (generation
    outputs: model text, where a string like 1.2.1.2 is content). gen (v2.5.6n): generation outputs, where a root is
    replaced only as the start of a path under it ("<root>/..."): $HOME alone is a short string that model text can
    contain, and its "~" replacement altered generations."""
    if ips is not None:
        for a in sorted(ips, key=len, reverse=True):
            t = re.sub(r"(?<![0-9.])" + re.escape(a) + r"(?![0-9])", "<ip>", t)
        t = PRIVATE_IP.sub("<ip>", t)
    """v2.5.5c: text for the public repository without this machine's paths, user or host names (B300 2026-10-18 02:1x
    Beijing: train.log, metrics.json, twr.json and gen_eval.log in the public repository carried them)."""
    for p, name in roots:
        t = t.replace(p + "/", name + "/") if gen else t.replace(p, name)

    def fix(m):
        words = {w.lower() for w in re.split(r"[^A-Za-z0-9_]+", m.group(0)) if w}
        return "<path>" if words & toks else m.group(0)
    t = re.sub(r"(?:/[A-Za-z0-9_.+=~-]+){2,}/?", fix, t)
    if toks:
        t = re.sub(r"(?i)(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, sorted(toks, key=len, reverse=True))) + r")(?![A-Za-z0-9_])", "<redacted>", t)
    return t


VOCAB_FILES = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer.model")
PATH_VALUE = re.compile(r"(?:~|\.{1,2}|\$[A-Z_]+)?/[^\s\"'<>]*")


def is_model_file(local, pin):
    """v2.5.6k: a file of a saved Hugging Face model directory (.../huggingface/ or huggingface_bf16/ in the repository
    path, or config.json next to it together with weights or a tokenizer)."""
    if re.search(r"/huggingface(?:_bf16)?/[^/]+$", "/" + pin):
        return True
    d = os.path.dirname(local)
    try:
        names = os.listdir(d)
    except OSError:
        return False
    return "config.json" in names and any(n.endswith(".safetensors") or n in VOCAB_FILES for n in names)


def scrub_model_file(name, data, roots, toks):
    """v2.5.6k: the text of a model directory file for the public repository, or None to upload it byte for byte. The
    vocabulary (tokenizer.json, vocab.json, merges.txt) is the base model's and goes up unchanged: the text scrub broke
    it (B300 2026-10-18: $HOME -> "~" also turned the vocabulary entry equal to $HOME into a second "~", and EVAL-e12b-P0 /
    -P2 failed 3 times each with "Token `~` out of vocabulary"). In a JSON config only string values that are paths
    (e.g. _name_or_path) are scrubbed, so the file stays the same config; other text files are scrubbed as usual."""
    if name in VOCAB_FILES:
        return None
    if not name.endswith(".json"):
        return scrub_text(data, roots, toks)
    try:
        obj = json.loads(data)
    except ValueError:
        return scrub_text(data, roots, toks)
    hit = []

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, str) and PATH_VALUE.fullmatch(x):
            y = scrub_text(x, roots, toks)
            if y != x:
                hit.append(1)
            return y
        return x
    out = walk(obj)
    return json.dumps(out, indent=2, ensure_ascii=False) + "\n" if hit else None


def put_public_unchanged(pairs, message):
    """v2.5.6n (scrubdiff fix): [(local file, path in the public repository)] uploaded byte for byte in one commit; the
    caller has checked every file (no forbidden token, no path of this machine)."""
    from huggingface_hub import CommitOperationAdd
    api, repo = hf("public")
    api.create_commit(repo_id=repo, repo_type="model", commit_message=message,
                      operations=[CommitOperationAdd(path_in_repo=d, path_or_fileobj=l) for l, d in pairs])


def _public_files(src, dst):
    if not os.path.isdir(src):
        return [(src, dst)]
    out = []
    for r, ds, fs in os.walk(src):
        for f in fs:
            rel = os.path.relpath(os.path.join(r, f), src).replace(os.sep, "/")
            if not rel.startswith(IGNORE):
                out.append((os.path.join(r, f), f"{dst.rstrip('/')}/{rel}"))
    return out


def _put_public(api, repo, src, dst):
    """One commit; text files go up scrubbed (scrub_text), everything else (weights) as it is."""
    from huggingface_hub import CommitOperationAdd
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from control import _forbidden_tokens
    roots, toks, ips = scrub_roots(), _forbidden_tokens(), own_ips()
    ops = []
    for local, pin in _public_files(src, dst):
        name = os.path.basename(local)
        if is_text(name, os.path.getsize(local)) and is_model_file(local, pin):
            t = scrub_model_file(name, open(local, "rb").read().decode("utf-8", "replace"), roots, toks)
            ops.append(CommitOperationAdd(path_in_repo=pin, path_or_fileobj=local if t is None else t.encode()))
        elif is_text(name, os.path.getsize(local)):
            data = open(local, "rb").read().decode("utf-8", "replace")
            gen = name.lower().endswith((".jsonl", ".part"))   # generation outputs: no address scrub
            ops.append(CommitOperationAdd(path_in_repo=pin, path_or_fileobj=scrub_text(data, roots, toks, None if gen else ips, gen=gen).encode()))
        else:
            ops.append(CommitOperationAdd(path_in_repo=pin, path_or_fileobj=local))
    if ops:
        api.create_commit(repo_id=repo, repo_type="model", operations=ops, commit_message=dst)


def _put(src, dst, force=None):
    dst = rename(dst)
    if BACKEND == "hf":
        kind = "public" if force == "public" else klass(dst)   # [v2.5.9z52] force: the pulse's public mirror
        api, repo = hf(kind)
        if kind == "private" and split_mode() and not private_ok(api, repo):
            raise PermissionError("private-class file refused: the private repository is public")
        if kind == "public":
            return _put_public(api, repo, src, dst)
        if os.path.isdir(src):
            api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dst, repo_type="model", commit_message=dst)
        else:
            api.upload_file(repo_id=repo, path_or_fileobj=src, path_in_repo=dst, repo_type="model", commit_message=dst)
    else:
        api, repo = ms()
        if os.path.isdir(src):
            api.upload_folder(repo_id=repo, folder_path=src, path_in_repo=dst, commit_message=dst)
        else:
            api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, commit_message=dst)


def done(task):
    if BACKEND != "hf":
        sys.exit(1)
    for kind in (("private", "public") if public_paused() else ("public",)):   # [v2.5.9z50] the private repository first
        if backoff_left() > 0:   # [v2.5.9z52]
            break
        try:
            api, repo = hf(kind)
            if api.file_exists(repo, f"runs/{task}/.done", repo_type="model"):
                sys.exit(0)
        except SystemExit:
            raise
        except Exception as e:
            if is_429(e):
                note_429()
            continue
    sys.exit(1)


if __name__ == "__main__":
    try:
        {"check": lambda: check(), "done": lambda: done(sys.argv[2]),
         "repo": lambda: print(fetch_repos() if sys.argv[2] == "public" else repos()[1]), "putmany": lambda: put_many(sys.argv[2]),
         "backoff": lambda: print(int(backoff_left()))}.get(sys.argv[1], lambda: put(sys.argv[2], sys.argv[3], (sys.argv[4:] or [None])[0]))()   # [v2.5.9z50] repo public: the fetch order
    except Exception as e:   # never echo the exception text verbatim: it can contain request headers
        print(f"{BACKEND} {sys.argv[1]} failed: {type(e).__name__}", file=sys.stderr); sys.exit(1)
