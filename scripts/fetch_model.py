"""fetch_model.py (runbook v2.5.4e): download one model directory of a Hugging Face repository completely, or say why not.

    python fetch_model.py <repo> <path prefix> <local dir> [--task NAME] [--dataset]

[v2.5.9e] --dataset: the repository is a dataset (repo_type "dataset"; the slow background pulls of run_tasks.sh's
"#fetch dataset:<owner>/<name>" lines).

The file list, sizes and LFS sha256 come from the Hub API (list_repo_tree). Endpoints are tried in the order of
HF_DOWNLOAD_ENDPOINTS (comma list; default https://huggingface.co, then https://hf-mirror.com), each up to 3 times, with
the token (HF_TOKEN) sent to huggingface.co only, never to a mirror; longer timeouts than the library's (HF_HUB_ETAG_TIMEOUT 60 s, HF_HUB_DOWNLOAD_TIMEOUT 120 s, unless set). The directory
counts as complete when every listed file exists locally with the listed size and every LFS file has the listed sha256;
a file of the wrong size is downloaded again. v2.5.7g: a prefix ending in .jsonl or .json names one file (PROBE-TERM's
archived generations; its metadata from get_paths_info, its record <local dir>/<prefix>.fetch_ok).
A complete directory gets <local dir>/<prefix>/.fetch_ok (the file list
with sizes and sha256), so a later call returns at once. One JSON line on stdout: ok, endpoint, files, bytes fetched,
seconds, MB/s, and the error of every failed attempt (type and the first 200 characters); with --task, also an event
fetch_done / fetch_failed. Exit 0 when complete, 1 when not.
(B300, 2026-10-17: snapshot_download through hf-mirror.com failed with FileMetadataError and through huggingface.co with
LocalEntryNotFoundError on the HEAD call; a directory with config.json but no weights was then taken as a model.)
[v2.5.8] One fetch per path at a time: an flock on <local dir>/.<prefix>.lock, the wait bounded by FETCH_LOCK_WAIT
(default FETCH_TIMEOUT + 300 s); the second caller then finds the files complete and downloads nothing (10-19: two
readings fetching one result: path blocked each other for 35 min). [v2.5.8] An overall cap per call: the download
runs in a child process (its own session), ended with its group after FETCH_TIMEOUT (default 3600 s); the files
already complete stay and the next call goes on from them. Both give an error LockTimeout / FetchTimeout.
"""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")       # read by huggingface_hub at import
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.pop("HF_HUB_OFFLINE", None)
# v2.5.7zh (reviewer 17:4x UTC, 10-19): Xet downloads through the node's proxy fail (the xet-read-token call: ProxyError;
# the t14 arms, three result: readings and the 32B fetch hung or failed on it), so fetches go over plain HTTP by default
# (HF_HUB_DISABLE_XET=1) and without hf_transfer, whose downloads do not resume (a partial file under
# .cache/huggingface/download is continued by the plain download). FETCH_XET=1 / FETCH_HF_TRANSFER=1 bring them back.
if os.environ.get("FETCH_XET") != "1":
    os.environ["HF_HUB_DISABLE_XET"] = "1"
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None and os.environ.get("FETCH_HF_TRANSFER") == "1":   # v2.5.5: multi-connection
    try:                                                    # downloads (B300 17:56: 7.9 MB/s single-connection from huggingface.co)
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        pass
elif os.environ.get("FETCH_HF_TRANSFER") != "1":
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

ENDPOINTS = [e.strip() for e in os.environ.get("HF_DOWNLOAD_ENDPOINTS", "https://huggingface.co,https://hf-mirror.com").split(",") if e.strip()]


def _fetch_endpoint():
    """v2.5.7zi (reviewer 18:1x UTC, pre-approved): FETCH_ENDPOINT (setenv; read from the environment or, so the running
    run_tasks loop sees it, from $OPD_ROOT/control.env) = the one endpoint to fetch from, also as HF_ENDPOINT; empty =
    the official one first, as before. No token goes to it (HF_TOKEN only ever to huggingface.co)."""
    v = os.environ.get("FETCH_ENDPOINT")
    if v is None:
        try:
            for line in open(os.path.join(os.environ.get("OPD_ROOT", ""), "control.env")):
                k, _, x = line.strip().removeprefix("export ").partition("=")
                if k == "FETCH_ENDPOINT":
                    v = x.strip().strip('"')
        except OSError:
            pass
    return (v or "").strip().rstrip("/")


if _fetch_endpoint():
    ENDPOINTS = [_fetch_endpoint()]
    os.environ["HF_ENDPOINT"] = ENDPOINTS[0]   # read by huggingface_hub at import (fetch() imports it later)
TRIES = int(os.environ.get("FETCH_TRIES", "3"))
FETCH_TIMEOUT = int(os.environ.get("FETCH_TIMEOUT", "3600"))                          # [v2.5.8]
FETCH_LOCK_WAIT = int(os.environ.get("FETCH_LOCK_WAIT", str(FETCH_TIMEOUT + 300)))   # [v2.5.8]


def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def single_file(prefix):
    return prefix.endswith((".jsonl", ".json"))


REPO_TYPE = None   # [v2.5.9e] "dataset" with --dataset


def listing(api, repo, prefix):
    """[(path, size, sha256 or None)] of the files under prefix (or of the one file prefix names)."""
    out = []
    entries = (api.get_paths_info(repo, [prefix], repo_type=REPO_TYPE) if single_file(prefix)
               else api.list_repo_tree(repo, path_in_repo=prefix or None, recursive=True, repo_type=REPO_TYPE))
    for f in entries:
        size = getattr(f, "size", None)
        if size is None:          # a folder
            continue
        lfs = getattr(f, "lfs", None)
        sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
        out.append((f.path, int(size), sha))
    return out


def missing(files, local):
    """Files not yet right locally (size; sha256 for LFS files)."""
    bad = []
    for path, size, sha in files:
        p = os.path.join(local, path)
        if not os.path.isfile(p) or os.path.getsize(p) != size or (sha and sha256(p) != sha):
            bad.append((path, size, sha))
    return bad


def night(t=None):   # [v2.5.9q] the window large pulls run in: 22:00-08:00 Beijing (UTC+8)
    h = (time.gmtime(t).tm_hour + 8) % 24
    return h >= 22 or h < 8


def uploads_timing_out():   # [v2.5.9e/9q] as fetch_pass: a result upload timed out in the last hour (results go first)
    p = os.path.join(os.environ.get("OPD_ROOT", ""), "logs", "heartbeat.err")
    try:
        if time.time() - os.path.getmtime(p) > 3600:
            return False
        return "UPLOAD_TIMEOUT" in "".join(open(p, errors="ignore").readlines()[-20:])
    except OSError:
        return False


def _cfg(name, default):
    """[v2.5.9r] a pull setting: $OPD_ROOT/control.env first (setenv reaches the running run_tasks loop without a restart, as
    FETCH_ENDPOINT), then the environment, then the default."""
    try:
        for line in open(os.path.join(os.environ.get("OPD_ROOT", ""), "control.env")):
            k, _, x = line.strip().removeprefix("export ").partition("=")
            if k == name and x.strip().strip('"'):
                return x.strip().strip('"')
    except OSError:
        pass
    return os.environ.get(name) or default


def day_max():   # [v2.5.9r] Runming 10-21 23:3x Beijing: by day as at night (nobody competes for the bandwidth): no day cap
    return int(float(_cfg("FETCH_DAY_MAX_BYTES", "1e13")))


def conns_now():   # [v2.5.9r] the same 32 by day and at night unless setenv says otherwise (PROBE-NET2's best)
    n = int(_cfg("FETCH_CONNS_NIGHT", "32") if night() else _cfg("FETCH_CONNS_DAY", "32"))
    cap = os.environ.get("FETCH_CONNS_CAP", "")   # [v2.5.9z43] (ops-lead 10-22, until the node-wide connection budget) one pull's
    return min(n, int(cap)) if cap.isdigit() and int(cap) > 0 else n   # ceiling, from its own environment only (not setenv)


def parallel_download(repo, path, size, local, ep):
    """[v2.5.9q] (FETCH_PARALLEL=1: run_tasks.sh's background pulls; Fable's rule 10-21 22:5x Beijing) one file in parallel
    Range segments (par_fetch.py), FETCH_CONNS_NIGHT at night and FETCH_CONNS_DAY by day (both 32; [v2.5.9r] the day like the
    night); a file over FETCH_DAY_MAX_BYTES ([v2.5.9r] 1e13: no cap) only at night (outside it: not started, or stopped
    at the end of the window, Deferred);
    every segment waits while result uploads time out. No token is sent (public repositories); [v2.5.9z25] a 401 / 403 from
    huggingface.co (a gated repository) makes fetch() take that file through hf_hub_download with the token instead."""
    from huggingface_hub import hf_hub_url
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import par_fetch
    big = size > day_max()
    if big and not night():
        raise par_fetch.Deferred(f"{path}: {size / 1e9:.1f} GB waits for the night window (22:00-08:00 Beijing)")
    conns = conns_now()
    dst = os.path.join(local, path); os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    return par_fetch.download(hf_hub_url(repo, path, repo_type=REPO_TYPE, endpoint=ep), dst, size, conns,
                              may_run=lambda: not uploads_timing_out(), stop=lambda: big and not night(),
                              log=lambda m: print(f"PAR_FETCH {path}: {m}", file=sys.stderr, flush=True))


def fetch(repo, prefix, local):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub import constants as K
    prefix = prefix.strip("/")
    ok_file = os.path.join(local, prefix + ".fetch_ok") if single_file(prefix) else os.path.join(local, prefix, ".fetch_ok")
    if os.path.isfile(ok_file):
        try:
            files = [tuple(x) for x in json.load(open(ok_file))["files"]]
            if all(os.path.isfile(os.path.join(local, p)) and os.path.getsize(os.path.join(local, p)) == s for p, s, _ in files):
                return {"ok": True, "cached": True, "files": len(files)}
        except (ValueError, KeyError, TypeError):
            pass
    errors, t0, got = [], time.time(), 0
    for ep in ENDPOINTS:
        tok = None if ep.rstrip("/") == "https://huggingface.co" else False   # HF_TOKEN goes to huggingface.co only, never to a mirror
        api = HfApi(endpoint=ep, token=tok)
        for attempt in range(1, TRIES + 1):
            try:
                files = listing(api, repo, prefix)
                if not files:
                    errors.append({"endpoint": ep, "attempt": attempt, "error": "NoFiles", "detail": f"nothing under {prefix}"})
                    break
                todo = missing(files, local)
                deferred, token_files = [], []
                for path, size, sha in sorted(todo, key=lambda x: x[1]):   # [v2.5.9q] small files first
                    p = os.path.join(local, path)
                    if os.path.isfile(p):
                        os.remove(p)      # wrong size or sha256: fetch again (only our own download directory)
                    if os.environ.get("FETCH_PARALLEL") == "1":   # [v2.5.9q]
                        import par_fetch
                        try:
                            got += parallel_download(repo, path, size, local, ep)["bytes"]
                        except par_fetch.Deferred as e:
                            deferred.append(str(e)); continue
                        except urllib.error.HTTPError as e:   # [v2.5.9z25] a gated repository (GPQA 10-22: 401 on every try): par_fetch
                            if e.code not in (401, 403) or tok is False:   # sends no token, so this file goes through hf_hub_download,
                                raise                                      # one connection, the token to huggingface.co only
                            for x in (p + ".part", p + ".part.segs"):
                                if os.path.isfile(x):
                                    os.remove(x)
                            hf_hub_download(repo, filename=path, local_dir=local, endpoint=ep, token=tok, repo_type=REPO_TYPE)
                            got += size; token_files.append(path)
                    else:
                        hf_hub_download(repo, filename=path, local_dir=local, endpoint=ep, token=tok, repo_type=REPO_TYPE)
                        got += size
                if deferred:   # [v2.5.9q] not a failure of this endpoint: the next call (after FETCH_RETRY_MIN) goes on
                    return {"ok": False, "deferred": True, "files": len(files), "bytes": got, "seconds": round(time.time() - t0, 1),
                            "errors": errors + [{"endpoint": ep, "error": "Deferred", "detail": "; ".join(deferred)[:300]}]}
                todo = missing(files, local)
                if not todo:
                    os.makedirs(os.path.dirname(ok_file), exist_ok=True)
                    json.dump({"repo": repo, "kind": REPO_TYPE or "model", "prefix": prefix, "files": files, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(ok_file, "w"))
                    s = time.time() - t0
                    return {"ok": True, "endpoint": ep, "hf_transfer": bool(getattr(K, "HF_HUB_ENABLE_HF_TRANSFER", False)),
                            "files": len(files), "bytes": got, "seconds": round(s, 1),
                            "mb_s": round(got / 1e6 / s, 1) if s > 0 and got else None, "errors": errors,
                            "token_files": len(token_files)}   # [v2.5.9z25] files fetched with the token after par_fetch's 401 / 403
                errors.append({"endpoint": ep, "attempt": attempt, "error": "Incomplete", "detail": ", ".join(p for p, _, _ in todo)[:200]})
            except Exception as e:   # every failure is recorded; the next attempt / endpoint follows
                errors.append({"endpoint": ep, "attempt": attempt, "error": type(e).__name__, "detail": str(e)[:200]})
                if getattr(K, "HF_HUB_ENABLE_HF_TRANSFER", False):   # the next attempts go without hf_transfer
                    K.HF_HUB_ENABLE_HF_TRANSFER = False; errors[-1]["hf_transfer_off"] = True
            time.sleep(min(30, 5 * attempt))
    return {"ok": False, "files": None, "bytes": got, "seconds": round(time.time() - t0, 1), "errors": errors}


def lock_path(prefix, local):
    """[v2.5.8] <local dir>/.<prefix with / as __>.lock (the whole directory when the prefix is empty)."""
    return os.path.join(local, "." + (prefix.strip("/").replace("/", "__") or "_all") + ".lock")


def fail(error, detail, t0):
    return {"ok": False, "files": None, "bytes": 0, "seconds": round(time.time() - t0, 1),
            "errors": [{"endpoint": "-", "attempt": 0, "error": error, "detail": detail}]}


def locked_fetch(repo, prefix, local):
    """[v2.5.8] fetch() under the path's lock, in a child process ended after FETCH_TIMEOUT."""
    t0 = time.time()
    os.makedirs(local, exist_ok=True)
    fd = os.open(lock_path(prefix, local), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > FETCH_LOCK_WAIT:
                    return fail("LockTimeout", f"another fetch held {os.path.basename(lock_path(prefix, local))} for {FETCH_LOCK_WAIT}s", t0)
                time.sleep(5)
        waited = round(time.time() - t0, 1)
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", repo, prefix, local] + (["--dataset"] if REPO_TYPE else []),
                             stdout=subprocess.PIPE, text=True, start_new_session=True)

        def on_term(signum, frame):   # [v2.5.8] a TERM to this call (timeout, kill) ends the download too
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            sys.exit(128 + signum)
        signal.signal(signal.SIGTERM, on_term)
        try:
            out, _ = p.communicate(timeout=FETCH_TIMEOUT)
        except subprocess.TimeoutExpired:
            for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 15)):
                try:
                    os.killpg(p.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    p.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    pass
            return fail("FetchTimeout", f"{prefix or repo} not complete after {FETCH_TIMEOUT}s", t0)
        r = None
        for line in reversed((out or "").splitlines()):
            try:
                r = json.loads(line)
                break
            except ValueError:
                continue
        if not isinstance(r, dict) or "ok" not in r:
            return fail("ChildFailed", f"exit {p.returncode}, no result line", t0)
        if waited >= 5:
            r["lock_wait_s"] = waited
        return r
    finally:
        os.close(fd)


def main():
    a = sys.argv[1:]
    global REPO_TYPE
    if "--dataset" in a:   # [v2.5.9e]
        a.remove("--dataset"); REPO_TYPE = "dataset"
    if a[:1] == ["--child"]:   # [v2.5.8] the download itself, run by locked_fetch
        print(json.dumps(fetch(*a[1:4])), flush=True)
        return
    task = None
    if "--task" in a:
        i = a.index("--task"); task = a[i + 1]; a = a[:i] + a[i + 2:]
    if len(a) != 3:
        sys.exit(__doc__)
    repo, prefix, local = a
    # [v2.5.9z50] a comma separated list of repositories (remote.py repo public while public uploads are paused: the private
    # one first): each in turn until one has the files; the result names the one that did and keeps every failed attempt
    tried, errs = [], []
    for rp in [x for x in repo.split(",") if x]:
        r = locked_fetch(rp, prefix, local)
        tried.append(rp); errs += r.get("errors") or []
        if r.get("ok"):
            break
    r.update({"repo": rp, "prefix": prefix, "repos_tried": len(tried)})
    if not r.get("ok"):
        r["errors"] = errs
    if not r.get("cached"):   # v2.5.5b: into control/transfers.jsonl like every upload
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from remote import record_transfer
            record_transfer({"dir": "down", "dst": prefix, "endpoint": r.get("endpoint"), "hf_transfer": r.get("hf_transfer"),
                             "files": r.get("files"), "bytes": r.get("bytes"), "seconds": r.get("seconds"), "ok": r["ok"],
                             "failed_attempts": len(r.get("errors") or [])})
        except Exception:
            pass
    print(json.dumps(r), flush=True)
    if task and not r.get("cached") and not r.get("deferred"):   # [v2.5.9q] a deferral (large file by day) is no event
        brief = {k: r.get(k) for k in ("endpoint", "hf_transfer", "files", "bytes", "seconds", "mb_s", "lock_wait_s")}
        errs = "; ".join(f"{e['endpoint'].split('//')[-1]}#{e['attempt']} {e['error']}" for e in r.get("errors", []))[:400]
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "control.py"), "event", task,
                        "fetch_done" if r["ok"] else "fetch_failed", f"{prefix} {json.dumps(brief)} errors: {errs or 'none'}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
