"""fetch_model.py (runbook v2.5.4e): download one model directory of a Hugging Face repository completely, or say why not.

    python fetch_model.py <repo> <path prefix> <local dir> [--task NAME]

The file list, sizes and LFS sha256 come from the Hub API (list_repo_tree). Endpoints are tried in the order of
HF_DOWNLOAD_ENDPOINTS (comma list; default https://huggingface.co, then https://hf-mirror.com), each up to 3 times, with
the token (HF_TOKEN) sent to huggingface.co only, never to a mirror; longer timeouts than the library's (HF_HUB_ETAG_TIMEOUT 60 s, HF_HUB_DOWNLOAD_TIMEOUT 120 s, unless set). The directory
counts as complete when every listed file exists locally with the listed size and every LFS file has the listed sha256;
a file of the wrong size is downloaded again. A complete directory gets <local dir>/<prefix>/.fetch_ok (the file list
with sizes and sha256), so a later call returns at once. One JSON line on stdout: ok, endpoint, files, bytes fetched,
seconds, MB/s, and the error of every failed attempt (type and the first 200 characters); with --task, also an event
fetch_done / fetch_failed. Exit 0 when complete, 1 when not.
(B300, 2026-10-17: snapshot_download through hf-mirror.com failed with FileMetadataError and through huggingface.co with
LocalEntryNotFoundError on the HEAD call; a directory with config.json but no weights was then taken as a model.)
"""
import hashlib
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")       # read by huggingface_hub at import
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.pop("HF_HUB_OFFLINE", None)
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") is None:   # v2.5.5: multi-connection downloads (install_uv.sh installs hf-transfer;
    try:                                                    # B300 17:56: 7.9 MB/s single-connection from huggingface.co)
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except ImportError:
        pass

ENDPOINTS = [e.strip() for e in os.environ.get("HF_DOWNLOAD_ENDPOINTS", "https://huggingface.co,https://hf-mirror.com").split(",") if e.strip()]
TRIES = int(os.environ.get("FETCH_TRIES", "3"))


def sha256(path, bs=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def listing(api, repo, prefix):
    """[(path, size, sha256 or None)] of the files under prefix."""
    out = []
    for f in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True):
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


def fetch(repo, prefix, local):
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub import constants as K
    prefix = prefix.strip("/")
    ok_file = os.path.join(local, prefix, ".fetch_ok")
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
                for path, size, sha in todo:
                    p = os.path.join(local, path)
                    if os.path.isfile(p):
                        os.remove(p)      # wrong size or sha256: fetch again (only our own download directory)
                    hf_hub_download(repo, filename=path, local_dir=local, endpoint=ep, token=tok)
                    got += size
                todo = missing(files, local)
                if not todo:
                    os.makedirs(os.path.dirname(ok_file), exist_ok=True)
                    json.dump({"repo": repo, "prefix": prefix, "files": files, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(ok_file, "w"))
                    s = time.time() - t0
                    return {"ok": True, "endpoint": ep, "hf_transfer": bool(getattr(K, "HF_HUB_ENABLE_HF_TRANSFER", False)),
                            "files": len(files), "bytes": got, "seconds": round(s, 1),
                            "mb_s": round(got / 1e6 / s, 1) if s > 0 and got else None, "errors": errors}
                errors.append({"endpoint": ep, "attempt": attempt, "error": "Incomplete", "detail": ", ".join(p for p, _, _ in todo)[:200]})
            except Exception as e:   # every failure is recorded; the next attempt / endpoint follows
                errors.append({"endpoint": ep, "attempt": attempt, "error": type(e).__name__, "detail": str(e)[:200]})
                if getattr(K, "HF_HUB_ENABLE_HF_TRANSFER", False):   # the next attempts go without hf_transfer
                    K.HF_HUB_ENABLE_HF_TRANSFER = False; errors[-1]["hf_transfer_off"] = True
            time.sleep(min(30, 5 * attempt))
    return {"ok": False, "files": None, "bytes": got, "seconds": round(time.time() - t0, 1), "errors": errors}


def main():
    a = sys.argv[1:]
    task = None
    if "--task" in a:
        i = a.index("--task"); task = a[i + 1]; a = a[:i] + a[i + 2:]
    if len(a) != 3:
        sys.exit(__doc__)
    repo, prefix, local = a
    r = fetch(repo, prefix, local)
    r.update({"repo": repo, "prefix": prefix})
    print(json.dumps(r), flush=True)
    if task and not r.get("cached"):
        brief = {k: r.get(k) for k in ("endpoint", "hf_transfer", "files", "bytes", "seconds", "mb_s")}
        errs = "; ".join(f"{e['endpoint'].split('//')[-1]}#{e['attempt']} {e['error']}" for e in r.get("errors", []))[:400]
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "control.py"), "event", task,
                        "fetch_done" if r["ok"] else "fetch_failed", f"{prefix} {json.dumps(brief)} errors: {errs or 'none'}"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
