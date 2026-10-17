"""Result-repository backend for the remote node: Hugging Face (HF_TOKEN, HF_REPO) or ModelScope (MS_TOKEN, MS_REPO),
chosen by UPLOAD_BACKEND. Credentials are read from the environment only and never printed or written to disk.
  python remote.py check                 -> exit 0 if the backend is usable (token valid, repo reachable)
  python remote.py put <local> <remote>  -> upload a file or a folder to <remote> in the repo
  python remote.py done <task>           -> exit 0 if runs/<task>/.done exists in the repo (HF only; ModelScope: no)
  python remote.py repo public|private   -> print the repository that class of files goes to
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


def klass(dst):
    return "public" if dst.lstrip("/").startswith(PUBLIC_PREFIXES) else "private"


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


def put(src, dst):
    t0, ok = time.time(), False
    try:
        _put(src, dst); ok = True
    finally:
        files = [os.path.join(r, f) for r, _, fs in os.walk(src) for f in fs] if os.path.isdir(src) else [src]
        record_transfer({"dir": "up", "dst": rename(dst), "files": len(files),
                         "bytes": sum(os.path.getsize(f) for f in files if os.path.isfile(f)), "seconds": round(time.time() - t0, 1), "ok": ok})


def _put(src, dst):
    dst = rename(dst)
    if BACKEND == "hf":
        kind = klass(dst)
        api, repo = hf(kind)
        if kind == "private" and split_mode() and not private_ok(api, repo):
            raise PermissionError("private-class file refused: the private repository is public")
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
    api, repo = hf("public")
    sys.exit(0 if api.file_exists(repo, f"runs/{task}/.done", repo_type="model") else 1)


if __name__ == "__main__":
    try:
        {"check": lambda: check(), "done": lambda: done(sys.argv[2]),
         "repo": lambda: print(repos()[0 if sys.argv[2] == "public" else 1])}.get(sys.argv[1], lambda: put(sys.argv[2], sys.argv[3]))()
    except Exception as e:   # never echo the exception text verbatim: it can contain request headers
        print(f"{BACKEND} {sys.argv[1]} failed: {type(e).__name__}", file=sys.stderr); sys.exit(1)
