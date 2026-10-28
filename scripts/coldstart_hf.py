"""[v2.5.9z100] cold_start.sh's Hugging Face side (requirements v26 §11): every call is read-only except the model downloads, and
none prints a token. COLDSTART_FAKE=<repo.json> replaces the Hub with a JSON file ({"<repo>": {"<path>": {"size", "sha256" | "blob_id",
"text"}}}) for the Mac tests.

  reach <public> <private>                        both repositories answer (and the private one is private)
  write <private> <node>                          the token can write: a small probe file committed to status/<node>/ (ops-lead A1)
  status <private> <node> <done.txt>              the node's last status: the tasks whose state is done, one per line
  models <manifest> <done.txt> <models dir>       download what the open, unfinished tasks need (the four base models always)
  labels <manifest> <private> <opd_root>          the probe labels from the private repository, checked against the Hub's hash
  weights <manifest> <done.txt> <public> <private> <out.json>   which weights unfinished tasks need are on the Hub, and where
  only <manifest> <task>[,<task>...] <out.json>   [v2.5.9z104] the manifest cut to those tasks (cold_start.sh --only)
"""
import fnmatch
import hashlib
import json
import os
import re
import sys


def fake():
    p = os.environ.get("COLDSTART_FAKE")
    return json.load(open(p)) if p else None


def info(repo, paths):
    """{path: {"size", "sha256" or "blob_id"}} of the paths present in repo."""
    f = fake()
    if f is not None:
        return {p: f.get(repo, {})[p] for p in paths if p in f.get(repo, {})}
    from huggingface_hub import HfApi
    out, api = {}, HfApi()
    for i in range(0, len(paths), 50):
        for x in api.get_paths_info(repo, paths[i:i + 50], repo_type="model"):
            lfs = getattr(x, "lfs", None)
            if getattr(x, "size", None) is None and lfs is None:
                continue   # a folder
            out[x.path] = {"size": getattr(x, "size", None), "sha256": (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None,
                           "blob_id": getattr(x, "blob_id", None)}
    return out


def download(repo, path, dest):
    f = fake()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if f is not None:
        open(dest, "w").write(f[repo][path].get("text", "")); return dest
    from huggingface_hub import hf_hub_download
    import shutil, tempfile
    with tempfile.TemporaryDirectory(dir=os.path.dirname(dest)) as t:
        shutil.copy(hf_hub_download(repo, path, cache_dir=t), dest)
    return dest


def git_blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def done_set(path):
    try:
        return {l.strip() for l in open(path) if l.strip()}
    except OSError:
        return set()


def cmd_reach(pub, priv):
    f = fake()
    if f is not None:
        ok = pub in f and priv in f
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            ok = bool(api.model_info(priv).private) and api.model_info(pub) is not None
        except Exception as e:
            print(f"REACH FAIL {type(e).__name__}"); return 1
    print("REACH " + ("PASS" if ok else "FAIL")); return 0 if ok else 1


def cmd_status(priv, node, out):
    path = f"status/{node}.json"
    try:
        tmp = download(priv, path, out + ".status.json")
        st = json.load(open(tmp))
    except Exception as e:
        print(f"STATUS none ({type(e).__name__}): every task counts as not done"); open(out, "w").write(""); return 0
    open(out + ".utc", "w").write(str(st.get("utc") or ""))   # the reused node's last sign of life (cold_start.sh's silent check)
    tasks = st.get("tasks") or {}
    items = tasks.items() if isinstance(tasks, dict) else ((v.get("task"), v) for v in tasks if isinstance(v, dict))   # as ctl.py reads it
    done = sorted(t for t, v in items if t and isinstance(v, dict) and v.get("state") == "done")
    open(out, "w").write("".join(t + "\n" for t in done))
    print(f"STATUS {len(done)} of {len(tasks)} tasks done (last status {st.get('utc')})"); return 0


def cmd_write(priv, node):
    f = fake()
    if f is not None:
        ok = bool(f.get("_writable", {}).get(priv))
    else:
        from huggingface_hub import HfApi
        import time
        try:
            HfApi().upload_file(path_or_fileobj=json.dumps({"coldstart_probe": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}).encode(),
                                path_in_repo=f"status/{node}/coldstart_probe.json", repo_id=priv, repo_type="model", commit_message="coldstart write probe")
            ok = True
        except Exception as e:
            print(f"WRITE FAIL {type(e).__name__}"); return 1
    print("WRITE " + ("PASS" if ok else "FAIL")); return 0 if ok else 1


def needed(m, done):
    return [x for x in m["models"] if x.get("always") or any(t not in done for t in x.get("tasks", []))]


def cmd_models(manifest, done_txt, models_dir):
    m, done = json.load(open(manifest)), done_set(done_txt)
    rc = 0
    for x in needed(m, done):
        dest = os.path.join(models_dir, x["name"])
        if x.get("by"):
            print(f"MODEL {x['name']} by {x['by']}"); continue
        if os.path.exists(os.path.join(dest, ".fetch_ok")):
            print(f"MODEL {x['name']} present"); continue
        if fake() is not None:
            os.makedirs(dest, exist_ok=True); open(os.path.join(dest, ".fetch_ok"), "w").write("fake\n"); print(f"MODEL {x['name']} fetched (fake)"); continue
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(x["hf"], local_dir=dest, max_workers=int(os.environ.get("COLDSTART_WORKERS", "8")))
            open(os.path.join(dest, ".fetch_ok"), "w").write(x["hf"] + "\n"); print(f"MODEL {x['name']} fetched")
        except Exception as e:
            rc = 1; print(f"MODEL {x['name']} FAIL {type(e).__name__}")
    return rc


def cmd_labels(manifest, priv, opd_root):
    m = json.load(open(manifest)); rc = 0
    for a in m["artifacts"]:
        have = info(priv, [a["repo_path"]]).get(a["repo_path"])
        if not have:
            print(f"LABELS {a['name']} MISSING in the private repository: rerun {a['rerun']} as a NEW label version"); rc = 2; continue
        dest = download(priv, a["repo_path"], os.path.join(opd_root, a["node_path"]))
        ok = (sha256(dest) == have["sha256"]) if have.get("sha256") else (have.get("blob_id") in (None, git_blob(dest)))
        print(f"LABELS {a['name']} {'PASS' if ok else 'FAIL hash differs'}"); rc = rc if ok else 1
        if not ok:
            os.remove(dest)
    return rc


def cmd_weights(manifest, done_txt, pub, priv, out):
    m, done = json.load(open(manifest)), done_set(done_txt)
    want = [w for w in m["weights"] if any(t not in done for t in w["tasks"])]
    paths = [w["path"] + "/config.json" for w in want]
    on_pub, on_priv = info(pub, paths), info(priv, paths)
    rows = [{"path": w["path"], "tasks": [t for t in w["tasks"] if t not in done],
             "where": "public" if p in on_pub else "private" if p in on_priv else "missing"} for w, p in zip(want, paths)]
    json.dump({"weights": rows}, open(out, "w"), indent=1)
    miss = [r for r in rows if r["where"] == "missing"]
    print(f"WEIGHTS {len(rows)} needed, {len(rows) - len(miss)} on the Hub, {len(miss)} missing" + (": " + ", ".join(r["path"].split("/")[1] for r in miss[:10]) if miss else ""))
    return 0


def cmd_only(manifest, tasks, out):
    """[v2.5.9z104] (Fable 10-28: a temporary H800 to adapt, one canary task) the manifest cut to <tasks>: the always-models and the
    models those tasks use; the labels whose used_by names one of them (a name or a glob such as E10b-P5route*); every other list
    of entries with "tasks" (weights, starts, pairs) keeps the entries of those tasks. Stdlib only, nothing fetched."""
    want = [t for t in tasks.split(",") if t]
    if not want or not all(re.fullmatch(r"[A-Za-z0-9._-]+", t) for t in want):
        print(f"ONLY bad task list {tasks!r}"); return 2
    m = json.load(open(manifest))
    uses = lambda x: bool(set(want) & set(x.get("tasks") or []))
    users = lambda a: [p.strip().split()[0] for p in str(a.get("used_by", "")).split(",") if p.strip()]
    r = dict(m, only=want)
    for k, v in m.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) and "tasks" in x for x in v):
            r[k] = [x for x in v if (k == "models" and x.get("always")) or uses(x)]
    r["artifacts"] = [a for a in m.get("artifacts", []) if any(fnmatch.fnmatchcase(t, u) for t in want for u in users(a))]
    json.dump(r, open(out, "w"), indent=1, sort_keys=True)
    print(f"ONLY {','.join(want)}: models {', '.join(x['name'] for x in r['models'])}; labels {len(r['artifacts'])}; "
          f"weights {len(r.get('weights', []))}")
    return 0


def main():
    c, a = sys.argv[1], sys.argv[2:]
    return {"reach": lambda: cmd_reach(*a), "write": lambda: cmd_write(*a), "status": lambda: cmd_status(*a), "models": lambda: cmd_models(*a),
            "labels": lambda: cmd_labels(*a), "weights": lambda: cmd_weights(*a), "only": lambda: cmd_only(*a)}[c]()


if __name__ == "__main__":
    sys.exit(main())
