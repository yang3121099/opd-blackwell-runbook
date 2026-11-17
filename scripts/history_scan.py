"""[v3.7] history_scan.py: earlier installs of ours on this machine's disks, found by the node itself (Fable 11-17; Runming: "every
new machine scans the history directories and reuses what it can"). A borrowed machine came up on a fresh local disk and downloaded
22 GB of base models and 20 GB of our own weights through a 14 MB/s proxy while two earlier installs of ours, with the same files,
sat on the shared disks it mounts. The earlier bounded walk of every mount (mount_probe.py) ended before it reached them.

Where it looks (paths never leave the node: they are in no control file, the result is node-local):
  1. what the node was told: node.env OLD_BASE_CANDS / FROM_OPD_ROOT / FETCH_LOCAL_ROOTS;
  2. hints: the directories this machine's own work lives in - the working directory and the script paths of the processes on its
     GPUs (the owner's filler runs from his code directory), the stored filler command, $HOME - and, for each, the same relative
     directory on every other real mount (a user's directory usually has the same name on each shared disk);
  3. below each hint, breadth first to HISTORY_DEPTH (4) levels, at most HISTORY_MAX_DIRS (6000) directories and HISTORY_SCAN_S
     (60) s in all: a directory is an install of ours when it holds one of paths.env / node.env / control.env / models.env and
     one of models / venv / runs / control / runbook. Heavy directories (models, runs, venvs, caches, .git ...) are not entered.
  4. without any hint: the same search from the root of each network mount.

What it writes (0600, node-local, "secret" in the names keeps them out of snapshot / upload):
  control/history_secret.json   every root: path, last change, complete base models, venvs present, fetched weights, data files
  control/history_secret.env    export OLD_BASE_CANDS=<told ones>:<found ones, newest first>   (env.sh sources it)
and what uses it: fetch_model.py copies a weight from such a root before it asks the Hub (z134, local_roots: size and sha256 are
still checked against the Hub's list); overnight.sh copies a complete base model from it before it downloads (v3.7). Data files
and venvs found there are reported, never taken over by themselves (an older prep_data.py, a venv built weeks ago).

Usage: python3 history_scan.py [--if-missing] [--print]      (stdlib only; called by env_probe.py and by overnight.sh's step 0)"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.environ.get("OPD_ROOT", "")
MARKS = {"paths.env", "node.env", "control.env", "models.env"}
PARTS = {"models", "venv", "runs", "control", "runbook"}
SKIP = {"models", "runs", "venv", "venv-fa", "venv-vl", "venv-ctl", "venv-ifeval", "cache", "data", "logs", "ckpt", "node_modules", "G-OPD", "runbook",
        "tmp", "hf_home", "__pycache__", "site-packages", "lib", "lib64", "bin", "share", "include", "datasets", "checkpoints", "outputs", "wandb",
        "proc", "sys", "dev", "adhoc_models", "opd-dr", "control"}
MAX_ENTRIES = int(os.environ.get("HISTORY_MAX_ENTRIES", 5000))
NET_FS = ("nfs", "nfs4", "ceph", "lustre", "gpfs", "cifs", "beegfs", "virtiofs", "glusterfs", "fuse.")
REAL_FS = ("ext2", "ext3", "ext4", "xfs", "btrfs", "zfs") + NET_FS


def mounts(text):
    """[(mount point, type)] of the real file systems, longest mount point first; '/' and bind mounts of single files are left out."""
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 3 and f[1] != "/" and f[2].startswith(REAL_FS):
            mp = f[1].replace("\\040", " ")
            if os.path.isdir(mp):
                out[mp] = f[2]
    return sorted(out.items(), key=lambda x: -len(x[0]))


def is_root(d):
    try:
        names = set(os.listdir(d))
    except OSError:
        return False
    return bool(names & MARKS) and bool(names & PARTS)


def complete_model(d):
    try:
        names = set(os.listdir(d))
        if "config.json" not in names:
            return False
        if "model.safetensors.index.json" in names:
            need = set(json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"].values())
        else:
            need = {"model.safetensors"}
        return all(f in names and os.path.getsize(os.path.join(d, f)) > 1000000 for f in need)
    except (OSError, ValueError, KeyError):
        return False


def describe(r):
    def ls(p):
        try:
            return sorted(os.listdir(p))
        except OSError:
            return []
    row = {"path": r, "models": [m for m in ls(os.path.join(r, "models")) if complete_model(os.path.join(r, "models", m))],
           "venvs": [v for v in ("venv", "venv-fa", "venv-vl") if os.path.isfile(os.path.join(r, v, "bin", "python")) or os.path.islink(os.path.join(r, v, "bin", "python"))],
           "data": len([f for f in ls(os.path.join(r, "data")) if f.endswith(".parquet")])}
    ck = [os.path.join(r, "adhoc_models", "result", "runs"), os.path.join(os.path.dirname(r), "opd-ckpt", "adhoc_models", "result", "runs"),
          os.path.join(os.path.dirname(r), "ckpt", "adhoc_models", "result", "runs")]
    try:
        m = re.search(r"^(?:export )?CKPT_ROOT=(\S+)", open(os.path.join(r, "paths.env")).read(), re.M)
        if m:
            ck.append(os.path.join(m.group(1).strip("'\""), "adhoc_models", "result", "runs"))
    except OSError:
        pass
    w = set()
    for c in ck:
        for arm in ls(c):
            for s in ls(os.path.join(c, arm)):
                if s.startswith("global_step_"):
                    w.add(arm + "/" + s)
    row["weights"] = len(w)
    try:
        row["changed"] = time.strftime("%Y-%m-%d", time.gmtime(max(os.stat(os.path.join(r, x)).st_mtime for x in ["."] + [n for n in ("runs", "logs", "control") if os.path.exists(os.path.join(r, n))])))
    except OSError:
        row["changed"] = None
    return row


def search(start, depth, budget, deadline, found, skip_roots):
    """Breadth first below `start`: installs of ours go to `found`; returns the directories it listed (for the budget)."""
    level, used = [start], 0
    for _ in range(depth + 1):
        nxt = []
        for d in level:
            if used >= budget or time.time() > deadline:
                return used, True
            rd = os.path.realpath(d)
            if rd in skip_roots or rd in found:
                continue
            if is_root(d):
                found[rd] = None
                continue
            if is_root(os.path.join(d, "run")):
                found[os.path.realpath(os.path.join(d, "run"))] = None
                continue
            used += 1
            try:
                with os.scandir(d) as it:
                    for n, e in enumerate(it):
                        if n >= MAX_ENTRIES or (n % 500 == 499 and time.time() > deadline):   # a directory of a million files is not ours
                            break
                        if e.name.startswith(".") or e.name in SKIP:
                            continue
                        try:
                            if e.is_dir(follow_symlinks=False):
                                nxt.append(e.path)
                        except OSError:
                            pass
            except OSError:
                continue
        # directories whose names look like ours first: a budget that runs out has looked at them
        level = sorted(nxt, key=lambda p: (not os.path.basename(p).startswith(("opd", "run")), p))
    return used, False


def gpu_pids():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        return sorted({int(x) for x in out.split() if x.isdigit()})
    except (OSError, subprocess.TimeoutExpired):
        return []


def hint_paths(env, pids):
    """Absolute paths this machine's own work points at (not ours: the owner's code directory is where an earlier install is likely)."""
    ps = []
    for k in ("OLD_BASE_CANDS", "FROM_OPD_ROOT", "FETCH_LOCAL_ROOTS"):
        ps += [p for p in env.get(k, "").split(":") if p]
    for f in ("gpu_fill_secret.cmd", "gpu_fill_secret.seen"):
        try:
            ps += re.findall(r"(?<![\w.=-])/[A-Za-z0-9_./-]{3,}", open(os.path.join(ROOT, "control", f)).read())
        except OSError:
            pass
    for p in pids:
        try:
            ps.append(os.readlink(f"/proc/{p}/cwd"))
            ps += [a for a in open(f"/proc/{p}/cmdline", "rb").read().decode("utf-8", "replace").split("\0") if a.startswith("/")]
        except OSError:
            pass
    ps.append(env.get("HOME", ""))
    return [p for p in ps if p.startswith("/")]


def hint_dirs(paths, mts):
    """The directories to search: each hint's ancestors one to three levels below its mount, and the same relative directory on
    every other real mount."""
    out = []
    for p in paths:
        p = os.path.normpath(p)
        mp = next((m for m, _ in mts if p == m or p.startswith(m + "/")), None)
        if not mp:
            continue
        rel = [x for x in os.path.relpath(p, mp).split("/") if x not in (".", "")]
        for n in (1, 2, 3):
            if len(rel) >= n:
                for m, _ in mts:
                    d = os.path.join(m, *rel[:n])
                    if os.path.isdir(d) and d not in out:
                        out.append(d)
    return sorted(out, key=lambda d: d.count("/"))   # a user's directory before its sub-directories: the search below covers them


def scan(mts, paths, env, deadline_s=None, budget=None, depth=None):
    t0 = time.time()
    deadline = t0 + float(deadline_s if deadline_s is not None else env.get("HISTORY_SCAN_S", 60))
    budget = int(budget if budget is not None else env.get("HISTORY_MAX_DIRS", 6000))
    depth = int(depth if depth is not None else env.get("HISTORY_DEPTH", 4))
    me = {os.path.realpath(ROOT)} if ROOT else set()
    told = [os.path.realpath(p) for k in ("OLD_BASE_CANDS", "FROM_OPD_ROOT", "FETCH_LOCAL_ROOTS") for p in env.get(k, "").split(":") if p and os.path.isdir(p)]
    found, used, cut = {}, 0, False
    hd = hint_dirs(paths, mts)
    starts = hd or [m for m, t in mts if t.startswith(NET_FS)]
    for s in starts:
        u, c = search(s, depth, budget - used, deadline, found, me)
        used += u
        cut = cut or c
        if used >= budget or time.time() > deadline:
            cut = True
            break
    rows = sorted((describe(r) for r in found if r not in me), key=lambda x: x["changed"] or "", reverse=True)
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "roots": rows, "told": [t for t in told if t not in me], "hints": len(hd),
            "dirs_listed": used, "truncated": cut, "seconds": round(time.time() - t0, 1)}


def summary(rep):
    rs = rep["roots"]
    return {"roots": len(rs), "models": sorted({m for r in rs for m in r["models"]}), "venvs": sorted({v for r in rs for v in r["venvs"]}),
            "weights": sum(r["weights"] for r in rs), "data": max([r["data"] for r in rs] or [0]), "hints": rep["hints"], "truncated": rep["truncated"],
            "seconds": rep["seconds"], "paths": [r["path"] for r in rs]}


def write(rep):
    c = os.path.join(ROOT, "control")
    os.makedirs(c, exist_ok=True)
    cands = []
    for p in rep["told"] + [r["path"] for r in rep["roots"]]:
        if p not in cands and re.fullmatch(r"/[A-Za-z0-9_./@+-]+", p):
            cands.append(p)
    for name, text in (("history_secret.json", json.dumps(rep, indent=1)), ("history_secret.env", f"export OLD_BASE_CANDS={':'.join(cands)}\n" if cands else "")):
        p = os.path.join(c, name)
        fd = os.open(p + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(p + ".tmp", p)


def run(env=None, pids=None):
    env = dict(os.environ) if env is None else env
    try:
        mts = mounts(open("/proc/mounts").read())
    except OSError:
        mts = []
    rep = scan(mts, hint_paths(env, gpu_pids() if pids is None else pids), env)
    write(rep)
    return rep


def main(argv):
    if not ROOT:
        sys.exit("OPD_ROOT is not set")
    p = os.path.join(ROOT, "control", "history_secret.json")
    if "--if-missing" in argv and os.path.exists(p):
        rep = json.load(open(p))
    else:
        rep = run()
    s = summary(rep)
    if "--print" in argv:
        print(json.dumps(rep, indent=1))
    print(f"HISTORY {s['roots']} earlier install(s): models {s['models']}, venvs {s['venvs']}, {s['weights']} fetched weight(s), {s['data']} data file(s); "
          f"{s['hints']} hint(s), {rep['dirs_listed']} directories in {s['seconds']} s{', cut short' if s['truncated'] else ''}")


if __name__ == "__main__":
    main(sys.argv[1:])
