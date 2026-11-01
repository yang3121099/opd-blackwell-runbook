"""[v2.5.9z123] mount_probe.py <out.json>: what persistent storage this machine sees beyond our own base, for the control command
`mounts` (Runming 11-02 00:0x Beijing via Fable: read-only - look, never write or delete). Standard library only.

  1. The real file systems of /proc/mounts (pseudo ones skipped): mount point, type, total / free GB (statvfs), and whether this
     user may read / write there (os.access; nothing is written, no test file).
  2. Beyond the mounts that hold $OPD_ROOT / $CKPT_ROOT (our current base) and the root file system, plus node.env's OLD_BASE_CANDS:
     a bounded walk (depth MP_DEPTH 5, MP_ENTRIES 300000 entries, MP_SECONDS 240 s in all) for directories named runs/ whose
     children look like our task names, then per task directory there: its saved steps (global_step_N under the task and under
     ckpt/, with safetensors under actor/huggingface*: GB) and its raw generation files (*_no_think_t*.jsonl, .part: count, GB).

The report holds paths, so control.py puts it to the PRIVATE repository only (logs/<node>/requests/<seq>/); the receipt carries
counts. Prints one summary line without paths."""
import json
import os
import re
import sys
import time

PSEUDO = {"proc", "sysfs", "tmpfs", "devtmpfs", "devpts", "mqueue", "cgroup", "cgroup2", "securityfs", "debugfs", "tracefs",
          "fusectl", "configfs", "pstore", "bpf", "hugetlbfs", "autofs", "nsfs", "binfmt_misc", "rpc_pipefs", "efivarfs", "ramfs",
          "selinuxfs", "squashfs", "nfsd", "fuse.gvfsd-fuse", "fuse.portal"}
TASK = re.compile(r"^(E\d{2}[a-e]?-|EVAL-|PROBE-|SFT-|DIAG-)")
STEP = re.compile(r"^global_step_(\d+)$")
RAW = re.compile(r"_no_think_t[0-9.]+\.jsonl(\.part)?$")


def real_mounts(text):
    """[(mount point, type)] of /proc/mounts' real file systems, each mount point once (the last entry wins), octal escapes undone."""
    out = {}
    for line in text.splitlines():
        f = line.split()
        if len(f) < 3 or f[2] in PSEUDO or f[1].startswith(("/proc", "/sys", "/dev", "/run")):
            continue
        out[f[1].encode().decode("unicode_escape")] = f[2]
    return sorted(out.items())


def fs_info(path):
    try:
        st = os.statvfs(path)
        return {"total_gb": round(st.f_blocks * st.f_frsize / 1e9, 1), "free_gb": round(st.f_bavail * st.f_frsize / 1e9, 1),
                "readable": os.access(path, os.R_OK | os.X_OK), "writable": os.access(path, os.W_OK)}
    except OSError as e:
        return {"error": type(e).__name__}


def holder(mounts, path):
    """The mount point a path lives on (the longest prefix)."""
    if not path:
        return None
    p = os.path.realpath(path)
    best = None
    for m, _ in mounts:
        if (p == m or p.startswith(m.rstrip("/") + "/")) and (best is None or len(m) > len(best)):
            best = m
    return best


def gb_of(d):
    n = 0
    for dp, _, fs in os.walk(d):
        for f in fs:
            try:
                n += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return round(n / 1e9, 2)


def task_steps(td):
    """[(step, place, GB)] of a task directory: global_step_N (directly or under ckpt/) whose actor/huggingface* holds safetensors."""
    out = []
    for place in ("", "ckpt"):
        base = os.path.join(td, place) if place else td
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for n in names:
            m = STEP.match(n)
            if not m:
                continue
            act = os.path.join(base, n, "actor")
            for hf in ("huggingface", "huggingface_bf16"):
                h = os.path.join(act, hf)
                try:
                    if any(f.endswith(".safetensors") for f in os.listdir(h)):
                        out.append({"step": int(m.group(1)), "place": (place + "/" if place else "") + hf, "gb": gb_of(h)})
                except OSError:
                    pass
    return out


def scan_runs(runs_dir):
    """Our task directories below one runs/: steps with weights, raw generation files (count, GB)."""
    tasks = {}
    for t in sorted(os.listdir(runs_dir)):
        td = os.path.join(runs_dir, t)
        if not TASK.match(t) or not os.path.isdir(td):
            continue
        raw_n = raw_b = 0
        try:
            for f in os.listdir(td):
                if RAW.search(f):
                    raw_n += 1
                    try:
                        raw_b += os.lstat(os.path.join(td, f)).st_size
                    except OSError:
                        pass
        except OSError:
            pass
        tasks[t] = {"steps": task_steps(td), "raw_files": raw_n, "raw_gb": round(raw_b / 1e9, 2)}
    return tasks


def find_runs(root, depth, budget, deadline):
    """runs/ directories (with at least one task-like child) at most `depth` levels below root; budget = [entries left]."""
    found, stack = [], [(root, 0)]
    while stack and budget[0] > 0 and time.time() < deadline:
        d, k = stack.pop()
        try:
            with os.scandir(d) as it:
                kids = [e for e in it if e.is_dir(follow_symlinks=False)]
        except OSError:
            continue
        budget[0] -= len(kids) + 1
        for e in kids:
            if e.name == "runs":
                try:
                    if any(TASK.match(x) for x in os.listdir(e.path)):
                        found.append(e.path); continue
                except OSError:
                    pass
            if k + 1 < depth and not e.name.startswith((".", "__pycache__")) and e.name not in ("venv", "venv-fa", "site-packages", "node_modules"):
                stack.append((e.path, k + 1))
    return found


def probe(mounts_text, env):
    t0 = time.time(); deadline = t0 + float(env.get("MP_SECONDS", "240"))
    mounts = real_mounts(mounts_text)
    ours = {holder(mounts, env.get("OPD_ROOT")), holder(mounts, env.get("CKPT_ROOT")), "/"}
    rep = {"mounts": [dict({"mount": m, "type": t, "ours": m in ours}, **fs_info(m)) for m, t in mounts], "scanned": [], "runs": {},
           "truncated": False}
    cands = [m for m, _ in mounts if m not in ours] + [c for c in env.get("OLD_BASE_CANDS", "").split(":") if c]
    budget = [int(env.get("MP_ENTRIES", "300000"))]
    for c in dict.fromkeys(cands):
        if not os.path.isdir(c) or time.time() >= deadline:
            continue
        rep["scanned"].append(c)
        for r in find_runs(c, int(env.get("MP_DEPTH", "5")), budget, deadline):
            rep["runs"][r] = scan_runs(r)
    rep["truncated"] = budget[0] <= 0 or time.time() >= deadline
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def summary(rep):
    tasks = [t for r in rep["runs"].values() for t in r.values()]
    steps = [s for t in tasks for s in t["steps"]]
    return (f"{len(rep['mounts'])} real mounts, {sum(1 for m in rep['mounts'] if not m['ours'])} beyond our base; "
            f"{len(rep['scanned'])} scanned; {len(rep['runs'])} runs/ dirs, {len(tasks)} task dirs; "
            f"{len(steps)} saved steps ({sum(s['gb'] for s in steps):.1f} GB); {sum(t['raw_files'] for t in tasks)} raw generation files "
            f"({sum(t['raw_gb'] for t in tasks):.1f} GB){'; walk truncated' if rep['truncated'] else ''}; {rep['seconds']} s")


def main():
    rep = probe(open("/proc/mounts").read(), dict(os.environ))
    json.dump(rep, open(sys.argv[1], "w"), indent=1)
    print(summary(rep))


if __name__ == "__main__":
    main()
