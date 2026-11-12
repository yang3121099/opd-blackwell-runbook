"""[z155] Three-tier checkpoints (lab WO-011; Fable's A1 design with two changes; Runming 11-12): the training writes to a LOCAL disk
($CKPT_LOCAL/<task>/ckpt), scripts/ckpt_mirror.sh copies each new step to the SHARED disk ($CKPT_ROOT/<task>/ckpt, what $OPD_ROOT/runs/<task>/ckpt
links to), and HF holds a resume state only on demand (push / pull). Stdlib only; every rule here is unit-tested (scripts/test_ckpt_lib.py).

Rules
- manifest.json in each step dir (the resume files: actor/huggingface* exports are copied but not covered): per-file size; sha256 of files <= BIG (1 GiB), a sampled sha256 for larger shards (the first and last
  64 MiB and 4 seeded random 64 MiB windows, offsets recorded); run_id = config.yaml's `started:` + sha256[:12] of the config text up to
  that line. A destination is re-checked against the manifest before it counts.
- mirror: when the local marker names a step newer than the shared one, copy it to global_step_N.part, verify, rename, then write the shared
  marker (atomically) - the shared marker only ever names a complete verified step. A copy in flight is never abandoned (a killed copy is
  finished on the next pass while the local step exists); after it the mirror jumps to the newest local step, skipping those between, and logs
  MIRROR_LAG <task> local N shared M behind K whenever it is behind. Shared keeps the newest 2 verified steps (the older deleted only after the
  new one passed); local keeps the newest 2 (verl's max_actor_ckpt_to_keep=2; this pruner is the backstop, never touching the step in flight).
- resume: only steps of the current run id; the highest verified step, local > shared > HF-pulled on equal steps; a local step without a
  manifest counts when the local dir's .run_id matches and the local marker covers it (verl writes the marker after a complete save). Another
  run id is ignored with one line; a local dir of another run is renamed aside (never deleted).
- space guard: a training does not start while the local disk has < 3 x the expected checkpoint size free (nothing is deleted).
- automatic interval (WO-011 amendment, Fable 19:4x): once per run id the mirror reads the train log's first checkpoint write
  (verl's timing_s/save_checkpoint) against the median timing_s/step of the steps without a save so far, appends
  CKPT_COST <task> save S s step T s ratio R -> save_freq n to the train log and records n in $OPD_ROOT/control/ckpt_interval.json;
  n = the smallest n >= 2 with S / (n x T) <= 0.16 (Fable A1 20:2x: 115 / 250 -> 3). run_e12.sh takes it at the next start / resume (verl's
  save_freq is fixed per process); only the final step is guaranteed (verl saves it), the intermediate upload steps are not forced (the
  registry uses the nearest checkpoint step) unless the arm sets CKPT_ALIGN=1: then n is raised to the first value dividing each of them.
  SAVE in control.env wins.
- final flush (z153's race, Fable's 8th node test): run_e12.sh calls `flush <task>` after the trainer exits and before finished: / .done,
  so the final step is on the shared disk (where upload_sync and run_queue_eval LOCAL_FIRST look) before either sees the run as finished;
  every mirror pass and the flush hold control/ckpt_mirror_pass.lock (one copier per step).
CLI: python ckpt_lib.py mirror | flush <task> | prepare <task> <config.yaml> | space <task> <start model> | push <task> | pull <task> [step]
     | interval <task> <steps> <upload steps, comma-separated>   (prints the save_freq to use, or nothing; CKPT_ALIGN=1 aligns)
"""
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time

BIG = 1 << 30
WIN = 64 << 20
KEEP = 2
MARKER = "latest_checkpointed_iteration.txt"


def log(*a):
    print(" ".join(str(x) for x in a), flush=True)


# ------------------------------------------------------------------------------------------------ identity and manifests
def run_id(config_path):
    """`started:` value + sha256[:12] of config.yaml up to and including that line (a resume appends `resumed:` after it: same id)."""
    text = open(config_path).read()
    m = re.search(r"^started: *(\S+) *$", text, re.M)
    if not m:
        return None
    return m.group(1) + "-" + hashlib.sha256(text[:m.end()].encode()).hexdigest()[:12]


def _files(d, resume_only=False):
    """Every file of a step dir (what is copied); resume_only: without the HF exports (actor/huggingface*), which upload_sync may thin
    out on the shared copy after the upload - the manifest covers what a resume needs."""
    out = []
    for root, _, fs in os.walk(d):
        for f in fs:
            rel = os.path.relpath(os.path.join(root, f), d)
            if rel.startswith("manifest") or rel.endswith(".cp") or (resume_only and rel.startswith(os.path.join("actor", "huggingface"))):
                continue
            out.append(rel)
    return sorted(out)


def _sha(path, windows=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if windows is None:
            for b in iter(lambda: f.read(8 << 20), b""):
                h.update(b)
        else:
            for off, ln in windows:
                f.seek(off); h.update(f.read(ln))
    return h.hexdigest()


def _windows(size, rel, big, win):
    rng = random.Random(f"{rel}:{size}")
    w = [(0, win), (max(0, size - win), min(win, size))]
    w += [(rng.randrange(0, max(1, size - win)), win) for _ in range(4)]
    return w


def manifest(step_dir, rid, step, big=BIG, win=WIN, source="train"):
    files = {}
    for rel in _files(step_dir, resume_only=True):
        p = os.path.join(step_dir, rel); size = os.path.getsize(p)
        if size <= big:
            files[rel] = {"size": size, "sha256": _sha(p)}
        else:
            w = _windows(size, rel, big, win)
            files[rel] = {"size": size, "sampled_sha256": _sha(p, w), "windows": w}
    return {"run_id": rid, "step": int(step), "source": source, "files": files, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def verify(step_dir, man):
    """'' when every file of the manifest is there with its size and hash (sampled where the manifest sampled); else the first problem."""
    for rel, e in man["files"].items():
        p = os.path.join(step_dir, rel)
        if not os.path.isfile(p):
            return f"missing {rel}"
        if os.path.getsize(p) != e["size"]:
            return f"size {rel}"
        got = _sha(p, [tuple(x) for x in e["windows"]]) if "windows" in e else _sha(p)
        if got != e.get("sampled_sha256", e.get("sha256")):
            return f"hash {rel}"
    return ""


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def write_marker(task_ckpt, step):
    tmp = os.path.join(task_ckpt, MARKER + ".tmp")
    with open(tmp, "w") as f:
        f.write(str(step)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, os.path.join(task_ckpt, MARKER))


def marker(task_ckpt):
    try:
        return int(open(os.path.join(task_ckpt, MARKER)).read().strip())
    except (OSError, ValueError):
        return 0


def steps(task_ckpt):
    out = []
    for n in os.listdir(task_ckpt) if os.path.isdir(task_ckpt) else []:
        m = re.fullmatch(r"global_step_(\d+)", n)
        if m and os.path.isdir(os.path.join(task_ckpt, n)):
            out.append(int(m.group(1)))
    return sorted(out)


def read_man(step_dir):
    try:
        return json.load(open(os.path.join(step_dir, "manifest.json")))
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ the mirror (local -> shared)
def _copy_tree(src, dst):
    for rel in _files(src):
        s, d = os.path.join(src, rel), os.path.join(dst, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if not (os.path.isfile(d) and os.path.getsize(d) == os.path.getsize(s)):   # a resumed copy keeps what is already whole
            shutil.copy2(s, d + ".cp"); os.replace(d + ".cp", d)


def prune(task_ckpt, keep=KEEP, protect=()):
    """Delete all but the newest `keep` verified steps (manifest present) and older unverified ones; never a protected step."""
    st = steps(task_ckpt)
    done = [s for s in st if read_man(os.path.join(task_ckpt, f"global_step_{s}"))]
    keepers = set(done[-keep:]) | set(protect)
    for s in st:
        if s not in keepers and (s in done or s < (done[-keep] if len(done) >= keep else -1)):
            shutil.rmtree(os.path.join(task_ckpt, f"global_step_{s}"), ignore_errors=True); log("PRUNED", task_ckpt, s)


def mirror_pass(task, local, shared, rid, big=BIG, win=WIN):
    """One pass for one task. Returns the step now named by the shared marker."""
    if rid is None:
        log("MIRROR_SKIP", task, "no run id (config.yaml without started:)"); return marker(shared)
    os.makedirs(shared, exist_ok=True)
    state = os.path.join(shared, ".mirror_inflight")
    while True:
        lm, sm = marker(local), marker(shared)
        try:
            inflight = int(open(state).read().strip())
        except (OSError, ValueError):
            inflight = None
        if inflight is None and lm <= sm:
            break
        if lm > sm:
            log(f"MIRROR_LAG {task} local {lm} shared {sm} behind {lm - sm}")
        step = inflight if inflight is not None else lm   # [Fable change 1] finish the copy in flight before anything newer
        src = os.path.join(local, f"global_step_{step}")
        part = os.path.join(shared, f"global_step_{step}.part")
        if not os.path.isdir(src):   # the local step is gone (pruned / disk lost): nothing left to finish
            log("MIRROR_ABANDON", task, step, "local step missing"); shutil.rmtree(part, ignore_errors=True); os.remove(state)
            if lm <= sm:
                break
            continue
        open(state, "w").write(str(step))
        man = read_man(src)
        if not man or man.get("run_id") != rid or man.get("step") != step:
            man = manifest(src, rid, step, big, win); write_json(os.path.join(src, "manifest.json"), man)
        _copy_tree(src, part); write_json(os.path.join(part, "manifest.json"), man)
        bad = verify(part, man)
        if bad:
            log("MIRROR_VERIFY_FAIL", task, step, bad); shutil.rmtree(part, ignore_errors=True); return marker(shared)   # retried next pass
        dst = os.path.join(shared, f"global_step_{step}")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.replace(part, dst)
        if step > marker(shared):
            write_marker(shared, step)
        os.remove(state)
        log("MIRRORED", task, step)
        prune(shared, KEEP)
        prune(local, KEEP, protect=(marker(local),))
    return marker(shared)


def mirror_all(local_root, shared_root, runs_root, state=None):
    state = state or os.path.join(os.path.dirname(runs_root), "control", "ckpt_interval.json")
    for task in sorted(os.listdir(local_root)) if os.path.isdir(local_root) else []:
        local = os.path.join(local_root, task, "ckpt")
        if not os.path.isdir(local):
            continue
        cfg = os.path.join(runs_root, task, "config.yaml")
        rid = run_id(cfg) if os.path.exists(cfg) else None
        try:
            mirror_pass(task, local, os.path.join(shared_root, task, "ckpt"), rid)
        except OSError as e:
            log("MIRROR_ERROR", task, type(e).__name__, e)
        try:
            cost_pass(task, os.path.join(runs_root, task, "train.log"), rid, state)
        except (OSError, ValueError) as e:
            log("CKPT_COST_ERROR", task, type(e).__name__, e)


class pass_lock:
    """an exclusive flock on <OPD_ROOT>/control/ckpt_mirror_pass.lock for one mirror pass / flush (blocking)."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        import fcntl
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, "a")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        self.f.close()


def flush(task, local, shared, rid, lock_path, max_s=1800, every=10):
    """mirror this task until the shared marker covers the local one and nothing is in flight; 0 = done, 1 = timeout / no run id."""
    if rid is None:
        log("CKPT_FLUSH_FAILED", task, "no run id"); return 1
    t0 = time.time()
    while True:
        with pass_lock(lock_path):
            sm = mirror_pass(task, local, shared, rid)
        lm = marker(local)
        if sm >= lm and not os.path.exists(os.path.join(shared, ".mirror_inflight")):
            log(f"CKPT_FLUSH_OK {task} local {lm} shared {sm}"); return 0
        if time.time() - t0 > max_s:
            log(f"CKPT_FLUSH_TIMEOUT {task} local {lm} shared {sm} after {int(time.time() - t0)} s"); return 1
        time.sleep(every)


# ------------------------------------------------------------------------------------------------ automatic interval
COST_CAP = 0.16   # Fable A1 20:2x (was 0.15)
_STEP = re.compile(r"(?:^|[ )])step:(\d+) - (.*)$")


def step_metrics(lines):
    """{step: {metric: value}} of verl's console lines `step:N - key:value - ...` (timing_s/* only)."""
    out = {}
    for line in lines:
        m = _STEP.search(line)
        if m:
            out[int(m.group(1))] = {k: float(v) for k, v in re.findall(r"(timing_s/[\w/]+):([-0-9.e]+)", m.group(2))}
    return out


def raw_interval(save_s, step_s, cap=COST_CAP):
    """the smallest n >= 2 with save_s / (n * step_s) <= cap."""
    n = 2
    while save_s > cap * n * step_s:
        n += 1
    return n


def aligned_interval(n, steps, uw):
    """n raised to the first value dividing every intermediate upload step (< steps); None if none <= steps does."""
    mids = [u for u in uw if 0 < u < steps]
    while n <= steps:
        if all(u % n == 0 for u in mids):
            return n
        n += 1
    return None


def measure(lines):
    """(save_s, median step_s, n) from the first save and the steps without one before it, or None while not measurable yet."""
    ms = step_metrics(lines)
    saves = sorted(s for s, d in ms.items() if "timing_s/save_checkpoint" in d)
    if not saves:
        return None
    first = saves[0]
    plain = sorted(d["timing_s/step"] for s, d in ms.items() if s < first and "timing_s/step" in d and "timing_s/save_checkpoint" not in d)
    if not plain:
        plain = sorted(d["timing_s/step"] for s, d in ms.items() if "timing_s/step" in d and "timing_s/save_checkpoint" not in d)
    if not plain:
        return None
    k = len(plain)
    med = plain[k // 2] if k % 2 else (plain[k // 2 - 1] + plain[k // 2]) / 2
    save = ms[first]["timing_s/save_checkpoint"]
    return save, med, raw_interval(save, med)


def cost_pass(task, train_log, rid, state):
    """once per run id: measure, append CKPT_COST to the train log, record the interval in the state file."""
    if not rid or not os.path.exists(train_log):
        return None
    st = json.load(open(state)) if os.path.exists(state) else {}
    if st.get(task, {}).get("run_id") == rid:
        return st[task]
    with open(train_log, errors="replace") as f:
        r = measure(f)
    if r is None:
        return None
    save, step, n = r
    line = f"CKPT_COST {task} save {save:.0f} s step {step:.0f} s ratio {save / step:.2f} -> save_freq {n}"
    with open(train_log, "a") as f:
        f.write(line + "\n")
    log(line)
    st[task] = {"run_id": rid, "save_s": round(save, 1), "step_s": round(step, 1), "ratio": round(save / step, 3), "save_freq": n,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.makedirs(os.path.dirname(state), exist_ok=True)
    write_json(state, st)
    return st[task]


def interval_for(task, steps, uw, state, align=False):
    """the save_freq run_e12.sh should use (the recorded n; aligned to the intermediate upload steps only with align = CKPT_ALIGN=1), or None."""
    st = json.load(open(state)) if os.path.exists(state) else {}
    n = st.get(task, {}).get("save_freq")
    if not n:
        return None
    return aligned_interval(int(n), steps, uw) if align else int(n)


# ------------------------------------------------------------------------------------------------ resume
def candidates(local, shared, rid, why):
    """[(step, rank, dir)] usable for this run: rank 0 local, 1 shared, 2 HF-pulled (a shared step whose manifest says source hf)."""
    out = []
    lrid = None
    try:
        lrid = open(os.path.join(local, ".run_id")).read().strip()
    except OSError:
        pass
    lm = marker(local)
    for s in steps(local):
        d = os.path.join(local, f"global_step_{s}"); man = read_man(d)
        if man:
            if man.get("run_id") != rid:
                why.append(f"local step {s}: run id {man.get('run_id')} is not this run's"); continue
            if verify(d, man):
                why.append(f"local step {s}: {verify(d, man)}"); continue
            out.append((s, 0, d))
        elif lrid == rid and s <= lm:
            out.append((s, 0, d))
    for s in steps(shared):
        d = os.path.join(shared, f"global_step_{s}"); man = read_man(d)
        if not man:
            continue
        if man.get("run_id") != rid:
            why.append(f"shared step {s}: run id {man.get('run_id')} is not this run's"); continue
        bad = verify(d, man)
        if bad:
            why.append(f"shared step {s}: {bad}"); continue
        out.append((s, 2 if man.get("source") == "hf" else 1, d))
    return out


def select_resume(local, shared, rid):
    why = []
    c = candidates(local, shared, rid, why)
    best = max(c, key=lambda x: (x[0], -x[1])) if c else None
    return best, why


def prepare(task, local, shared, rid):
    """Before the trainer starts: a local dir of another run is renamed aside; the best step of this run is made the local resume point."""
    os.makedirs(os.path.dirname(local) or ".", exist_ok=True)
    stamp = os.path.join(local, ".run_id")
    if os.path.isdir(local) and os.path.exists(stamp) and open(stamp).read().strip() != rid:
        old = local + ".old_" + open(stamp).read().strip()[-12:]
        os.replace(local, old); log("CKPT_LOCAL_ASIDE", task, "another run's local dir moved to", os.path.basename(old))
    os.makedirs(local, exist_ok=True)
    if not os.path.exists(stamp):
        open(stamp, "w").write(rid)
    best, why = select_resume(local, shared, rid)
    for w in why:
        log("CKPT_RESUME_SKIP", task, w)
    if not best:
        log("CKPT_RESUME", task, "none (step 0)"); return 0
    step, rank, d = best
    if rank > 0 and os.path.realpath(local) != os.path.realpath(shared):
        dst = os.path.join(local, f"global_step_{step}")
        shutil.rmtree(dst + ".part", ignore_errors=True); _copy_tree(d, dst + ".part"); os.replace(dst + ".part", dst)
    write_marker(local, step)
    log("CKPT_RESUME", task, step, ["local", "shared", "hf"][rank])
    return step


# ------------------------------------------------------------------------------------------------ space guard
def du(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) if os.path.isdir(path) else 0


def expected_bytes(local, shared, start_model, est_gb=None):
    """The newest existing step's size (local or shared); else CKPT_EST_GB; else 10 x the start model's weights (bf16 weights, fp32 actor +
    optimizer state + an HF copy)."""
    for d in (local, shared):
        st = steps(d)
        if st:
            return du(os.path.join(d, f"global_step_{st[-1]}"))
    if est_gb:
        return int(float(est_gb) * (1 << 30))
    return 10 * du(start_model)


def space_ok(local_root, need):
    os.makedirs(local_root, exist_ok=True)
    free = shutil.disk_usage(local_root).free
    return free >= 3 * need, free


# ------------------------------------------------------------------------------------------------ HF on demand
def push(task, shared, repo, api=None):
    """The newest verified shared step -> <repo> resume/<task>/global_step_N/ with a full-sha256 manifest (manifest_full.json)."""
    st = [s for s in steps(shared) if read_man(os.path.join(shared, f"global_step_{s}"))]
    if not st:
        log("CKPT_PUSH", task, "no verified shared step"); return None
    d = os.path.join(shared, f"global_step_{st[-1]}"); man = read_man(d)
    if verify(d, man):
        log("CKPT_PUSH", task, "shared step failed verification:", verify(d, man)); return None
    full = manifest(d, man["run_id"], st[-1], big=float("inf"), source="hf")
    write_json(os.path.join(d, "manifest_full.json"), full)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    api.upload_folder(repo_id=repo, folder_path=d, path_in_repo=f"resume/{task}/global_step_{st[-1]}", commit_message=f"resume {task} s{st[-1]}")
    log("CKPT_PUSH", task, st[-1]); return st[-1]


def place_pulled(task, src, shared):
    """A downloaded resume dir (with manifest_full.json) -> verified, then placed in the shared layout as source hf."""
    full = json.load(open(os.path.join(src, "manifest_full.json")))
    bad = verify(src, full)
    if bad:
        log("CKPT_PULL", task, "verification failed:", bad); return None
    step = full["step"]; dst = os.path.join(shared, f"global_step_{step}")
    if os.path.isdir(dst) and read_man(dst):
        log("CKPT_PULL", task, step, "already in the shared layout"); return step
    part = dst + ".part"
    shutil.rmtree(part, ignore_errors=True); _copy_tree(src, part)
    write_json(os.path.join(part, "manifest.json"), full)
    os.replace(part, dst)
    if step > marker(shared):
        write_marker(shared, step)
    log("CKPT_PULL", task, step); return step


def pull(task, shared, repo, step=None, cache=None):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    if step is None:
        names = [f.path for f in api.list_repo_tree(repo, path_in_repo=f"resume/{task}")]
        st = sorted(int(m.group(1)) for n in names for m in [re.search(r"global_step_(\d+)$", n)] if m)
        if not st:
            log("CKPT_PULL", task, "nothing in the repo"); return None
        step = st[-1]
    root = snapshot_download(repo, allow_patterns=[f"resume/{task}/global_step_{step}/*"], local_dir=cache or os.path.join(shared, ".pull"))
    return place_pulled(task, os.path.join(root, f"resume/{task}/global_step_{step}"), shared)


def main(argv):
    env = os.environ
    local_root = env.get("CKPT_LOCAL", ""); shared_root = env.get("CKPT_ROOT", ""); runs = os.path.join(env.get("OPD_ROOT", ""), "runs")
    cmd = argv[0] if argv else ""
    lock = os.path.join(env.get("OPD_ROOT", ""), "control", "ckpt_mirror_pass.lock")
    if cmd == "mirror":
        with pass_lock(lock):
            return mirror_all(local_root, shared_root, runs) or 0
    if cmd == "flush":
        if not local_root:
            return 0
        task = argv[1]; cfg = os.path.join(runs, task, "config.yaml")
        return flush(task, os.path.join(local_root, task, "ckpt"), os.path.join(shared_root, task, "ckpt"),
                     run_id(cfg) if os.path.exists(cfg) else None, lock, int(env.get("CKPT_FLUSH_MAX_S", "1800")))
    if cmd == "prepare":
        task, cfg = argv[1], argv[2]
        return 0 if prepare(task, os.path.join(local_root, task, "ckpt"), os.path.join(shared_root, task, "ckpt"), run_id(cfg)) is not None else 1
    if cmd == "space":
        task, start = argv[1], argv[2]
        need = expected_bytes(os.path.join(local_root, task, "ckpt"), os.path.join(shared_root, task, "ckpt"), start, env.get("CKPT_EST_GB"))
        ok, free = space_ok(local_root, need)
        print(f"{'CKPT_SPACE_OK' if ok else 'CKPT_SPACE_LOW'} {task}: free {free / 2**30:.1f} GiB, need 3 x {need / 2**30:.1f} GiB")
        return 0 if ok else 1
    if cmd == "interval":
        uw = [int(x) for x in argv[3].split(",") if x.strip().isdigit()] if len(argv) > 3 else []
        n = interval_for(argv[1], int(argv[2]), uw, os.path.join(env.get("OPD_ROOT", ""), "control", "ckpt_interval.json"),
                        align=env.get("CKPT_ALIGN") == "1")
        if n:
            print(n)
        return 0
    if cmd in ("push", "pull"):
        import subprocess
        repo = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote.py"), "repo", "private"],
                              capture_output=True, text=True).stdout.strip()
        if not repo:
            print("no private repo configured"); return 1
        shared = os.path.join(shared_root, argv[1], "ckpt")
        if cmd == "push":
            return 0 if push(argv[1], shared, repo) is not None else 1
        return 0 if pull(argv[1], shared, repo, int(argv[2]) if len(argv) > 2 else None) is not None else 1
    print(__doc__.split("CLI:")[1].strip()); return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
