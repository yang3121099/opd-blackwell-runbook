"""[v2.6 S0] (Runming 10-25 "polish, not patches"; Fable F7 / ops-lead O7: the one structural exception on main during the
v2.6 migration) the recorders behind scripts/dryrun_shims/<tool>. dryrun_json.py runs a 2.5.9 task inside a sandbox whose
PATH starts with dryrun_shims/; every tool there execs this file as `dryrun_shim.py <tool> <argv...>` with the node's real
python3 (DRYRUN_REAL_PY). Each call appends one JSON line to $DRYRUN_REC: {seq, tool, cls, argv, venv, env}.

The python classes follow ops-lead's O7 list. Training / evaluation / upload / fetch entry points are recorded and not run.
Small python that decides a branch (a gate report read, control_rules.check_eval_args) runs for real, on the sandbox's
files (py_forward). control.py / remote.py subcommands are listed one by one in CONTROL / REMOTE. Nothing here touches the
network or a GPU, and nothing writes outside $DRYRUN_SB.

stdlib only (it runs on the node's system python and on the Mac's for the tests)."""
import hashlib
import json
import os
import re
import sys
import time

ENV_PREFIXES = ("OPD_", "VLLM_", "NCCL_", "RAY_", "HYDRA_", "TOKENIZERS_")
ENV_KEYS = ("UPLOAD_WEIGHTS", "KEEP_BF16", "FIXED_MICRO_BSZ", "MAX_TOK_GPU", "TEACHER_OFFLOAD", "CUDA_VISIBLE_DEVICES",
            "PYTHONPATH", "SMOKE", "PROBE_TASK", "VENV", "PROBE_ENV", "SMOKE_STEPS")
# control.py subcommands the 2.5.9 runners call (task.sh, run_arm.sh, run_e12.sh, run_queue_eval.sh, env.sh, dr.sh, probe_*.sh):
# what each one does here. "record" = recorded, exit 0, nothing printed (side effects only on the node: events, reaps);
# "node" = prints the fixed node token (its real output is only used in upload paths, which are normalized anyway).
CONTROL = {"event": "record", "reap": "record", "rcpt": "record", "node": "node"}
# remote.py: backoff = a query (the sandbox has no backoff state: 0); repo = the destination class name (normalized);
# put / pend / putmany = uploads (recorded, never run)
REMOTE = {"backoff": "zero", "repo": "class", "put": "record", "pend": "record", "putmany": "record"}
NODE_TOKEN = "${NODE}"   # never the real node id (control.py node hashes the host name)
# code/train / code/agent jobs a runner may call by bare name (from inside their directory)
DATA_JOBS = {"prep_mmk12.py", "prep_tools_parquet.py", "build_sft_data.py", "gen_teacher_traces.py"}
# a heredoc that inspects or patches the installed verl (run_sft.sh's attention patch): recorded, as a *_patch.py call
PATCH_HEREDOC = re.compile(r"find_spec\(\s*['\"]verl|^\s*import verl|^\s*from verl", re.M)
# a heredoc that imports test modules (the probes' unit-test runners): recorded and reported as passed, as a test_*.py call
UNIT_TEST_HEREDOC = re.compile(r"""import_module\(["']test_|^\s*import test_|^\s*from test_\S+ import|for modname in \(["']test_""", re.M)


def _sb():
    return os.environ.get("DRYRUN_SB", "")


def record(tool, cls, argv, **extra):
    env = {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES) or k in ENV_KEYS}
    venv = os.path.basename(os.environ.get("VIRTUAL_ENV", "")) or None
    rec = {"tool": tool, "cls": cls, "argv": list(argv), "venv": venv, "env": env}
    rec.update(extra)
    path = os.environ.get("DRYRUN_REC")
    if not path:
        return
    with open(path, "a") as f:   # one line per call; the task's processes run one at a time here (no setsid / nohup)
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def _clock():
    """the pinned clock: DRYRUN_T0 plus one second per call (deterministic, and a wait loop on `date +%s` still ends)."""
    p = os.path.join(_sb(), "clock")
    n = 0
    try:
        n = int(open(p).read().strip() or 0)
    except (OSError, ValueError):
        pass
    try:
        open(p, "w").write(str(n + 1))
    except OSError:
        pass
    return int(os.environ.get("DRYRUN_T0", "1767225600")) + n


def _in_sb(path):
    sb = os.path.realpath(_sb())
    return bool(sb) and os.path.realpath(path).startswith(sb + os.sep)


# ------------------------------------------------------------------------------------------------------------- python
def _opt(argv, name):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def classify_python(argv):
    """argv after the interpreter -> (class, detail). The first matching row wins (v26_impl_plan §2.4, ops-lead O7)."""
    a = [x for x in argv if x not in ("-I", "-u", "-B", "-E", "-s")]
    if not a:
        return "unknown", None
    if a[0] == "-m":
        mod = a[1] if len(a) > 1 else ""
        if mod.startswith("verl.trainer.") or mod in ("torch.distributed.run", "torch.distributed.launch"):
            return "train_cmd", mod
        if mod == "pytest":
            return "unit_test", mod
        if mod in ("pip", "ensurepip", "venv"):   # the venv builds: never run in the sandbox (they would fetch and install)
            return "pip", mod
        return "unknown", mod
    if a[0] == "-":   # a heredoc (`python3 - <<PY`): the probes' checks and report writers, run on the sandbox's files
        return "py_forward", "stdin"
    if a[0] == "-c":
        code = a[1] if len(a) > 1 else ""
        if re.search(r"\bimport flash_attn\b", code):
            return "probe_fa", None
        if "snapshot_download" in code:
            return "hf_cache", None
        m = re.search(r"find_spec\(['\"]([A-Za-z0-9_.]+)['\"]\)\.origin", code)
        if m:   # where an installed module's file lives (the probes patch / test it): a stand-in file in the sandbox venv
            return "find_spec", m.group(1)
        return "py_forward", "-c"
    script = a[0]
    base = os.path.basename(script)
    if base == "control.py":
        sub = a[1] if len(a) > 1 else ""
        return ("control", sub) if sub in CONTROL else ("control_unlisted", sub)
    if base == "remote.py":
        sub = a[1] if len(a) > 1 else ""
        return ("remote", sub) if sub in REMOTE else ("remote_unlisted", sub)
    if re.search(r"(^|/)code/patches/[a-z0-9_]+_patch\.py$", script):
        return "patch", base
    if base.startswith("test_") and base.endswith(".py"):
        return "unit_test", base
    if base in ("fetch_model.py", "par_fetch.py", "wheel_fetch.py"):
        return "fetch", base
    if re.search(r"(^|/)code/eval/[a-z0-9_]+\.py$", script):
        return "gen_eval", base
    if re.search(r"(^|/)code/(analysis|leakage)/[a-z0-9_]+\.py$", script):
        return "analysis", base
    if re.search(r"(^|/)code/weights/[a-z0-9_]+\.py$", script) or base == "to_bf16.py":   # weights built / converted on the node only
        return "weights_op", base
    if re.search(r"(^|/)code/(train|agent)/[a-z0-9_]+\.py$", script) or base in DATA_JOBS:   # data preparation, teacher traces
        return "data_job", base
    if re.search(r"(^|/)scripts/[a-z0-9_]+\.py$", script):
        return "py_forward", base
    return "unknown", base


def _gen_eval_out(argv):
    """gen_eval / gen_eval_vl / forced_stop / ifeval_grade: write the files the runner looks for next (an empty jsonl and a
    metrics file) under --out, inside the sandbox only."""
    out, tag, bench = _opt(argv, "--out"), _opt(argv, "--tag"), _opt(argv, "--bench")
    if not (out and tag and bench and _in_sb(out)):
        return
    t = _opt(argv, "--temperature") or "0.6"
    try:
        t = f"{float(t):.1f}"
    except ValueError:
        pass
    mode = "think" if "--think" in argv else "no_think"
    os.makedirs(out, exist_ok=True)
    stem = os.path.join(out, f"{tag}_{bench}_{mode}_t{t}")
    open(stem + ".jsonl", "a").close()
    if not os.path.exists(stem + ".metrics.json"):
        json.dump({"avg_acc": 0.0, "dryrun": True}, open(stem + ".metrics.json", "w"))


def _fetch_dest(argv):
    """fetch_model.py <repo> <remote path> <dest dir> ...: the destination's weights and .fetch_ok (as a real fetch leaves)."""
    pos = [x for x in argv[1:] if not x.startswith("--")]
    if len(pos) < 3:
        return None
    dest = os.path.join(pos[2], pos[1]) if not pos[1].startswith("/") else pos[2]
    if not _in_sb(dest):
        return None
    os.makedirs(dest, exist_ok=True)
    open(os.path.join(dest, "model.safetensors"), "a").close()
    open(os.path.join(dest, ".fetch_ok"), "a").close()
    return dest


def run_python(tool, argv):
    cls, detail = classify_python(argv)
    if cls == "train_cmd":
        record(tool, cls, argv, module=detail)
        steps = 0
        for x in argv:
            m = re.match(r"trainer\.total_training_steps=(\d+)$", x)
            if m:
                steps = int(m.group(1))
        for k in range(1, steps + 1):   # the runner tees this into train.log and derives completed_steps / .done from it
            print(f"step:{k} - timing_s/step:1.0 - response_length/mean:100.0", flush=True)
        return 0
    if cls == "probe_fa":
        venv = os.path.basename(os.environ.get("VIRTUAL_ENV", "")) or "none"
        ans = dict(kv.split(":", 1) for kv in os.environ.get("DRYRUN_FA", "").split(",") if ":" in kv)
        record(tool, cls, argv, answer=ans.get(venv, "0"))
        return 0 if ans.get(venv, "0") == "1" else 1
    if cls == "hf_cache":
        repo = argv[-1] if argv else ""
        record(tool, cls, argv)
        print(os.path.join(os.environ.get("HF_HOME", _sb()), repo.replace("/", "__")))
        return 0
    if cls == "control":
        record(tool, cls, argv, sub=detail)
        if CONTROL[detail] == "node":
            print(NODE_TOKEN)
        return 0
    if cls == "remote":
        record(tool, cls, argv, sub=detail)
        mode = REMOTE[detail]
        if mode == "zero":
            print(0)
        elif mode == "class":
            print(argv[-1] if argv and argv[-1] in ("public", "private") else "private")
        return 0
    if cls == "fetch":
        dest = _fetch_dest(argv)
        record(tool, cls, argv, dest=dest)
        return 0
    if cls == "gen_eval":
        record(tool, cls, argv)
        _gen_eval_out(argv)
        return 0
    if cls == "weights_op":   # recorded, not run; its output gets stand-in weights so the runner's next step reads a model
        out = _opt(argv, "--out") or (argv[-1] if len(argv) > 2 and not argv[-1].startswith("-") else None)
        if out and _in_sb(out):
            os.makedirs(out, exist_ok=True)
            open(os.path.join(out, "model.safetensors"), "a").close()
        record(tool, cls, argv)
        return 0
    if cls in ("patch", "unit_test", "analysis", "data_job"):
        record(tool, cls, argv)
        if detail == "build_sft_data.py":   # run_sft.sh tees its stdout into sft_data.json and reads the step plan from it
            print(json.dumps({"steps_for_budget": 100, "steps_per_epoch": 50, "total_epochs_arg": 2, "dryrun": True}))
        return 0
    if cls == "pip":
        record(tool, cls, argv)
        if "--version" in argv:
            print("pip 24.0 (dryrun)")
        return 0
    if cls == "find_spec":
        venv = os.environ.get("VIRTUAL_ENV") or os.path.join(_sb(), "opd", "venv")
        path = os.path.join(venv, "site", *detail.split(".")) + ".py"
        if _in_sb(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "a").close()
        record(tool, cls, argv, module=detail)
        print(path)
        return 0
    if cls == "py_forward":
        real = os.environ.get("DRYRUN_REAL_PY") or sys.executable
        rest = [x for x in argv if x not in ("-I", "-u", "-B", "-E", "-s")]
        if detail == "stdin":   # a heredoc: read it to tell a unit-test runner (recorded, as test_*.py) from a check (run)
            code = sys.stdin.read()
            sha = hashlib.sha256(code.encode()).hexdigest()[:16]
            if UNIT_TEST_HEREDOC.search(code):
                record(tool, "unit_test", argv, heredoc_sha=sha)
                print("1 passed, 0 failed (dryrun: recorded, not run)")
                return 0
            if PATCH_HEREDOC.search(code):
                record(tool, "patch", argv, heredoc_sha=sha)
                return 0
            record(tool, cls, argv, heredoc_sha=sha)
            sys.stdout.flush()
            os.execv(real, [real, "-c", code] + rest[1:])
        record(tool, cls, argv)
        sys.stdout.flush()
        os.execv(real, [real] + list(argv))
    record(tool, cls, argv, detail=detail)   # unknown / control_unlisted / remote_unlisted: findings, exit 0
    return 0


# -------------------------------------------------------------------------------------------------------- other tools
GPU_NAME = "NVIDIA B300 SXM6 AC"


def run_nvidia_smi(argv):
    record("nvidia-smi", "gpu", argv)
    q = _opt(argv, "--query-gpu")
    if not q:
        return 0   # --query-compute-apps etc.: nothing running
    fmt = _opt(argv, "--format") or ""
    idx = _opt(argv, "-i")
    total = os.environ.get("DRYRUN_GPU_MIB", "275040")
    gpus = [int(i) for i in idx.split(",")] if idx else list(range(int(os.environ.get("DRYRUN_NGPU", "8"))))
    vals = {"name": GPU_NAME, "memory.used": "0", "memory.total": total, "utilization.gpu": "0", "compute_cap": "10.3",
            "memory.free": total}
    for g in gpus:
        row = []
        for f in q.split(","):
            v = str(g) if f == "index" else vals.get(f, "0")
            if "nounits" not in fmt and f.startswith("memory."):
                v += " MiB"
            row.append(v)
        print(", ".join(row))
    return 0


def run_date(argv):
    record("date", "clock", argv)
    t = _clock()
    utc = True   # TZ=UTC in the sandbox
    fmt = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-u", "--utc"):
            utc = True
        elif a == "-d" and i + 1 < len(argv) and argv[i + 1].startswith("@"):
            t = int(float(argv[i + 1][1:])); i += 1
        elif a.startswith("+"):
            fmt = a[1:]
        i += 1
    tm = time.gmtime(t) if utc else time.localtime(t)
    if fmt is None:
        print(time.strftime("%a %b %d %H:%M:%S UTC %Y", tm))
        return 0
    fmt = fmt.replace("%s", str(t)).replace("%N", "000000000")
    print(time.strftime(fmt, tm))
    return 0


def run_df(argv):
    """POSIX df for the sandbox: one filesystem, mounted at the sandbox root, with DRYRUN_DF_FREE_KB free (fixed)."""
    record("df", "disk", argv)
    free = int(os.environ.get("DRYRUN_DF_FREE_KB", str(4 * 1024 ** 3)))   # 4 TB
    print("Filesystem 1024-blocks Used Available Capacity Mounted on")
    print(f"sandbox {free * 2} {free} {free} 50% {os.path.realpath(_sb()) or '/'}")
    return 0


def run_du(argv):
    record("du", "disk", argv)
    paths = [a for a in argv if not a.startswith("-")] or ["."]
    for p in paths:
        print(f"0\t{p}")
    return 0


def main(argv):
    if len(argv) < 2:
        return 2
    tool, args = argv[1], argv[2:]
    if tool in ("python", "python3"):
        return run_python(tool, args)
    if tool == "torchrun":   # run_sft.sh: torchrun ... -m verl.trainer.fsdp_sft_trainer ...: a training command, recorded
        mod = args[args.index("-m") + 1] if "-m" in args and args.index("-m") + 1 < len(args) else None
        record(tool, "train_cmd", args, module=mod)
        d, n = _opt(args, "trainer.default_local_dir"), _opt(args, "trainer.total_training_steps")
        if d and n and _in_sb(d):   # the final checkpoint run_sft.sh checks for (empty stand-in weights, sandbox only)
            h = os.path.join(d, f"global_step_{n}", "huggingface")
            os.makedirs(h, exist_ok=True)
            open(os.path.join(h, "model.safetensors"), "a").close()
        return 0
    if tool in ("setsid", "nohup"):
        record(tool, "bg", args)   # not run: the upload_sync / dr_sync / heartbeat launches are recorded with their env
        return 0
    if tool == "pgrep":
        record(tool, "proc", args)
        return 1   # nothing running
    if tool == "nvidia-smi":
        return run_nvidia_smi(args)
    if tool == "sleep":
        record(tool, "clock", args)
        return 0
    if tool == "date":
        return run_date(args)
    if tool == "df":
        return run_df(args)
    if tool == "du":
        return run_du(args)
    if tool == "rsync":
        record(tool, "dr_sync", args)
        return 0
    if tool in ("huggingface-cli", "curl", "wget", "git", "aria2c"):
        record(tool, "net", args)   # a network attempt is a corpus finding; nothing leaves the sandbox
        return 1
    record(tool, "unknown", args)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
