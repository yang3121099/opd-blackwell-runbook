"""[v2.6 S0] the golden-corpus recorder (v26_impl_plan §2; Fable F7, ops-lead O7: the one structural exception on main during
the v2.6 migration, new files only plus the name CORPUS-RECORD).

For every task name it runs the real 2.5.9 `scripts/task.sh <gpu> <task>` inside a sandbox (a fresh temporary $OPD_ROOT /
$CKPT_ROOT / $HOME / $HF_HOME / $DR_ROOT under one base directory on the checkpoint disk) whose PATH starts with
scripts/dryrun_shims/. The shims record every training command, patch, upload launch, fetch, evaluation and query, and run
nothing that trains, uploads, fetches or touches a GPU or the network (dryrun_shim.py). A refusal ("<gate> has not passed",
"WAIT_INPUTS ...", "missing start checkpoint <path>") plants exactly what it names and the task runs again (the fixpoint, at
most MAX_ITER times); a refusal no row of PLANTS covers stops the task as a finding (never a guessed plant). What the task did
is collected into one JSON record per task and profile (normalized: dryrun_normalize.py).

Profiles: clean = an empty control.env and synthetic plants; node = a copy of this node's control.env and copies of its real
small inputs (gate reports, label files; <= 1 MB each, read-only; weights never: empty files mirrored at the paths that exist),
so a refusal the node would give is recorded as given.

Usage: python3 dryrun_json.py [--all | <task> ...] [--profile clean|node|both] [--out <dir>] [--base <dir>] [--runbook <dir>]
         [--per-task-sec 60] [--total-sec 7200] [--task-dir <dir>] [--no-upload]
Output: <out>/corpus/<profile>/<task>.json, <out>/index.json, <out>/findings.json, <out>/corpus.tgz; with --task-dir (the
CORPUS-RECORD task) also report.json, .done and one put of <out> (index, findings, the tarball) to the private repository.
stdlib only."""
import argparse
import concurrent.futures
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True   # nothing written under the runbook, __pycache__ included
sys.path.insert(0, HERE)
import dryrun_normalize as N  # noqa: E402

SHIMS = ("python", "python3", "torchrun", "setsid", "nohup", "pgrep", "nvidia-smi", "sleep", "date", "df", "du", "rsync",
         "huggingface-cli", "curl", "wget", "git", "aria2c")
VENVS = ("venv", "venv-fa", "venv-vl", "venv-ifeval")
T0 = 1767225600   # 2026-01-01T00:00:00Z: the sandbox clock's origin
MAX_ITER = 12
SMALL = 1 << 20   # node-profile copies: at most 1 MB a file
CLEAN_FA = {"venv": "0", "venv-fa": "1", "venv-vl": "0", "venv-ifeval": "0"}   # the clean profile's flash-attn answers
MODELS_ENV_KEYS = {"STUDENT_06B": "Qwen3-0.6B", "STUDENT_17B": "Qwen3-1.7B", "STUDENT_4B": "Qwen3-4B", "TEACHER": "Qwen3-8B"}

# refusals -> what to plant (v26_impl_plan §2.5). Each pattern is tried in order; {task} is the task name.
REFUSALS = [
    ("gate", r"^(?P<gate>[A-Z0-9][A-Za-z0-9-]*) has not passed: (?P<task>\S+) not started$"),
    ("gate_check", r"^(?P<gate>[A-Z0-9][A-Za-z0-9-]*) has not checked (?P<what>.+): (?P<task>\S+) not started$"),
    ("start", r"^missing (?:start )?checkpoint (?P<path>\S+)$"),   # run_arm / run_e12: start; run_eval: the checkpoint read
    ("model_missing", r"^MODEL_MISSING (?P<name>[A-Za-z0-9._-]+)$"),   # run_vlm.sh: a model dir without .fetch_ok
    ("adhoc_fetch", r"^WAIT_INPUTS (?P<task>\S+): start .* \(no local copy, no adhoc_models/result/(?P<rel>\S+)/\.fetch_ok\)$"),
    ("fetch_ok", r"^WAIT_INPUTS (?P<task>\S+): (?P<name>[A-Za-z0-9._-]+) not in \(no \.fetch_ok(?: or config\.json)?\)$"),
    ("node_file", r"^WAIT_INPUTS (?P<task>\S+): (?P<path>runs/\S+) not on this node$"),
    ("never", r"^WAIT_INPUTS (?P<task>\S+): (?P<what>a venv-vl python is running)$"),   # pgrep finds nothing in the sandbox
    ("backoff", r"^WAIT_INPUTS (?P<task>\S+): .* \(the HF 429 backoff lasts\)$"),   # the sandbox's backoff is 0: never here
    ("never", r"^WAIT_INPUTS (?P<task>\S+): (?P<what>venv-\S+ not built) \(\S+\)$"),   # the sandbox's venvs all exist: never here
    ("wait_other", r"^WAIT_INPUTS (?P<task>\S+): (?P<what>.*)$"),
]
# gates whose check reads more than "result": PASS, and the files a gate's arms read beside its report (config.yaml lines);
# clean-profile values (the node profile copies the real files)
GATE_REPORTS = {"PROBE-MIXP": {"result": "PASS", "mix_check": {"share": {"0.6": 0.6, "0.75": 0.75, "0.9": 0.9}}}}
GATE_FILES = {"PROBE-SELFROUTE": {"out/selfroute.json": {"threshold": 0.39, "train": {"agreement": 0.746}}},
              "PROBE-VLM": {"check_b.json": {"suffix": "think"}}}   # run_vlm.sh reads the teacher suffix from it
# inputs a runner reads with no refusal to learn them from (a failed `ls` under pipefail ends it): planted before the first run;
# the node profile mirrors them (an empty file at the same name) when the node has them
INPUT_PLANTS = [
    (r"^E11b-(?!P0$)", [("runs/E11b-P0/e11b_P0_math500_T0.6_math500_no_think_t0.6.jsonl", "")]),   # run_eval.sh: the twr reference
    (r"^E12b-P1$", [("runs/SFT-P1/final_step", "100\n"),   # run_e12.sh: $(cat SFT-P1/final_step) under set -e, then its
                    ("runs/SFT-P1/ckpt/global_step_100/huggingface/model.safetensors", "")]),   # weights (else a silent ls exit)
]
# the names a `<name> not in (no .fetch_ok)` refusal uses for a dataset kept elsewhere
DATA_ALIASES = {"Geometry3K": "data/hf/hiyouga__geometry3k"}
# refusal-shaped lines the fixpoint never needs to act on: per-row notes of a reading that goes on, a probe's own report
# check, and inputs a node must be given by hand (a faithful finding). Test T8 accepts a runner message that matches one.
NOT_TERMINAL = [
    (r"report missing", "a probe's own report check after its run"),
    (r"^PAIR_REF_MISSING ", "a reading's pair without its reference: the reading goes on"),
    (r"^MODEL_MISSING \$TAG: ", "one #eval row without its model: the next row goes on"),
    (r"^FORCED_MODEL_MISSING ", "one forced row without its model: the next row goes on"),
    (r"^TWR_SKIPPED ", "one twr row skipped: the next row goes on"),
    (r"^TOOL_DATA_MISSING ", "E10b-P5tool needs TOOL_DATA set by hand: a faithful finding"),
    (r"^\$\{F:-", "a file name, not a message"),
]
# every other read of $OPD_ROOT/runs/... in run_arm.sh / run_e12.sh, and how the fixpoint covers it (test T8 fails on a new one)
KNOWN_READS = [
    (r"^\S+/ckpt/global_step_\S*/actor/huggingface(_bf16)?$", "start: the 'missing start checkpoint' refusal plants it"),
    (r"^SFT-P1/ckpt/global_step_\$?$", "start: E12b-P1's SFT-P1 chain ends in a 'missing start checkpoint' refusal"),
    (r"^SFT-P1/final_step$", "INPUT_PLANTS: a cat under set -e (E12b-P1); the start chain's refusal follows"),
    (r"^PROBE-DIFF17/labels_17\.jsonl\}?$", "env: exported as OPD_ROUTE_LABELS, read by the training only"),
    (r"^PROBE-SELFROUTE/labels_probe_17\.jsonl\}?$", "node_file: the 'not on this node' refusal plants it"),
    (r"^PROBE-SELFROUTE/out/selfroute\.json$", "GATE_FILES: planted with PROBE-SELFROUTE's report"),
    (r"^PROBE-T32/escalation\.txt$", "absent by default (the default path); deliberately not planted"),
    (r"^E11b-P0/e11b_P0_math500_T0\.6_\S*$", "INPUT_PLANTS: run_eval.sh's twr reference (an ls under pipefail)"),
    (r"^PROBE-VLM/check_b\.json$", "GATE_FILES: planted with PROBE-VLM's report"),
]


# ------------------------------------------------------------------------------------------------------------ task list
def main_case_names(task_sh):
    """task.sh's main case: [(line, [alternatives])] (as the shadow test reads it)."""
    L = open(task_sh).read().split("\n")
    start = next(i for i, l in enumerate(L) if re.match(r'^case "?\$T"? in', l))
    end = next(i for i in range(start + 1, len(L)) if L[i] == "esac")
    return [(i + 1, m.group(1).split("|")) for i in range(start + 1, end) for m in [re.match(r"^  ([^ )#][^)]*)\)", L[i])] if m]


def queue_tasks(queue_txt):
    out = []
    for l in open(queue_txt).read().split("\n"):
        m = re.match(r"^(\S+)\s+(?:open|auto|locked)\s+(.*)$", l)
        if m and not l.startswith("#"):
            out += [t.split(":")[0] for t in m.group(2).split()]
    return out


def queue_gpus(queue_txt):
    g = {}
    for l in open(queue_txt).read().split("\n"):
        m = re.match(r"^#gpus\s+(\S+)\s+(\d+)\s*$", l)
        if m:
            g[m.group(1)] = int(m.group(2))
    return g


def cpu_tasks(run_tasks_sh):
    """the names of every `<a>|<b>) return 0;;` branch of run_tasks.sh's one-line cpu_task()."""
    m = re.search(r"^cpu_task\(\) \{ case \$1 in (.*?) esac", open(run_tasks_sh).read(), re.M)
    return {n for br in re.findall(r"([A-Za-z0-9._|-]+)\) return 0;;", m.group(1)) for n in br.split("|")} if m else set()


def all_tasks(runbook):
    """task.sh's literal names, every task of the queue's stage lines; and the family globs no listed name covers."""
    case = main_case_names(os.path.join(runbook, "scripts", "task.sh"))
    lit = [a for _, alts in case for a in alts if not re.search(r"[*?\[]", a)]
    # [v2.5.9z88e] the parentheses: `-` binds tighter than `|`, so SELF used to leave the queue's names only, never task.sh's
    # literals - CORPUS-RECORD, a task.sh literal, was recorded and its sandbox ran a real nested recording (10-26 06:48Z)
    names = sorted((set(lit) | set(queue_tasks(os.path.join(runbook, "tasks", "queue.txt")))) - set(SELF))
    globs = [a for _, alts in case for a in alts if re.search(r"[*?\[]", a) and a != "*"]
    uncovered = [g for g in globs if not any(fnmatch.fnmatchcase(n, g) for n in names)]
    return names, uncovered


# [ops-lead O7] the node's own loops must not mistake a sandboxed task for a real one: every process scan of run_tasks.sh /
# control.py / heartbeat.sh anchors the task name at the end of the command line (`task.sh <gpu> <task>$`) and gpu_held
# matches the real GPU indices. So the sandbox calls task.sh with GPU indices from SANDBOX_GPU0 (no node has a GPU 90) and a
# trailing SANDBOX_MARK argument (task.sh reads $1 / $2 only): none of those scans can match a sandboxed process.
SANDBOX_GPU0 = 90
SELF = ("CORPUS-RECORD", "BUILD-RB-IDX1")   # never recorded: the recorder itself (a nested recording), and [v2.5.9z88e] a task
# that would signal the node's own loops
SANDBOX_MARK = "--dryrun-sandbox"


def gpu_arg(task, cpu, gpus):
    if task in cpu:
        return "cpu"
    n = gpus.get(task, 1)
    return ",".join(str(SANDBOX_GPU0 + i) for i in range(n))


# -------------------------------------------------------------------------------------------------------------- sandbox
def _guard(path, base):
    rp, rb = os.path.realpath(path), os.path.realpath(base)
    if not (rp == rb or rp.startswith(rb + os.sep)):
        raise SystemExit(f"refused: {path} outside the sandbox base")
    return rp


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _weights(d):
    os.makedirs(d, exist_ok=True)
    _touch(os.path.join(d, "model.safetensors"))
    _touch(os.path.join(d, "config.json"), "{}\n")


def model_names(runbook):
    """every models/<name> the scripts or the queue name (the local models a task may look for)."""
    names = set(MODELS_ENV_KEYS.values())
    for rel in ("scripts/run_arm.sh", "scripts/run_e12.sh", "scripts/run_queue_eval.sh", "scripts/env.sh", "tasks/queue.txt"):
        p = os.path.join(runbook, rel)
        if os.path.exists(p):
            names |= set(re.findall(r"models/([A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9])", open(p).read()))
    return sorted(n for n in names if not n.endswith((".env", ".txt", ".json")))


class Real:
    """what the node profile copies from: this node's $OPD_ROOT and $CKPT_ROOT (read-only)."""
    def __init__(self, opd, ckpt=""):
        self.opd, self.ckpt = opd, ckpt

    def file(self, rel):
        p = os.path.join(self.opd, rel) if self.opd else ""
        if p and os.path.isfile(p) and os.path.getsize(p) <= SMALL:
            return open(p, "rb").read()
        return None

    def has(self, rel, root=None):
        r = self.opd if root is None else root
        return bool(r) and os.path.exists(os.path.join(r, rel))

    def weights(self, rel, root=None):
        r = self.opd if root is None else root
        d = os.path.join(r, rel) if r else ""
        return bool(d) and os.path.isdir(d) and any(f.endswith(".safetensors") for f in os.listdir(d))


def make_sandbox(sb, base, profile, runbook, plants, real, control_env):
    _guard(sb, base)
    if os.path.exists(sb):
        shutil.rmtree(_guard(sb, base))
    opd, ck = os.path.join(sb, "opd"), os.path.join(sb, "ckpt")
    for d in (opd, ck, os.path.join(sb, "hm"), os.path.join(sb, "hf"), os.path.join(sb, "dr"), os.path.join(sb, "tmp"),
              os.path.join(opd, "runs"), os.path.join(opd, "logs"), os.path.join(opd, "control"), os.path.join(opd, "data"),
              os.path.join(opd, "G-OPD", "verl")):   # the runners cd into the trainer's checkout before the training command
        os.makedirs(d, exist_ok=True)
    shims = os.path.join(runbook, "scripts", "dryrun_shims")   # used in place (read-only): no new executables per sandbox
    for v in VENVS:   # each venv: an activate that keeps the shims first, and python / python3 that are the python shim
        b = os.path.join(opd, v, "bin")
        os.makedirs(b, exist_ok=True)
        _touch(os.path.join(b, "activate"), f'export VIRTUAL_ENV="{os.path.join(opd, v)}"\nexport PATH="{shims}:$PATH"\n'
                                             'deactivate() { :; }\n')
        for py in ("python", "python3"):
            os.symlink(os.path.join(shims, "python"), os.path.join(b, py))
    models = {}
    for k, name in MODELS_ENV_KEYS.items():
        models[k] = os.path.join(opd, "models", name)
    _touch(os.path.join(opd, "models.env"), "".join(f"{k}={v}\n" for k, v in sorted(models.items())))
    for name in model_names(runbook):
        d = os.path.join(opd, "models", name)
        always = name in MODELS_ENV_KEYS.values()   # models.env's four: present on every node (wherever models.env points)
        if profile == "clean" or always or real.weights(os.path.join("models", name)):
            _weights(d)
        if profile == "clean" or always or real.has(os.path.join("models", name, ".fetch_ok")):
            _touch(os.path.join(d, ".fetch_ok"))
    _touch(os.path.join(opd, "paths.env"), f"CKPT_ROOT={ck}\n")
    env_txt = control_env if profile == "node" else ""
    # the sandbox's own roots last (env.sh sources control.env last): a copied CKPT_ROOT / DR_ROOT / HF_HOME cannot escape
    env_txt += f"\n# [dryrun_json] sandbox roots\nexport CKPT_ROOT={ck} DR_ROOT={os.path.join(sb, 'dr')} HF_HOME={os.path.join(sb, 'hf')}\n"
    _touch(os.path.join(opd, "control.env"), env_txt)
    for rel, data in plants:
        p = _guard(os.path.join(sb, rel), sb)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    return opd, ck


def child_env(sb, runbook, real_py, fa):
    e = {"PATH": f"{os.path.join(runbook, 'scripts', 'dryrun_shims')}:/usr/bin:/bin", "HOME": os.path.join(sb, "hm"),
         "OPD_ROOT": os.path.join(sb, "opd"), "CKPT_ROOT": os.path.join(sb, "ckpt"), "CKPT_ROOT_FROM": "paths.env",
         "HF_HOME": os.path.join(sb, "hf"), "DR_ROOT": os.path.join(sb, "dr"), "TMPDIR": os.path.join(sb, "tmp"),
         "TZ": "UTC", "LC_ALL": "C", "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1", "RUNBOOK": runbook, "PS4": "+TASKSH ",
         "DRYRUN_REC": os.path.join(sb, "rec.jsonl"), "DRYRUN_SB": sb, "DRYRUN_REAL_PY": real_py,
         "DRYRUN_SHIM_PY": os.path.join(runbook, "scripts", "dryrun_shim.py"), "DRYRUN_FA": fa, "DRYRUN_T0": str(T0)}
    return e


def run_once(sb, runbook, task, g, env, sec):
    cmd = ["bash", "-x", os.path.join(runbook, "scripts", "task.sh"), g, task, SANDBOX_MARK]
    if shutil.which("nice"):
        cmd = ["nice", "-n", "19"] + cmd
    p = subprocess.Popen(cmd, cwd=os.path.join(sb, "opd"), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=sec)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        out, err = p.communicate()
        rc = "timeout"
    return rc, out.decode(errors="replace"), err.decode(errors="replace")


def classify(out_lines, task):
    """the refusal this attempt ended with -> (kind, groupdict) or None."""
    for line in reversed(out_lines):
        line = line.strip()
        for kind, pat in REFUSALS:
            m = re.match(pat, line)
            if m and m.groupdict().get("task", task) == task:
                return kind, m.groupdict()
    return None


# --------------------------------------------------------------------------------------------------------------- plants
def plan_plant(kind, g, profile, real, sb):
    """(plants [(rel to sb, bytes, source)], finding or None) for one refusal."""
    if kind == "gate":
        gate = g["gate"]
        rel = os.path.join("runs", gate, "report.json")
        if profile == "node":
            data = real.file(rel)
            if data is None:
                return [], f"node_gate_absent:{gate}"
            ps = [(os.path.join("opd", rel), data, "node"), (os.path.join("opd", "runs", gate, ".done"), b"", "node")]
            for f in GATE_FILES.get(gate, {}):
                d = real.file(os.path.join("runs", gate, f))
                if d is not None:
                    ps.append((os.path.join("opd", "runs", gate, f), d, "node"))
            return ps, None
        rep = GATE_REPORTS.get(gate, {"result": "PASS"})
        ps = [(os.path.join("opd", rel), (json.dumps(rep) + "\n").encode(), "synthetic"),
              (os.path.join("opd", "runs", gate, ".done"), b"", "synthetic")]
        for f, v in GATE_FILES.get(gate, {}).items():
            ps.append((os.path.join("opd", "runs", gate, f), (json.dumps(v) + "\n").encode(), "synthetic"))
        return ps, None
    if kind == "gate_check":
        gate = g["gate"]
        if gate not in GATE_REPORTS:
            return [], f"gate_check_unlisted:{gate}"
        if profile == "node":
            data = real.file(os.path.join("runs", gate, "report.json"))
            return ([(os.path.join("opd", "runs", gate, "report.json"), data, "node")], None) if data is not None else ([], f"node_gate_absent:{gate}")
        return [(os.path.join("opd", "runs", gate, "report.json"), (json.dumps(GATE_REPORTS[gate]) + "\n").encode(), "synthetic")], None
    if kind == "start":
        p = g["path"]
        opd = os.path.join(sb, "opd")
        for root, tag in ((opd, "opd"), (os.path.join(sb, "ckpt"), "ckpt")):
            if os.path.realpath(p).startswith(os.path.realpath(root) + os.sep):
                rel = os.path.relpath(os.path.realpath(p), os.path.realpath(root))
                if profile == "node" and not real.weights(rel, real.opd if tag == "opd" else real.ckpt):
                    return [], f"node_start_absent:{rel}"
                return [(os.path.join(tag, rel, "model.safetensors"), b"", "synthetic" if profile == "clean" else "mirror")], None
        return [], f"start_outside_sandbox:{p}"
    if kind == "adhoc_fetch":
        rel = os.path.join("adhoc_models", "result", g["rel"])
        if profile == "node" and not real.has(os.path.join(rel, ".fetch_ok"), real.ckpt):
            return [], f"node_input_absent:{rel}"
        src = "synthetic" if profile == "clean" else "mirror"
        return [(os.path.join("ckpt", rel, ".fetch_ok"), b"", src), (os.path.join("ckpt", rel, "model.safetensors"), b"", src)], None
    if kind in ("fetch_ok", "model_missing"):
        name = g["name"]
        rel = DATA_ALIASES.get(name) or (os.path.join("data", "hf", name) if "__" in name else os.path.join("models", name))
        if profile == "node" and not real.has(os.path.join(rel, ".fetch_ok")):
            return [], f"node_input_absent:{rel}"
        src = "synthetic" if profile == "clean" else "mirror"
        ps = [(os.path.join("opd", rel, ".fetch_ok"), b"", src)]
        if not rel.startswith("data"):
            ps += [(os.path.join("opd", rel, "model.safetensors"), b"", src), (os.path.join("opd", rel, "config.json"), b"{}\n", src)]
        return ps, None
    if kind == "node_file":
        rel = g["path"]
        if profile == "node":
            data = real.file(rel)
            if data is None:
                return [], f"node_input_absent:{rel}"
            return [(os.path.join("opd", rel), data, "node")], None
        return [(os.path.join("opd", rel), b'{"id": 0}\n', "synthetic")], None
    return [], f"{kind}:{g.get('what', '')}"


# ------------------------------------------------------------------------------------------------------------- collect
def branch_not_covered(calls):
    """the branches a recording cannot show (ops-lead O7; v26_impl_plan §2.4): run_tasks.sh's placement is never on the
    task.sh path; a recorded-only call's other outcome (an HF backoff, a failed pend) is never taken; evaluations and
    analyses are recorded, not run."""
    out = {"placement (run_tasks.sh: place_run / run_placeable / runs_ckpt)"}
    for r in calls:
        if r["cls"] == "remote" and r.get("sub") == "backoff":
            out.add("remote.py backoff > 0")
        if r["cls"] == "remote" and r.get("sub") == "pend":
            out.add("remote.py pend failure")
        if r["cls"] in ("gen_eval", "analysis", "weights_op"):
            out.add("evaluation / analysis / weights not run")
        if r["cls"] == "unit_test":
            out.add("unit tests not run (reported passed)")
    return sorted(out)



def read_rec(path):
    out = []
    if os.path.exists(path):
        for l in open(path):
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    return out


def _ctx(sb, runbook, g):
    """the normalization of one sandbox: its roots, the runbook, and the real interpreter's own paths (a forwarded python's
    traceback names its standard library)"""
    pys = sorted({p for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(sys.executable)) if p and p != "/"},
                 key=len, reverse=True)
    ctx = N.Ctx(sb, os.path.join(sb, "opd"), os.path.join(sb, "ckpt"), os.path.join(sb, "hm"), os.path.join(sb, "hf"),
                os.path.join(sb, "dr"), runbook, g, T0)
    ctx.pairs = sorted(set(ctx.pairs + [(p, "${PYTHON}") for p in pys] + [(os.path.realpath(p), "${PYTHON}") for p in pys]),
                       key=lambda x: -len(x[0]))
    return ctx


def runner_of(err, runbook):
    for l in err.split("\n"):
        m = re.match(r"^\++TASKSH (?:env [^ ]+ )*(bash|python3?) (\S+/scripts/\S+)(.*)$", l)
        if m and not m.group(2).endswith("/env.sh"):
            return m.group(2), m.group(3).split()
    return None, []


def collect(task, profile, g, sb, runbook, rc, out, err, iters, gates, plants, finding, head, rules):
    recs = read_rec(os.path.join(sb, "rec.jsonl"))
    calls = [r for r in recs if r.get("cls") != "clock"]
    trains = [r for r in calls if r["cls"] == "train_cmd"]
    cfg_p = os.path.join(sb, "opd", "runs", task, "config.yaml")
    cfg = open(cfg_p).read() if os.path.isfile(cfg_p) else None
    tail = []
    if cfg is not None:
        keep = []
        for l in cfg.split("\n"):
            (tail if re.match(r"^(completed_steps|finished|stopped|resumed):", l) else keep).append(l)
        cfg = "\n".join(keep)
    runner, rargs = runner_of(err, runbook)
    first = next((l for l in out.split("\n") if l.strip()), "")
    up = [r for r in calls if r["cls"] == "bg" and any("upload_sync.sh" in a for a in r["argv"])]
    smokes = {}
    for r in trains:
        if r["env"].get("SMOKE") == "1":
            smokes.setdefault(r["env"].get("PROBE_TASK", "?"), []).append(r)
    rec = {
        "schema": 1, "task": task, "profile": profile, "runbook_head": head, "rules_version": rules,
        "dispatch": {"runner": runner, "args": rargs, "lane": "cpu" if g == "cpu" else "gpu", "gpu_arg": g, "exit": rc,
                     "first_line": first},
        "result": finding or "ok", "iterations": iters, "gates": gates,
        "plants": [{"path": p, "source": s, "sha256": hashlib.sha256(d).hexdigest()} for p, d, s in plants],
        "train": [{"module": r.get("module"), "argv": r["argv"], "venv": r["venv"], "env": r["env"]}
                  for r in trains if r["env"].get("SMOKE") != "1"],
        "smokes": {k: [{"argv": r["argv"], "venv": r["venv"], "env": r["env"]} for r in v] for k, v in sorted(smokes.items())},
        "patches": [r["argv"][0] for r in calls if r["cls"] == "patch"],
        "upload": {"launch": [{"argv": r["argv"], "env": {k: r["env"][k] for k in ("UPLOAD_WEIGHTS", "KEEP_BF16") if k in r["env"]}}
                              for r in up], "restart": None},
        "config_yaml": cfg, "config_tail": tail,
        "eval": [r["argv"] for r in calls if r["cls"] == "gen_eval"],
        "uncovered": sorted({f"{r['cls']}:{r.get('sub') or r.get('detail') or (r['argv'][:1] or [''])[0]}" for r in calls
                             if r["cls"] in ("unknown", "control_unlisted", "remote_unlisted", "net")}),
        "branch_not_covered": branch_not_covered(calls),
        "calls": calls,
    }
    ctx = _ctx(sb, runbook, g)
    rec = N.normalize(rec, ctx)
    rec["dispatch"]["args"] = N.argv_gpu(rec["dispatch"]["args"], g, ctx)
    return rec


# ------------------------------------------------------------------------------------------------------------ one task
def input_plants(task, profile, real):
    out = []
    for pat, rels in INPUT_PLANTS:
        if re.match(pat, task):
            for rel, text in rels:
                rel = rel.format(task=task)
                if profile == "clean":
                    out.append((os.path.join("opd", rel), text.encode(), "synthetic"))
                elif real.has(rel):
                    data = real.file(rel) if text else b""
                    out.append((os.path.join("opd", rel), data if data is not None else text.encode(), "node" if data else "mirror"))
    return out


def record_task(task, profile, g, base, runbook, real, control_env, real_py, fa, sec, head, rules):
    plants, gates, finding, iters = input_plants(task, profile, real), [], None, 0
    rc, out, err, sb = None, "", "", None
    for it in range(1, MAX_ITER + 1):
        iters = it
        sb = os.path.join(base, profile, task)
        make_sandbox(sb, base, profile, runbook, [(p, d) for p, d, _ in plants], real, control_env)
        rc, out, err = run_once(sb, runbook, task, g, child_env(sb, runbook, real_py, fa), sec)
        if rc == 0:
            break
        if rc == "timeout":
            finding = "timeout"
            break
        r = classify(out.split("\n"), task)
        if r is None:
            finding = f"exit:{rc}"
            break
        kind, gd = r
        new, f = plan_plant(kind, gd, profile, real, sb)
        if f or not new:
            finding = f or f"{kind}:no_plant"
            break
        if kind in ("gate", "gate_check"):
            gates.append(gd["gate"])
        if any(p == q for p, _, _ in plants for q, _, _ in new):
            finding = f"{kind}:repeat"   # the same plant again: the check reads something the plant does not give
            break
        plants += new
    else:
        finding = "max_iterations"
    rec = collect(task, profile, g, sb, runbook, rc, out, err, iters, gates, plants, finding, head, rules)
    if finding:
        ctx = _ctx(sb, runbook, g)
        rec["stdout_tail"] = N.normalize(out.strip().split("\n")[-20:], ctx)
        rec["stderr_tail"] = N.normalize([l for l in err.strip().split("\n") if not l.startswith("+")][-20:], ctx)
    return rec


def flash_attn_answers(opd):
    """the node profile's answer to `import flash_attn` per venv, measured once with each real venv's python."""
    ans = {}
    for v in VENVS:
        py = os.path.join(opd, v, "bin", "python") if opd else ""
        if py and os.path.exists(py):
            try:
                ans[v] = "1" if subprocess.run([py, "-c", "import flash_attn"], stdout=subprocess.DEVNULL,
                                               stderr=subprocess.DEVNULL, timeout=120).returncode == 0 else "0"
            except (OSError, subprocess.TimeoutExpired):
                ans[v] = "0"
        else:
            ans[v] = "0"
    return ans


def git(runbook, *a):
    try:
        return subprocess.run(["git", "-C", runbook] + list(a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              timeout=30).stdout.decode().strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def runbook_state(runbook):
    """the read-only check: git's view plus every file's mtime under the runbook (not .git)."""
    st = git(runbook, "status", "--porcelain")
    newest = 0.0
    for d, dirs, files in os.walk(runbook):
        dirs[:] = [x for x in dirs if x != ".git"]
        for f in files:
            try:
                newest = max(newest, os.lstat(os.path.join(d, f)).st_mtime)
            except OSError:
                pass
    return st, newest


def main(argv=None):
    if os.environ.get("DRYRUN_SB") or os.environ.get("DRYRUN_REAL_PY"):   # [v2.5.9z88e] never inside a sandbox (no nested recording)
        print("CORPUS_RECORD_REFUSED inside a sandbox: no nested recording")
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--profile", default="both", choices=("clean", "node", "both"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--runbook", default=os.path.dirname(HERE))
    ap.add_argument("--per-task-sec", type=int, default=60)
    ap.add_argument("--total-sec", type=int, default=7200)
    ap.add_argument("--jobs", type=int, default=4)   # tasks recorded at once (each in its own sandbox; nice 19, one CPU-lane task)
    ap.add_argument("--task-dir", default=None)
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args(argv)
    runbook = os.path.realpath(a.runbook)
    real_opd = os.environ.get("OPD_ROOT", "")
    ck = os.environ.get("CKPT_ROOT", "")
    base = a.base or (os.path.join(ck, "dryrun_json.tmp") if ck else None)
    if not base or os.path.basename(base.rstrip("/")) != "dryrun_json.tmp":
        raise SystemExit("refused: the sandbox base must be a directory named dryrun_json.tmp (default: $CKPT_ROOT/dryrun_json.tmp)")
    if ck and not a.base and not os.path.realpath(base).startswith(os.path.realpath(ck) + os.sep):
        raise SystemExit("refused: the sandbox base is not on the checkpoint disk")
    out = a.out or os.path.join(os.path.dirname(base), "dryrun_json.out")
    if os.path.basename(out.rstrip("/")) != "dryrun_json.out":
        raise SystemExit("refused: the output directory must be named dryrun_json.out (default: beside the sandbox base)")
    if ck and not os.path.realpath(out).startswith(os.path.realpath(ck) + os.sep):
        raise SystemExit("refused: the output directory is not on the checkpoint disk")
    if os.path.exists(base):
        shutil.rmtree(_guard(base, base))
    os.makedirs(base)
    for sub in ("corpus", "up"):   # [N1] each recording starts empty: a renamed or removed task's old JSON never reaches the tarball
        if os.path.exists(os.path.join(out, sub)):
            shutil.rmtree(_guard(os.path.join(out, sub), out))
    names, uncovered = all_tasks(runbook) if a.all else (a.tasks, [])
    cpu = cpu_tasks(os.path.join(runbook, "scripts", "run_tasks.sh"))
    gpus = queue_gpus(os.path.join(runbook, "tasks", "queue.txt"))
    head = git(runbook, "rev-parse", "--short", "HEAD") or "unknown"
    m = re.search(r'^RULES_VERSION = "([0-9.]+)"', open(os.path.join(runbook, "scripts", "control_rules.py")).read(), re.M)
    rules = m.group(1) if m else "unknown"
    real = Real(real_opd, ck)
    control_env = ""
    if real_opd and os.path.isfile(os.path.join(real_opd, "control.env")):
        control_env = open(os.path.join(real_opd, "control.env")).read()
    profiles = ["clean", "node"] if a.profile == "both" else [a.profile]
    fa_node = ",".join(f"{k}:{v}" for k, v in flash_attn_answers(real_opd).items()) if "node" in profiles else ""
    fa_clean = ",".join(f"{k}:{v}" for k, v in CLEAN_FA.items())
    real_py = sys.executable
    st0 = runbook_state(runbook)
    t_start = time.time()
    index, findings = {}, [{"kind": "glob_uncovered", "glob": gl} for gl in uncovered]
    for prof in profiles:
        os.makedirs(os.path.join(out, "corpus", prof), exist_ok=True)
        fa = fa_node if prof == "node" else fa_clean

        def one(t):
            if time.time() - t_start > a.total_sec:
                return t, None
            return t, record_task(t, prof, gpu_arg(t, cpu, gpus), base, runbook, real, control_env, real_py, fa,
                                  a.per_task_sec, head, rules)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            done = list(ex.map(one, names))   # in name order: the outputs do not depend on the job count
        for t, rec in done:
            if rec is None:
                findings.append({"kind": "total_time_cap", "profile": prof, "task": t})
                continue
            txt = N.dumps(rec)
            open(os.path.join(out, "corpus", prof, f"{t}.json"), "w").write(txt)
            index.setdefault(t, {})[prof] = {"sha256": hashlib.sha256(txt.encode()).hexdigest(), "iterations": rec["iterations"],
                                             "result": rec["result"]}
            if rec["result"] != "ok":
                findings.append({"kind": "task", "profile": prof, "task": t, "result": rec["result"]})
            for u in rec["uncovered"]:
                findings.append({"kind": "uncovered", "profile": prof, "task": t, "call": u})
            shutil.rmtree(_guard(os.path.join(base, prof, t), base), ignore_errors=True)
    st1 = runbook_state(runbook)
    if st1[0] != st0[0] or st1[1] > st0[1]:
        findings.append({"kind": "runbook_changed", "before": st0[0], "after": st1[0]})
    hits = []
    lits = N.machine_literals([real_opd, ck, os.environ.get("HOME", ""), runbook, base, out])
    for d, _, files in os.walk(os.path.join(out, "corpus")):
        for f in files:
            hits += [{"kind": "leak", "file": os.path.relpath(os.path.join(d, f), out), "what": w, "match": "<redacted>"}
                     for w, _ in N.leak_hits(open(os.path.join(d, f)).read(), lits)]
    findings += hits
    meta = {"schema": 1, "runbook_head": head, "rules_version": rules, "profiles": profiles, "tasks": len(names),
            "seconds": round(time.time() - t_start, 1)}
    open(os.path.join(out, "index.json"), "w").write(N.dumps({"meta": meta, "tasks": index}))
    open(os.path.join(out, "findings.json"), "w").write(N.dumps(findings))
    with tarfile.open(os.path.join(out, "corpus.tgz"), "w:gz") as tf:
        tf.add(os.path.join(out, "corpus"), arcname="corpus")
    shutil.rmtree(_guard(base, base), ignore_errors=True)
    blocked = [f for f in findings if f["kind"] in ("leak", "runbook_changed")]
    up_ok = None
    if a.task_dir and not a.no_upload and not blocked:
        stage = os.path.join(out, "up")
        os.makedirs(stage, exist_ok=True)
        for f in ("index.json", "findings.json", "corpus.tgz"):
            shutil.copy(os.path.join(out, f), stage)
        up_ok = subprocess.run([sys.executable, os.path.join(runbook, "scripts", "remote.py"), "put", stage,
                                f"corpus/{head}"], timeout=1800).returncode == 0
    if a.task_dir:
        res = "FAIL" if blocked or up_ok is False else "PASS"
        why = (f"{len(names)} tasks x {len(profiles)} profiles in {meta['seconds']} s; {len(findings)} findings"
               + ("; refused to upload: " + ", ".join(sorted({f['kind'] for f in blocked})) if blocked else "")
               + ("" if up_ok is None else ("; uploaded to the private repository" if up_ok else "; the upload failed")))
        os.makedirs(a.task_dir, exist_ok=True)
        json.dump({"result": res, "reason": why, "meta": meta, "findings": len(findings)}, open(os.path.join(a.task_dir, "report.json"), "w"), indent=1)
        open(os.path.join(a.task_dir, ".done"), "w").close()
        print(f"CORPUS_RECORD {res} {why}")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
