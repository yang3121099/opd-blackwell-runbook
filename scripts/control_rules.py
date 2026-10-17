"""Rules of the remote-control plane (runbook v2.5, D-004 / R-012), shared by the node (control.py) and the project's
ctl.py, so a command the node would refuse is refused before it is pushed. Any change to this file is a change of
what the node accepts: the node takes a pulled version only if control/CHANGES.md names its RULES_VERSION (R-012).
"""
import calendar
import re
import time

RULES_VERSION = "2.5.4"   # 2.5.1: HF_REPO_PUBLIC / HF_REPO_PRIVATE / HEARTBEAT_SEC; 2.5.2: FIXED_MICRO_BSZ; 2.5.3: VENV=fa; 2.5.4: reap

META = set(";&|`$<>\\\"'(){}*?!#~\n\r")
TASK_RE = re.compile(r"^[A-Za-z0-9._-]+$")
KEY_RE = re.compile(r"^(VLLM_[A-Z0-9_]+|TRITON_[A-Z0-9_]+|TORCHINDUCTOR_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|VLLM_UTIL|TEACHER_OFFLOAD"
                    r"|MAX_TOK_GPU|KEEP_LOCAL_HF|MAX_JOBS|SAVE|CKPT_ROOT|CKPT_CANDIDATES\+|UPLOAD_WEIGHTS|KEEP_BF16|ARCHIVE_ROOT"
                    r"|HF_REPO_PUBLIC|HF_REPO_PRIVATE|HEARTBEAT_SEC|FIXED_MICRO_BSZ|VENV)$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")   # owner/name of an HF repository
VAL_RE = re.compile(r"^[A-Za-z0-9_.,:/=+-]*$")
BENCHES = {"math500", "amc23", "aime24", "aime25", "olympiadbench", "minervamath"}
SIGNALS = {"TERM", "KILL", "STOP", "CONT"}
LAB_ARCHIVE = "yang31210999/opd-lab-ckpts"   # eval source besides Qwen/*, the result repository and the two local roots
MAX_UPLOAD_BYTES = 20 * 2 ** 30
MAX_PER_PULL, MAX_SIGNALS_PER_PULL, MAX_KILLPGID_NOFORCE_PER_PULL = 20, 10, 1
H2, H24 = 2 * 3600, 24 * 3600

# verb -> (min args, max args, default lifetime in s): R-012 "expiry by default"
VERBS = {
    "ps": (0, 0, H24), "snapshot": (0, 1, H24), "logs": (2, 2, H24),
    "pause": (1, 1, H24), "resume": (1, 1, H24), "prio": (2, 2, H24), "set": (1, 1, H24), "setenv": (1, 1, H24),
    "kill": (1, 1, H2), "run": (1, 1, H2), "eval": (2, 4, H2), "update": (0, 0, H2), "upload": (1, 1, H2),
    "restore": (2, 2, H2), "rerun_verify": (0, 0, H2),
    "signal": (2, 3, H2), "kill-pgid": (1, 2, H2), "cancel": (1, 1, H24), "heartbeat": (0, 0, H2), "move-ckpt": (2, 2, H2), "reap": (1, 1, H2),
}
TASK_VERBS = {"pause", "resume", "kill", "prio", "restore", "logs", "run", "move-ckpt"}

# processes never signalled, whatever the command says (D-004 §5 rule 3); matched on the command line
NEVER_CMD = re.compile(r"(^|/)(init|systemd[^ ]*|sshd|containerd[^ ]*|dockerd|supervisord|cron|crond|tmux|screen|agetty|login)( |$)"
                       r"|jupyter|/opt/portal-aio|instance_portal|tunnel_manager|syncthing|caddy|log-tee"
                       r"|scripts/control\.py|scripts/run_tasks\.sh", re.I)
# processes of a training (STOP/CONT refused on them: Ray/NCCL heartbeats time out while frozen, R-012)
TRAINING_CMD = re.compile(r"verl\.trainer\.main_ppo|ray::|raylet|gcs_server|/tmp/ray_gpu|run_arm\.sh|run_e12\.sh|run_sft\.sh"
                          r"|torchrun|fsdp_sft_trainer")
EVAL_CMD = re.compile(r"code/eval/gen_eval\.py|run_eval\.sh|run_adhoc_eval\.sh|twr_alpha\.py")


class Reject(Exception):
    """A command the node does not run; the message goes into the receipt."""


def _ts(s):
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%MZ", "%Y-%m-%dT%H:%M:%S"):
        try:
            return calendar.timegm(time.strptime(s, fmt))
        except (TypeError, ValueError):
            continue
    return None


def parse(entry):
    """entry = one item of control/<node>.yaml: {seq, cmd, reason, utc, expires?, force?, confirm_foreign?}.
    Returns (verb, args, flags) or raises Reject."""
    if not isinstance(entry, dict):
        raise Reject("entry is not a mapping")
    cmd, reason = entry.get("cmd"), entry.get("reason")
    if not isinstance(cmd, str) or not cmd.strip():
        raise Reject("cmd missing")
    if any(ch in META for ch in cmd):
        raise Reject("shell metacharacter in cmd")
    if not isinstance(reason, str) or not reason.strip():
        raise Reject("reason missing (required for every command)")
    parts = cmd.split()
    verb, args = parts[0], parts[1:]
    if verb not in VERBS:
        raise Reject(f"unknown verb {verb!r}")
    lo, hi, _ = VERBS[verb]
    flags = {"force": False}
    if verb in ("signal", "kill-pgid") and args and args[-1] == "force":
        flags["force"] = True; args = args[:-1]; hi -= 1
    if not lo <= len(args) <= hi:
        raise Reject(f"{verb} takes {lo}-{hi} argument(s)")
    if verb in TASK_VERBS and not TASK_RE.match(args[0]):
        raise Reject("bad task name")
    if verb == "logs" and not (args[1].isdigit() and 1 <= int(args[1]) <= 2000):
        raise Reject("logs <task> <n>, n 1-2000")
    if verb == "prio" and not re.fullmatch(r"\d{1,4}", args[1]):
        raise Reject("prio must be an integer 0-9999")
    if verb in ("set", "setenv"):
        if "=" not in args[0]:
            raise Reject(f"{verb} KEY=VAL")
        k, v = args[0].split("=", 1)
        if not KEY_RE.match(k):
            raise Reject(f"key {k} not allowed")
        if not VAL_RE.match(v):
            raise Reject("value has characters outside [A-Za-z0-9_.,:/=+-]")
    if verb == "set":
        k, v = args[0].split("=", 1)
        if k in ("HF_REPO_PUBLIC", "HF_REPO_PRIVATE") and v and not REPO_RE.match(v):
            raise Reject(f"{k} must be <owner>/<name>")
        if k == "HEARTBEAT_SEC" and not (v.isdigit() and 60 <= int(v) <= 3600):
            raise Reject("HEARTBEAT_SEC must be an integer 60-3600")
    if verb in ("set", "setenv"):
        k, v = args[0].split("=", 1)
        if k == "FIXED_MICRO_BSZ" and v and not (v.isdigit() and 1 <= int(v) <= 16):
            raise Reject("FIXED_MICRO_BSZ must be an integer 1-16 (empty: back to the token budget)")
        if k == "VENV" and v not in ("", "fa"):
            raise Reject("VENV must be fa (the venv-fa that FA-PROBE built) or empty (the venv in use)")
    if verb == "eval":
        check_eval_args(args)
    if verb == "signal":
        if not args[0].isdigit() or int(args[0]) <= 1:
            raise Reject("signal <pid> <TERM|KILL|STOP|CONT> [force]; pid > 1")
        if args[1] not in SIGNALS:
            raise Reject(f"signal must be one of {sorted(SIGNALS)}")
    if verb == "kill-pgid" and (not args[0].isdigit() or int(args[0]) <= 1):
        raise Reject("kill-pgid <pgid> [force]; pgid > 1")
    if verb == "reap" and not (args[0].isdigit() and 0 <= int(args[0]) <= 7):
        raise Reject("reap <gpu index 0-7>")
    if verb == "cancel" and not (args[0].isdigit() and int(args[0]) > 0):
        raise Reject("cancel <seq>")
    flags["confirm_foreign"] = bool(entry.get("confirm_foreign"))
    return verb, args, flags


def check_eval_args(args):
    model, bench = args[0], args[1]
    ok_model = (re.fullmatch(r"Qwen/[A-Za-z0-9._-]+", model)
                or re.fullmatch(rf"{re.escape(LAB_ARCHIVE)}/(runs|ckpts)/[A-Za-z0-9._/-]+", model)
                or re.fullmatch(r"result:runs/[A-Za-z0-9._/-]+", model)
                or re.fullmatch(r"(local:)?[A-Za-z0-9._/-]+", model) and ".." not in model and not model.startswith("/"))
    if not ok_model or ".." in model:
        raise Reject("model must be Qwen/<name>, " + LAB_ARCHIVE + "/runs|ckpts/..., result:runs/... or a path under $OPD_ROOT/$CKPT_ROOT")
    if bench not in BENCHES:
        raise Reject(f"bench must be one of {sorted(BENCHES)}")
    if len(args) > 2 and not (args[2].isdigit() and 1 <= int(args[2]) <= 16):
        raise Reject("n must be 1-16")
    if len(args) > 3 and args[3] not in ("0.6", "1.0"):
        raise Reject("temperature must be 0.6 or 1.0")


def expiry(entry, verb):
    """Deadline (epoch s) of a command: its 'expires', else its 'utc' + the verb's default lifetime (R-012). A command
    with neither is treated as already expired for the 2 h verbs (a node must never run a stale kill)."""
    if entry.get("expires"):
        t = _ts(str(entry["expires"]))
        if t is None:
            raise Reject("expires is not a UTC time like 2026-10-17T12:00Z")
        return t
    t = _ts(str(entry.get("utc") or ""))
    if t is None:
        return 0 if VERBS[verb][2] == H2 else float("inf")
    return t + VERBS[verb][2]
