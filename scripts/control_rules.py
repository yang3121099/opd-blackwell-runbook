"""Rules of the remote-control plane (runbook v2.5, D-004 / R-012), shared by the node (control.py) and the project's
ctl.py, so a command the node would refuse is refused before it is pushed. Any change to this file is a change of
what the node accepts: the node takes a pulled version only if control/CHANGES.md names its RULES_VERSION (R-012).
"""
import calendar
import re
import time

RULES_VERSION = "2.5.22"   # 2.5.1: HF_REPO_PUBLIC / HF_REPO_PRIVATE / HEARTBEAT_SEC; 2.5.2: FIXED_MICRO_BSZ; 2.5.3: VENV=fa; 2.5.4: reap; 2.5.5: rerun; 2.5.6: republish; 2.5.7: rerun failed / skipped tasks; 2.5.8: E10c; 2.5.9: publish, leakscan; 2.5.10: fa-build; 2.5.11: scrubdiff; 2.5.12: SPEED-PROBE, setenv FUSED / GC; 2.5.13: du, rm-ours, dr-keep, setenv DR / DR_EVERY / DR_MBPS / E12B_GC; 2.5.14: persist-env, rebuild-drill; 2.5.15: E10d (4B), PROBE-4B; 2.5.16: FA-SPEED probes, setenv E12B_FA; 2.5.17: setenv E10D_FA; 2.5.18: FA-SPEED-TWIN; 2.5.19: #eval max_tokens, E10c dyn6144 / -s43, PROBE-TERM; 2.5.20: #eval mode, #forced, E10b-P5top10 / P5rand10m, PROBE-TOP10; 2.5.21: PROBE-TERM2; 2.5.22: PROBE-TERM3

META = set(";&|`$<>\\\"'(){}*?!#~\n\r")
TASK_RE = re.compile(r"^[A-Za-z0-9._-]+$")
KEY_RE = re.compile(r"^(VLLM_[A-Z0-9_]+|TRITON_[A-Z0-9_]+|TORCHINDUCTOR_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|VLLM_UTIL|TEACHER_OFFLOAD"
                    r"|MAX_TOK_GPU|KEEP_LOCAL_HF|MAX_JOBS|SAVE|CKPT_ROOT|CKPT_CANDIDATES\+|UPLOAD_WEIGHTS|KEEP_BF16|ARCHIVE_ROOT"
                    r"|HF_REPO_PUBLIC|HF_REPO_PRIVATE|HEARTBEAT_SEC|FIXED_MICRO_BSZ|VENV|FUSED|GC|E12B_GC|E12B_FA|E10D_FA|DR|DR_EVERY|DR_MBPS)$")
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
    "signal": (2, 3, H2), "kill-pgid": (1, 2, H2), "cancel": (1, 1, H24), "heartbeat": (0, 0, H2), "move-ckpt": (2, 2, H2), "reap": (1, 1, H2), "rerun": (1, 1, H2), "republish": (1, 1, H2), "publish": (1, 1, H2), "leakscan": (0, 0, H2), "fa-build": (0, 0, H2), "scrubdiff": (0, 1, H2), "du": (1, 2, H2), "rm-ours": (2, 2, H2), "dr-keep": (1, 1, H2), "persist-env": (0, 0, H2), "rebuild-drill": (0, 0, H2),
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
        if k in ("FUSED", "GC", "E12B_GC", "E12B_FA", "E10D_FA") and v not in ("", "0", "1"):
            raise Reject(f"{k} must be 0, 1 or empty (v2.5.6o: FUSED=1 fused log-prob kernels, GC=0 no gradient checkpointing)")
        if k == "DR" and v not in ("", "0", "1"):
            raise Reject("DR must be 0, 1 or empty (v2.5.7: resume state mirrored to the share; default on)")
        if k == "DR_EVERY" and v and not (v.isdigit() and 10 <= int(v) <= 100):
            raise Reject("DR_EVERY must be an integer 10-100 (steps between resume states)")
        if k == "DR_MBPS" and v and not (v.isdigit() and 50 <= int(v) <= 1000):
            raise Reject("DR_MBPS must be an integer 50-1000 (mirror bandwidth limit, MB/s)")
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
    if verb == "publish" and not re.fullmatch(r"[A-Za-z0-9._-]+", args[0]):
        raise Reject("publish <task>")
    if verb == "republish" and not re.fullmatch(r"[A-Za-z0-9._-]+", args[0]):
        raise Reject("republish <task>")
    if verb == "scrubdiff" and args and args[0] != "fix":
        raise Reject("scrubdiff [fix]")
    if verb in ("du", "rm-ours"):   # v2.5.7: symbolic prefixes only (control files are public: no paths in them)
        allowed = ("opd_root", "dr", "old_base") if verb == "du" else ("dr", "old_base")
        if args[0] not in allowed:
            raise Reject(f"{verb} <{'|'.join(allowed)}> {'[' if verb == 'du' else '<'}relative path{']' if verb == 'du' else '>'}")
        if len(args) > 1 and (not re.fullmatch(r"[A-Za-z0-9._/-]+", args[1]) or args[1].startswith("/") or ".." in args[1].split("/")):
            raise Reject("relative path of [A-Za-z0-9._/-], no leading / and no ..")
    if verb == "dr-keep" and not TASK_RE.match(args[0]):
        raise Reject("dr-keep <task>")
    if verb == "rerun" and not re.fullmatch(r"FA-PROBE|SPEED-PROBE|PROBE-4B|PROBE-TERM[23]?|PROBE-TOP10|FA-SPEED-(?:17|4B|E12|TWIN)|EVAL-[A-Za-z0-9._-]+|SFT-[A-Za-z0-9]+|E1[012]b-[A-Za-z0-9._-]+|E10[cd]-[A-Za-z0-9._-]+|E0[23]-[A-Za-z0-9._-]+", args[0]):
        raise Reject("rerun <FA-PROBE | SPEED-PROBE | EVAL-name | SFT-* | E10b/E11b/E12b-* | E02/E03-*>")
    if verb == "cancel" and not (args[0].isdigit() and int(args[0]) > 0):
        raise Reject("cancel <seq>")
    flags["confirm_foreign"] = bool(entry.get("confirm_foreign"))
    return verb, args, flags


MAX_TOKENS = ("2048", "4096", "8192", "16384", "32768")   # v2.5.7g (reviewer 18:4x UTC): budget-dependence readings


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
    if len(args) > 4 and args[4] not in MAX_TOKENS:   # v2.5.7g: the generation budget of a queue #eval line (7th field)
        raise Reject(f"max_tokens must be one of {', '.join(MAX_TOKENS)}")
    if len(args) > 5 and args[5] not in ("think", "no_think"):   # v2.5.7h: the mode of a queue #eval line (8th field)
        raise Reject("mode must be think or no_think")


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
