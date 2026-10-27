"""Rules of the remote-control plane (runbook v2.5, D-004 / R-012), shared by the node (control.py) and the project's
ctl.py, so a command the node would refuse is refused before it is pushed. Any change to this file is a change of
what the node accepts: the node takes a pulled version only if control/CHANGES.md names its RULES_VERSION (R-012).
"""
import calendar
import os
import re
import time

RULES_VERSION = "2.5.103"   # 2.5.103: rerun DIAG-T32-TAILS / DIAG-PROBE-GATE / DIAG-WALL-BIMODAL / DIAG-VERIFY-ONSET / DIAG-VLM-LOOP2 (z99); 2.5.102: publish-redacted <task> <file> (z96); 2.5.101: rerun PROBE-E12E-16K (z97); 2.5.100: rerun VERIFY-PRIV-LFS (z96a); 2.5.99: rerun E15-VLM-P2lo / E15-VLM-P5tail-lo / PROBE-VLM-COMPLETE / PROBE-VLM-TAIL / DIAG-VLM-TRUNC (z89); 2.5.98: rerun E10e-* / E12e-* / PROBE-E12E-8K / PROBE-TAIL-8B (z91); 2.5.1: HF_REPO_PUBLIC / HF_REPO_PRIVATE / HEARTBEAT_SEC; 2.5.2: FIXED_MICRO_BSZ; 2.5.3: VENV=fa; 2.5.4: reap; 2.5.5: rerun; 2.5.6: republish; 2.5.7: rerun failed / skipped tasks; 2.5.8: E10c; 2.5.9: publish, leakscan; 2.5.10: fa-build; 2.5.11: scrubdiff; 2.5.12: SPEED-PROBE, setenv FUSED / GC; 2.5.13: du, rm-ours, dr-keep, setenv DR / DR_EVERY / DR_MBPS / E12B_GC; 2.5.14: persist-env, rebuild-drill; 2.5.15: E10d (4B), PROBE-4B; 2.5.16: FA-SPEED probes, setenv E12B_FA; 2.5.17: setenv E10D_FA; 2.5.18: FA-SPEED-TWIN; 2.5.19: #eval max_tokens, E10c dyn6144 / -s43, PROBE-TERM; 2.5.20: #eval mode, #forced, E10b-P5top10 / P5rand10m, PROBE-TOP10; 2.5.21: PROBE-TERM2; 2.5.22: PROBE-TERM3; 2.5.23: E12d, E12b-P2-s43, E10d -s43, STRICT-C1; 2.5.24: E10b / E11b -t14, PROBE-T14; 2.5.25: E10b-P5think, PROBE-P5THINK; 2.5.26: E10b-P5self, PROBE-P5SELF; 2.5.27: du ckpt; 2.5.28: PROBE-TERM4; 2.5.29: setenv DRILL / REMOTE_FAULT; 2.5.30: E10b-P2self, PROBE-P2SELF; 2.5.31: setenv FETCH_ENDPOINT; 2.5.32: PROBE-TERM4-4B; 2.5.33 (runbook 2.5.8): setenv LOCAL_LRU_GB / OUTBOX / PREFETCH_PAR / PREFETCH_TIMEOUT / FETCH_TIMEOUT / CPU_MAX; 2.5.34 (runbook 2.5.8a): E12b-P5self, E12c-*, PROBE-E12C; 2.5.35 (runbook 2.5.8b): E12b-P5-kl, PROBE-E12B-KL; 2.5.36 (runbook 2.5.8c): E12b-P5-stop, PROBE-E12B-STOP; 2.5.37 (runbook 2.5.8e): E10b-P5self-s43, E12b-P5self-s43, E10d-P5self, PROBE-P5SELF-4B; 2.5.38 (runbook 2.5.8h): E12b-P5-8k / -16k, PROBE-E12B-8K / -16K; 2.5.39 (runbook 2.5.8i): E10d-X1 / X2, PROBE-X2507-X1 / X2; 2.5.40 (runbook 2.5.8l): E10d-X4, PROBE-X2507-X4; 2.5.41 (runbook 2.5.8m): E10b-P5lenS / L / R, PROBE-LENSEL; 2.5.42 (runbook 2.5.9a): PROBE-2GPU (#gpus); 2.5.43 (runbook 2.5.9b): PROBE-TOOL; 2.5.44 (runbook 2.5.9c): PROBE-NET (#git lines); 2.5.45 (runbook 2.5.9f): E12b-P5-stop2, PROBE-E12B-STOP2; [2.5.46] rerun PROBE-NET2; [2.5.47] rerun PROBE-ROLLOUT-U60/U75/U85; [2.5.48] rerun PROBE-STOPMERGE; [2.5.49] rerun PROBE-TAIL; [2.5.50] setenv FETCH_CONNS_DAY / FETCH_CONNS_NIGHT / FETCH_DAY_MAX_BYTES; [2.5.51] rerun PROBE-POLISH; [2.5.52] rerun PROBE-P5SELF-06B / PROBE-TERM4-06B; [2.5.53] E12b-P5-r3, rerun PROBE-E12B-R3; [2.5.54] E12b-P5-16k-kl; [2.5.55] BUILD-VENV-VL, PROBE-VLLM-VL, setenv VENV_VL_REBUILD; [2.5.56] E10b-P5route / -P5mix, PROBE-DIFF17(-4B), PROBE-ROUTE; [2.5.57] READINGS-EXTRA; [2.5.58] E15-VLM-P5, PROBE-VLM, benches geo3k / mathvista; [2.5.59] PROBE-STOPMERGE-UNIT; [2.5.60] E10d-P5-t32, PROBE-T32; [2.5.61] setenv GPUS_HOLD_MIN / GPUS_SOON_MIN; [2.5.62] benches gpqa / mmlupro1k, temperature 0; [2.5.63] PROBE-WDIFF; [2.5.64] TOKCHECK-X5; [2.5.65] VENV-VL-NVRTC; [2.5.66] benches mathvision / mathverse; [2.5.67] setenv PT32_MAX_HOURS; [2.5.68] E10b-P5mix75 / -P5mix90, PROBE-MIXP; [2.5.69] E12b-P5-8k-s44, E10b-P5tail-s43; [2.5.70] E12b-P5-prog / -prog-up, PROBE-PROG; [2.5.71] E10b-P5route-inv, PROBE-ROUTE-INV; [2.5.72] E16-X5 / -X5e, PROBE-X5 (run and rerun), max_tokens 3072; [2.5.73] E10b-P5mix60; [2.5.74] setenv DISK_ALERT_GB / CKPT_ALERT_GB / CTRL_FALLBACK_LOW_GB / CTRL_FALLBACK_HIGH_GB; [2.5.75] setenv PUBLIC_UPLOADS; [2.5.76] setenv STATUS_PUT_S / PROCS_PUT_S / EVENTS_PUT_S / RESYNC_PUT_S / PULSE_PUT_S / PULSE_MIRROR_S / PULSE_PUBLIC_MIRROR / RUNS_PUT_S; [2.5.77] setenv CONTROL_DIRECT / CONTROL_PUT_TIMEOUT / DIAG_TAILS_S; [2.5.78] setenv GH_HB / GH_HB_SEC / GH_HB_URGENT_MIN_S; [2.5.79] setenv WEIGHTS_REPLAY / WEIGHTS_REPLAY_MBPS / WEIGHTS_REPLAY_MAX_S / WEIGHTS_REPLAY_BATCH / REPLAY_OUTBOX_MAX_AGE / WEIGHTS_REPLAY_COOL_S / WEIGHTS_REPLAY_COOL_MAX_S; [2.5.80] setenv GPU_RELEASE; [2.5.81] E10b-P5-8k / -P5smerge-8k, PROBE-P5-8K / PROBE-SMERGE-8K; [2.5.82] E10e-P5self, PROBE-8B; [2.5.83] setenv READINGS_DISK_MIN_GB; [2.5.84] setenv CTRL_ON_CKPT / RUNS_ON_CKPT / RUNS_MOVE_MAX_MB / TASKS_DISK_MIN_GB / TASKS_DISK_CRIT_GB; [2.5.85] inventory [pend]; [2.5.86] E10b-P5bud2k / -P5bud3k, PROBE-TAIL-BUD; [2.5.87] PROBE-DIFFPROBE; [2.5.88] PROBE-PSTAR; [2.5.89] E10b-P5-dense / -P5-akl, PROBE-DENSE; [2.5.90] E10b-P5selfroute, E10c-P5tail-c50, PROBE-SELFROUTE; [2.5.91] E10b-P5softmix50, PROBE-SOFTMIX; [2.5.92] bench ifeval, BUILD-VENV-IFEVAL, setenv VENV_IFEVAL_REBUILD; [2.5.93] E12b-P5tail-8k, E10d-P5tail, PROBE-TAIL-4B; [2.5.94] PROBE-DENSE-4B, PROBE-E12C-8K, PROBE-E12C-16K (rerun); [2.5.95] CORPUS-RECORD (rerun; v2.6 S0); [2.5.96] pend-weight; [2.5.97] HEARTBEAT-UNSTICK (run / rerun)

META = set(";&|`$<>\\\"'(){}*?!#~\n\r")
TASK_RE = re.compile(r"^[A-Za-z0-9._-]+$")
KEY_RE = re.compile(r"^(VLLM_[A-Z0-9_]+|TRITON_[A-Z0-9_]+|TORCHINDUCTOR_[A-Z0-9_]+|NCCL_[A-Z0-9_]+|VLLM_UTIL|TEACHER_OFFLOAD"
                    r"|MAX_TOK_GPU|KEEP_LOCAL_HF|MAX_JOBS|SAVE|CKPT_ROOT|CKPT_CANDIDATES\+|UPLOAD_WEIGHTS|KEEP_BF16|ARCHIVE_ROOT"
                    r"|HF_REPO_PUBLIC|HF_REPO_PRIVATE|HEARTBEAT_SEC|FIXED_MICRO_BSZ|VENV|FUSED|GC|E12B_GC|E12B_FA|E10D_FA|DR|DR_EVERY|DR_MBPS|DRILL|REMOTE_FAULT|FETCH_ENDPOINT|GPUS_HOLD_MIN|GPUS_SOON_MIN"
                    r"|LOCAL_LRU_GB|OUTBOX|PREFETCH_PAR|PREFETCH_TIMEOUT|FETCH_TIMEOUT|CPU_MAX|FETCH_CONNS_DAY|FETCH_CONNS_NIGHT|FETCH_DAY_MAX_BYTES|VENV_VL_REBUILD|VENV_IFEVAL_REBUILD|PT32_MAX_HOURS|DISK_ALERT_GB|CKPT_ALERT_GB|CTRL_FALLBACK_LOW_GB|CTRL_FALLBACK_HIGH_GB|PUBLIC_UPLOADS"
                    r"|STATUS_PUT_S|PROCS_PUT_S|EVENTS_PUT_S|RESYNC_PUT_S|PULSE_PUT_S|PULSE_MIRROR_S|PULSE_PUBLIC_MIRROR|RUNS_PUT_S"
                    r"|CONTROL_DIRECT|CONTROL_PUT_TIMEOUT|DIAG_TAILS_S"
                    r"|GH_HB|GH_HB_SEC|GH_HB_URGENT_MIN_S"
                    r"|WEIGHTS_REPLAY|WEIGHTS_REPLAY_MBPS|WEIGHTS_REPLAY_MAX_S|WEIGHTS_REPLAY_BATCH|REPLAY_OUTBOX_MAX_AGE|WEIGHTS_REPLAY_COOL_S|WEIGHTS_REPLAY_COOL_MAX_S"
                    r"|GPU_RELEASE|READINGS_DISK_MIN_GB|CTRL_ON_CKPT|RUNS_ON_CKPT|RUNS_MOVE_MAX_MB|TASKS_DISK_MIN_GB|TASKS_DISK_CRIT_GB)$")   # [2.5.80] a released GPU free for the next task   # [2.5.79] the weights replay   # [2.5.78] the GitHub fallback heartbeat   # [2.5.77] the control plane goes direct   # [2.5.76] the upload throttles (HF 429, 10-22)   # [2.5.74] the disk lines   # [v2.5.8] 2.5.33; [2.5.50] the parallel pulls
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")   # owner/name of an HF repository
VAL_RE = re.compile(r"^[A-Za-z0-9_.,:/=+-]*$")
# [v2.5.9z11] task names task.sh dispatches that `run` / `rerun` deliberately do not accept (the queue starts them); every other
# task.sh name must be accepted by both (test_v251.test_task_sh_names_accepted_by_run_and_rerun)
QUEUE_ONLY = {"E01-opsa17-think", "E01-opsa17-nothink", "E01-base17-think", "E02-opsa17-c-1", "E02-opsa17-c2", "E02-opsa4-c2",
              "E05-fstop-base06", "E05-fstop-art06"}   # the lab's zero-shot runs on this node (v2.5.6j / l): queue rows only
BENCHES = {"math500", "amc23", "aime24", "aime25", "olympiadbench", "minervamath", "geo3k", "mathvista", "gpqa", "mmlupro1k",
           "mathvision", "mathverse", "ifeval"}   # [2.5.58] the image benches (gen_eval_vl.py); [2.5.66] mathvision / mathverse (v2.5.9z33's benches); [2.5.92] ifeval (graded in venv-ifeval)
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
    "signal": (2, 3, H2), "kill-pgid": (1, 2, H2), "cancel": (1, 1, H24), "heartbeat": (0, 0, H2), "move-ckpt": (2, 2, H2), "reap": (1, 1, H2), "rerun": (1, 1, H2), "republish": (1, 1, H2), "publish": (1, 1, H2), "publish-redacted": (2, 2, H2), "pend-weight": (2, 2, H2), "leakscan": (0, 0, H2), "fa-build": (0, 0, H2), "scrubdiff": (0, 1, H2), "du": (1, 2, H2), "inventory": (0, 1, H2), "rm-ours": (2, 2, H2), "dr-keep": (1, 1, H2), "persist-env": (0, 0, H2), "rebuild-drill": (0, 0, H2),
}
TASK_VERBS = {"pause", "resume", "kill", "prio", "restore", "logs", "run", "move-ckpt"}

# processes never signalled, whatever the command says (D-004 §5 rule 3); system processes matched on the command line
NEVER_CMD = re.compile(r"(^|/)(init|systemd[^ ]*|sshd|containerd[^ ]*|dockerd|supervisord|cron|crond|tmux|screen|agetty|login)( |$)"
                       r"|jupyter|/opt/portal-aio|instance_portal|tunnel_manager|syncthing|caddy|log-tee", re.I)
# [v2.5.8] our control loops, matched on the process's own program (program()), not on its arguments: a task session's
# wrapper `bash -c "bash .../task.sh G T ...; python scripts/control.py event ..."` named control.py only as an argument,
# so kill-pgid refused every group run_tasks launched (10-19)
NEVER_PROG = re.compile(r"scripts/(control\.py|run_tasks\.sh|heartbeat\.sh|watchdog\.sh|uploader\.sh)", re.I)
INTERP = re.compile(r"^((ba|da|z)?sh|python[0-9.]*)$")


def program(cmd):
    """[v2.5.8] A process's own program: argv[0], and for an interpreter (sh / bash / python) its script (the first
    argument that is not an option), or "-c" / "-m <module>"."""
    a = (cmd or "").split()
    if not a:
        return ""
    out = [a[0]]
    if INTERP.match(os.path.basename(a[0])):
        for i in range(1, len(a)):
            if a[i] == "-c":
                out.append("-c"); break
            if a[i] == "-m":
                out += a[i:i + 2]; break
            if not a[i].startswith("-"):
                out.append(a[i]); break
    return " ".join(out)


def protected(cmd):
    """[v2.5.8] True for a system process (NEVER_CMD on the command line) or one of our loops (NEVER_PROG on program())."""
    return bool(NEVER_CMD.search(cmd or "") or NEVER_PROG.search(program(cmd)))
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
        if k in ("GPUS_HOLD_MIN", "GPUS_SOON_MIN") and v and not (v.isdigit() and 1 <= int(v) <= 1440):   # [2.5.61] minutes
            raise Reject(f"{k} must be an integer 1-1440 (empty: back to 30)")
        if k in ("FUSED", "GC", "E12B_GC", "E12B_FA", "E10D_FA") and v not in ("", "0", "1"):
            raise Reject(f"{k} must be 0, 1 or empty (v2.5.6o: FUSED=1 fused log-prob kernels, GC=0 no gradient checkpointing)")
        if k == "VENV_VL_REBUILD" and v not in ("", "0", "1"):   # [2.5.55]
            raise Reject("VENV_VL_REBUILD must be 0, 1 or empty")
        if k == "VENV_IFEVAL_REBUILD" and v not in ("", "0", "1"):   # [2.5.92]
            raise Reject("VENV_IFEVAL_REBUILD must be 0, 1 or empty")
        if k in ("FETCH_CONNS_DAY", "FETCH_CONNS_NIGHT") and v and not (v.isdigit() and 1 <= int(v) <= 128):   # [2.5.50]
            raise Reject(f"{k} must be an integer 1-128 (empty: back to the default 32)")
        if k == "FETCH_DAY_MAX_BYTES" and v and not re.fullmatch(r"[0-9]+(\.[0-9]+)?(e[0-9]{1,2})?", v):
            raise Reject("FETCH_DAY_MAX_BYTES must be a number of bytes such as 1e9 or 1e13 (empty: back to the default 1e13)")
        if k == "DR" and v not in ("", "0", "1"):
            raise Reject("DR must be 0, 1 or empty (v2.5.7: resume state mirrored to the share; default on)")
        if k == "DR_EVERY" and v and not (v.isdigit() and 10 <= int(v) <= 100):
            raise Reject("DR_EVERY must be an integer 10-100 (steps between resume states)")
        if k == "DRILL" and v not in ("", "0", "1"):   # 2.5.29 (v2.5.7zb): fault-injection drills only with DRILL=1
            raise Reject("DRILL must be 0, 1 or empty")
        if k == "REMOTE_FAULT" and v not in ("", "hang_once", "hang_once_hard"):
            raise Reject("REMOTE_FAULT must be hang_once, hang_once_hard or empty")
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
    if verb == "publish-redacted" and not (re.fullmatch(r"[A-Za-z0-9._-]+", args[0]) and re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*", args[1])
                                           and ".." not in args[1].split("/")):   # [v2.5.9z96]
        raise Reject("publish-redacted <task> <file under runs/<task>/>")
    if verb == "scrubdiff" and args and args[0] != "fix":
        raise Reject("scrubdiff [fix]")
    if verb in ("du", "rm-ours"):   # v2.5.7: symbolic prefixes only (control files are public: no paths in them)
        allowed = ("opd_root", "dr", "old_base", "ckpt") if verb == "du" else ("dr", "old_base")   # 2.5.27: du ckpt (read-only)
        if args[0] not in allowed:
            raise Reject(f"{verb} <{'|'.join(allowed)}> {'[' if verb == 'du' else '<'}relative path{']' if verb == 'du' else '>'}")
        if len(args) > 1 and (not re.fullmatch(r"[A-Za-z0-9._/-]+", args[1]) or args[1].startswith("/") or ".." in args[1].split("/")):
            raise Reject("relative path of [A-Za-z0-9._/-], no leading / and no ..")
    if verb == "inventory" and args and args[0] != "pend":   # [2.5.85]
        raise Reject("inventory [pend]")
    if verb == "dr-keep" and not TASK_RE.match(args[0]):
        raise Reject("dr-keep <task>")
    if verb == "pend-weight" and not (TASK_RE.match(args[0]) and args[1].isdigit() and 0 < int(args[1]) < 100000):   # [2.5.96]
        raise Reject("pend-weight <task> <step>")
    if verb == "rerun" and not re.fullmatch(r"FA-PROBE|SPEED-PROBE|PROBE-4B|PROBE-TERM(?:[234]|4-4B|4-06B)?|PROBE-TOP10|STRICT-C1|PROBE-T14|PROBE-T32|PROBE-P5THINK|PROBE-P5SELF|PROBE-P5SELF-4B|PROBE-P5SELF-06B|PROBE-P2SELF|PROBE-E12C|PROBE-E12B-KL|PROBE-E12B-STOP|PROBE-E12B-STOP2|PROBE-E12B-8K|PROBE-E12B-16K|PROBE-E12B-R3|PROBE-X2507-X[124]|PROBE-LENSEL|PROBE-2GPU|PROBE-TOOL|PROBE-NET2?|BUILD-VENV-VL|BUILD-VENV-IFEVAL|PROBE-VLLM-VL|READINGS-EXTRA|PROBE-VLM|E15-VLM-P5|E15-VLM-P2lo|E15-VLM-P5tail-lo|PROBE-VLM-COMPLETE|PROBE-VLM-TAIL|DIAG-VLM-TRUNC|DIAG-T32-TAILS|DIAG-PROBE-GATE|DIAG-WALL-BIMODAL|DIAG-VERIFY-ONSET|DIAG-VLM-LOOP2|VERIFY-PRIV-LFS|PROBE-ROLLOUT-U(?:60|75|85)|PROBE-STOPMERGE(?:-UNIT)?|PROBE-P5-8K|PROBE-SMERGE-8K|E10e-P5self|PROBE-8B|PROBE-WDIFF|TOKCHECK-X5|VENV-VL-NVRTC|PROBE-MIXP|PROBE-PROG|PROBE-ROUTE-INV|PROBE-X5|E16-X5e?|PROBE-TAIL|PROBE-TAIL-BUD|PROBE-TAIL-4B|PROBE-DENSE-4B|PROBE-E12C-8K|PROBE-E12C-16K|CORPUS-RECORD|HEARTBEAT-UNSTICK|PROBE-DIFF17|PROBE-DIFF17-4B|PROBE-DIFFPROBE|PROBE-PSTAR|PROBE-DENSE|PROBE-SELFROUTE|PROBE-SOFTMIX|PROBE-ROUTE|PROBE-POLISH|FA-SPEED-(?:17|4B|E12|TWIN)|EVAL-[A-Za-z0-9._-]+|SFT-[A-Za-z0-9]+|E1[012]b-[A-Za-z0-9._-]+|E12[cde]-[A-Za-z0-9._-]+|E10[cde]-[A-Za-z0-9._-]+|PROBE-E12E-8K|PROBE-E12E-16K|PROBE-TAIL-8B|E0[23]-[A-Za-z0-9._-]+", args[0]):
        raise Reject("rerun <FA-PROBE | SPEED-PROBE | EVAL-name | SFT-* | E10b/E10c/E10d/E10e/E11b/E12b/E12c/E12d/E12e-* | E02/E03-*>")
    if verb == "cancel" and not (args[0].isdigit() and int(args[0]) > 0):
        raise Reject("cancel <seq>")
    flags["confirm_foreign"] = bool(entry.get("confirm_foreign"))
    return verb, args, flags


MAX_TOKENS = ("2048", "4096", "8192", "16384", "32768", "3072")   # v2.5.7g (reviewer 18:4x UTC): budget-dependence readings; [2.5.72] 3072: Q-057's 4096-position student (prompt + 3072)


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
    if len(args) > 3 and args[3] not in ("0.6", "1.0", "0"):   # [2.5.62] 0: greedy
        raise Reject("temperature must be 0.6, 1.0 or 0")
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
