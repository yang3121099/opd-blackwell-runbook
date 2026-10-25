"""[v2.6 S0] the golden corpus's normalization and leak check (v26_impl_plan §2.7).

normalize(obj, ctx) rewrites every string in a record so two recordings of the same behaviour compare equal and no path,
host or user name is kept: the sandbox prefixes -> ${OPD_ROOT} / ${CKPT_ROOT} / ${HOME} / ${HF_HOME} / ${DR_ROOT} / ${SB};
the real runbook -> ${RUNBOOK}; GPU lists (CUDA_VISIBLE_DEVICES, the task's GPU argument) -> G1 / G1,G2 ...; the pinned
clock's epochs and ISO times -> ${T0}+n. leak_hits(text) lists what the lab's leak scans look for; the recorder refuses to
upload a corpus with any hit (findings.json names them).

stdlib only; used by dryrun_json.py and its tests."""
import json
import os
import re
import time

# generic patterns only (no machine's name is written into the runbook): this machine's own names - the user, the host,
# its real roots - come in at run time as literals (machine_literals; the same idea as control.py's _forbidden_tokens)
LEAK = [
    (r"/(home|root|Users)/[A-Za-z0-9._-]+", "a home directory"),
    (r"\bhf_[A-Za-z0-9]{20,}", "an HF token"),
    (r"\bghp_[A-Za-z0-9]{20,}", "a GitHub token"),
    (r"(?i)password\s*[=:]", "a password"),
    (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "an IP address"),
]


def leak_hits(text, extra=()):
    """[(what, match)] for every leak pattern in text; extra = literal strings (the node id, user and host names)."""
    out = []
    for pat, what in LEAK:
        for m in re.finditer(pat, text):
            out.append((what, m.group(0)))
    for s in extra:
        if s and s in text:
            out.append(("a literal", s))
    return out


def machine_literals(paths):
    """the strings of this machine that must not appear in the corpus: its real roots, the user and the host names."""
    out = {p for p in paths if p and len(p) > 1}
    out |= {os.path.realpath(p) for p in paths if p and len(p) > 1}
    try:
        import getpass
        out.add(getpass.getuser())
    except Exception:
        pass
    import socket
    h = socket.gethostname()   # (not getfqdn: a reverse lookup can hang for seconds)
    if h and h not in ("localhost", "localhost.localdomain"):
        out.add(h)
    return sorted(x for x in out if len(x) >= 5)   # shorter ones are generic (control.py's rule)


class Ctx:
    """the prefixes and values one recording rewrites; longest prefix first."""
    def __init__(self, sb, opd, ckpt, home, hf, dr, runbook, gpu_arg, t0, extra_literals=()):
        self.pairs = sorted([(p, tok) for p, tok in ((opd, "${OPD_ROOT}"), (ckpt, "${CKPT_ROOT}"), (home, "${HOME}"),
                                                       (hf, "${HF_HOME}"), (dr, "${DR_ROOT}"), (runbook, "${RUNBOOK}"),
                                                       (sb, "${SB}")) if p],
                            key=lambda x: -len(x[0]))
        real = [(os.path.realpath(p), tok) for p, tok in self.pairs]
        self.pairs = sorted(set(self.pairs + [(p, t) for p, t in real if p]), key=lambda x: -len(x[0]))
        self.gpu_arg = gpu_arg
        self.t0 = t0
        self.extra = [x for x in extra_literals if x]

    def gpus(self, s):
        """'0' -> 'G1', '3,5' -> 'G1,G2': the task's GPU list by position (the node picks the GPUs, the recipe their count)."""
        parts = s.split(",")
        return ",".join(f"G{i + 1}" for i in range(len(parts)))

    def text(self, s):
        for p, tok in self.pairs:
            s = s.replace(p, tok)
        for x in self.extra:
            s = s.replace(x, "${LITERAL}")
        s = re.sub(r"\b1[67]\d{8}\b", lambda m: self._epoch(int(m.group(0))), s)
        s = re.sub(r"\b20\d\d-\d\d-\d\dT\d\d:\d\d:\d\dZ\b", lambda m: self._iso(m.group(0)), s)
        return s

    def _epoch(self, n):
        d = n - self.t0
        return f"${{T0}}+{d}" if 0 <= d < 10 ** 6 else str(n)

    def _iso(self, s):
        try:
            n = int(time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone)
        except (ValueError, OverflowError):
            return s
        d = n - self.t0
        return f"${{T0}}+{d}" if 0 <= d < 10 ** 6 else s


def normalize(obj, ctx, key=None):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "CUDA_VISIBLE_DEVICES" and isinstance(v, str) and re.fullmatch(r"[0-9,]+", v):
                out[k] = ctx.gpus(v)
            else:
                out[k] = normalize(v, ctx, k)
        return out
    if isinstance(obj, list):
        return [normalize(x, ctx, key) for x in obj]
    if isinstance(obj, str):
        return ctx.text(obj)
    return obj


def argv_gpu(argv, gpu_arg, ctx):
    """a runner's argv with the task's GPU argument (its first positional) normalized."""
    return [ctx.gpus(a) if a == gpu_arg and re.fullmatch(r"[0-9,]+", a) else a for a in argv]


def dumps(obj):
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
