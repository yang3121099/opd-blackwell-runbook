"""Per-prompt teacher template for OPD (runbook [v2.5.9x], Q-041 E10b-P5route / Q-042 E10b-P5mix; research_program §3c).

The 8B teacher scores each response ONCE, in one of two templates chosen per prompt: the thinking template with '<think>\\n'
(P5's teacher) or the non-thinking one (P2's teacher). Hooked into G-OPD's prepare_ref_model_inputs, where the P5 patch already
sets `_think` per sample (route_patch.py; a no-op unless OPD_ROUTE is set):
  route: the prompt's pass rate r from PROBE-DIFF17 (the 1.7B base, non-thinking, k=8): r >= 0.5 (easy) -> non-thinking teacher,
         otherwise (hard) -> thinking teacher; [v2.5.9z45] (Q-063, the reviewer 10-22, E10b-P5route-inv) OPD_ROUTE_INVERT=1 reverses
         it: r >= 0.5 (easy) -> thinking teacher, otherwise (hard) -> non-thinking; the log line then says "invert": true;
  mix:   a fixed coin per prompt (sha1 of OPD_SEED and the prompt's key, p = 0.5): heads -> thinking teacher; the same prompt always
         gets the same teacher, independent of the labels. [v2.5.9z38] (Q-060, the reviewer 10-22) OPD_ROUTE_P = p: thinking teacher
         when int(sha1(...)[:8], 16) / 2**32 < p (E10b-P5mix75 / -P5mix90: 0.75 / 0.9; the prompts with the thinking teacher at a
         smaller p are a subset of those at a larger one); unset: the parity coin E10b-P5mix ran with (p = 0.5), unchanged.
  count: [v2.5.9z9] (Q-053, PROBE-T32's smoke) the template unchanged (the P5 patch's own choice, OPD_REF_THINK), the call only
         logged: the teacher's input rows per call against the batch.
  prefix: [v2.5.9z108] (Q-083 E10b-P5mode, research_program §3i; beyond_mainline §1) the student's prompt carries a visible mode
         token at the start of its last user message, '[long] ' or '[short] ' (mode_data.py writes them into the training parquet
         with mix's coin): '[long]' -> thinking teacher (P5's), '[short]' -> non-thinking teacher (P2's); a prompt without a token
         -> thinking teacher, counted unknown (PROBE-MODE requires none). The decision's key is the prompt WITHOUT the token (the
         same key as mix's coin and the labels). teacher_view(messages) gives the teacher the prompt without the token unless
         OPD_MODE_STRIP=0 (mode_patch.py's hook, right after think_for); the log row holds the token instead of r.
A prompt without a label is routed to the thinking teacher (P5's default) and counted as unknown; PROBE-ROUTE requires none.
Logged per call (one call per training step: the reference inputs of the whole batch), OPD_ROUTE_LOG JSON lines: rows, think,
plain, unknown, and per row [key, r, think].

Environment: OPD_ROUTE route|mix; OPD_ROUTE_LABELS (labels_17.jsonl of PROBE-DIFF17); OPD_ROUTE_THRESH (0.5); OPD_SEED (1234);
OPD_ROUTE_P ([v2.5.9z38] mix's p, unset = the parity coin); OPD_ROUTE_INVERT ([v2.5.9z45] 1: route reversed; mix / count ignore it);
OPD_ROUTE_LOG (path, optional); OPD_MODE_STRIP ([v2.5.9z108] prefix mode: 0 leaves the token in the teacher's prompt; default 1).
"""
import hashlib
import json
import os
import time

MODES = ("route", "mix", "count", "prefix")
TOKENS = {"[long]": True, "[short]": False}   # [v2.5.9z108] the mode token -> thinking teacher
_labels = None
_rows = []
_calls = 0
_stripped = 0   # [v2.5.9z108] prompts teacher_view gave the teacher without their token, this call


def key(messages):
    """The same key as diff_labels.key: sha1 (16 hex) of the last user message's content."""
    msgs = list(messages)
    users = [m for m in msgs if (m.get("role") if isinstance(m, dict) else None) == "user"]
    text = users[-1]["content"] if users else json.dumps(msgs, sort_keys=True, default=str)
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]


def load_labels(path):
    """{key: r} of the training prompts (split train) in a labels_17.jsonl."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "train":
                out[r["key"]] = float(r["r"])
    return out


def coin(k, seed, p=None):
    """mix: True (thinking teacher), fixed per (seed, prompt): about half the prompts by parity (p None, E10b-P5mix), or ([v2.5.9z38])
    the prompts whose hash fraction is below p."""
    h = int(hashlib.sha1(f"{seed}:{k}".encode()).hexdigest()[:8], 16)
    return h % 2 == 0 if p is None else h / 2 ** 32 < p


def decide(k, mode, labels, thresh=0.5, seed=1234, p=None, invert=False):
    """(think, r or None): the template for the prompt with key k ([v2.5.9z45] invert: route's rule reversed; unknown stays thinking)."""
    if mode not in MODES:
        raise ValueError(f"OPD_ROUTE must be one of {MODES}, got {mode!r}")
    r = labels.get(k)
    if mode == "mix":
        return coin(k, seed, p), r
    if r is None:
        return True, None
    return (r >= thresh if invert else r < thresh), r


def split_prefix(text):
    """[v2.5.9z108] (token or None, the text without it): a token is '[long]' or '[short]' followed by one space at the very start."""
    t = str(text)
    for tok in TOKENS:
        if t.startswith(tok + " "):
            return tok, t[len(tok) + 1:]
    return None, t


def prefixed(text, think):
    """[v2.5.9z108] the prompt text with its mode token: '[long] ' (thinking teacher) or '[short] ' (non-thinking)."""
    return ("[long] " if think else "[short] ") + str(text)


def _last_user(messages):
    msgs = list(messages)
    for i in range(len(msgs) - 1, -1, -1):
        if (msgs[i].get("role") if isinstance(msgs[i], dict) else None) == "user":
            return msgs, i
    return msgs, None


def strip_messages(messages):
    """[v2.5.9z108] a NEW list whose last user message (a new dict) has no mode token; the input is never changed (the dataset's
    rows may be shared with the student's side). Unchanged copy when there is no token."""
    msgs, i = _last_user(messages)
    if i is None:
        return msgs
    tok, rest = split_prefix(msgs[i].get("content", ""))
    if tok is None:
        return msgs
    msgs[i] = dict(msgs[i], content=rest)
    return msgs


def teacher_view(messages):
    """[v2.5.9z108] mode_patch.py's hook: in prefix mode the teacher scores the prompt without the token (OPD_MODE_STRIP=0 keeps it)."""
    global _stripped
    if os.environ.get("OPD_ROUTE") != "prefix" or os.environ.get("OPD_MODE_STRIP", "1") == "0":
        return messages
    out = strip_messages(messages)
    _stripped += int(out != list(messages))
    return out


def inverted():
    """[v2.5.9z45] route reversed (OPD_ROUTE_INVERT=1, route mode only)."""
    return os.environ.get("OPD_ROUTE") == "route" and os.environ.get("OPD_ROUTE_INVERT") == "1"


def think_for(messages):
    """The hook, called once per sample inside prepare_ref_model_inputs: True = thinking template."""
    global _labels
    mode = os.environ.get("OPD_ROUTE", "")
    if _labels is None:
        p = os.environ.get("OPD_ROUTE_LABELS", "")
        _labels = load_labels(p) if mode == "route" or (p and os.path.exists(p)) else {}
    k = key(messages)
    if mode == "count":   # [v2.5.9z9] P5's template as without the hook
        think, r = os.environ.get("OPD_REF_THINK") == "1", None
    elif mode == "prefix":   # [v2.5.9z108] the token decides; the key and the log are of the prompt without it
        msgs, i = _last_user(messages)
        tok = split_prefix(msgs[i].get("content", ""))[0] if i is not None else None
        k = key(strip_messages(messages))
        think, r = TOKENS.get(tok, True), tok
    else:
        rp = os.environ.get("OPD_ROUTE_P")   # [v2.5.9z38]
        think, r = decide(k, mode, _labels, float(os.environ.get("OPD_ROUTE_THRESH", "0.5")), int(os.environ.get("OPD_SEED", "1234")),
                          float(rp) if rp else None, inverted())
    _rows.append([k, r, bool(think)])
    return think


def flush(n_batch):
    """At the end of prepare_ref_model_inputs: one log line for the call, the counts reset."""
    global _rows, _calls, _stripped
    _calls += 1
    rows, _rows = _rows, []
    stripped, _stripped = _stripped, 0
    rec = {"call": _calls, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": os.environ.get("OPD_ROUTE", ""),
           "batch": int(n_batch), "rows": len(rows), "think": sum(1 for x in rows if x[2]), "plain": sum(1 for x in rows if not x[2]),
           "unknown": sum(1 for x in rows if x[1] is None) if os.environ.get("OPD_ROUTE") in ("route", "prefix") else 0, "decisions": rows}
    if inverted():
        rec["invert"] = True   # [v2.5.9z45]
    if os.environ.get("OPD_ROUTE") == "prefix":
        rec["stripped"] = stripped   # [v2.5.9z108] rows the teacher saw without their token (PROBE-MODE: = rows)
    log = os.environ.get("OPD_ROUTE_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec
