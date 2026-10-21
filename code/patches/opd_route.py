"""Per-prompt teacher template for OPD (runbook [v2.5.9x], Q-041 E10b-P5route / Q-042 E10b-P5mix; research_program §3c).

The 8B teacher scores each response ONCE, in one of two templates chosen per prompt: the thinking template with '<think>\\n'
(P5's teacher) or the non-thinking one (P2's teacher). Hooked into G-OPD's prepare_ref_model_inputs, where the P5 patch already
sets `_think` per sample (route_patch.py; a no-op unless OPD_ROUTE is set):
  route: the prompt's pass rate r from PROBE-DIFF17 (the 1.7B base, non-thinking, k=8): r >= 0.5 (easy) -> non-thinking teacher,
         otherwise (hard) -> thinking teacher;
  mix:   a fixed coin per prompt (sha1 of OPD_SEED and the prompt's key, p = 0.5): heads -> thinking teacher; the same prompt always
         gets the same teacher, independent of the labels.
  count: [v2.5.9z9] (Q-053, PROBE-T32's smoke) the template unchanged (the P5 patch's own choice, OPD_REF_THINK), the call only
         logged: the teacher's input rows per call against the batch.
A prompt without a label is routed to the thinking teacher (P5's default) and counted as unknown; PROBE-ROUTE requires none.
Logged per call (one call per training step: the reference inputs of the whole batch), OPD_ROUTE_LOG JSON lines: rows, think,
plain, unknown, and per row [key, r, think].

Environment: OPD_ROUTE route|mix; OPD_ROUTE_LABELS (labels_17.jsonl of PROBE-DIFF17); OPD_ROUTE_THRESH (0.5); OPD_SEED (1234);
OPD_ROUTE_LOG (path, optional).
"""
import hashlib
import json
import os
import time

MODES = ("route", "mix", "count")
_labels = None
_rows = []
_calls = 0


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


def coin(k, seed):
    """mix: True (thinking teacher) for about half the prompts, fixed per (seed, prompt)."""
    return int(hashlib.sha1(f"{seed}:{k}".encode()).hexdigest()[:8], 16) % 2 == 0


def decide(k, mode, labels, thresh=0.5, seed=1234):
    """(think, r or None): the template for the prompt with key k."""
    if mode not in MODES:
        raise ValueError(f"OPD_ROUTE must be one of {MODES}, got {mode!r}")
    r = labels.get(k)
    if mode == "mix":
        return coin(k, seed), r
    if r is None:
        return True, None
    return r < thresh, r


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
    else:
        think, r = decide(k, mode, _labels, float(os.environ.get("OPD_ROUTE_THRESH", "0.5")), int(os.environ.get("OPD_SEED", "1234")))
    _rows.append([k, r, bool(think)])
    return think


def flush(n_batch):
    """At the end of prepare_ref_model_inputs: one log line for the call, the counts reset."""
    global _rows, _calls
    _calls += 1
    rows, _rows = _rows, []
    rec = {"call": _calls, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": os.environ.get("OPD_ROUTE", ""),
           "batch": int(n_batch), "rows": len(rows), "think": sum(1 for x in rows if x[2]), "plain": sum(1 for x in rows if not x[2]),
           "unknown": sum(1 for x in rows if x[1] is None) if os.environ.get("OPD_ROUTE") == "route" else 0, "decisions": rows}
    log = os.environ.get("OPD_ROUTE_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec
