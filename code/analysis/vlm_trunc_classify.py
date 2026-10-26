"""DIAG-VLM-TRUNC (runbook [v2.5.9z89]; Q-080's first diagnostic, Fable's ruling 10-26 23:0x UTC): what the VLM's truncated answers are.

Each truncated sample (finish_reason "length") of a gen_eval jsonl falls in one of three classes:
  loop        an n-gram loop: some window of WINDOW (20) consecutive token ids occurs at least REPEATS (3) times in the response
  unfinished  no complete \\boxed{...} and no loop: reasoning that ran out of room
  box_more    a complete \\boxed{...}, no loop, and the answer goes on after it
(a loop with a box is a loop; loop_with_box counts those). Per input: samples, truncated, the share of each class among the truncated.
Usage: python vlm_trunc_classify.py <out.json> <label>=<jsonl> [...]   (CPU only; reads the files, writes out.json)
"""
import json
import os
import sys

WINDOW = int(os.environ.get("TRUNC_WINDOW", "20"))
REPEATS = int(os.environ.get("TRUNC_REPEATS", "3"))


def has_loop(ids, window=WINDOW, repeats=REPEATS):
    if len(ids) < window * repeats:
        return False
    seen = {}
    for i in range(len(ids) - window + 1):
        k = tuple(ids[i:i + window])
        seen[k] = seen.get(k, 0) + 1
        if seen[k] >= repeats:
            return True
    return False


def box_end(text):
    """The index just after the first complete \\boxed{...} (balanced braces), or -1."""
    i = text.find("\\boxed{")
    while i >= 0:
        j, depth = i + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
        if depth == 0:
            return j
        i = text.find("\\boxed{", i + 1)
    return -1


def classify(rec):
    ids = rec.get("token_ids") or []
    loop = has_loop(ids)
    b = box_end(rec.get("text") or "")
    if loop:
        return "loop", b >= 0
    return ("box_more" if b >= 0 else "unfinished"), False


def summarize(path):
    n = tr = lwb = 0
    cls = {"loop": 0, "unfinished": 0, "box_more": 0}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line); n += 1
            if rec.get("finish_reason") != "length":
                continue
            tr += 1
            c, wb = classify(rec); cls[c] += 1; lwb += wb
    return {"samples": n, "truncated": tr, "truncated_frac": round(tr / max(n, 1), 4), "counts": cls,
            "shares": {k: round(v / max(tr, 1), 4) for k, v in cls.items()}, "loop_with_box": lwb}


def main():
    out, pairs = sys.argv[1], sys.argv[2:]
    res = {"window": WINDOW, "repeats": REPEATS, "inputs": {}}
    for p in pairs:
        label, path = p.split("=", 1)
        res["inputs"][label] = summarize(path)
    json.dump(res, open(out, "w"), indent=1)
    for label, r in res["inputs"].items():
        print(f"TRUNC {label}: {r['truncated']}/{r['samples']} truncated; loop {r['shares']['loop']:.1%}, unfinished {r['shares']['unfinished']:.1%}, box then more {r['shares']['box_more']:.1%}")


if __name__ == "__main__":
    main()
