"""[v2.5.9z101] outcomes.jsonl (Fable's ruling on requirements v26 §11, 10-27 (2): HF keeps no generated text): per reading one small
file, no text, from which accuracies, truncation rates, lengths and the problem-level paired bootstraps can be recomputed when the
raw generations are gone.

  reading   a gen_eval jsonl -> one line per sample: {"id", "s", "ok", "n", "trunc", "box", "answer_norm"} (problem id, sample,
            correct, tokens, ended at the length limit, holds a complete \\boxed{}, the final answer - answer_norm below)
            [z145] (Fable 11-11 D8: the restatement after the student's own </think> is redundant) + two cut readings per sample:
            "nc" tokens before the first </think> (None: none written), "okc" graded on the text cut there (its last box; no </think>:
            the full answer), "nfb" tokens up to the end of the first complete box (None: no box), "okfb" graded on the text cut there.
            Token counts are exact: </think> is one token in token_ids; the box end is mapped through the tokenizer's byte pieces
            (--tokenizer, default $OPD_ROOT/models/Qwen3-1.7B; without one, or when text != the decoded ids, nfb stays None).
  rollouts  a verl rollout dump (<run>/rollouts/<step>.jsonl: input, output, score, step) -> one line per response: {"step", "key",
            "score", "chars", "box", "n", "answer_norm"} (key = diff_labels.key: sha1 of the last user message; n = tokens, with
            --tokenizer)
  backfill  for each listed object of the private repository (the cleanup CSV: tier, task, filename, ...): download it, write its
            outcomes under <outdir>/<task>/, delete the download (a small disk); resumable (an existing output is kept)

Usage: python outcomes.py reading <in.jsonl> <out.jsonl>
       python outcomes.py rollouts <in.jsonl> <out.jsonl> [--tokenizer <dir or id>]
       python outcomes.py backfill <csv> <repo> <outdir> [--tiers D,C] [--tasks a,b] [--tokenizer <dir or id>]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_trunc_classify import box_end  # noqa: E402

USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
WRAP = re.compile(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}")
DROP = re.compile(r"\\left|\\right|\\[!,;:]|\$|\^\{?\\circ\}?|\s+")
ANSWER_MAX = 120


def answer_norm(text):
    """[Fable 10-27 11:4x UTC] The final answer as a short string, no reasoning text: the content of the last complete \\boxed{...}
    (gen_eval grades the last box), with \\text{} / \\mathrm{} unwrapped, \\dfrac / \\tfrac as \\frac, no \\left / \\right / spacing
    commands / $ / degree marks / whitespace / trailing period, cut at ANSWER_MAX characters; None without a complete box."""
    text = text or ""
    i = text.rfind("\\boxed{")
    while i >= 0:
        j, depth = i + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
        if depth == 0:
            a = text[i + len("\\boxed{"):j - 1]
            a = WRAP.sub(r"\1", a).replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
            return DROP.sub("", a).rstrip(".")[:ANSWER_MAX]
        i = text.rfind("\\boxed{", 0, i)
    return None


def rows(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def key(rendered):
    users = USER.findall(rendered or "")
    return hashlib.sha1((users[-1] if users else (rendered or "")).encode("utf-8")).hexdigest()[:16]


CLOSE = "</think>"
CLOSE_ID = 151668   # Qwen3's </think>, one added token (the tokenizer's id wins when one is given)


def byte_pieces(tok):
    """[z145] token id -> its bytes in the decoded text (byte-level BPE; added / special tokens as their literal text)."""
    try:
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    except Exception:
        return None
    dec = {v: k for k, v in bytes_to_unicode().items()}
    added = set(getattr(tok, "added_tokens_decoder", {}) or {})
    cache = {}
    def piece(i):
        if i not in cache:
            t = tok.convert_ids_to_tokens(int(i))
            cache[i] = t.encode("utf-8") if (i in added or t is None or any(ch not in dec for ch in t)) else bytes(dec[ch] for ch in t)
        return cache[i]
    return piece


def token_at(text, ids, char_end, piece):
    """[z145] tokens up to and including the one that holds text[char_end - 1], exactly; None when the pieces do not rebuild text."""
    if piece is None or not ids:
        return None
    tb = text.encode("utf-8"); target, buf = len(text[:char_end].encode("utf-8")), bytearray()
    for k, i in enumerate(ids):   # the prefix must rebuild the text byte for byte (trailing tokens such as EOS may follow)
        buf += piece(i)
        if len(buf) >= target:
            return k + 1 if bytes(buf[:target]) == tb[:target] else None
    return None


def cut_fields(r, piece=None, grade=None, close_id=CLOSE_ID):
    """[z145] {"nc", "okc", "nfb", "okfb"} of one gen_eval record (Fable 11-11 D8)."""
    t, ids, gold, ok = r.get("text") or "", r.get("token_ids") or [], r.get("gold"), r.get("correct")
    def g(x):
        if grade is None:
            return None
        try:
            v = grade(x, gold)
            return bool(v[0] if isinstance(v, tuple) else v)
        except Exception:
            return False
    c = t.find(CLOSE)
    nc = ids.index(close_id) if close_id in ids else None
    if c < 0:
        nc, okc = None, ok
    else:
        if nc is None:   # written as pieces, not the one token: map the character position
            k = token_at(t, ids, c + len(CLOSE), piece); nc = None if k is None else max(0, k - 1)
        okc = g(t[:c])
    e = box_end(t)
    nfb, okfb = (token_at(t, ids, e, piece), g(t[:e])) if e >= 0 else (None, False if grade else None)
    return {"nc": nc, "okc": okc, "nfb": nfb, "okfb": okfb}


def reading(src, out, piece=None, grade=None, close_id=CLOSE_ID):
    n = 0
    with open(out + ".tmp", "w") as f:
        for r in rows(src):
            d = {"id": r.get("pid"), "s": r.get("sample"), "ok": r.get("correct"), "n": r.get("n_tokens"),
                 "trunc": r.get("finish_reason") == "length", "box": box_end(r.get("text") or "") >= 0,
                 "answer_norm": answer_norm(r.get("text"))}
            d.update(cut_fields(r, piece, grade, close_id))
            f.write(json.dumps(d) + "\n"); n += 1
    os.replace(out + ".tmp", out); return n


def reading_tools(tokenizer=None):
    """[z145] (piece, grade, close_id) for the CLI: the tokenizer (default $OPD_ROOT/models/Qwen3-1.7B) and gen_eval.grade, each
    None when not loadable (the fields that need it stay None)."""
    name = tokenizer or os.path.join(os.environ.get("OPD_ROOT", ""), "models", "Qwen3-1.7B")
    piece, close_id = None, CLOSE_ID
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        piece = byte_pieces(tok); cid = tok.convert_tokens_to_ids(CLOSE)
        close_id = cid if isinstance(cid, int) and cid >= 0 else CLOSE_ID
    except Exception:
        pass
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
        from gen_eval import grade
    except Exception:
        grade = None
    return piece, grade, close_id


def rollouts(src, out, count=None):
    recs = list(rows(src))
    ns = count([r.get("output") or "" for r in recs]) if count else [None] * len(recs)
    with open(out + ".tmp", "w") as f:
        for r, k in zip(recs, ns):
            o = r.get("output") or ""
            f.write(json.dumps({"step": r.get("step"), "key": key(r.get("input")), "score": r.get("score"), "chars": len(o),
                                "box": box_end(o) >= 0, "n": k, "answer_norm": answer_norm(o)}) + "\n")
    os.replace(out + ".tmp", out); return len(recs)


def kind(filename):
    return "rollouts" if "/rollouts/" in filename else "reading"


def out_path(outdir, task, filename):
    base = os.path.basename(filename)[:-len(".jsonl")]
    sub = "rollouts" if kind(filename) == "rollouts" else ""
    return os.path.join(outdir, task, sub, base + ".outcomes.jsonl")


def backfill(csv_path, repo, outdir, tiers=None, tasks=None, count=None, fetch=None):
    """fetch(repo, filename, tmpdir) -> local path; the default downloads from the Hub. Returns {done, kept, failed}."""
    if fetch is None:
        from huggingface_hub import hf_hub_download
        fetch = lambda rp, fn, d: hf_hub_download(rp, fn, cache_dir=d)
    res = {"done": 0, "kept": 0, "failed": []}
    os.makedirs(outdir, exist_ok=True); log = open(os.path.join(outdir, "backfill_log.jsonl"), "a")
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if tiers and r["tier"][0] not in tiers or tasks and r["task"] not in tasks:
            continue
        out = out_path(outdir, r["task"], r["filename"])
        if os.path.exists(out):
            res["kept"] += 1; continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = tempfile.mkdtemp(dir=os.environ.get("OUTCOMES_TMP"))
        try:
            src = fetch(repo, r["filename"], tmp)
            n = rollouts(src, out, count) if kind(r["filename"]) == "rollouts" else reading(src, out)
            res["done"] += 1; log.write(json.dumps({"filename": r["filename"], "lines": n, "out": os.path.relpath(out, outdir)}) + "\n"); log.flush()
        except Exception as e:
            res["failed"].append({"filename": r["filename"], "why": type(e).__name__})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)   # the download goes at once (a 50 GB disk)
    return res


def tokenizer_count(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    return lambda texts: [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]] if texts else []


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("args", nargs="+")
    ap.add_argument("--tokenizer"); ap.add_argument("--tiers"); ap.add_argument("--tasks")
    a = ap.parse_args(argv)
    count = tokenizer_count(a.tokenizer) if a.tokenizer and a.cmd != "reading" else None
    if a.cmd == "reading":
        piece, grade, cid = reading_tools(a.tokenizer)
        print(f"OUTCOMES {reading(*a.args, piece=piece, grade=grade, close_id=cid)} lines")
    elif a.cmd == "rollouts":
        print(f"OUTCOMES {rollouts(*a.args, count)} lines")
    elif a.cmd == "backfill":
        r = backfill(*a.args, tiers=set(a.tiers.split(",")) if a.tiers else None, tasks=set(a.tasks.split(",")) if a.tasks else None, count=count)
        print(f"BACKFILL done {r['done']}, kept {r['kept']}, failed {len(r['failed'])}" + (f": {r['failed'][:5]}" if r["failed"] else ""))
        return 1 if r["failed"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
