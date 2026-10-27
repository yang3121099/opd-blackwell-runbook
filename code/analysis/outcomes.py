"""[v2.5.9z101] outcomes.jsonl (Fable's ruling on requirements v26 §11, 10-27 (2): HF keeps no generated text): per reading one small
file, no text, from which accuracies, truncation rates, lengths and the problem-level paired bootstraps can be recomputed when the
raw generations are gone.

  reading   a gen_eval jsonl -> one line per sample: {"id", "s", "ok", "n", "trunc", "box"} (problem id, sample, correct, tokens,
            ended at the length limit, holds a complete \\boxed{})
  rollouts  a verl rollout dump (<run>/rollouts/<step>.jsonl: input, output, score, step) -> one line per response: {"step", "key",
            "score", "chars", "box", "n"} (key = diff_labels.key: sha1 of the last user message; n = tokens, with --tokenizer)
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


def rows(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def key(rendered):
    users = USER.findall(rendered or "")
    return hashlib.sha1((users[-1] if users else (rendered or "")).encode("utf-8")).hexdigest()[:16]


def reading(src, out):
    n = 0
    with open(out + ".tmp", "w") as f:
        for r in rows(src):
            f.write(json.dumps({"id": r.get("pid"), "s": r.get("sample"), "ok": r.get("correct"), "n": r.get("n_tokens"),
                                "trunc": r.get("finish_reason") == "length", "box": box_end(r.get("text") or "") >= 0}) + "\n"); n += 1
    os.replace(out + ".tmp", out); return n


def rollouts(src, out, count=None):
    recs = list(rows(src))
    ns = count([r.get("output") or "" for r in recs]) if count else [None] * len(recs)
    with open(out + ".tmp", "w") as f:
        for r, k in zip(recs, ns):
            o = r.get("output") or ""
            f.write(json.dumps({"step": r.get("step"), "key": key(r.get("input")), "score": r.get("score"), "chars": len(o),
                                "box": box_end(o) >= 0, "n": k}) + "\n")
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
    count = tokenizer_count(a.tokenizer) if a.tokenizer else None
    if a.cmd == "reading":
        print(f"OUTCOMES {reading(*a.args)} lines")
    elif a.cmd == "rollouts":
        print(f"OUTCOMES {rollouts(*a.args, count)} lines")
    elif a.cmd == "backfill":
        r = backfill(*a.args, tiers=set(a.tiers.split(",")) if a.tiers else None, tasks=set(a.tasks.split(",")) if a.tasks else None, count=count)
        print(f"BACKFILL done {r['done']}, kept {r['kept']}, failed {len(r['failed'])}" + (f": {r['failed'][:5]}" if r["failed"] else ""))
        return 1 if r["failed"] else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
