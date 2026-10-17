"""EVAL-v1 generation + grading (see lab/protocols/EVAL-v1.md).

Generates n samples per problem with vLLM, grades with Math-Verify, and keeps every
raw sample plus the fields needed for termination analysis (E05/E11).

Example:
  python gen_eval.py --model Qwen/Qwen3-0.6B --bench math500 --mode no_think \
      --n 4 --temperature 0.7 --top_p 0.8 --top_k 20 --max_tokens 16384 --out runs/g1
"""
import argparse
import json
import math
import os
import re
import time
from collections import defaultdict

PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# (hf repo, split, problem field, answer field). First one that loads wins.
BENCH_SOURCES = {
    "math500": [("HuggingFaceH4/MATH-500", "test", "problem", "answer")],
    "aime24": [("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
               ("math-ai/aime24", "test", "problem", "solution")],
    "aime25": [("math-ai/aime25", "test", "problem", "answer"),
               ("yentinglin/aime_2025", "train", "problem", "answer"),
               ("opencompass/AIME2025", "test", "question", "answer")],
    "amc23": [("math-ai/amc23", "test", "question", "answer"),
              ("AI-MO/aimo-validation-amc", "train", "problem", "answer")],
    # evaluation suite v2 (E11 amendment 2, 2026-10-17): OlympiadBench text-only English maths competition subset
    # (674 problems; final_answer is a list such as "['$\\frac{1}{2}$']") and Minerva Math (272 problems)
    # (repo, split, problem field, answer field, subset, revision): revisions pinned (E11 amendment 2)
    "olympiadbench": [("Hothan/OlympiadBench", "train", "question", "final_answer", "OE_TO_maths_en_COMP",
                       "91184b52131e7fc9455fef848035173aea8cc01a")],
    "minervamath": [("math-ai/minervamath", "test", "question", "answer", None, "ee46ddc498933b1977577953250ca5c66be64f96")],
}


def olympiad_answer(raw):
    """OlympiadBench final_answer: a Python list as text; drop every $ and join several items with ', ' (Math-Verify then
    compares the tuple)."""
    import ast
    try:
        items = ast.literal_eval(raw) if isinstance(raw, str) else list(raw)
    except (ValueError, SyntaxError):
        items = [raw]
    return ", ".join(str(a).replace("$", "").strip() for a in items)   # every $ goes (one gold is '$221$,$8$')


def load_bench(name, limit=None):
    from datasets import load_dataset
    errors = []
    for src in BENCH_SOURCES[name]:
        repo, split, pf, af = src[:4]
        cfg = src[4] if len(src) > 4 else None
        rev = src[5] if len(src) > 5 else None
        try:
            ds = load_dataset(repo, cfg, split=split, revision=rev) if (cfg or rev) else load_dataset(repo, split=split)
        except Exception as e:  # try next source
            errors.append(f"{repo}: {e}")
            continue
        rows = []
        for i, ex in enumerate(ds):
            ans = str(ex[af]).strip()
            if name == "aime24" and af == "solution":
                m = re.findall(r"\\boxed\{([^}]*)\}", ans)
                ans = m[-1] if m else ans
            if name == "olympiadbench":
                ans = olympiad_answer(ex[af])
            if name == "minervamath":
                ans = ans.replace("$", "").strip()
            rows.append({"pid": f"{name}-{i}", "problem": ex[pf], "answer": ans,
                         "level": ex.get("level"), "source": repo})
        return rows[:limit] if limit else rows
    raise RuntimeError(f"no source loaded for {name}: {errors}")


GRADER_VERSION = 2  # v2 (R-001 items 6-7): last-boxed only, no boxed -> wrong; answer-context matching


def last_boxed(text):
    i = text.rfind("\\boxed{")
    return text[i:] if i >= 0 else None


def grade(text, gold):
    """Grade ONLY the last \\boxed{...}; responses without a boxed answer are wrong (EVAL-v1)."""
    from math_verify import parse, verify
    seg = last_boxed(text)
    if seg is None:
        return False, None
    try:
        g = parse(f"${gold}$")
        p = parse(seg)
        ok = bool(verify(g, p)) if p else False
        pred = str(p[-1]) if p else None
    except Exception:
        ok, pred = False, None
    return ok, pred


def first_correct_pos(text, gold, loose=False):
    """Char fraction where the gold value first appears IN AN ANSWER CONTEXT.
    strict (default, used for E05 classes): \\boxed{gold}, 'answer is gold' / 'answer: gold'.
    loose: additionally 'gold' closing an equation ('= gold$'), which also hits intermediate steps
    (manual check of 20 contexts in Q-001 reply). Bare numbers inside computations never count."""
    g = gold.strip()
    n = max(len(text), 1)
    if re.fullmatch(r"-?\d+", g):
        num = rf"(?<![\d.]){re.escape(g)}(?![\d.])"
        pats = [rf"\\boxed\{{\s*{num}\s*\}}",
                rf"[Aa]nswer(?:\s+is|:)\**\s*\$?\\?\(?\s*{num}"]
        if loose:
            pats.append(rf"=\s*{num}\s*(?:\$|\\\]|\\\)|\.\s|\n|$)")
        hits = [m.start() for p in pats for m in re.finditer(p, text)]
        return round(min(hits) / n, 4) if hits else None
    for m in re.finditer(r"\\boxed\{", text):
        ok, _ = grade(text[m.start(): m.start() + 200], gold)
        if ok:
            return round(m.start() / n, 4)
    return None


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def make_record(r, si, c, mode):
    text = c.text
    ok, pred = grade(text, r["answer"])
    trunc = c.finish_reason == "length"
    return {"pid": r["pid"], "sample": si, "level": r["level"], "gold": r["answer"],
            "pred": pred, "correct": ok, "n_tokens": len(c.token_ids),
            "finish_reason": c.finish_reason, "terminated": not trunc,
            "think_closed": ("</think>" in text) if mode == "think" else None,
            "first_correct_pos": first_correct_pos(text, r["answer"]),
            "first_value_pos_loose": first_correct_pos(text, r["answer"], loose=True),
            "grader_version": GRADER_VERSION, "token_ids": list(c.token_ids), "text": text}


def compute_metrics(args, rows, recs, gen_s):
    per_prob = defaultdict(list)
    by_level = defaultdict(list)
    n_tok, n_trunc = [], 0
    for rec in recs:
        per_prob[rec["pid"]].append(rec["correct"])
        if rec["level"] is not None:
            by_level[rec["level"]].append((rec["correct"], rec["n_tokens"]))
        n_tok.append(rec["n_tokens"])
        n_trunc += rec["finish_reason"] == "length"
    acc = sum(sum(v) for v in per_prob.values()) / sum(len(v) for v in per_prob.values())
    return {
        "model": args.model, "bench": args.bench, "mode": args.mode, "n": args.n,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "max_tokens": args.max_tokens, "num_problems": len(rows), "avg_acc": round(100 * acc, 2),
        "pass_at_k": {k: round(100 * sum(pass_at_k(len(v), sum(v), k) for v in per_prob.values()) / len(per_prob), 2)
                      for k in [1, 2, 4, 8, 16, 32, 64] if k <= args.n},
        "mean_tokens": round(sum(n_tok) / len(n_tok), 1), "trunc_rate": round(n_trunc / len(n_tok), 4),
        "by_level": {str(l): {"acc": round(100 * sum(x for x, _ in v) / len(v), 2),
                              "mean_tokens": round(sum(t for _, t in v) / len(v), 1)}
                     for l, v in sorted(by_level.items())},
        "gen_seconds": round(gen_s, 1), "source": rows[0]["source"],
    }


# --chunk K (reviewer 03:3x / Runming, 2026-10-17: a preempted spot box must not lose a whole evaluation): problems are
# generated K at a time in fixed blocks (0..K-1, K..2K-1, ...); each finished block is appended to <stem>.jsonl.part and
# flushed, and <stem>.jsonl.part.meta records the arguments and the generation seconds so far. A rerun with the same
# arguments keeps every complete block, drops a block that was only partly written, and generates the rest; the last
# block renames .part to <stem>.jsonl and writes the metrics exactly as an unchunked run does. Blocks never change size,
# so every problem is generated in the same batch as in an uninterrupted chunked run.
RESUME_KEYS = ["model", "bench", "mode", "n", "temperature", "top_p", "top_k", "max_tokens", "limit", "seed", "chunk"]


def resume_state(part, meta, args, rows, K):
    want = {k: getattr(args, k) for k in RESUME_KEYS}
    if not os.path.exists(part):
        return [], 0.0
    m = json.load(open(meta)) if os.path.exists(meta) else None
    if not m or m.get("args") != want:
        raise SystemExit(f"RESUME_REFUSED {part}: written with other arguments ({m and m.get('args')}); move it aside first")
    recs = []
    for line in open(part):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:   # a line cut by the kill
            break
    by_pid = defaultdict(int)
    for r in recs:
        by_pid[r["pid"]] += 1
    blocks = 0
    for b in range(0, len(rows), K):
        if all(by_pid.get(r["pid"], 0) == args.n for r in rows[b:b + K]):
            blocks += 1
        else:
            break
    keep_pids = {r["pid"] for r in rows[:blocks * K]}
    recs = [r for r in recs if r["pid"] in keep_pids]
    with open(part + ".tmp", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(part + ".tmp", part)
    return recs, float(m.get("gen_seconds_done", 0.0)) if blocks else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True, choices=list(BENCH_SOURCES))
    ap.add_argument("--mode", required=True, choices=["think", "no_think"])
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0, help="problems per resumable block (0: one pass, no resume)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_bench(args.bench, args.limit)
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": r["problem"] + PROMPT_SUFFIX}]
        kw = {"enable_thinking": args.mode == "think"}
        try:
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))
        except TypeError:  # templates without enable_thinking
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    os.makedirs(args.out, exist_ok=True)
    stem = f"{args.tag or os.path.basename(args.model.rstrip('/'))}_{args.bench}_{args.mode}_t{args.temperature}"
    final = os.path.join(args.out, stem + ".jsonl")
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_tokens + 2048, seed=args.seed, trust_remote_code=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k if args.top_k > 0 else -1, max_tokens=args.max_tokens, seed=args.seed)
    if args.chunk <= 0:
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        gen_s = time.time() - t0
        recs = [make_record(r, si, c, args.mode) for r, o in zip(rows, outs) for si, c in enumerate(o.outputs)]
        with open(final, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        K, part = args.chunk, final + ".part"
        meta = part + ".meta"
        recs, gen_s = resume_state(part, meta, args, rows, K)
        done_rows = len({r["pid"] for r in recs})
        if not os.path.exists(meta):   # before the first block, so a .part never exists without its arguments
            with open(meta, "w") as f:
                json.dump({"args": {k: getattr(args, k) for k in RESUME_KEYS}, "gen_seconds_done": 0.0, "problems_done": 0}, f)
            open(part, "a").close()
        if done_rows:
            print(f"RESUME {stem}: {done_rows}/{len(rows)} problems kept from {part}", flush=True)
        for b in range(done_rows, len(rows), K):
            t0 = time.time()
            outs = llm.generate(prompts[b:b + K], sp)
            gen_s += time.time() - t0
            block = [make_record(r, si, c, args.mode) for r, o in zip(rows[b:b + K], outs) for si, c in enumerate(o.outputs)]
            with open(part, "a") as f:
                for rec in block:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            recs += block
            with open(meta + ".tmp", "w") as f:
                json.dump({"args": {k: getattr(args, k) for k in RESUME_KEYS}, "gen_seconds_done": gen_s,
                           "problems_done": min(b + K, len(rows))}, f)
            os.replace(meta + ".tmp", meta)
            print(f"BLOCK {stem}: {min(b + K, len(rows))}/{len(rows)} problems", flush=True)
        os.replace(part, final)
        os.remove(meta)

    metrics = compute_metrics(args, rows, recs, gen_s)
    with open(os.path.join(args.out, stem + ".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ["model", "bench", "mode", "avg_acc", "mean_tokens", "trunc_rate", "gen_seconds"]}), flush=True)


def _reap_children(grace=5):
    """End every descendant of this process (vLLM's engine core and its workers). os._exit alone leaves the engine core
    running as an orphan that keeps the GPU (B300 2026-10-17: ~248 GB each; the next engine on that GPU failed)."""
    import signal
    if not os.path.isdir("/proc"):
        return []
    kids = defaultdict(list)
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                kids[int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])].append(int(p))
            except (OSError, ValueError, IndexError):
                pass
    out, todo = [], [os.getpid()]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c); todo.append(c)
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0)):
        for c in out:
            try:
                os.kill(c, sig)
            except OSError:
                pass
        if out:
            time.sleep(wait)
    return out


if __name__ == "__main__":
    main()
    # vLLM v1 can hang on interpreter shutdown while holding the GPU (seen on the 2x H100, 11 min idle).
    # All outputs are written and closed above, so exit hard, after ending vLLM's engine-core child (2026-10-18).
    import sys as _sys
    _sys.stdout.flush(); _sys.stderr.flush()
    _reap_children()
    os._exit(0)
