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
}


def load_bench(name, limit=None):
    from datasets import load_dataset
    errors = []
    for repo, split, pf, af in BENCH_SOURCES[name]:
        try:
            ds = load_dataset(repo, split=split)
        except Exception as e:  # try next source
            errors.append(f"{repo}: {e}")
            continue
        rows = []
        for i, ex in enumerate(ds):
            ans = str(ex[af]).strip()
            if name == "aime24" and af == "solution":
                m = re.findall(r"\\boxed\{([^}]*)\}", ans)
                ans = m[-1] if m else ans
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

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_model_len=args.max_tokens + 2048, seed=args.seed, trust_remote_code=True)
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                        top_k=args.top_k if args.top_k > 0 else -1, max_tokens=args.max_tokens, seed=args.seed)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    os.makedirs(args.out, exist_ok=True)
    stem = f"{args.tag or os.path.basename(args.model.rstrip('/'))}_{args.bench}_{args.mode}_t{args.temperature}"
    per_prob = defaultdict(list)
    by_level = defaultdict(list)
    n_tok, n_trunc = [], 0
    with open(os.path.join(args.out, stem + ".jsonl"), "w") as f:
        for r, o in zip(rows, outs):
            for si, c in enumerate(o.outputs):
                text = c.text
                ok, pred = grade(text, r["answer"])
                trunc = c.finish_reason == "length"
                rec = {"pid": r["pid"], "sample": si, "level": r["level"], "gold": r["answer"],
                       "pred": pred, "correct": ok, "n_tokens": len(c.token_ids),
                       "finish_reason": c.finish_reason, "terminated": not trunc,
                       "think_closed": ("</think>" in text) if args.mode == "think" else None,
                       "first_correct_pos": first_correct_pos(text, r["answer"]),
                       "first_value_pos_loose": first_correct_pos(text, r["answer"], loose=True),
                       "grader_version": GRADER_VERSION, "token_ids": list(c.token_ids), "text": text}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                per_prob[r["pid"]].append(ok)
                if r["level"] is not None:
                    by_level[r["level"]].append((ok, len(c.token_ids)))
                n_tok.append(len(c.token_ids))
                n_trunc += trunc

    acc = sum(sum(v) for v in per_prob.values()) / sum(len(v) for v in per_prob.values())
    metrics = {
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
    with open(os.path.join(args.out, stem + ".metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in ["model", "bench", "mode", "avg_acc", "mean_tokens", "trunc_rate", "gen_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
    # vLLM v1 can hang on interpreter shutdown while holding the GPU (seen on the 2x H100, 11 min idle).
    # All outputs are written and closed above, so exit hard.
    import sys as _sys
    _sys.stdout.flush(); _sys.stderr.flush()
    os._exit(0)
