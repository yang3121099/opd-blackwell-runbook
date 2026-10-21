"""PROBE-DIFF17's difficulty labels (runbook [v2.5.9x], Q-041 / Q-042 / Q-050; the reviewer's pre-registration, research_program §3c).

A base model (Qwen3-1.7B, or Qwen3-4B for the k_4b column) on its NON-thinking template samples k=8 answers per prompt (T=1.0,
top-p 1, 4096 new tokens, vLLM seed 0); the pass rate r = correct / 8 (the training reward reward_mathverify.compute_score for the
training prompts, gen_eval.grade for MATH-500). Prompts:
  - train: the prompts an E10b arm meets in its first STEPS (50) steps of TBS (128), in the order verl's sampler gives them:
    dapo_train.parquet, the rows whose prompt fits MAXP (1024) tokens on the student's template (filter_overlong_prompts),
    torch.randperm(n, generator seeded with data.seed 42) -- RandomSampler's order for one epoch; the first STEPS x TBS.
  - math500: all 500, with gen_eval's prompt (problem + its suffix), for the readings split easy / hard.
Each prompt is keyed by key(): the sha1 of its (last) user message, the same function opd_route.py applies to the trainer's
raw_prompt. Output <out>/labels_<tag>.jsonl (one line per prompt: split, pos (sampler position or MATH-500 index), key, k, n, r,
mean_len, trunc) and <out>/report_<tag>.json (counts, the histogram of k, the easy share r >= 0.5, mean length, truncation, timing).
Usage: python diff_labels.py --model <path> --tag 17|4b --train <dapo_train.parquet> --out <dir> [--steps 50 --tbs 128 --seed 42
       --maxp 1024 --k 8 --max_tokens 4096 --limit N]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def key(messages):
    """sha1 (16 hex) of the last user message's content: the prompt's identity on both sides (labels, trainer)."""
    users = [m for m in messages if (m.get("role") if isinstance(m, dict) else None) == "user"]
    text = users[-1]["content"] if users else json.dumps(list(messages), sort_keys=True, default=str)
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:16]


def sampler_order(n, seed):
    """verl's RandomSampler order for one epoch: torch.randperm with a generator seeded by data.seed."""
    import torch
    g = torch.Generator(); g.manual_seed(int(seed))
    return torch.randperm(n, generator=g).tolist()


def select_train(raw, tokenizer, steps, tbs, seed, maxp, chat_kwargs):
    """raw: [(messages, ground truth, data_source)] in parquet order -> ([(pos, messages, gt, src)] for the first steps x tbs
    in sampler order, rows in, rows that fit)."""
    rows = [x for x in raw if len(tokenizer.apply_chat_template(x[0], add_generation_prompt=True, **chat_kwargs)) <= maxp]   # filter_overlong_prompts, before the sampler
    order = sampler_order(len(rows), seed)[: steps * tbs]
    return [(pos, *rows[i]) for pos, i in enumerate(order)], len(raw), len(rows)


def train_prompts(parquet, tokenizer, steps, tbs, seed, maxp, chat_kwargs):
    """select_train over dapo_train.parquet."""
    import pandas as pd
    df = pd.read_parquet(parquet)
    raw = [([dict(m) for m in r["prompt"]], str(r["reward_model"]["ground_truth"]), r.get("data_source", "dapo_mv")) for _, r in df.iterrows()]
    return select_train(raw, tokenizer, steps, tbs, seed, maxp, chat_kwargs)


def math500_prompts():
    sys.path.insert(0, os.path.join(HERE, "..", "eval"))
    import gen_eval
    return [(i, [{"role": "user", "content": r["problem"] + gen_eval.PROMPT_SUFFIX}], r["answer"], "math500")
            for i, r in enumerate(gen_eval.load_bench("math500"))]


def grader(split):
    if split == "train":
        sys.path.insert(0, os.path.join(HERE, "..", "train"))
        import reward_mathverify
        return lambda text, gold, src: float(reward_mathverify.compute_score(src, text, gold))
    sys.path.insert(0, os.path.join(HERE, "..", "eval"))
    import gen_eval
    return lambda text, gold, src: float(bool(gen_eval.grade(text, gold)))


def summarize(rows, k):
    hist = Counter(r["k"] for r in rows)
    n = max(1, len(rows))
    return {"prompts": len(rows), "hist_k": {str(i): hist.get(i, 0) for i in range(k + 1)},
            "easy_share": round(sum(1 for r in rows if r["r"] >= 0.5) / n, 4), "mean_r": round(sum(r["r"] for r in rows) / n, 4),
            "mean_len": round(sum(r["mean_len"] for r in rows) / n, 1), "trunc_share": round(sum(r["trunc"] for r in rows) / (n * k), 4)}


def label(llm, tokenizer, prompts, split, k, max_tokens, chat_kwargs, sp_factory):
    grade = grader(split)
    texts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False, **chat_kwargs) for _, m, _, _ in prompts]
    outs = llm.generate(texts, sp_factory(k, max_tokens))
    rows = []
    for (pos, msgs, gold, src), o in zip(prompts, outs):
        c = [grade(x.text, gold, src) for x in o.outputs]
        lens = [len(x.token_ids) for x in o.outputs]
        rows.append({"split": split, "pos": pos, "key": key(msgs), "k": int(sum(1 for v in c if v > 0.5)), "n": len(c),
                     "r": round(sum(1 for v in c if v > 0.5) / max(1, len(c)), 4), "mean_len": round(sum(lens) / max(1, len(lens)), 1),
                     "trunc": int(sum(1 for x in o.outputs if x.finish_reason == "length"))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True, choices=["17", "4b"])
    ap.add_argument("--train", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--tbs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--maxp", type=int, default=1024)
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=0, help="tests / smoke: only the first N prompts of each split")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import vllm
    tok = AutoTokenizer.from_pretrained(a.model)
    ck = {"enable_thinking": False}
    t0 = time.time()
    tr, n_rows, n_fit = train_prompts(a.train, tok, a.steps, a.tbs, a.seed, a.maxp, ck)
    mp = math500_prompts()
    if a.limit:
        tr, mp = tr[: a.limit], mp[: a.limit]
    llm = vllm.LLM(model=a.model, seed=0, max_model_len=a.maxp + a.max_tokens + 64, gpu_memory_utilization=float(os.environ.get("DIFF_UTIL", "0.85")))
    sp = lambda k, m: vllm.SamplingParams(n=k, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=m, seed=0)
    os.makedirs(a.out, exist_ok=True)
    rows_tr = label(llm, tok, tr, "train", a.k, a.max_tokens, ck, sp); t1 = time.time()
    rows_mp = label(llm, tok, mp, "math500", a.k, a.max_tokens, ck, sp); t2 = time.time()
    with open(os.path.join(a.out, f"labels_{a.tag}.jsonl"), "w") as f:
        for r in rows_tr + rows_mp:
            f.write(json.dumps(r) + "\n")
    rep = {"tag": a.tag, "model": os.path.basename(a.model.rstrip("/")), "k": a.k, "temperature": 1.0, "max_tokens": a.max_tokens,
           "template": "non-thinking", "train_rows": n_rows, "train_rows_fit": n_fit, "train_selected": len(rows_tr),
           "steps": a.steps, "tbs": a.tbs, "seed": a.seed, "train": summarize(rows_tr, a.k), "math500": summarize(rows_mp, a.k),
           "seconds": {"train": round(t1 - t0, 1), "math500": round(t2 - t1, 1)}, "vllm": vllm.__version__}
    json.dump(rep, open(os.path.join(a.out, f"report_{a.tag}.json"), "w"), indent=1)
    print("DIFF_LABELS", json.dumps({"train": rep["train"], "math500": rep["math500"]}))


if __name__ == "__main__":
    main()
