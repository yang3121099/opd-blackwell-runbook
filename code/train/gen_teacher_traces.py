"""P1/P1' teacher traces (E10 wave-1b amendment, 2026-10-16): Qwen3-8B, no-think template, one trace per DAPO training
prompt, graded with the training reward function. Inference only.

Usage: python gen_teacher_traces.py --shard i --nshards k [--limit N] --out runs/p1_traces
Writes <out>/shard<i>.jsonl with: index, prompt (chat messages), ground_truth, text, token_ids, n_tokens,
finish_reason, correct.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_mathverify import compute_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.environ.get("OPD_ROOT", os.path.expanduser("~/opd-run")), "data", "dapo_train.parquet"))
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(a.data)
    rows = [(i, df.iloc[i]) for i in range(len(df)) if i % a.nshards == a.shard]
    if a.limit:
        rows = rows[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.teacher)
    prompts = [tok.apply_chat_template(list(r["prompt"]), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False) for _, r in rows]
    llm = LLM(model=a.teacher, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_tokens + 2048, seed=a.seed)
    sp = SamplingParams(n=1, temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"shard{a.shard}.jsonl")
    n_ok = n_term = n_tok = 0
    with open(path, "w") as f:
        for (i, r), o in zip(rows, outs):
            c = o.outputs[0]
            gt = str(r["reward_model"]["ground_truth"])
            ok = compute_score("dapo", c.text, gt) > 0.5
            term = c.finish_reason != "length"
            n_ok += ok; n_term += term; n_tok += len(c.token_ids)
            f.write(json.dumps({"index": int(i), "prompt": [dict(m) for m in r["prompt"]], "ground_truth": gt,
                                "text": c.text, "token_ids": list(c.token_ids), "n_tokens": len(c.token_ids),
                                "finish_reason": c.finish_reason, "correct": bool(ok)}, ensure_ascii=False) + "\n")
    n = max(len(rows), 1)
    print(json.dumps({"shard": a.shard, "n": len(rows), "acc": round(n_ok / n, 4), "terminated": round(n_term / n, 4),
                      "mean_tokens": round(n_tok / n, 1), "gen_seconds": round(time.time() - t0, 1)}), flush=True)
    os._exit(0)   # vLLM engine children can hang at shutdown (see gen_eval.py)


if __name__ == "__main__":
    main()
