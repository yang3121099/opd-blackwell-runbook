"""E05-p3 forced stop (queue v3 V3-c3 c; lab registry E05): at budget B, every sample of a gen_eval jsonl (thinking mode,
token_ids kept) that has not terminated within B tokens is cut at B, "\\n</think>\\n\\n**Final Answer**\\n\\\\boxed{" is
appended (E05.yaml forced_stop), and the model continues for at most 64 tokens; the answer is graded with gen_eval's
grader. Per budget: accuracy without forcing (a truncated sample counts wrong), accuracy with forcing, how many samples
were forced and how many of them became correct. E05-p3's claim (forcing Base-0.6B recovers >= 1/3 of Art's gain) is
read from the Base and Art outputs together.

    python forced_stop.py --model M --jsonl GEN.jsonl --bench aime25 --budgets 8192,16384,32768 --out OUT.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FORCE = "\n</think>\n\n**Final Answer**\n\\boxed{"


def plan(records, budget):
    """(kept, forced): kept = [(record, correct)] for samples that terminated within the budget; forced = records cut
    at the budget (their first `budget` tokens are the prefix to continue)."""
    kept, forced = [], []
    for r in records:
        n = len(r["token_ids"])
        if n <= budget and r.get("finish_reason") != "length":
            kept.append((r, bool(r["correct"])))
        else:
            forced.append(r)
    return kept, forced


def summarize(records, budget, forced_correct):
    """forced_correct: {(pid, sample): bool} for the forced samples."""
    kept, forced = plan(records, budget)
    n = len(records)
    base = sum(c for _, c in kept)
    fc = sum(1 for r in forced if forced_correct.get((r["pid"], r["sample"])))
    return {"budget": budget, "n": n, "acc_no_force": round(100 * base / n, 2), "acc_forced": round(100 * (base + fc) / n, 2),
            "n_forced": len(forced), "forced_correct": fc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--jsonl", required=True); ap.add_argument("--bench", required=True)
    ap.add_argument("--budgets", default="8192,16384,32768"); ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gpu_mem", type=float, default=0.85); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from gen_eval import PROMPT_SUFFIX, _reap_children, grade, load_bench
    rows = {r["pid"]: r for r in load_bench(a.bench)}
    recs = [json.loads(l) for l in open(a.jsonl)]
    tok = AutoTokenizer.from_pretrained(a.model)
    force_ids = tok(FORCE, add_special_tokens=False)["input_ids"]
    budgets = [int(b) for b in a.budgets.split(",")]
    jobs = []
    for b in budgets:
        _, forced = plan(recs, b)
        for r in forced:
            msgs = [{"role": "user", "content": rows[r["pid"]]["problem"] + PROMPT_SUFFIX}]
            p = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=True)
            jobs.append((b, r, list(p) + r["token_ids"][:b] + force_ids))
    out = {"model": a.model, "jsonl": a.jsonl, "bench": a.bench, "force": FORCE, "max_new": a.max_new, "budgets": []}
    if jobs:
        llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=max(len(j[2]) for j in jobs) + a.max_new + 16, seed=a.seed)
        sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=a.max_new, seed=a.seed)
        res = llm.generate([TokensPrompt(prompt_token_ids=j[2]) for j in jobs], sp)
    else:
        res = []
    for b in budgets:
        fc = {}
        for (bb, r, _), o in zip(jobs, res):
            if bb == b:
                fc[(r["pid"], r["sample"])] = grade("\\boxed{" + o.outputs[0].text, r["gold"])[0]
        out["budgets"].append(summarize(recs, b, fc))
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["budgets"]), flush=True)
    _reap_children()   # as gen_eval: no vLLM engine core outlives the task (v2.5.6e)
    os._exit(0)


if __name__ == "__main__":
    main()
