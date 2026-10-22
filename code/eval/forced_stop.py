"""E05-p3 forced stop (queue v3 V3-c3 c; lab registry E05): at budget B, every sample of a gen_eval jsonl (thinking mode,
token_ids kept) that has not terminated within B tokens is cut at B, "\\n</think>\\n\\n**Final Answer**\\n\\\\boxed{" is
appended (E05.yaml forced_stop), and the model continues for at most 64 tokens; the answer is graded with gen_eval's
grader. Per budget: accuracy without forcing (a truncated sample counts wrong), accuracy with forcing, how many samples
were forced and how many of them became correct. E05-p3's claim (forcing Base-0.6B recovers >= 1/3 of Art's gain) is
read from the Base and Art outputs together.

    python forced_stop.py --model M --jsonl GEN.jsonl --bench aime25 --budgets 8192,16384,32768 --out OUT.json

v2.5.7h (reviewer 2026-10-18 19:3x UTC, budget readings B, lab Q-013): --force_preset budB --mode think|no_think: the
non-thinking prompt template for no_think readings, the stop string "\\n\\n**Final Answer**\\n\\\\boxed{" (no_think, or a
thinking sample whose prefix already closed </think>) or "\\n</think>\\n\\n\\\\boxed{" (thinking), at most --max_new tokens
(budB: 32); --samples writes one line per forced sample (pid, sample, budget, forced_correct, pred); budgets above the
reading's own max_tokens (its .metrics.json) are left out. The defaults (preset e05, think, 64 tokens) are E05's.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FORCE = "\n</think>\n\n**Final Answer**\n\\boxed{"
FORCE_BUDB = {"think": "\n</think>\n\n\\boxed{", "no_think": "\n\n**Final Answer**\n\\boxed{"}


def force_text(preset, mode, prefix_text=""):
    """The stop string appended at the cut (E05: always FORCE)."""
    if preset == "e05":
        return FORCE
    return FORCE_BUDB["no_think"] if mode == "no_think" or "</think>" in prefix_text else FORCE_BUDB["think"]


def budgets_upto(budgets, max_tokens):
    """The budgets a reading of max_tokens can be forced at: those up to it and ([v2.5.9z46] Q-057's 3072 readings) the cap itself."""
    return [b for b in budgets if b <= max_tokens] + ([max_tokens] if max_tokens not in budgets else [])


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
    ap.add_argument("--mode", choices=["think", "no_think"], default="think"); ap.add_argument("--force_preset", choices=["e05", "budB"], default="e05")
    ap.add_argument("--top_k", type=int, default=20); ap.add_argument("--samples", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams, TokensPrompt
    from gen_eval import PROMPT_SUFFIX, _reap_children, grade, load_bench
    rows = {r["pid"]: r for r in load_bench(a.bench)}
    recs = [json.loads(l) for l in open(a.jsonl)]
    tok = AutoTokenizer.from_pretrained(a.model)
    force_ids = {}
    budgets = [int(b) for b in a.budgets.split(",")]
    mpath = a.jsonl[:-len(".jsonl")] + ".metrics.json"
    if a.force_preset != "e05" and os.path.isfile(mpath) and json.load(open(mpath)).get("max_tokens"):
        budgets = budgets_upto(budgets, int(json.load(open(mpath))["max_tokens"]))
    jobs = []
    for b in budgets:
        _, forced = plan(recs, b)
        for r in forced:
            msgs = [{"role": "user", "content": rows[r["pid"]]["problem"] + PROMPT_SUFFIX}]
            p = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=a.mode == "think")
            f = force_text(a.force_preset, a.mode, tok.decode(r["token_ids"][:b]) if a.force_preset != "e05" else "")
            if f not in force_ids:
                force_ids[f] = tok(f, add_special_tokens=False)["input_ids"]
            jobs.append((b, r, list(p) + r["token_ids"][:b] + force_ids[f]))
    out = {"model": a.model, "jsonl": a.jsonl, "bench": a.bench, "force": FORCE if a.force_preset == "e05" else FORCE_BUDB,
           "mode": a.mode, "preset": a.force_preset, "max_new": a.max_new, "budgets": []}
    if jobs:
        llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=max(len(j[2]) for j in jobs) + a.max_new + 16, seed=a.seed)
        sp = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=a.top_k if a.top_k > 0 else -1, max_tokens=a.max_new, seed=a.seed)
        res = llm.generate([TokensPrompt(prompt_token_ids=j[2]) for j in jobs], sp)
    else:
        res = []
    sf = open(a.samples, "w") if a.samples else None
    for b in budgets:
        fc = {}
        for (bb, r, _), o in zip(jobs, res):
            if bb == b:
                ok, pred = grade("\\boxed{" + o.outputs[0].text, r["gold"])
                fc[(r["pid"], r["sample"])] = ok
                if sf:
                    sf.write(json.dumps({"pid": r["pid"], "sample": r["sample"], "budget": b, "forced_correct": bool(ok), "pred": pred}) + "\n")
        out["budgets"].append(summarize(recs, b, fc))
    if sf:
        sf.close()
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["budgets"]), flush=True)
    _reap_children()   # as gen_eval: no vLLM engine core outlives the task (v2.5.6e)
    os._exit(0)


if __name__ == "__main__":
    main()
