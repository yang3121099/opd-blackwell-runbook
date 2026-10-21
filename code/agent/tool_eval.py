"""Tool-call evaluation with a budget curve (runbook [v2.5.9b], research_program §1.7: "score - per-turn budget" curve).

Generates once per prompt (vLLM, n samples, the largest budget), then reads every sample at each budget B as if generation
had stopped after B tokens (cut at B tokens and graded again, as budget_curve.py does for MATH-500): score
(tool_grader.grade, v2 since [v2.5.9z19]), legal-call rate (a call parsed and no malformed block), BFCL-style AST match with
the reference calls (over the samples that have reference calls; [v2.5.9z19] rules-only samples such as Auto-ClawEval's are
left out of it, n_ast counts the rest), forbidden-call rate, and mean tokens. Prompts come from the parquet of
prep_tools_parquet.py (system tool block + user turn), rendered with the model's own template (non-thinking:
enable_thinking=False; think: True).

Usage: python tool_eval.py --model M --data VAL.parquet --out OUT.json [--samples OUT.jsonl] [--n 4] [--temperature 0.6]
                            [--budgets 512,1024,1536,4096] [--mode no_think|think]
"""
import argparse
import json

import tool_grader as G


def evaluate(records, budgets, cut):
    """records: [{"id", "ground_truth", "ids": [token ids], "schemas": {name: [params]}}]; cut(ids, B) -> text of the first B
    tokens. Returns {budget: metrics} with score, legal, forbidden (shares in %) and mean tokens over all samples, ast over
    the n_ast samples with reference calls (None without any)."""
    out = {}
    for b in budgets:
        n = sc = lg = am = na = fb = tok = 0
        for r in records:
            gt = json.loads(r["ground_truth"]) if isinstance(r["ground_truth"], str) else r["ground_truth"]
            text = cut(r["ids"], b)
            g = G.grade(text, gt)
            n += 1; sc += g["score"]; lg += g["legal"]; fb += bool(g["forbidden"]); tok += min(len(r["ids"]), b)
            if gt.get("ref_calls"):
                na += 1; am += G.ast_match(G.parse_calls(text)[0], gt["ref_calls"], r.get("schemas"))
        out[str(b)] = {"score": round(100 * sc / n, 2), "legal": round(100 * lg / n, 2),
                       "ast": round(100 * am / na, 2) if na else None, "n_ast": na,
                       "forbidden": round(100 * fb / n, 2), "mean_tokens": round(tok / n, 1), "n": n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--samples"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--budgets", default="512,1024,1536,4096"); ap.add_argument("--mode", default="no_think", choices=["no_think", "think"])
    a = ap.parse_args()
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    budgets = sorted(int(x) for x in a.budgets.split(","))
    tok = AutoTokenizer.from_pretrained(a.model)
    df = pd.read_parquet(a.data)
    prompts, meta = [], []
    for _, row in df.iterrows():
        msgs = [dict(m) for m in row["prompt"]]
        prompts.append(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=a.mode == "think"))
        tools = json.loads(row["extra_info"]["tools"])
        meta.append((row["extra_info"]["id"], row["reward_model"]["ground_truth"], G.schemas(tools)))
    llm = LLM(model=a.model, gpu_memory_utilization=0.8, max_model_len=4096 + max(budgets))
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=max(budgets), seed=0)
    res = llm.generate(prompts, sp)
    records = []
    for (rid, gt, schemas), r in zip(meta, res):
        for o in r.outputs:
            records.append({"id": rid, "ground_truth": gt, "ids": list(o.token_ids), "schemas": schemas})
    cut = lambda ids, b: tok.decode(ids[:b], skip_special_tokens=False)
    curve = evaluate(records, budgets, cut)
    json.dump({"model_tag": a.model.rstrip("/").split("/")[-1], "mode": a.mode, "n": a.n, "temperature": a.temperature,
               "prompts": len(prompts), "budgets": curve}, open(a.out, "w"), indent=1)
    if a.samples:
        with open(a.samples, "w") as f:
            for r in records:
                f.write(json.dumps({"id": r["id"], "tokens": len(r["ids"]), "text": tok.decode(r["ids"], skip_special_tokens=False)}) + "\n")
    print("TOOL_EVAL", json.dumps(curve))


if __name__ == "__main__":
    main()
