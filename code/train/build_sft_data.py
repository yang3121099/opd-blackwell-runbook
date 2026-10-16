"""Build the P1 / P1' SFT parquet from teacher-trace shards (E10 amendment 7; R-005 A3).

P1  = traces that are correct AND terminated (rejection-sampling fine-tuning).
P1' = all terminated traces, correct or not. Truncated traces go into neither (SFT would append an EOS at the cut).
Columns: prompt (the user message text, rendered later with the no-think template), response (teacher text).
Also prints the step count that matches the token budget at 128 sequences per step, and the epoch count.
Usage: python build_sft_data.py --mode p1|p1prime --out data/sft_p1.parquet shard0.jsonl [shard1.jsonl ...]
"""
import argparse
import json
import math

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["p1", "p1prime"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget_tokens", type=float, default=49.2e6)   # P2's student tokens over 150 steps
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    rows, n_all, n_term, n_ok = [], 0, 0, 0
    for path in a.shards:
        opener = (lambda p: __import__("gzip").open(p, "rt", encoding="utf-8")) if path.endswith(".gz") else open
        for line in opener(path):
            r = json.loads(line)
            n_all += 1
            term = r["finish_reason"] != "length"
            n_term += term; n_ok += term and r["correct"]
            if not term or (a.mode == "p1" and not r["correct"]):
                continue
            user = [m for m in r["prompt"] if m["role"] == "user"][-1]["content"]
            rows.append({"prompt": user, "response": r["text"], "n_tokens": r["n_tokens"], "index": r["index"]})
    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    df[["prompt", "response"]].to_parquet(a.out)
    df.head(64)[["prompt", "response"]].to_parquet(a.out.replace(".parquet", "_val64.parquet"))
    tok = int(df["n_tokens"].sum())
    steps = math.ceil(a.budget_tokens / (a.batch * df["n_tokens"].mean()))
    spe = math.ceil(len(df) / a.batch)
    print(json.dumps({"mode": a.mode, "traces_in": n_all, "terminated": n_term, "correct_and_terminated": n_ok,
                      "kept": len(df), "tokens_per_epoch": tok, "mean_tokens": round(df["n_tokens"].mean(), 1),
                      "steps_for_budget": steps, "steps_per_epoch": spe, "epochs": round(steps / spe, 2),
                      "total_epochs_arg": math.ceil(steps / spe)}))


if __name__ == "__main__":
    main()
