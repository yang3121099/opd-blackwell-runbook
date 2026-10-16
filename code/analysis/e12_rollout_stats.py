"""Per-step GRPO group diagnostics for E12 (R-007 section 2), from the rollout dumps verl writes with
trainer.rollout_data_dir (<run>/rollouts/<step>.jsonl: input, output, gts, score, step).
Groups are the 8 responses to the same prompt. Per step: share of all-correct, all-wrong and mixed groups
(only mixed groups carry a GRPO gradient), mean reward, and mean response length in tokens.
Truncation rate and entropy are read from the console log instead (response_length/clip_ratio, actor/entropy).
Usage: python e12_rollout_stats.py <run dir> [--tokenizer Qwen/Qwen3-0.6B] [--out <csv>]
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rows = []
    files = sorted(glob.glob(os.path.join(a.run, "rollouts", "*.jsonl")), key=lambda p: int(os.path.basename(p)[:-6]))
    for path in files:
        groups = defaultdict(list)
        lens = []
        for line in open(path):
            r = json.loads(line)
            groups[r["input"]].append(float(r["score"]) > 0.5)
            lens.append(len(tok(r["output"], add_special_tokens=False)["input_ids"]))
        g = list(groups.values())
        n = len(g)
        allc = sum(all(x) for x in g); allw = sum(not any(x) for x in g)
        rows.append({"step": int(os.path.basename(path)[:-6]), "groups": n,
                     "group_size": round(sum(len(x) for x in g) / max(n, 1), 2),
                     "all_correct": round(allc / n, 4), "all_wrong": round(allw / n, 4),
                     "mixed": round((n - allc - allw) / n, 4),
                     "reward_mean": round(sum(sum(x) for x in g) / max(sum(len(x) for x in g), 1), 4),
                     "len_tok_mean": round(sum(lens) / max(len(lens), 1), 1)})
    out = a.out or os.path.join(a.run, "rollout_stats.csv")
    with open(out, "w") as f:
        f.write(",".join(rows[0].keys()) + "\n" if rows else "")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    print(json.dumps(rows[-1] if rows else {}), "->", out)


if __name__ == "__main__":
    main()
