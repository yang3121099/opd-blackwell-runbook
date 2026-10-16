"""Build verl parquet files for E10/E12.

- DAPO-Math-17k: deduplicate, strip DAPO's 'Answer:' instructions, re-wrap with our boxed suffix.
  Hold out 512 prompts (seeded) for E11 coverage diagnostics.
- MATH500 (first 100, fixed order) as the online validation set.
Writes to $OPD_ROOT/data/ (default ~/opd-run/data).
"""
import json
import os
import random
import re

import pandas as pd
from datasets import load_dataset

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
OUT = os.path.join(os.environ.get("OPD_ROOT", os.path.expanduser("~/opd-run")), "data")
os.makedirs(OUT, exist_ok=True)


def strip_dapo(text):
    m = re.search(r"\n\n(.*)\n\nRemember to put your answer", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return re.sub(r"^Solve the following math problem.*?\n\n", "", text, flags=re.S).strip()


def row(problem, answer, source, idx, split):
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": problem + SUFFIX}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(answer)},
        "extra_info": {"split": split, "index": idx},
    }


dapo = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
seen, rows = set(), []
for ex in dapo:
    content = ex["prompt"][0]["content"]
    prob = strip_dapo(content)
    if prob in seen:
        continue
    seen.add(prob)
    rows.append((prob, ex["reward_model"]["ground_truth"]))
random.Random(42).shuffle(rows)
hold, train = rows[:512], rows[512:]
pd.DataFrame([row(p, a, "dapo_mv", i, "train") for i, (p, a) in enumerate(train)]).to_parquet(f"{OUT}/dapo_train.parquet")
pd.DataFrame([row(p, a, "dapo_mv", i, "holdout") for i, (p, a) in enumerate(hold)]).to_parquet(f"{OUT}/dapo_holdout512.parquet")

m5 = load_dataset("HuggingFaceH4/MATH-500", split="test")
pd.DataFrame([row(ex["problem"], ex["answer"], "math500", i, "val") for i, ex in enumerate(m5) if i < 100]).to_parquet(f"{OUT}/math500_100.parquet")

print(json.dumps({"dapo_unique": len(rows), "train": len(train), "holdout": len(hold),
                  "example_problem": train[0][0][:300], "example_answer": train[0][1]}, ensure_ascii=False, indent=1))
