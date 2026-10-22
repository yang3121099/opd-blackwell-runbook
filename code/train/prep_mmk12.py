"""MMK12 for verl's multi-modal RL (runbook [v2.5.9z33]; the reviewer's Q-051 revision 10-22, Runming: the mainstream setting). Reads
the node's copy of FanqingM/MMK12 (data/hf/FanqingM__MMK12, its .fetch_ok; data/train-*.parquet 15,616 rows, data/test-*.parquet 2,000;
fields id, question, answer, subject, image) and writes data/mmk12_train.parquet, and data/mmk12_val.parquet (VAL_N rows of the test
split drawn with a fixed seed, for the in-training validation only), as verl's RLHFDataset reads them with data.image_key=images:
prompt = one user message, '<image>' first, then the question and the maths data's answer instruction (gen_eval's PROMPT_SUFFIX);
images = [the row's image] (bytes as they are); reward_model.ground_truth = answer; data_source FanqingM/MMK12; extra_info split /
index / id / subject. Rows without an image or with an empty answer are dropped and counted. Prompts over 2048 tokens are left to
verl's filter_overlong_prompts (run_vlm.sh). Prints the counts (kept, dropped by reason, subjects). [ops-lead's A1 conditions] Each file
is written to <name>.tmp and renamed into place (os.replace), the validation file first, so a train file present is a complete one
(run_vlm.sh makes it only when missing); a missing train or test split, or a train split with no row kept, exits non-zero with
nothing written.
Usage: python prep_mmk12.py <src dir> <out dir> [val_n]
"""
import glob
import json
import os
import random
import sys
from collections import Counter

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
VAL_N, VAL_SEED = 300, 20261022


def has_image(x):
    if x is None:
        return False
    if isinstance(x, dict):
        return bool(x.get("bytes")) or bool(x.get("path"))
    return bool(x)


def rows_of(df, split):
    """(rows, counts): the verl rows of one split and what was kept / dropped."""
    out, c = [], Counter()
    for i, r in enumerate(df.to_dict("records")):
        ans = "" if r.get("answer") is None else str(r["answer"]).strip()
        if not has_image(r.get("image")):
            c["dropped_no_image"] += 1
            continue
        if not ans:
            c["dropped_empty_answer"] += 1
            continue
        q = str(r.get("question") or "").replace("<image>", "").strip()
        out.append({"data_source": "FanqingM/MMK12", "prompt": [{"role": "user", "content": "<image>" + q + SUFFIX}], "images": [r["image"]],
                    "ability": "math", "reward_model": {"style": "rule", "ground_truth": ans},
                    "extra_info": {"split": split, "index": i, "id": str(r.get("id")), "subject": str(r.get("subject"))}})
        c["kept"] += 1; c["subject:" + str(r.get("subject"))] += 1
    return out, c


def val_rows(rows, n=VAL_N, seed=VAL_SEED):
    """n rows of the test split, drawn with a fixed seed, in their order."""
    if len(rows) <= n:
        return rows
    keep = sorted(random.Random(seed).sample(range(len(rows)), n))
    return [rows[k] for k in keep]


def write_atomic(df, path):
    """df to path through path.tmp and os.replace: an interrupted write leaves no file under the final name."""
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)


def main(src, out, val_n=VAL_N):
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    counts, rows = {}, {}
    for split in ("train", "test"):
        fs = sorted(glob.glob(os.path.join(src, "**", f"{split}-*.parquet"), recursive=True))
        if not fs:
            raise SystemExit(f"PREP_MMK12_FAILED no {split}-*.parquet under {os.path.basename(src.rstrip('/'))}")
        rows[split], c = rows_of(pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True), split)
        counts[split] = dict(c)
    if not rows["train"]:
        raise SystemExit(f"PREP_MMK12_FAILED no train row kept: {json.dumps(counts['train'])}")
    val = val_rows(rows["test"], val_n); counts["test"]["val_rows"] = len(val)
    write_atomic(pd.DataFrame(val), os.path.join(out, "mmk12_val.parquet"))   # first: a train file present is a complete pair
    write_atomic(pd.DataFrame(rows["train"]), os.path.join(out, "mmk12_train.parquet"))
    print("PREP_MMK12", json.dumps(counts))
    return counts


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else VAL_N)
