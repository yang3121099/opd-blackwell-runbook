"""Geometry3K for verl's multi-modal RL (runbook [v2.5.9z], Q-051; the reviewer's E15-VLM-P5 data). Reads the node's copy of
hiyouga/geometry3k (data/hf/hiyouga__geometry3k, its .fetch_ok; the parquet splits train / validation / test with images, problem,
answer) and writes data/geo3k_<split>.parquet rows as verl's RLHFDataset reads them with data.image_key=images: prompt = one user
message, the problem with its '<image>' placeholder first (added when missing) and the same answer instruction as the maths data
(gen_eval's PROMPT_SUFFIX), images = the row's images (bytes kept as they are), reward_model.ground_truth = answer, data_source
hiyouga/geometry3k, extra_info split / index. Prints the row counts. Usage: python prep_geo3k.py <src dir> <out dir>
"""
import glob
import json
import os
import sys

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def rows_of(df, split):
    out = []
    for i, r in enumerate(df.to_dict("records")):
        problem = str(r["problem"]).strip()
        if "<image>" not in problem:
            problem = "<image>" + problem
        imgs = r["images"]
        imgs = list(imgs) if not isinstance(imgs, dict) else [imgs]
        out.append({"data_source": "hiyouga/geometry3k", "prompt": [{"role": "user", "content": problem + SUFFIX}], "images": imgs,
                    "ability": "math", "reward_model": {"style": "rule", "ground_truth": str(r["answer"]).strip()},
                    "extra_info": {"split": split, "index": i}})
    return out


def main(src, out):
    import pandas as pd
    os.makedirs(out, exist_ok=True)
    counts = {}
    for split in ("train", "validation", "test"):
        fs = sorted(glob.glob(os.path.join(src, "**", f"{split}-*.parquet"), recursive=True))
        if not fs:
            continue
        df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        rows = rows_of(df, split)
        pd.DataFrame(rows).to_parquet(os.path.join(out, f"geo3k_{split}.parquet"))
        counts[split] = len(rows)
    print("PREP_GEO3K", json.dumps(counts))
    return counts


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
