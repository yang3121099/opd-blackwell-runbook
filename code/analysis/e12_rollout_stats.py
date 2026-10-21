"""Per-step GRPO group diagnostics for E12 (R-007 section 2), from the rollout dumps verl writes with
trainer.rollout_data_dir (<run>/rollouts/<step>.jsonl: input, output, gts, score, step).
Groups are the 8 responses to the same prompt. Per step: share of all-correct, all-wrong and mixed groups
(only mixed groups carry a GRPO gradient), mean reward, and mean response length in tokens.
Truncation rate and entropy are read from the console log instead (response_length/clip_ratio, actor/entropy).
[v2.5.9f] Also the reward's spread (its standard deviation over all responses, and the share of groups whose rewards are all
equal: no GRPO gradient, the failure of Q-021's stop reward) and, when the reward function returns its parts (reward_stop2.py:
tail, cut, bonus in the dump), the tail after the first box of the correct responses (mean, p50, p90), the cut share and the
mean bonus.
Usage: python e12_rollout_stats.py <run dir> [--tokenizer Qwen/Qwen3-0.6B] [--out <csv>]
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def _pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else ""


def step_stats(step, recs, ntok):
    """One row of the CSV from one step's dump records (input, output, score[, tail, cut, bonus])."""
    groups, scores, lens = defaultdict(list), defaultdict(list), []
    for r in recs:
        groups[r["input"]].append(float(r["score"]) > 0.5)
        scores[r["input"]].append(round(float(r["score"]), 6))
        lens.append(ntok(r["output"]))
    g = list(groups.values()); n = len(g)
    allc = sum(all(x) for x in g); allw = sum(not any(x) for x in g)
    flat = [x for v in scores.values() for x in v]
    mu = sum(flat) / max(len(flat), 1)
    row = {"step": step, "groups": n, "group_size": round(sum(len(x) for x in g) / max(n, 1), 2),
           "all_correct": round(allc / n, 4) if n else "", "all_wrong": round(allw / n, 4) if n else "",
           "mixed": round((n - allc - allw) / n, 4) if n else "",
           "reward_mean": round(sum(sum(x) for x in g) / max(sum(len(x) for x in g), 1), 4),
           "len_tok_mean": round(sum(lens) / max(len(lens), 1), 1),
           # [v2.5.9f] the reward's spread; the parts of a reward that returns them
           "reward_std": round((sum((x - mu) ** 2 for x in flat) / max(len(flat), 1)) ** 0.5, 4),
           "zero_var_groups": round(sum(len(set(v)) == 1 for v in scores.values()) / n, 4) if n else ""}
    tails = [int(r["tail"]) for r in recs if "tail" in r and float(r.get("acc", r["score"] > 0.5)) >= 1 and int(r["tail"]) >= 0]
    has = any("tail" in r for r in recs)
    row.update({"tail_mean_correct": round(sum(tails) / len(tails), 1) if tails else "", "tail_p50_correct": _pct(tails, 0.5),
                "tail_p90_correct": _pct(tails, 0.9),
                "cut_share": round(sum(int(r.get("cut", 0)) for r in recs) / len(recs), 4) if has and recs else "",
                "bonus_mean": round(sum(float(r.get("bonus", 0)) for r in recs) / len(recs), 4) if has and recs else ""})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    files = sorted(glob.glob(os.path.join(a.run, "rollouts", "*.jsonl")), key=lambda p: int(os.path.basename(p)[:-6]))
    ntok = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
    rows = [step_stats(int(os.path.basename(p)[:-6]), [json.loads(x) for x in open(p)], ntok) for p in files]
    out = a.out or os.path.join(a.run, "rollout_stats.csv")
    with open(out, "w") as f:
        f.write(",".join(rows[0].keys()) + "\n" if rows else "")
        for r in rows:
            f.write(",".join(str(v) for v in r.values()) + "\n")
    print(json.dumps(rows[-1] if rows else {}), "->", out)


if __name__ == "__main__":
    main()
