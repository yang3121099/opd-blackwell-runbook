"""[v2.5.9z108] (Q-083 E10b-P5mode, research_program §3i; beyond_mainline §1) the training parquet with a visible mode token per prompt.

Every row of dapo_train.parquet keeps its place and its fields; only the last user message of its prompt gets '[long] ' or '[short] '
in front (opd_route.prefixed), by mix's coin: opd_route.coin(key of the prompt as it is, OPD_SEED), so the '[long]' prompts are exactly
the prompts E10b-P5mix gave the thinking teacher (Q-083 differs from it only by the visible token). A prompt that already carries a
token is refused (the input must be the plain parquet). Writes <out> atomically and prints
"MODE_DATA rows <n> long <a> short <b> share <a/n> seed <s>"; an existing <out> with the same header line in <out>.json is kept.

Usage: python mode_data.py <dapo_train.parquet> <out.parquet> [--seed 1234]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "patches"))
import opd_route as R  # noqa: E402


def mode_prompt(messages, seed):
    """(the prompt with its token, think): a new list; the last user message is a new dict."""
    msgs, i = R._last_user(messages)
    if i is None:
        raise ValueError("a prompt without a user message")
    if R.split_prefix(msgs[i].get("content", ""))[0] is not None:
        raise ValueError("a prompt already carries a mode token")
    think = R.coin(R.key(msgs), seed)
    msgs[i] = dict(msgs[i], content=R.prefixed(msgs[i].get("content", ""), think))
    return msgs, think


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=int(os.environ.get("OPD_SEED", "1234")))
    a = ap.parse_args()
    import pandas as pd
    df = pd.read_parquet(a.train)
    prompts, longs = [], 0
    for _, r in df.iterrows():
        m, think = mode_prompt([dict(x) for x in r["prompt"]], a.seed)
        prompts.append(m); longs += int(think)
    head = {"rows": len(df), "long": longs, "short": len(df) - longs, "share": round(longs / max(len(df), 1), 4), "seed": a.seed,
            "source": os.path.basename(a.train)}
    meta = a.out + ".json"
    if os.path.exists(a.out) and os.path.exists(meta) and json.load(open(meta)) == head:
        print("MODE_DATA kept " + " ".join(f"{k} {head[k]}" for k in ("rows", "long", "short", "share", "seed")))
        return
    df = df.copy(); df["prompt"] = prompts
    tmp = f"{a.out}.tmp.{os.getpid()}"
    df.to_parquet(tmp, index=False); os.replace(tmp, a.out)
    json.dump(head, open(meta, "w"))
    print("MODE_DATA " + " ".join(f"{k} {head[k]}" for k in ("rows", "long", "short", "share", "seed")))


if __name__ == "__main__":
    main()
