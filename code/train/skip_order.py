"""[v2.5.9z43] (Q-058, the reviewer's pre-registration 10-22: the progressive-window arms continue a run's step 25 for 25 steps)
A run started from a checkpoint's weights meets verl's data from the first batch again. E12b's sampler is torch.randperm(n,
generator seeded with data.seed 4242) over the rows of dapo_train.parquet whose prompt fits 1024 tokens on the non-thinking
template (filter_overlong_prompts, before the sampler; the model of code/analysis/diff_labels.py), 64 prompts a step. So a
continuation of step 25 would see steps 1-25's 1600 prompts again, where the run it is compared with (the same recipe for 50
steps) sees positions 1600-3199. build writes the training file a continuation reads instead: the fitting rows, placed so
that the same sampler (same seed, same n) yields the parent's order from position K = skip_steps x tbs on:
    new[perm[i]] = fit[perm[(i + K) mod n]]
check compares the sampler model with a run's rollout dumps (verl's rollouts/<step>.jsonl: "input" is the decoded prompt,
"user\\n<content>\\nassistant\\n..."): the user messages predicted for a batch of a file are that step's distinct prompts
(whitespace-normalised).

    python skip_order.py build --train <parquet> --tokenizer <dir> --out <parquet> [--skip_steps 25 --tbs 64 --seed 4242 --maxp 1024]
    python skip_order.py check --train <parquet> --tokenizer <dir> --pair <batch>:<rollouts jsonl> [--pair ...] [--tbs 64 --seed 4242 --maxp 1024]

build prints BUILD_OK with the counts (rows, fitting rows, K) and writes <out> through <out>.tmp and os.replace; it exits
non-zero when the file it wrote does not keep every row (a different n would change the order). check prints one CHECK_OK /
CHECK_FAIL line per pair (predicted, found, missing, extra) and exits non-zero on any failure.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT_KWARGS = {"enable_thinking": False}   # E12b: +data.apply_chat_template_kwargs.enable_thinking=False


def sampler_order(n, seed):
    """verl's RandomSampler order for one epoch (diff_labels.sampler_order: torch.randperm, generator seeded by data.seed)."""
    sys.path.insert(0, os.path.join(HERE, "..", "analysis"))
    import diff_labels
    return diff_labels.sampler_order(n, seed)


def norm(s):
    return " ".join(str(s).split())


def user_content(messages):
    """The last user message of a parquet prompt (a list of {role, content})."""
    users = [m for m in messages if (m.get("role") if isinstance(m, dict) else None) == "user"]
    return str(users[-1]["content"]) if users else ""


def input_content(text):
    """The user message of a rollout dump's decoded prompt ("user\\n<content>\\nassistant\\n<think>...")."""
    t = str(text)
    if t.startswith("user\n"):
        t = t[len("user\n"):]
    head, sep, _ = t.rpartition("\nassistant\n")
    return head if sep else t


def messages_of(df):
    return [[dict(m) for m in p] for p in df["prompt"]]


def fit_rows(msgs, tokenizer, maxp, chat_kwargs=CHAT_KWARGS):
    """Indices of the prompts that fit maxp tokens on the generation template (verl's filter_overlong_prompts)."""
    return [i for i, m in enumerate(msgs) if len(tokenizer.apply_chat_template(m, add_generation_prompt=True, **chat_kwargs)) <= maxp]


def skip_index(fit, perm, k):
    """Row order of the continuation file: position perm[i] holds fit[perm[(i + k) mod n]]."""
    n = len(fit)
    assert sorted(perm) == list(range(n)), "perm is not a permutation of the fitting rows"
    new = [None] * n
    for i in range(n):
        new[perm[i]] = fit[perm[(i + k) % n]]
    return new


def predicted(msgs, fit, perm, batch, tbs):
    """The user messages of one batch of a file under the sampler model."""
    return [user_content(msgs[fit[perm[p]]]) for p in range(batch * tbs, (batch + 1) * tbs)]


def compare(pred, inputs):
    """(ok, detail): the predicted batch against a step's rollout prompts (each prompt n times in the dump)."""
    want = {norm(x) for x in pred}
    got = {norm(input_content(x)) for x in inputs}
    missing, extra = want - got, got - want
    return (len(want) == len(pred) and not missing and not extra,
            {"predicted": len(pred), "distinct_predicted": len(want), "found": len(got), "missing": len(missing), "extra": len(extra)})


def write_atomic(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)


def load(train, tokenizer, maxp, seed, perm_fn=None):
    import pandas as pd
    df = pd.read_parquet(train)
    msgs = messages_of(df)
    fit = fit_rows(msgs, tokenizer, maxp)
    perm = (perm_fn or sampler_order)(len(fit), seed)
    return df, msgs, fit, perm


def build(train, tokenizer, out, skip_steps=25, tbs=64, seed=4242, maxp=1024, perm_fn=None):
    df, msgs, fit, perm = load(train, tokenizer, maxp, seed, perm_fn)
    k = skip_steps * tbs
    new = skip_index(fit, perm, k)
    write_atomic(df.iloc[new].reset_index(drop=True), out)
    _, msgs2, fit2, perm2 = load(out, tokenizer, maxp, seed, perm_fn)
    r = {"rows": len(df), "fit": len(fit), "skip": k, "seed": seed, "out_rows": len(msgs2), "out_fit": len(fit2)}
    r["kept_every_row"] = len(fit2) == len(fit) == len(msgs2)
    # the first batch of the new file is the parent's batch skip_steps (positions k .. k + tbs - 1)
    r["first_batch_is_parent_batch"] = predicted(msgs2, fit2, perm2, 0, tbs) == predicted(msgs, fit, perm, skip_steps, tbs)
    return r


def check(train, tokenizer, pairs, tbs=64, seed=4242, maxp=1024, perm_fn=None):
    """pairs: [(batch index, rollouts jsonl)] -> [(ok, detail)]."""
    _, msgs, fit, perm = load(train, tokenizer, maxp, seed, perm_fn)
    out = []
    for b, path in pairs:
        inputs = [json.loads(x)["input"] for x in open(path) if x.strip()]
        ok, d = compare(predicted(msgs, fit, perm, b, tbs), inputs)
        d.update({"batch": b, "rollouts": os.path.basename(os.path.dirname(os.path.dirname(path))) + "/" + os.path.basename(path)})
        out.append((ok, d))
    return out


def tokenizer_of(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out")
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--skip_steps", type=int, default=25)
    ap.add_argument("--tbs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--maxp", type=int, default=1024)
    a = ap.parse_args(argv)
    tok = tokenizer_of(a.tokenizer)
    if a.cmd == "build":
        r = build(a.train, tok, a.out, a.skip_steps, a.tbs, a.seed, a.maxp)
        ok = r["kept_every_row"] and r["first_batch_is_parent_batch"]
        print(("BUILD_OK " if ok else "BUILD_FAIL ") + json.dumps(r))
        return 0 if ok else 1
    pairs = [(int(p.split(":", 1)[0]), p.split(":", 1)[1]) for p in a.pair]
    res = check(a.train, tok, pairs, a.tbs, a.seed, a.maxp)
    for ok, d in res:
        print(("CHECK_OK " if ok else "CHECK_FAIL ") + json.dumps(d))
    return 0 if res and all(ok for ok, _ in res) else 1


if __name__ == "__main__":
    sys.exit(main())
