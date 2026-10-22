"""[v2.5.9z43] skip_order: the continuation file keeps the parent's order from position K, the rollout prompts are read back,
the check fails on a wrong batch. No pytest fixtures: PROBE-PROG calls these functions directly with the python the arms
train in (torch and pyarrow there: the two node tests run; here they return early)."""
import json
import os
import random
import tempfile

import skip_order as S


class Tok:
    """apply_chat_template's length = the content's length (+ 10 for the template)."""
    def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
        assert add_generation_prompt and kw == {"enable_thinking": False}
        return list(range(10 + len(messages[-1]["content"])))


def _perm(n, seed):
    p = list(range(n)); random.Random(seed).shuffle(p); return p


def _frame(n=40, long_every=7):
    import pandas as pd
    rows = []
    for i in range(n):
        q = f"Problem {i}: find x." + (" pad" * 300 if i % long_every == 3 else "")   # some rows too long for maxp
        rows.append({"data_source": "dapo", "prompt": [{"role": "user", "content": q + "\nPlease reason step by step."}],
                     "reward_model": {"ground_truth": str(i)}, "extra_info": {"index": i}})
    return pd.DataFrame(rows)


def _rollout(path, contents, n=8):
    with open(path, "w") as f:
        for c in contents:
            for _ in range(n):
                f.write(json.dumps({"input": "user\n" + c + "\nassistant\n<think>\n\n</think>\n\n", "output": "x", "score": 0.0}) + "\n")


class _FakeIO:
    """read_parquet / to_parquet without pyarrow: the file holds a key into a dict (so os.replace moves it), restored on exit."""
    def __enter__(self):
        import pandas as pd
        self.pd, self.frames = pd, {}
        self.r, self.w = pd.read_parquet, pd.DataFrame.to_parquet
        pd.read_parquet = lambda p, *a, **k: self.frames[open(p).read()].copy()
        def to_parquet(df, p, *a, **k):
            key = str(len(self.frames)); self.frames[key] = df.copy(); open(p, "w").write(key)
        pd.DataFrame.to_parquet = to_parquet
        return self

    def put(self, p, df):
        self.pd.DataFrame.to_parquet(df, p)

    def get(self, p):
        return self.frames[open(p).read()]

    def __exit__(self, *exc):
        self.pd.read_parquet, self.pd.DataFrame.to_parquet = self.r, self.w


def test_skip_index_continues_the_order():
    fit = [i for i in range(30) if i % 4 != 1]; n = len(fit)
    perm = _perm(n, 7)
    for k in (0, 5, n - 1, 3 * 4):
        new = S.skip_index(fit, perm, k)
        assert sorted(new) == sorted(fit)
        assert [new[perm[i]] for i in range(n)] == [fit[perm[(i + k) % n]] for i in range(n)]


def test_input_content_and_compare():
    assert S.input_content("user\nFind x.\nPlease reason.\nassistant\n<think>\n\n</think>\n\n") == "Find x.\nPlease reason."
    assert S.input_content("user\nA\nassistant\nB\nassistant\n") == "A\nassistant\nB"   # the last assistant turn is the cut
    ok, d = S.compare(["a  b", "c"], ["user\na b\nassistant\n"] * 8 + ["user\nc\nassistant\n"] * 8)
    assert ok and d == {"predicted": 2, "distinct_predicted": 2, "found": 2, "missing": 0, "extra": 0}
    assert not S.compare(["a", "c"], ["user\na\nassistant\n"])[0]                      # missing
    assert not S.compare(["a"], ["user\na\nassistant\n", "user\nz\nassistant\n"])[0]    # extra
    assert not S.compare(["a", "a"], ["user\na\nassistant\n"])[0]                       # a batch of 2 with one prompt


def test_build_then_check_against_rollouts():
    tmp = tempfile.mkdtemp(); src, out = os.path.join(tmp, "train.parquet"), os.path.join(tmp, "skip.parquet")
    with _FakeIO() as io:
        io.put(src, _frame())
        r = S.build(src, Tok(), out, skip_steps=3, tbs=4, seed=4242, maxp=60, perm_fn=_perm)
        assert r["kept_every_row"] and r["first_batch_is_parent_batch"] and r["skip"] == 12 and r["fit"] < r["rows"] == 40
        _, msgs, fit, perm = S.load(src, Tok(), 60, 4242, _perm)
        os.makedirs(os.path.join(tmp, "P", "rollouts"))
        r1, r4 = os.path.join(tmp, "P", "rollouts", "1.jsonl"), os.path.join(tmp, "P", "rollouts", "4.jsonl")
        _rollout(r1, S.predicted(msgs, fit, perm, 0, 4)); _rollout(r4, S.predicted(msgs, fit, perm, 3, 4))
        res = S.check(src, Tok(), [(0, r1), (3, r4)], tbs=4, seed=4242, maxp=60, perm_fn=_perm)
        assert [ok for ok, _ in res] == [True, True] and res[1][1]["rollouts"] == "P/4.jsonl" and res[1][1]["batch"] == 3
        # the continuation file: its batch 0 is the parent's step 4, not step 1; its batch 1 the parent's batch 4
        assert [ok for ok, _ in S.check(out, Tok(), [(0, r4), (0, r1)], tbs=4, seed=4242, maxp=60, perm_fn=_perm)] == [True, False]
        _, m2, f2, p2 = S.load(out, Tok(), 60, 4242, _perm)
        assert S.predicted(m2, f2, p2, 1, 4) == S.predicted(msgs, fit, perm, 4, 4)
        assert list(io.get(out).columns) == list(io.get(src).columns)


def test_build_reports_a_changed_n():
    tmp = tempfile.mkdtemp(); src, out = os.path.join(tmp, "train.parquet"), os.path.join(tmp, "skip.parquet")

    class Flaky(Tok):   # a filter that keeps fewer rows of the new file (as a different template would)
        calls = 0
        def apply_chat_template(self, messages, add_generation_prompt=True, **kw):
            Flaky.calls += 1
            return list(range(10 if Flaky.calls <= 40 or Flaky.calls % 5 else 999))
    with _FakeIO() as io:
        io.put(src, _frame())
        r = S.build(src, Flaky(), out, skip_steps=1, tbs=4, seed=1, maxp=60, perm_fn=_perm)
        assert not r["kept_every_row"]


def test_main_exit_codes():
    tmp = tempfile.mkdtemp(); src, out = os.path.join(tmp, "train.parquet"), os.path.join(tmp, "skip.parquet")
    with _FakeIO() as io:
        io.put(src, _frame())
        tok0, so0 = S.tokenizer_of, S.sampler_order
        S.tokenizer_of, S.sampler_order = (lambda p: Tok()), _perm
        try:
            assert S.main(["build", "--train", src, "--tokenizer", "t", "--out", out, "--skip_steps", "2", "--tbs", "4", "--maxp", "60"]) == 0
            _, msgs, fit, perm = S.load(src, Tok(), 60, 4242, _perm)
            r = os.path.join(tmp, "3.jsonl"); _rollout(r, S.predicted(msgs, fit, perm, 2, 4))
            assert S.main(["check", "--train", out, "--tokenizer", "t", "--pair", f"0:{r}", "--tbs", "4", "--maxp", "60"]) == 0
            assert S.main(["check", "--train", src, "--tokenizer", "t", "--pair", f"0:{r}", "--tbs", "4", "--maxp", "60"]) == 1
            assert S.main(["check", "--train", src, "--tokenizer", "t", "--tbs", "4", "--maxp", "60"]) == 1   # no pair: nothing checked
        finally:
            S.tokenizer_of, S.sampler_order = tok0, so0


def test_sampler_order_is_randomsampler_4242():   # the node (torch)
    try:
        import torch
    except ImportError:
        return
    g = torch.Generator(); g.manual_seed(4242)
    assert S.sampler_order(5000, 4242) == list(torch.utils.data.RandomSampler(list(range(5000)), generator=g))


def test_parquet_roundtrip_keeps_order():   # the node (pyarrow, torch)
    try:
        import pyarrow  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return
    tmp = tempfile.mkdtemp(); src, out = os.path.join(tmp, "train.parquet"), os.path.join(tmp, "skip.parquet")
    _frame(200).to_parquet(src)
    r = S.build(src, Tok(), out, skip_steps=5, tbs=8, seed=4242, maxp=60)
    assert r["kept_every_row"] and r["first_batch_is_parent_batch"] and not os.path.exists(out + ".tmp")
