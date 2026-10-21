"""[v2.5.9f] Unit tests of e12_rollout_stats.step_stats (no tokenizer: one token per word)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e12_rollout_stats as E  # noqa: E402

ntok = lambda t: len(t.split())


def test_old_columns_and_spread():
    recs = [{"input": "a", "output": "x y", "score": 1.0}, {"input": "a", "output": "x", "score": 0.0},
            {"input": "b", "output": "x y z", "score": 1.0}, {"input": "b", "output": "x", "score": 1.0}]
    r = E.step_stats(3, recs, ntok)
    assert (r["step"], r["groups"], r["all_correct"], r["mixed"], r["reward_mean"], r["len_tok_mean"]) == (3, 2, 0.5, 0.5, 0.75, 1.8)
    assert r["zero_var_groups"] == 0.5 and abs(r["reward_std"] - 0.433) < 1e-3 and r["tail_mean_correct"] == "" and r["cut_share"] == ""


def test_stop2_parts():   # correct rollouts with different tails: different rewards, no zero-variance group
    recs = [{"input": "a", "output": "o", "score": 1.5, "acc": 1.0, "tail": 0, "cut": 0, "bonus": 0.5},
            {"input": "a", "output": "o", "score": 1.25, "acc": 1.0, "tail": 512, "cut": 0, "bonus": 0.25},
            {"input": "b", "output": "o", "score": 1.0, "acc": 1.0, "tail": 1024, "cut": 1, "bonus": 0.0},
            {"input": "b", "output": "o", "score": 0.0, "acc": 0.0, "tail": -1, "cut": 0, "bonus": 0.0}]
    r = E.step_stats(1, recs, ntok)
    assert r["zero_var_groups"] == 0.0 and r["tail_mean_correct"] == 512.0 and r["tail_p50_correct"] == 512 and r["tail_p90_correct"] == 1024
    assert r["cut_share"] == 0.25 and r["bonus_mean"] == 0.1875 and r["all_correct"] == 0.5


def test_collect_settle_and_cache(tmp_path=None):   # [v2.5.9g] a fresh dump is skipped; a cached step is not read again
    import json, tempfile, time
    run = tmp_path or tempfile.mkdtemp(); run = str(run); os.makedirs(os.path.join(run, "rollouts"))
    rec = lambda s: json.dumps({"input": "a", "output": "x y", "score": s}) + "\n"
    for step in (1, 2, 10):
        open(os.path.join(run, "rollouts", f"{step}.jsonl"), "w").write(rec(1.0) + rec(0.0))
    old = time.time() - 300
    for step in (1, 2):
        os.utime(os.path.join(run, "rollouts", f"{step}.jsonl"), (old, old))
    calls = []
    fac = lambda: calls.append(1) or ntok
    rows = E.collect(run, fac, settle=60)
    assert [r["step"] for r in rows] == [1, 2] and len(calls) == 1   # step 10 is being written: skipped
    rows = E.collect(run, fac, settle=0)
    assert [r["step"] for r in rows] == [1, 2, 10] and len(calls) == 2   # steps 1 and 2 from the cache, 10 read
    rows = E.collect(run, lambda: (_ for _ in ()).throw(AssertionError("tokenizer loaded")), settle=0)
    assert [r["step"] for r in rows] == [1, 2, 10] and rows[0]["zero_var_groups"] == 0.0
    open(os.path.join(run, "rollouts", "2.jsonl"), "w").write(rec(1.0) + rec(1.0))   # a dump rewritten after its cache: read again
    future = time.time() + 5
    os.utime(os.path.join(run, "rollouts", "2.jsonl"), (future, future))
    rows = E.collect(run, fac, settle=0, now=future + 1)
    assert rows[1]["zero_var_groups"] == 1.0 and len(calls) == 3
