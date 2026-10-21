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
