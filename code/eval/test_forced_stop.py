"""Unit tests of forced_stop.py's bookkeeping (no model): which samples are forced at a budget, and the accuracies."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forced_stop as F  # noqa: E402


def rec(pid, s, n, fin, ok):
    return {"pid": pid, "sample": s, "token_ids": list(range(n)), "finish_reason": fin, "correct": ok, "gold": "5"}


R = [rec("a", 0, 5000, "stop", True), rec("a", 1, 12000, "stop", True), rec("b", 0, 32768, "length", False), rec("b", 1, 20000, "stop", False)]


def test_plan_by_budget():
    kept, forced = F.plan(R, 8192)
    assert [r["sample"] for r, _ in kept] == [0] and len(forced) == 3          # 12000 / 32768 / 20000 tokens: cut at 8K
    kept, forced = F.plan(R, 32768)
    assert len(kept) == 3 and [(r["pid"], r["sample"]) for r in forced] == [("b", 0)]   # only the 32K truncation


def test_summarize():
    s = F.summarize(R, 16384, {("b", 0): True, ("b", 1): False})
    assert s == {"budget": 16384, "n": 4, "acc_no_force": 50.0, "acc_forced": 75.0, "n_forced": 2, "forced_correct": 1}
    assert F.FORCE == "\n</think>\n\n**Final Answer**\n\\boxed{"
