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


def test_budb_stop_strings():   # v2.5.7h (Q-013); E05 keeps FORCE whatever the mode
    assert F.force_text("e05", "no_think") == F.FORCE and F.force_text("e05", "think", "a</think>b") == F.FORCE
    assert F.force_text("budB", "no_think") == "\n\n**Final Answer**\n\\boxed{"
    assert F.force_text("budB", "think", "still thinking") == "\n</think>\n\n\\boxed{"
    assert F.force_text("budB", "think", "done</think>answer") == "\n\n**Final Answer**\n\\boxed{"   # think already closed


def test_budgets_upto():   # [v2.5.9z46] a 3072 reading is forced at 2048 and at its cap; the usual caps unchanged
    B = [2048, 4096, 8192, 16384, 32768]
    assert F.budgets_upto(B, 3072) == [2048, 3072] and F.budgets_upto(B, 16384) == [2048, 4096, 8192, 16384] and F.budgets_upto(B, 32768) == B
