"""[v2.5.8c] Unit tests of reward_stop.py (CPU, no model, no network): a stub math_verify (exact match of the boxed
content) and a whitespace tokenizer stand in for the real ones; the node's probe checks the real ones.
Run: python -m pytest -q code/train/test_reward_stop.py"""
import importlib.util
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _content(s):
    i = s.rfind("\\boxed{")
    if i < 0:
        return s.strip("$")
    k, depth = i + 7, 1
    while k < len(s) and depth:
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        k += 1
    return s[i + 7:k - 1] if not depth else None


@pytest.fixture
def rs(monkeypatch):
    mv = types.ModuleType("math_verify")
    mv.parse = lambda s: [c] if (c := _content(s)) else []
    mv.verify = lambda gold, pred: [g.replace(" ", "") for g in gold] == [p.replace(" ", "") for p in pred]
    monkeypatch.setitem(sys.modules, "math_verify", mv)
    monkeypatch.setenv("OPD_REWARD_MAXLEN", "4096")
    spec = importlib.util.spec_from_file_location("reward_stop_t", os.path.join(HERE, "reward_stop.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "_ntok", lambda text: len(text.split()))      # one token per word
    return m


def words(n):
    return " ".join(["w"] * n)


def test_single_box_ending_soon_gets_the_bonus(rs):
    assert rs.compute_score("math", words(200) + " so \\boxed{42} " + words(29), "42") == 1.5


def test_wrong_or_missing_answer_gets_nothing(rs):
    assert rs.compute_score("math", words(10) + " \\boxed{41} done", "42") == 0.0
    assert rs.compute_score("math", words(10) + " the answer is 42", "42") == 0.0


def test_long_tail_after_the_first_box_gets_no_bonus(rs):
    assert rs.compute_score("math", "\\boxed{42} " + words(64), "42") == 1.5          # 64 tokens after: still within
    assert rs.compute_score("math", "\\boxed{42} " + words(65), "42") == 1.0


def test_inconsistent_boxes_get_no_bonus(rs):
    # the last box is right, an earlier one is not: correct (1.0) but no bonus
    assert rs.compute_score("math", "\\boxed{41} wait \\boxed{42} ok", "42") == 1.0


def test_repeated_identical_boxes_count_from_the_first(rs):
    s = "\\boxed{42} " + words(30) + " \\boxed{ 4 2 } " + words(20)                  # whitespace-insensitive
    assert rs.compute_score("math", s, "42") == 1.5
    s = "\\boxed{42} " + words(60) + " \\boxed{42} " + words(20)                      # 82 tokens after the first box
    assert rs.compute_score("math", s, "42") == 1.0


def test_a_response_cut_at_the_length_limit_gets_no_bonus(rs, monkeypatch):
    monkeypatch.setenv("OPD_REWARD_MAXLEN", "40")
    s = words(31) + " \\boxed{42} " + words(8)                                        # 40 tokens: cut, not ended
    assert len(s.split()) == 40 and rs.compute_score("math", s, "42") == 1.0
    s = words(20) + " \\boxed{42} " + words(8)                                        # 29 tokens: ended
    assert rs.compute_score("math", s, "42") == 1.5


def test_nested_braces(rs):
    assert rs.boxes("a \\boxed{\\frac{1}{2}} b \\boxed{x")[0][2] == "\\frac{1}{2}"
    assert len(rs.boxes("a \\boxed{\\frac{1}{2}} b \\boxed{x")) == 1                  # the unclosed box is not a box
    assert rs.compute_score("math", "so \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}") == 1.5


def test_grading_is_reward_mathverify_s(rs):
    base = rs._base()
    for s in ("\\boxed{42} " + words(100), "\\boxed{41}", "no box", "\\boxed{41} \\boxed{42}"):
        assert rs.compute_score("math", s, "42") - base("math", s, "42") in (0.0, 0.5)
