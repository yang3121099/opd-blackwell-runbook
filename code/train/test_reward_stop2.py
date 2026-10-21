"""[v2.5.9f] Unit tests of reward_stop2.py (CPU, no model, no network): a stub math_verify (exact match of the boxed content)
and a one-token-per-word counter stand in for the real ones; PROBE-E12B-STOP2 checks the real ones on the node.
Run: python -m pytest -q code/train/test_reward_stop2.py"""
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
def r2(monkeypatch):
    mv = types.ModuleType("math_verify")
    mv.parse = lambda s: [c] if (c := _content(s)) else []
    mv.verify = lambda gold, pred: [g.replace(" ", "") for g in gold] == [p.replace(" ", "") for p in pred]
    monkeypatch.setitem(sys.modules, "math_verify", mv)
    monkeypatch.setenv("OPD_REWARD_MAXLEN", "4096")
    spec = importlib.util.spec_from_file_location("reward_stop2_t", os.path.join(HERE, "reward_stop2.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    s = m._stop()
    monkeypatch.setattr(s, "_ntok", lambda text: len(text.split()))      # one token per word
    return m


def words(n):
    return " ".join(["w"] * n)


def test_reward_is_continuous_in_the_tail(r2):
    f = lambda t: r2.compute_score("math", words(50) + " so \\boxed{42} " + words(t), "42")["score"]
    assert f(0) == 1.5 and f(512) == 1.25 and f(1024) == 1.0 and f(3000) == 1.0
    assert f(10) > f(100) > f(500) > 1.0                                  # two correct rollouts differ when their tails differ


def test_wrong_missing_and_cut(r2):
    w = r2.compute_score("math", words(10) + " \\boxed{41} " + words(3), "42")
    assert w["score"] == 0.0 and w["acc"] == 0.0 and w["bonus"] == 0.0 and w["tail"] == 3
    n = r2.compute_score("math", "the answer is 42", "42")
    assert n == {"score": 0.0, "acc": 0.0, "tail": -1, "cut": 0, "bonus": 0.0}
    c = r2.compute_score("math", "\\boxed{42} " + words(4094), "42")      # 4095 tokens: cut at the 4096 limit
    assert c["cut"] == 1 and c["tail"] == 1024 and c["score"] == 1.0


def test_tail_counts_from_the_first_box(r2):
    d = r2.compute_score("math", "\\boxed{41} " + words(100) + " \\boxed{42} " + words(5), "42")
    assert d["acc"] == 1.0 and d["tail"] == 106 and abs(d["score"] - (1 + 0.5 * (1 - 106 / 1024))) < 1e-9


def test_scale_from_the_environment(monkeypatch, r2):
    monkeypatch.setattr(r2, "SCALE", 512.0)
    assert r2.compute_score("math", "\\boxed{42} " + words(256), "42")["score"] == 1.25
