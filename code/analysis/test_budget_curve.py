"""Tests of budget_curve.py (runbook v2.5.7g) with a stub tokenizer and grader (no model, no Math-Verify)."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_curve as BC  # noqa: E402


def stub_grade(text, gold):
    return f"[{gold}]" in text, None


def decode(ids):
    return "".join(chr(i) for i in ids)


def rec(pid, s, text, gold, ended=True):
    ids = [ord(c) for c in text]
    return {"pid": pid, "sample": s, "gold": gold, "correct": stub_grade(text, gold)[0] and ended, "n_tokens": len(ids),
            "finish_reason": "stop" if ended else "length", "token_ids": ids}


def test_at_budget_cuts_and_regrades():
    r = [rec(0, 0, "ab[7]", "7"), rec(1, 0, "xxxxxxxx[3]", "3"), rec(2, 0, "[5]yyyyyyy", "5", ended=False)]
    rows, k = BC.at_budget(r, 5, decode, stub_grade)
    assert rows == [(0, True, False, 5, 0), (1, False, True, 5, 0), (2, True, True, 5, 0)] and k == 2   # [5] survives the cut
    rows, k = BC.at_budget(r, 100, decode, stub_grade)
    assert [c for _, c, _, _, _ in rows] == [True, True, False] and [t for _, _, t, _, _ in rows] == [False, False, True] and k == 0


def test_crossing():
    c = BC.crossing([2048, 4096, 8192, 16384], [-3.0, -1.0, 1.0, 2.0], [-5, -3, -0.5, 0.2])
    assert c["first_positive"] == 8192 and c["first_ci_positive"] == 16384 and 4096 < c["log2_interp"] < 8192 and c["log2_interp"] == 5793
    assert BC.crossing([2048, 4096], [1.0, 2.0], [0.1, 0.2])["positive_at_all"] is True
    c = BC.crossing([2048, 4096], [-1.0, None], [None, None])
    assert c["first_positive"] is None and c["positive_at_all"] is False


def test_main_end_to_end(tmp_path, monkeypatch):
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: types.SimpleNamespace(decode=lambda ids, skip_special_tokens=True: decode(ids)))
    fake_ge = types.ModuleType("gen_eval"); fake_ge.grade = stub_grade
    monkeypatch.setitem(sys.modules, "transformers", fake_tf); monkeypatch.setitem(sys.modules, "gen_eval", fake_ge)
    def write(name, recs, mt):
        p = tmp_path / f"{name}.jsonl"; p.write_text("".join(json.dumps(x) + "\n" for x in recs))
        (tmp_path / f"{name}.metrics.json").write_text(json.dumps({"max_tokens": mt})); return str(p)
    # base: right but long (12 chars); reading: right and short (4 chars) on half the problems, wrong otherwise
    base = write("base", [rec(p, s, "z" * 9 + f"[{p % 10}]", str(p % 10)) for p in range(20) for s in range(2)], 16)
    rd = write("p5", [rec(p, s, (f"[{p % 10}]" if p % 2 == 0 else "[x]") + "q", str(p % 10)) for p in range(20) for s in range(2)], 16)
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["budget_curve.py", "--out", str(out), "--budgets", "4,8,16,32", "--tokenizer", "t", base, rd, str(tmp_path / "gone.jsonl")])
    BC.main()
    r = json.load(open(out))
    assert r["base"] == "base" and r["missing"] == ["gone"] and "32" not in r["readings"]["base"]["by_budget"]   # above max_tokens
    assert r["readings"]["base"]["by_budget"]["8"]["acc"] == 0.0 and r["readings"]["base"]["by_budget"]["16"]["acc"] == 100.0
    assert r["readings"]["base"]["by_budget"]["8"]["trunc_rate"] == 1.0 and r["readings"]["p5"]["by_budget"]["8"]["trunc_rate"] == 0.0
    pr = r["pairs_vs_base"]["p5"]
    assert pr["8"]["diff"] == 50.0 and pr["16"]["diff"] == -50.0
    assert r["crossing"]["p5"]["first_positive"] is None and r["crossing"]["p5"]["first_ci_positive"] == 4


def test_forced_and_finished(tmp_path, monkeypatch):   # v2.5.7h (Q-013)
    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda p: types.SimpleNamespace(decode=lambda ids, skip_special_tokens=True: decode(ids)))
    fake_ge = types.ModuleType("gen_eval"); fake_ge.grade = stub_grade
    monkeypatch.setitem(sys.modules, "transformers", fake_tf); monkeypatch.setitem(sys.modules, "gen_eval", fake_ge)
    def write(name, recs):
        p = tmp_path / f"{name}.jsonl"; p.write_text("".join(json.dumps(x) + "\n" for x in recs))
        (tmp_path / f"{name}.metrics.json").write_text(json.dumps({"max_tokens": 16})); return str(p)
    # base: right but long (12 chars) everywhere; p5: short and right on even problems
    base = write("base", [rec(p, s, "z" * 9 + f"[{p % 10}]", str(p % 10)) for p in range(10) for s in range(2)])
    rd = write("p5", [rec(p, s, (f"[{p % 10}]" if p % 2 == 0 else "[x]") + "q", str(p % 10)) for p in range(10) for s in range(2)])
    # forced answers at 8 tokens: the base recovers on every problem but 0 (sample 1 of problem 1 has none -> wrong, missing)
    (tmp_path / "base.forced.jsonl").write_text("".join(json.dumps({"pid": p, "sample": s, "budget": 8, "forced_correct": p != 0})
                                                       + "\n" for p in range(10) for s in range(2) if (p, s) != (1, 1)))
    out = tmp_path / "o.json"
    monkeypatch.setattr(sys, "argv", ["budget_curve.py", "--out", str(out), "--budgets", "8,16", "--tokenizer", "t", "--forced_dir", str(tmp_path), base, rd])
    BC.main()
    r = json.load(open(out))
    b8 = r["readings"]["base"]["by_budget"]["8"]
    assert b8["acc"] == 0.0 and b8["finished_rate"] == 0.0 and b8["acc_finished"] is None and b8["acc_forced"] == 85.0 and b8["forced_missing"] == 1
    p8 = r["readings"]["p5"]["by_budget"]["8"]
    assert p8["finished_rate"] == 1.0 and p8["acc_finished"] == 50.0 and "acc_forced" not in p8 and r["readings"]["p5"]["forced"] is False
    assert r["pairs_forced_vs_base"] == {}                   # p5 has no forced answers: no forced pair
