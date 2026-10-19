"""Tests of strict_contract.py (runbook v2.5.7o): box counting, the answer part of thinking readings, the contract
clauses, the scan of a runs directory and the paired comparison under both graders (no model, no Math-Verify)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strict_contract as SC  # noqa: E402


def test_boxes():
    assert SC.boxes("x \\boxed{5} y") == (["5"], 0)
    assert SC.boxes("\\boxed{\\frac{1}{2}} and \\boxed{ 1/2 }") == (["\\frac{1}{2}", " 1/2 "], 0)
    assert SC.boxes("\\boxed{3} then \\boxed{\\frac{1}{") == (["3"], 1)          # cut inside the second box
    assert SC.boxes("no answer") == ([], 0)


def test_answer_part():
    assert SC.answer_part("a \\boxed{1}", False) == "a \\boxed{1}"
    assert SC.answer_part("<think>try \\boxed{2}</think>\n\\boxed{1}", True) == "\n\\boxed{1}"
    assert SC.answer_part("<think>still thinking \\boxed{2}", True) is None


def rec(pid, text, correct, fr="stop", sample=0):
    return {"pid": pid, "sample": sample, "text": text, "correct": correct, "finish_reason": fr, "grader_version": 2}


def test_judge_clauses():
    j = SC.judge(rec("a", "so \\boxed{4}", True), False)
    assert j["strict"] and j["same"]
    assert not SC.judge(rec("a", "so \\boxed{4}", True, "length"), False)["strict"]           # not ended at EOS
    j = SC.judge(rec("a", "\\boxed{4} check: \\boxed{ 4 }", True), False)
    assert not j["strict"] and j["same"] and j["nbox"] == 2                                   # two boxes, same content
    j = SC.judge(rec("a", "\\boxed{3} no wait \\boxed{4}", True), False)
    assert not j["strict"] and not j["same"]                                                  # two different boxes
    assert not SC.judge(rec("a", "\\boxed{4}", False), False)["strict"]                       # wrong stays wrong
    # thinking: a box inside the thought does not count, the answer part needs its own
    assert SC.judge(rec("a", "<think>\\boxed{3}</think> \\boxed{4}", True), True)["strict"]
    assert not SC.judge(rec("a", "<think>\\boxed{4}</think> it is 4", True), True)["strict"]


def write_reading(d, task, stem, recs, bench="math500", mode="no_think", sub=""):
    p = os.path.join(d, task, sub); os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, stem + ".jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump({"bench": bench, "mode": mode, "temperature": 0.6, "max_tokens": 16384, "avg_acc": 0.0, "model": "/node/path/m"},
              open(os.path.join(p, stem + ".metrics.json"), "w"))


def test_scan_and_main(tmp_path, monkeypatch):
    runs = str(tmp_path / "runs")
    # A: 4 problems x 2 samples, all correct; half of them with two different boxes or cut at the limit
    A = [rec(f"p{i}", "\\boxed{1}", True, sample=0) for i in range(4)] + \
        [rec("p0", "\\boxed{2} \\boxed{1}", True, sample=1), rec("p1", "\\boxed{1}", True, "length", 1),
         rec("p2", "\\boxed{1} \\boxed{1}", True, sample=1), rec("p3", "\\boxed{1}", True, sample=1)]
    B = [rec(f"p{i}", "\\boxed{1}", i < 2, sample=s) for i in range(4) for s in range(2)]
    write_reading(runs, "EVAL-x", "a_math500_no_think_t0.6", A)
    write_reading(runs, "EVAL-x", "b_math500_no_think_t0.6", B)
    write_reading(runs, "PROBE-TERM", "term_c_math500_no_think_t0.6", B, sub="gen")
    write_reading(runs, "EVAL-y", "d_aime24_no_think_t0.6", B, bench="aime24")                # not MATH500
    open(os.path.join(runs, "EVAL-x", "a_math500_no_think_t0.6.forced.jsonl"), "w").write("{}\n")   # never a reading
    os.makedirs(os.path.join(runs, "EVAL-z")); open(os.path.join(runs, "EVAL-z", "e_math500.jsonl"), "w").write("{}\n")  # no metrics
    names = [os.path.relpath(p, runs) for p in SC.scan(runs)]
    assert names == ["EVAL-x/a_math500_no_think_t0.6.jsonl", "EVAL-x/b_math500_no_think_t0.6.jsonl",
                     "PROBE-TERM/gen/term_c_math500_no_think_t0.6.jsonl"]
    out, tab = tmp_path / "r.json", tmp_path / "t.txt"
    monkeypatch.setattr(sys, "argv", ["strict_contract.py", "--scan", runs, "--out", str(out), "--table", str(tab),
                                      "--pair", "EVAL-x/a_math500_no_think_t0.6.jsonl", "EVAL-x/b_math500_no_think_t0.6.jsonl",
                                      "--pair", "EVAL-x/a_math500_no_think_t0.6.jsonl", "EVAL-q/none.jsonl"])
    SC.main()
    r = json.load(open(out))
    a = {x["name"]: x for x in r["readings"]}["EVAL-x/a_math500_no_think_t0.6.jsonl"]
    assert a["acc"] == 100.0 and a["acc_contract"] == 62.5 and a["diff"] == -37.5 and a["acc_same"] == 75.0
    assert a["share"]["eos"] == 87.5 and a["share"]["multi_box"] == 25.0 and a["share"]["one_box"] == 75.0
    assert a["of_correct_rejected"] == {"not_eos": 12.5, "multi": 25.0, "none": 0.0, "unclosed": 0.0}
    assert "model" not in a and "/node/" not in open(out).read()                             # path-free report
    p = r["pairs"][0]
    assert p["lenient"]["diff"] == 50.0 and p["contract"]["diff"] == 12.5 and p["diff_change"] == -37.5
    assert r["pairs"][1]["error"] == "reading missing"
    t = open(tab).read()
    assert "EVAL-x/a_math500_no_think_t0.6.jsonl" in t and "-37.50" in t and "reading missing" in t
