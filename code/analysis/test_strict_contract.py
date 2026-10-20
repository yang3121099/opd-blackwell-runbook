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


def test_ready():
    # the first box holding the final answer ends at char 12; a correct sample whose answer is never boxed earlier
    t = "x \\boxed{3} y \\boxed{4} z \\boxed{ 4 }"
    assert SC.ready(t, ["3", "4", " 4 "], True) == (t.index("\\boxed{4}") + len("\\boxed{4}"), 2, len(t))
    assert SC.ready(t, ["3", "4", " 4 "], False)[0] is None
    assert SC.ready("no box", [], True) == (None, None, 6)
    assert SC.q([5, 1, 3, 2, 4], .5) == 3 and SC.q([], .5) is None


def test_think_structure(tmp_path, monkeypatch):
    # v2.5.7x: a thinking reading whose answer is boxed inside the thought and again in the summary
    monkeypatch.setattr(SC, "_GRADE", lambda text, gold: (text.endswith("{" + gold + "}"), None))
    recs = [{"pid": "p0", "sample": 0, "text": "<think>so \\boxed{4}</think> answer \\boxed{4}", "correct": True, "finish_reason": "stop", "n_tokens": 40, "gold": "4"},
            {"pid": "p1", "sample": 0, "text": "<think>hmm</think> \\boxed{7}", "correct": True, "finish_reason": "stop", "n_tokens": 30, "gold": "7"}]
    runs = str(tmp_path / "runs")
    write_reading(runs, "EVAL-t", "t_math500_think_t0.6", recs, mode="think")
    r, _ = SC.rejudge(os.path.join(runs, "EVAL-t", "t_math500_think_t0.6.jsonl"), "t")
    a = r["answer_ready"]
    assert a["correct_multi_box_same_share"] == 50.0 and a["correct_closing_think"] == 2 and a["answer_box_before_think_close_share"] == 50.0
    assert a["post_think_tok_p50"] is not None


def test_stop_position_and_tail(tmp_path):
    # v2.5.7ze: correct + EOS samples only; tokens after the last box; endings with the answer and digits masked
    recs = [{"pid": "p0", "sample": 0, "text": "a long derivation \\boxed{12} then \\boxed{12}.", "correct": True, "finish_reason": "stop", "n_tokens": 30},
            {"pid": "p1", "sample": 0, "text": "b long derivation \\boxed{34} then \\boxed{34}.", "correct": True, "finish_reason": "stop", "n_tokens": 30},
            {"pid": "p2", "sample": 0, "text": "\\boxed{3} and more text 1", "correct": True, "finish_reason": "length", "n_tokens": 50},
            {"pid": "p3", "sample": 0, "text": "\\boxed{9}", "correct": False, "finish_reason": "stop", "n_tokens": 5}]
    f = tmp_path / "x.jsonl"; f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    a = SC.rejudge(str(f), "x")[0]["answer_ready"]
    assert a["stop_n"] == 2 and a["stop_after_last_box_tok_p50"] in (1, 2) and a["stop_tail_top5"][0][1] == 2
    assert a["stop_tail_top5"][0][0] == "n \\boxed{99} then \\boxed{<A>}." and a["stop_tail_top5"][0][2] == 100.0
    assert SC.tail_pattern("x \\boxed{42} 2024 end", SC.boxes_at("x \\boxed{42} 2024 end")[0]) == "x \\boxed{<A>} 9999 end"


def test_first_box(monkeypatch):
    monkeypatch.setattr(SC, "_GRADE", lambda text, gold: (text.endswith("{" + gold + "}"), None))   # v2.5.7w: stub grader
    r = {"text": "\\boxed{2} no \\boxed{1}", "correct": True, "gold": "1"}
    assert SC.first_box(r) == (False, len("\\boxed{2}"))                 # graded again: the first box says 2
    r = {"text": "\\boxed{1} so \\boxed{ 1 }", "correct": True, "gold": "1"}
    assert SC.first_box(r) == (True, len("\\boxed{1}"))                  # same content: the last box's grade
    assert SC.first_box({"text": "none", "correct": False}) == (False, None)
    assert SC.hist([0, 255, 256, 99999]) [:2] == [2, 1] and SC.hist([99999])[-1] == 1 and len(SC.hist([])) == 128


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
    assert not j["strict"] and not j["same"] and j["inconsistent"]                            # two different boxes
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
    monkeypatch.setattr(SC, "_GRADE", lambda text, gold: (text.endswith("{" + gold + "}"), None))
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
    write_reading(runs, "EVAL-w", "w_math500_no_think_t0.6", [rec("p0", "no", False)])        # v2.5.7t: no correct sample
    open(os.path.join(runs, "EVAL-x", "a_math500_no_think_t0.6.forced.jsonl"), "w").write("{}\n")   # never a reading
    os.makedirs(os.path.join(runs, "EVAL-z")); open(os.path.join(runs, "EVAL-z", "e_math500.jsonl"), "w").write("{}\n")  # no metrics
    names = [os.path.relpath(p, runs) for p in SC.scan(runs)]
    assert names == ["EVAL-w/w_math500_no_think_t0.6.jsonl", "EVAL-x/a_math500_no_think_t0.6.jsonl", "EVAL-x/b_math500_no_think_t0.6.jsonl",
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
    assert a["of_correct_rejected"] == {"not_eos": 12.5, "multi": 25.0, "none": 0.0, "unclosed": 0.0, "inconsistent": 12.5}
    # v2.5.7s: same loses 25 points: 12.5 not ended at EOS (p1 s1), 12.5 boxes that disagree (p0 s1)
    assert a["same_loss_points_exclusive"] == {"not_eos": 12.5, "unclosed": 0.0, "no_box": 0.0, "inconsistent": 12.5}
    assert round(sum(a["same_loss_points_exclusive"].values()), 2) == round(a["acc"] - a["acc_same"], 2)
    assert "model" not in a and "/node/" not in open(out).read()                             # path-free report
    assert a["answer_ready"]["correct_with_answer_box"] == 8 and a["answer_ready"]["after_gt_500_share"] == 0.0   # v2.5.7v
    ar = a["answer_ready"]   # v2.5.7w: p0 s1's first box (2) is wrong; the rest open with the final answer
    assert ar["first_box_acc"] == 87.5 and sum(ar["ready_tok_hist"]) == 8 and len(ar["after_tok_hist"]) == 128 and ar["bin_tokens"] == 256
    # [v2.5.8d] p0 s1 changed its answer from a wrong first box (2) to a right last one (1); p2 s1 repeats the same box
    assert ar["box_change"] == {"n": 1, "share": 12.5, "first_right_last_wrong": 0, "first_wrong_last_right": 1, "both_right": 0, "both_wrong": 0,
                                "first_right_last_wrong_points": 0.0, "first_wrong_last_right_points": 12.5}
    p = r["pairs"][0]
    assert p["lenient"]["diff"] == 50.0 and p["contract"]["diff"] == 12.5 and p["diff_change"] == -37.5
    assert r["pairs"][1]["error"] == "reading missing"
    t = open(tab).read()
    assert "EVAL-x/a_math500_no_think_t0.6.jsonl" in t and "-37.50" in t and "reading missing" in t
    assert "answer ready (v2.5.7v)" in t
    w = {x["name"]: x for x in r["readings"]}["EVAL-w/w_math500_no_think_t0.6.jsonl"]
    assert w["acc"] == 0.0 and w["of_correct_rejected"]["not_eos"] is None and "EVAL-w/w_math500" in t.split("same (the C1")[1]


def test_box_change(tmp_path, monkeypatch):   # [v2.5.8d] which way did a changed answer go?
    monkeypatch.setattr(SC, "_GRADE", lambda text, gold: (text.endswith("{" + gold + "}"), None))
    g = lambda r: dict(r, gold="1")
    R = [g(rec("p0", "\\boxed{1} wait \\boxed{2}", False)),      # right, then changed to wrong
         g(rec("p1", "\\boxed{2} wait \\boxed{1}", True)),       # wrong, then changed to right
         g(rec("p2", "\\boxed{3} wait \\boxed{2}", False)),      # wrong both times
         g(rec("p3", "\\boxed{1} so \\boxed{ 1 }", True)),       # the same answer twice: not a change
         g(rec("p4", "\\boxed{1}", True))]
    write_reading(str(tmp_path), "EVAL-c", "c_math500_no_think_t0.6", R)
    res, _ = SC.rejudge(str(tmp_path / "EVAL-c" / "c_math500_no_think_t0.6.jsonl"), "c")
    bc, ar = res["answer_ready"]["box_change"], res["answer_ready"]
    assert (bc["n"], bc["first_right_last_wrong"], bc["first_wrong_last_right"], bc["both_wrong"], bc["both_right"]) == (3, 1, 1, 1, 0)
    assert bc["share"] == 60.0 and bc["first_right_last_wrong_points"] == 20.0
    assert round(bc["first_right_last_wrong_points"] - bc["first_wrong_last_right_points"], 2) == round(ar["first_box_acc"] - res["acc"], 2)
