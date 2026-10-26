"""[v2.5.9z89] DIAG-VLM-TRUNC's classes on synthetic gen_eval records (no model, CPU only)."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "analysis"))
import vlm_trunc_classify as V  # noqa: E402


def test_loop_window_and_repeats():
    assert V.has_loop(list(range(20)) * 3) and not V.has_loop(list(range(20)) * 2)          # 3 x a 20-token window
    assert V.has_loop([1, 2, 3] * 30) and not V.has_loop(list(range(200)))                 # a short period repeats too


def test_box_end_balanced():
    t = "so \\boxed{\\frac{1}{2}} done"
    assert t[V.box_end(t):] == " done" and V.box_end("no box") == -1 and V.box_end("\\boxed{1") == -1


def test_classes_and_summary():
    recs = [{"finish_reason": "length", "token_ids": list(range(20)) * 3, "text": "x"},                      # loop
            {"finish_reason": "length", "token_ids": list(range(20)) * 3, "text": "\\boxed{3} and"},          # loop with a box
            {"finish_reason": "length", "token_ids": list(range(100)), "text": "thinking..."},               # unfinished
            {"finish_reason": "length", "token_ids": list(range(100)), "text": "\\boxed{3} wait, check"},    # box then more
            {"finish_reason": "stop", "token_ids": [1], "text": "\\boxed{3}"}]                               # not truncated
    d = tempfile.mkdtemp(); p = os.path.join(d, "a.jsonl")
    open(p, "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
    r = V.summarize(p)
    assert r["samples"] == 5 and r["truncated"] == 4 and r["counts"] == {"loop": 2, "unfinished": 1, "box_more": 1}
    assert r["shares"]["loop"] == 0.5 and r["loop_with_box"] == 1
    out = os.path.join(d, "o.json"); sys.argv = ["x", out, f"base_mathvista={p}"]; V.main()
    assert json.load(open(out))["inputs"]["base_mathvista"]["truncated"] == 4
