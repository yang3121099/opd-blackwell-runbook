"""[v2.5.9x1] readings_extra.py on a fake runs directory: splits 1-100 / 101-500 / all / easy / hard, the frontier rows from a budget
curve and a metrics.json, missing inputs listed, outputs where the upload expects them."""
import json
import os
import tempfile

import readings_extra as X


def setup():
    runs = tempfile.mkdtemp(); out = tempfile.mkdtemp()
    os.makedirs(os.path.join(runs, "EVAL-a")); os.makedirs(os.path.join(runs, "PROBE-DIFF17")); os.makedirs(os.path.join(runs, "EVAL-b"))
    with open(os.path.join(runs, "EVAL-a", "a_math500.jsonl"), "w") as f:
        for i in range(500):
            for s in range(2):   # 1-100: all correct, 100 tokens; 101-500: half correct, 300 tokens; one truncated sample at 499
                f.write(json.dumps({"pid": f"math500-{i}", "sample": s, "correct": i < 100 or s == 0, "n_tokens": 100 if i < 100 else 300,
                                    "finish_reason": "length" if (i == 499 and s == 1) else "stop", "text": "x" * 50, "token_ids": [1] * 5}) + "\n")
    json.dump({"avg_acc": 60.0}, open(os.path.join(runs, "EVAL-a", "a_math500.metrics.json"), "w"))
    with open(os.path.join(runs, "PROBE-DIFF17", "labels_17.jsonl"), "w") as f:
        for i in range(500):
            f.write(json.dumps({"split": "math500", "pos": i, "key": "k", "k": 8 if i < 250 else 0, "n": 8, "r": 1.0 if i < 250 else 0.0}) + "\n")
        f.write(json.dumps({"split": "train", "pos": 0, "key": "t", "k": 0, "n": 8, "r": 0.0}) + "\n")
    json.dump({"readings": {"ref_x": {"by_budget": {"16384": {"acc": 72.6, "trunc_rate": 0.09, "mean_tokens": 5372.9}}}}},
              open(os.path.join(runs, "EVAL-b", "budget_curve.json"), "w"))
    json.dump({"mean_tokens": 1449.0, "trunc_rate": 0.01, "avg_acc": 55.7, "max_tokens": 16384}, open(os.path.join(runs, "EVAL-b", "m.metrics.json"), "w"))
    q = os.path.join(out, "queue.txt")
    open(q, "w").write("#split READINGS-EXTRA 17_A EVAL-a/a_math500.jsonl\n#split READINGS-EXTRA 17_gone EVAL-z/z.jsonl\n"
                       "#frontier READINGS-EXTRA 06think EVAL-b/budget_curve.json ref_x\n#frontier READINGS-EXTRA Pstop EVAL-b/m.metrics.json\n"
                       "#frontier READINGS-EXTRA bad EVAL-b/budget_curve.json not_there\n#eval EVAL-a x y math500 4 0.6\n")
    return runs, out, q


def test_split_and_frontier():
    runs, out, q = setup()
    s = X.main(runs, q, out)
    a = s["split"]["17_A"]
    assert a["1-100"] == {"acc": 100.0, "mean_tokens": 100.0, "trunc_rate": 0.0, "n_problems": 100, "n_samples": 200}
    assert a["101-500"]["acc"] == 50.0 and a["101-500"]["mean_tokens"] == 300.0 and a["101-500"]["n_problems"] == 400 and a["101-500"]["trunc_rate"] == round(1 / 800, 4)
    assert a["all"]["acc"] == 60.0 and a["metrics_avg_acc"] == 60.0
    assert a["easy"]["n_problems"] == 250 and a["hard"]["n_problems"] == 250 and a["easy"]["acc"] == 70.0 and a["easy"]["mean_tokens"] == 220.0 and a["hard"]["acc"] == 50.0
    assert s["frontier"]["06think"]["mean_tokens_16k"] == 5372.9 and s["frontier"]["06think"]["source"] == "EVAL-b/budget_curve.json [ref_x]"
    assert s["frontier"]["Pstop"] == {"mean_tokens_16k": 1449.0, "trunc_16k": 0.01, "acc_16k": 55.7, "how": "metrics.json (max_tokens 16384)", "source": "EVAL-b/m.metrics.json"}
    assert "EVAL-z/z.jsonl" in s["missing"] and any(m.startswith("EVAL-b/budget_curve.json") for m in s["missing"])
    up = os.path.join(out, "up", "runs")
    assert json.load(open(os.path.join(up, "EVAL-a", "a_math500.split.json")))["reading"] == "EVAL-a/a_math500.jsonl"
    assert os.path.exists(os.path.join(runs, "EVAL-a", "a_math500.split.json"))
    assert set(json.load(open(os.path.join(up, "READINGS-EXTRA", "frontier_fill.json")))) == {"06think", "Pstop"}
    assert "17_A | 100.0 / 100.0 | 50.0 / 300.0 | 60.0 (60.0) | 70.0 / 220.0" in open(os.path.join(out, "summary.txt")).read()


def test_no_labels_no_easy_rows():
    runs, out, q = setup()
    os.remove(os.path.join(runs, "PROBE-DIFF17", "labels_17.jsonl"))
    a = X.main(runs, q, out)["split"]["17_A"]
    assert "easy" not in a and a["1-100"]["acc"] == 100.0


def test_refused_paths_and_the_metrics_mismatch():   # [v2.5.9z3] ops-lead's A1 notes on 9x1
    runs, out, q = setup()
    json.dump({"avg_acc": 61.0}, open(os.path.join(runs, "EVAL-a", "a_math500.metrics.json"), "w"))   # all 500 is 60.0
    open(q, "a").write("#split READINGS-EXTRA bad ../etc/x.jsonl\n#frontier READINGS-EXTRA bad2 /abs/m.metrics.json\n")
    s = X.main(runs, q, out)
    assert "../etc/x.jsonl: refused path" in s["missing"] and "/abs/m.metrics.json: refused path" in s["missing"]
    assert s["mismatch"] == ["17_A: all 60.0 vs metrics 61.0"] and "mismatch: 17_A" in open(os.path.join(out, "summary.txt")).read()


def test_pairq_paired_bootstrap():   # [v2.5.9z5] problem-level pairs of two readings, the lab's method
    runs, out, q = setup()
    os.makedirs(os.path.join(runs, "EVAL-c"))
    with open(os.path.join(runs, "EVAL-c", "c_aime.jsonl"), "w") as f, open(os.path.join(runs, "EVAL-c", "d_aime.jsonl"), "w") as g:
        for i in range(30):
            for s in range(4):
                f.write(json.dumps({"pid": f"aime24-{i}", "sample": s, "correct": i < 20}) + "\n")   # a: 20 / 30 solved
                g.write(json.dumps({"pid": f"aime24-{i}", "sample": s, "correct": i < 15}) + "\n")   # b: 15 / 30
    open(q, "a").write("#pairq READINGS-EXTRA c_vs_d EVAL-c/c_aime.jsonl EVAL-c/d_aime.jsonl\n#pairq READINGS-EXTRA gone EVAL-c/c_aime.jsonl EVAL-z/none.jsonl\n#pairq READINGS-EXTRA bad ../x.jsonl EVAL-c/d_aime.jsonl\n")
    s = X.main(runs, q, out)
    p = s["pairs"]["c_vs_d"]
    assert p["acc_a"] == 66.67 and p["acc_b"] == 50.0 and p["diff"] == 16.67 and p["wins"] == 5 and p["losses"] == 0 and p["problems"] == 30
    assert p["ci95"][0] > 0 and p["a"] == "EVAL-c/c_aime.jsonl"
    assert "EVAL-z/none.jsonl" in s["missing"] and any("refused path" in m for m in s["missing"])
    assert json.load(open(os.path.join(out, "up", "runs", "READINGS-EXTRA", "pairs.json")))["c_vs_d"]["diff"] == 16.67
    assert "c_vs_d | 66.67 | 50.0 | 16.67 [" in open(os.path.join(out, "summary.txt")).read()
