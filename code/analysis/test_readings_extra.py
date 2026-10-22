"""[v2.5.9x1] readings_extra.py on a fake runs directory: splits 1-100 / 101-500 / all / easy / hard, the frontier rows from a budget
curve and a metrics.json, missing inputs listed, outputs where the upload expects them."""
import json
import random
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


def test_coverq_problem_coverage():   # [v2.5.9z7] (Q-052) problems one reading solves at least once and the other never
    runs, out, q = setup()
    os.makedirs(os.path.join(runs, "EVAL-c"))
    solved = {"a": {0, 1, 2, 5}, "b": {0, 1, 3, 4, 6}, "t": {3, 5}, "s": {4}}   # by problem index; problem 7: nobody
    for k, sol in solved.items():
        with open(os.path.join(runs, "EVAL-c", f"{k}.jsonl"), "w") as f:
            for i in range(8 if k != "s" else 7):                              # the extra s lacks problem 7
                for smp in range(4):
                    f.write(json.dumps({"pid": f"aime25-{i}", "sample": smp, "correct": i in sol and smp < (1 if k == "a" else 2)}) + "\n")
    open(q, "a").write("#coverq READINGS-EXTRA q052_aime25 EVAL-c/a.jsonl EVAL-c/b.jsonl teacher=EVAL-c/t.jsonl self=EVAL-c/s.jsonl gone=EVAL-z/x.jsonl\n"
                       "#coverq READINGS-EXTRA nob EVAL-c/a.jsonl EVAL-z/none.jsonl\n#coverq READINGS-EXTRA bad EVAL-c/a.jsonl EVAL-c/b.jsonl t=../x.jsonl\n")
    s = X.main(runs, q, out)
    c = s["coverage"]["q052_aime25"]
    assert (c["problems"], c["a_solved"], c["b_solved"], c["both"], c["neither"]) == (8, 4, 5, 2, 1)
    assert [r["pid"] for r in c["b_only"]] == ["aime25-3", "aime25-4", "aime25-6"] and [r["pid"] for r in c["a_only"]] == ["aime25-2", "aime25-5"]
    r3 = c["b_only"][0]
    assert r3["a"] == [0, 4] and r3["b"] == [2, 4] and r3["teacher"] == [2, 4] and r3["self"] == [0, 4] and "gone" not in r3
    assert c["b_only"][1]["self"] == [2, 4] and c["a_only"][1]["teacher"] == [2, 4] and c["a_only"][1]["a"] == [1, 4]
    assert c["extras_missing"] == ["gone"] and "EVAL-z/x.jsonl" in s["missing"] and "EVAL-z/none.jsonl" in s["missing"]
    assert "nob" not in s["coverage"] and "bad" not in s["coverage"] and "bad: refused path" in s["missing"]
    assert json.load(open(os.path.join(out, "up", "runs", "READINGS-EXTRA", "coverage.json")))["q052_aime25"]["b_solved"] == 5
    t = open(os.path.join(out, "summary.txt")).read()
    assert "q052_aime25 | 8 | 4 / 5 / 2 / 1 | aime25-3 a 0/4 b 2/4 teacher 2/4 self 0/4 gone —; aime25-4 " in t


def test_coverq_extra_without_a_problem_is_null():
    A, B, E = {"p-0": [0, 4], "p-1": [1, 4]}, {"p-0": [3, 4], "p-1": [1, 4], "p-2": [1, 4]}, {"p-1": [4, 4]}
    c = X.coverage(A, B, {"e": E})
    assert c["b_only"] == [{"pid": "p-0", "a": [0, 4], "b": [3, 4], "e": None}] and c["not_in_both"] == ["p-2"] and c["problems"] == 2


def test_lensel_steps_and_first_drop():   # [v2.5.9z13] (Q-030 check 1)
    runs, out, q = setup()
    d = os.path.join(runs, "E10b-P5lenS"); os.makedirs(d)
    with open(os.path.join(d, "lensel.jsonl"), "w") as f:
        for step, eos_kept in ((1, 4), (2, 4), (3, 3), (4, 1)):
            for g in range(2):
                eos = [1] * eos_kept + [0] * (16 - eos_kept)
                f.write(json.dumps({"step": step, "group": g, "mode": "short", "len": [100] * eos_kept + [4096] * (16 - eos_kept),
                                    "trunc": [1 - e for e in eos], "eos": eos, "score": [0.0] * 16, "kept": [0, 1, 2, 3]}) + "\n")
    open(q, "a").write("#lensel READINGS-EXTRA q030 E10b-P5lenS E10b-P5lenX\n#lensel READINGS-EXTRA bad ../x\n")
    s = X.main(runs, q, out)
    st = s["lensel"]["q030"]["E10b-P5lenS"]
    assert st["steps"][1] == {"kept": [100.0, 0.0, 100], "all": [25.0, 75.0, 3097], "groups": 2}
    assert st["steps"][3]["kept"][0] == 75.0 and st["kept_eos_below_90_from"] == 3
    assert "E10b-P5lenX/lensel.jsonl" in s["missing"] and any("refused path" in m for m in s["missing"])
    assert json.load(open(os.path.join(out, "up", "runs", "READINGS-EXTRA", "lensel.json")))["q030"]["E10b-P5lenS"]["kept_eos_below_90_from"] == 3
    assert "lensel q030: kept EOS below 90% from step E10b-P5lenS 3" in open(os.path.join(out, "summary.txt")).read()


def test_tails_loop_and_long_fixed_seed():   # [v2.5.9z13] (Q-030 check 2)
    runs, out, q = setup()
    d = os.path.join(runs, "EVAL-len-S"); os.makedirs(d)
    words = "the area of the triangle is half the base times the height so we compute each side length carefully and then".split()
    with open(os.path.join(d, "s.jsonl"), "w") as f:
        for i in range(10):
            if i % 2:   # a loop: one unit repeated to the end
                text = "Solve step by step. " + "Wait, let me recheck the sum of 3 and 4. " * 60
            else:       # long reasoning: no repeated unit at the end
                rr = random.Random(i); text = " ".join(rr.choice(words) + str(rr.randrange(1000)) for _ in range(400))
            f.write(json.dumps({"pid": f"math500-{i}", "sample": 0, "n_tokens": len(text) // 4, "finish_reason": "length", "correct": False, "text": text}) + "\n")   # ~4 characters a token
    open(q, "a").write("#tails READINGS-EXTRA lenS16k EVAL-len-S/s.jsonl 10 0 500\n#tails READINGS-EXTRA few EVAL-len-S/s.jsonl 4 7 500\n#tails READINGS-EXTRA nonum EVAL-len-S/s.jsonl x 0 500\n")
    s = X.main(runs, q, out)
    t = s["tails"]["lenS16k"]
    assert t["n"] == 10 and t["counts"] == {"loop": 5, "long": 5} and t["truncated"] == 10 and t["seed"] == 0
    assert "Wait, let me recheck" in t["example"]["loop"] and len(t["example"]["loop"]) <= 300 and len(t["example"]["long"]) <= 300
    loops = [x for x in t["samples"] if x["class"] == "loop"]
    assert all(x["period"] == len("Wait, let me recheck the sum of 3 and 4. ") for x in loops) and all(x["pid"] in ("math500-1", "math500-3", "math500-5", "math500-7", "math500-9") for x in loops)
    a, b = s["tails"]["few"], X.main(runs, q, out)["tails"]["few"]
    assert a["samples"] == b["samples"] and a["n"] == 4                       # the same seed, the same samples
    assert any("nonum" in m for m in s["missing"])
    assert "tails lenS16k: 10 of 10 (seed 0, last ~500 tokens): loop 5, long 5, truncated 10" in open(os.path.join(out, "summary.txt")).read()


def test_loop_period_edges():
    assert X.loop_period("abcdefgh" * 3) == 8 and X.loop_period("x" * 30) == 8   # a run of one character: its shortest unit of 8
    assert X.loop_period("no repetition at the end of this sentence at all") is None


def _m500(path, correct_of, tokens_of, n=4, max_tokens=16384):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in range(500):
            for s in range(n):
                f.write(json.dumps({"pid": f"math500-{i}", "sample": s, "correct": correct_of(i, s), "n_tokens": tokens_of(i, s), "finish_reason": "stop"}) + "\n")
    json.dump({"max_tokens": max_tokens}, open(path[:-len(".jsonl")] + ".metrics.json", "w"))


def test_fingerprint_auto_named_capped_and_pairs():   # [v2.5.9z14] (Q-056)
    runs, out, q = setup()
    _m500(os.path.join(runs, "EVAL-a", "a_s50_math500_T0.6_math500_no_think_t0.6.jsonl"), lambda i, s: i % 2 == 0, lambda i, s: 1000 + i)
    _m500(os.path.join(runs, "EVAL-b", "b_math500_T0.6_math500_no_think_t0.6.jsonl"), lambda i, s: i % 2 == 0 and s < 3, lambda i, s: 2000 + 2 * i)
    _m500(os.path.join(runs, "EVAL-c", "anchor_math500_T0.6_math500_no_think_t0.6.jsonl"), lambda i, s: True, lambda i, s: 10, n=1, max_tokens=2048)   # an anchor: left out
    # a 32K thinking reading: odd problems run past 16K -> at the 16K budget they are wrong
    _m500(os.path.join(runs, "EVAL-t", "t_think_math500_T0.6_32k_math500_think_t0.6.jsonl"), lambda i, s: True, lambda i, s: 20000 if i % 2 else 9000, max_tokens=32768)
    open(q, "a").write("#fpauto READINGS-EXTRA q056\n#fpmodel READINGS-EXTRA q056 T_think EVAL-t/t_think_math500_T0.6_32k_math500_think_t0.6.jsonl 16384\n"
                       "#fppair READINGS-EXTRA q056 a_s50 T_think b nobody\n")
    s = X.main(runs, q, out)
    fp = s["fingerprint"]["q056"]
    assert sorted(fp["models"]) == ["T_think", "a_s50", "b"]                        # the anchor is not a 16K avg@4 reading
    assert fp["models"]["T_think"]["acc"] == 50.0 and fp["models"]["T_think"]["mean_len"] == 12692.0   # (16384 + 9000) / 2
    assert fp["models"]["a_s50"]["acc"] == 50.0 and fp["models"]["b"]["acc"] == 37.5
    ab = next(r for r in fp["pairs"] if r["b"] == "b")
    assert ab["all"]["phi"] == 1.0 and ab["all"]["agree"] == 1.0 and ab["all"]["r_len"] == 1.0 and ab["all"]["n"] == 500
    assert ab["1-100"]["n"] == 100 and ab["101-500"]["n"] == 400
    at = next(r for r in fp["pairs"] if r["b"] == "T_think")
    assert at["all"]["r_acc"] == 1.0 and abs(at["all"]["r_len"]) < 0.05   # the cap makes the 32K reading's accuracy that of a; lengths unrelated
    assert any("q056 pair a_s50 / nobody" in m for m in s["missing"])
    full = json.load(open(os.path.join(out, "up", "runs", "READINGS-EXTRA", "fingerprint.json")))
    assert full["q056"]["matrix"]["all"]["a_s50"]["b"]["phi"] == 1.0 and "matrix" not in s["fingerprint"]["q056"]
    t = open(os.path.join(out, "summary.txt")).read()
    assert "fingerprint q056 (3 models)" in t and "a_s50 vs b | 1.0 / 1.0 / 1.0 (0.5) / 1.0 |" in t and ab["all"]["chance"] == 0.5


def test_fp_compare_constant_and_parts():
    va = {i: (1.0, 100.0) for i in range(500)}
    vb = {i: (0.5 if i < 100 else 0.0, float(i)) for i in range(500)}
    c = X.fp_compare(va, vb)
    assert c["r_acc"] is None and c["phi"] is None and c["agree"] == 0.2 and c["n"] == 500   # a constant vector has no correlation
    assert c["chance"] == 0.2                                   # [v2.5.9z17] pa = 1, pb = 0.2: 1 x 0.2 + 0 x 0.8
    assert X.fp_compare(va, vb, "1-100")["agree"] == 1.0 and X.fp_compare(va, vb, "101-500")["agree"] == 0.0
