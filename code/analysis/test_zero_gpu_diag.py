"""[v2.5.9z99] zero_gpu_diag.py on synthetic readings and rollout dumps (CPU, no tokenizer: token counts and decoding are stubbed)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zero_gpu_diag as Z  # noqa: E402


def write(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return str(path)


def test_loop_info_and_box_start():
    ids = list(range(30)) + [7, 8, 9, 10, 11] * 20
    li = Z.loop_info(ids)
    assert li and li["start"] == 30 and li["period"] == 5 and li["unit"] == [7, 8, 9, 10, 11]
    assert Z.loop_info(list(range(200))) is None
    assert Z.box_start("so \\boxed{x} then \\boxed{y}") == 3 and Z.box_start("no box") == -1


def test_t32tails(tmp_path):
    loop = {"pid": "math500-1", "sample": 0, "n_tokens": 100, "finish_reason": "length", "token_ids": [1, 2, 3, 4] * 25, "text": "abc " * 25}
    unfinished = {"pid": "math500-2", "sample": 0, "n_tokens": 90, "finish_reason": "length", "token_ids": list(range(90)), "text": "still going"}
    box_more = {"pid": "math500-3", "sample": 1, "n_tokens": 90, "finish_reason": "length", "token_ids": list(range(90)), "text": "\\boxed{4} and more"}
    done = {"pid": "math500-4", "sample": 0, "n_tokens": 10, "finish_reason": "stop", "token_ids": [1], "text": "\\boxed{1}"}
    r = Z.t32tails(write(tmp_path / "x.jsonl", [loop, unfinished, box_more, done]), n=20)
    assert r["samples"] == 4 and r["truncated"] == 3 and r["counts"] == {"loop": 1, "unfinished": 1, "box_more": 1}
    assert len(r["drawn"]) == 3 and r["drawn_loop"] == 1 and r["drawn_continuation"] == 2
    d = {x["pid"]: x for x in r["drawn"]}
    assert d["math500-1"]["loop_period"] == 4 and d["math500-3"]["box"] is True and d["math500-2"]["tail"] == "still going"


def test_probegate_policies_and_hypotheses():
    pids = [f"math500-{i}" for i in range(4)]
    nothink = dict(zip(pids, [(1.0, 100), (1.0, 100), (0.0, 100), (0.0, 100)]))
    think = dict(zip(pids, [(1.0, 1000), (1.0, 1000), (1.0, 1000), (0.5, 1000)]))
    probe = dict(zip(pids, [0.9, 0.8, 0.2, 0.1]))   # the easy two -> no-think
    truth = dict(zip(pids, [1.0, 1.0, 0.0, 0.0]))
    r = Z.probegate(nothink, think, probe, truth, boot=200)
    assert r["always_think"] == {"acc": 87.5, "mean_len": 1000.0, "think_share": 1.0}
    assert r["probe_gate"] == {"acc": 87.5, "mean_len": 550.0, "think_share": 0.5}
    assert r["never_think"]["acc"] == 50.0 and r["oracle"]["acc"] == 87.5
    assert r["H1"] is True and r["H2"] is True and r["problems"] == 4      # 550 <= 0.7 x 1000; equal to the oracle
    probe_bad = dict(zip(pids, [0.1, 0.2, 0.8, 0.9]))                      # the probe inverted: the hard two without thinking
    b = Z.probegate(nothink, think, probe_bad, truth, boot=200)
    assert b["probe_gate"]["acc"] == 50.0 and b["H1"] is False and b["H2"] is False


def test_math500_labels(tmp_path):
    p = write(tmp_path / "l.jsonl", [{"split": "math500", "pos": 3, "r": 0.7, "r_true": 0.5}, {"split": "train", "pos": 0, "key": "k", "r": 0.1}])
    assert Z.math500_labels(p) == {"math500-3": 0.7} and Z.math500_labels(p, "r_true") == {"math500-3": 0.5}


def test_wallbimodal_mixture_and_hard_first(tmp_path):
    run = tmp_path / "runs" / "E10b-P5x"
    easy_in = "<|im_start|>user\neasy problem<|im_end|>\n<|im_start|>assistant\n"
    hard_in = "<|im_start|>user\nhard problem<|im_end|>\n<|im_start|>assistant\n"
    short, long_ = "w " * 5, "w " * 50                                     # 5 / 50 "tokens" (words), threshold 20
    for st in range(1, 13):
        hard_long = st >= 4; easy_long = st >= 8                           # the hard prompt goes long first
        recs = [{"input": hard_in, "output": long_ if hard_long else short}] * 4 + [{"input": easy_in, "output": long_ if easy_long else short}] * 4
        write(run / "rollouts" / f"{st}.jsonl", recs)
    easy = {Z.prompt_key(easy_in): True, Z.prompt_key(hard_in): False}
    r = Z.wallbimodal([str(run)], lambda texts: [len(t.split()) for t in texts], easy, thresh=20)
    a = r["arms"]["E10b-P5x"]
    assert [s["f_long"] for s in a["steps"]][:9] == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0]
    assert a["wall_step"] == 4 and a["wall_gap_hard_minus_easy"] == 1.0 and a["H_A2"] is True
    assert a["rise"] == [3, 8] and a["component_move"] == {"short_mean": 0.0, "long_mean": 0.0} and a["H_A1"] is True   # a mixture
    assert a["first_long_median_hard"] == 4 and a["first_long_median_easy"] == 8 and a["hard_lead_steps"] == 4


def test_verifyonset_gate(tmp_path):
    good = ["x" * 50 + " Let me verify this. " + "y" * 30 + " \\boxed{1}" for _ in range(10)]   # marker at ~0.5 of the length, before the box
    early = ["Let me check. " + "z" * 200 + "\\boxed{2}" for _ in range(10)]                 # at the very start: too early
    none = ["no marker \\boxed{3}" for _ in range(10)]
    a = write(tmp_path / "a.jsonl", [{"text": t} for t in good]); b = write(tmp_path / "b.jsonl", [{"text": t} for t in early])
    c = write(tmp_path / "c.jsonl", [{"text": t} for t in none])
    r = Z.verifyonset([("good", a)], n=10)
    g = r["readings"]["good"]
    assert g["first_before_box_share"] == 1.0 and 0.40 <= g["median_rel_len"] <= 0.85 and g["gate"] is True and r["gate"] is True
    assert g["candidates"]["verify_kw"]["present"] == 1.0 and g["candidates"]["wait"]["present"] == 0.0
    assert Z.verifyonset([("early", b)], n=10)["readings"]["early"]["gate"] is False                    # median ~0
    assert Z.verifyonset([("good", a), ("none", c)], n=10)["gate"] is False                              # every reading must pass


def test_vlmloop2_pairs_and_kinds(tmp_path):
    decode = lambda ids: " ".join("+" if i == 1 else "verify" if i == 2 else "w" for i in ids)
    calc_loop = list(range(200, 210)) + [1, 1, 3, 1] * 20                                # a unit of '+' signs: calc
    verify_loop = list(range(100, 140)) + [2, 5, 6, 7] * 20
    A = [{"pid": "p1", "finish_reason": "length", "token_ids": calc_loop, "text": "x"},
         {"pid": "p2", "finish_reason": "length", "token_ids": verify_loop, "text": "\\boxed{1} x"},
         {"pid": "p3", "finish_reason": "stop", "token_ids": [1], "text": "\\boxed{2}"}]
    B = [{"pid": "p1", "finish_reason": "length", "token_ids": calc_loop, "text": "x"},
         {"pid": "p2", "finish_reason": "length", "token_ids": list(range(100)), "text": "x"},
         {"pid": "p3", "finish_reason": "stop", "token_ids": [1], "text": "\\boxed{2}"}]
    pa, pb = write(tmp_path / "a.jsonl", A), write(tmp_path / "b.jsonl", B)
    r = Z.vlmloop2([("mv", ("t0", pa), ("t06", pb))], [], decode)
    assert r["pairs"]["mv"] == {"problems": 3, "both": 1, "only_t0": 1, "only_t06": 0, "neither": 1}
    ra = r["readings"]["t0"]
    assert ra["loops"] == 2 and ra["kinds"] == {"calc": 1, "verify": 1} and ra["box_before_share"] == 0.5 and ra["period_median"] == 4
