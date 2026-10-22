"""[v2.5.9x] diff_labels.py without vLLM: the overlong filter and the sampler selection of the training prompts, the summary."""
import os
import tempfile

import diff_labels as D


class Tok:   # tokens = characters, so the 1024-token filter is a length filter
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=True, **kw):
        s = "".join(m["content"] for m in msgs)
        return s if tokenize is False else list(s)


def test_select_train_filter_then_order(monkeypatch):
    raw = [([{"role": "user", "content": ("x" * (2000 if i == 3 else 10)) + str(i)}], str(i), "dapo_mv") for i in range(10)]
    monkeypatch.setattr(D, "sampler_order", lambda n, seed: list(range(n))[::-1])   # the order is torch's on the node
    sel, n_rows, n_fit = D.select_train(raw, Tok(), steps=2, tbs=3, seed=42, maxp=1024, chat_kwargs={})
    assert n_rows == 10 and n_fit == 9 and len(sel) == 6
    assert [s[0] for s in sel] == list(range(6)) and sel[0][2] == "9" and all(s[2] != "3" for s in sel)   # row 3 filtered out
    assert sel[0][1] == [{"role": "user", "content": "x" * 10 + "9"}]


def test_summarize_hist_easy_share():
    rows = [{"k": k, "r": k / 8, "mean_len": 100.0, "trunc": 1 if k == 0 else 0} for k in (0, 0, 4, 8)]
    s = D.summarize(rows, 8)
    assert s["prompts"] == 4 and s["hist_k"]["0"] == 2 and s["hist_k"]["8"] == 1 and s["easy_share"] == 0.5
    assert s["trunc_share"] == round(2 / 32, 4) and s["mean_r"] == round((0 + 0 + 0.5 + 1) / 4, 4)


def test_math500_grader_takes_ok_of_the_tuple(monkeypatch):   # [v2.5.9z27] bool((False, None)) is True: every answer was right
    import sys
    import types
    stub = types.SimpleNamespace(grade=lambda text, gold: (text == gold, text))
    monkeypatch.setitem(sys.modules, "gen_eval", stub)
    g = D.grader("math500")
    assert g("4", "4", "math500") == 1.0 and g("5", "4", "math500") == 0.0


def test_redo_math500_keeps_the_train_lines_byte_for_byte():   # [v2.5.9z27] opd_route.py reads them while E10b-P5route trains
    import json
    d = tempfile.mkdtemp()
    train = ['{"split": "train", "pos": %d, "key": "t%d",  "k": 3, "n": 8, "r": 0.375}\n' % (i, i) for i in range(5)]
    with open(os.path.join(d, "labels_17.jsonl"), "w") as f:
        f.writelines(train + [json.dumps({"split": "math500", "pos": i, "k": 8, "n": 8, "r": 1.0}) + "\n" for i in range(3)])
    json.dump({"tag": "17", "train": {"mean_r": 0.2973}, "math500": {"mean_r": 1.0}, "seconds": {"train": 7131.6, "math500": 238.2}},
              open(os.path.join(d, "report_17.json"), "w"))
    new = [{"split": "math500", "pos": i, "key": f"m{i}", "k": k, "n": 8, "r": k / 8, "mean_len": 800.0, "trunc": 0} for i, k in enumerate((8, 2, 5))]
    rep, n_tr = D.redo_math500(d, "17", new, 8, 240.0)
    lines = open(os.path.join(d, "labels_17.jsonl")).readlines()
    assert n_tr == 5 and lines[:5] == train and [json.loads(x)["r"] for x in lines[5:]] == [1.0, 0.25, 0.625]
    assert rep["train"] == {"mean_r": 0.2973} and rep["math500"]["mean_r"] == round((1 + 0.25 + 0.625) / 3, 4) and rep["math500_grader"] == "v2"
    assert rep["seconds"] == {"train": 7131.6, "math500": 240.0} and json.load(open(os.path.join(d, "report_17.json")))["math500_grader"] == "v2"
    assert not os.path.exists(os.path.join(d, "labels_17.jsonl.tmp"))
