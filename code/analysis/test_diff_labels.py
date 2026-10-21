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
