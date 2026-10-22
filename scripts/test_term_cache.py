"""[v2.5.9z61] (ops-lead 10-22) term_probe.py --cache_dir: a rerun for one new row measures only that row; a model whose inputs
(tag, model path, generations path / size / mtime, template, options) and code are unchanged is taken from the last run with its
per-problem values (the pairs come out the same) and its samples; a changed generations file is measured again. torch,
transformers and gen_eval are stubbed and the per-token measures replaced by cheap stand-ins: the test is the bookkeeping."""
import json
import os
import sys
import tempfile
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "analysis"))


@pytest.fixture
def tp(monkeypatch):
    loads = []
    model = types.SimpleNamespace(cuda=lambda: types.SimpleNamespace(eval=lambda: "model"))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(bfloat16="bf16", cuda=types.SimpleNamespace(empty_cache=lambda: None)))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda p, **k: (loads.append(p), model)[1]),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda p: "tok")))
    monkeypatch.setitem(sys.modules, "gen_eval", types.SimpleNamespace(PROMPT_SUFFIX="", load_bench=lambda b: [{"pid": i} for i in range(4)]))
    import term_probe as T
    monkeypatch.setattr(T, "prompts", lambda tok, rows, think, suf: {i: "p" for i in range(4)})
    monkeypatch.setattr(T, "token_bytes", lambda tok, ids: [b"x"] * len(ids))
    monkeypatch.setattr(T, "anchor_positions", lambda pieces, ids: (0, 1, "nl"))
    monkeypatch.setattr(T, "first_boxed_end", lambda b: 0)
    monkeypatch.setattr(T, "markers", lambda pieces, end: (4, 1))
    monkeypatch.setattr(T, "env_positions", lambda pieces, ids: (1, 2, "env_nl", "ctx"))
    monkeypatch.setattr(T, "stop_probs", lambda m, p, ids, ks: {k: (0.1 * (ids[0] % 7), 0.2) for k in ks})
    return T, loads


def gens(d, name, seed):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for pid in range(4):
            f.write(json.dumps({"pid": pid, "sample": 0, "correct": True, "n_tokens": 20, "finish_reason": "stop", "token_ids": [seed + pid] * 20}) + "\n")
    return p


def run(T, d, rows, monkeypatch):
    inp = os.path.join(d, "inputs.tsv")
    open(inp, "w").write("".join(f"{t}\t{m}\t{g}\tM-{t}\tsrc\n" for t, m, g in rows))
    monkeypatch.setattr(sys, "argv", ["term_probe.py", "--teacher", "t", "--inputs", inp, "--out", os.path.join(d, "report.json"),
                                      "--samples", os.path.join(d, "samples.jsonl"), "--no_teacher", "--cache_dir", os.path.join(d, "cache")])
    T.main()
    return json.load(open(os.path.join(d, "report.json"))), open(os.path.join(d, "samples.jsonl")).read().splitlines()


def test_a_new_row_measures_only_that_row(tp, monkeypatch):
    T, loads = tp
    d = tempfile.mkdtemp()
    a, b, c = gens(d, "a.jsonl", 1), gens(d, "b.jsonl", 3), gens(d, "c.jsonl", 5)
    r1, s1 = run(T, d, [("17_A", "/m/a", a), ("17_B", "/m/b", b)], monkeypatch)
    assert loads == ["/m/a", "/m/b"] and len(s1) == 8 and "17_A - 17_B" in r1["pairs"]
    loads.clear()
    r2, s2 = run(T, d, [("17_A", "/m/a", a), ("17_B", "/m/b", b), ("17_C", "/m/c", c)], monkeypatch)
    assert loads == ["/m/c"]                                                        # only the new row loaded and measured
    assert r2["models"]["17_A"]["reused"] and r2["models"]["17_B"]["reused"] and "reused" not in r2["models"]["17_C"]
    assert r2["pairs"]["17_A - 17_B"] == r1["pairs"]["17_A - 17_B"] and "17_B - 17_C" in r2["pairs"]   # the same per-problem values
    for t in ("17_A", "17_B"):
        assert {k: v for k, v in r2["models"][t].items() if k not in ("reused", "source")} == {k: v for k, v in r1["models"][t].items() if k != "source"}
        assert r2["models"][t]["source"] == f"reused (measured {r1['models'][t]['measured_utc']})"
    assert r1["models"]["17_A"]["source"] == "measured" and r2["models"]["17_C"]["source"] == "measured"
    assert s2[:8] == s1 and len(s2) == 12                                           # the samples carried over


def test_changed_inputs_are_measured_again(tp, monkeypatch):
    T, loads = tp
    d = tempfile.mkdtemp(); a = gens(d, "a.jsonl", 1)
    run(T, d, [("17_A", "/m/a", a)], monkeypatch); loads.clear()
    time.sleep(1.1); gens(d, "a.jsonl", 2)                                         # the reading regenerated
    run(T, d, [("17_A", "/m/a", a)], monkeypatch); assert loads == ["/m/a"]
    loads.clear(); run(T, d, [("17_A", "/m/a2", a)], monkeypatch); assert loads == ["/m/a2"]   # another model path
    loads.clear(); run(T, d, [("17_A", "/m/a2", a)], monkeypatch); assert loads == []          # unchanged: reused
    w = os.path.join(d, "wdir"); os.makedirs(w); open(os.path.join(w, "model.safetensors"), "w").write("1")
    loads.clear(); run(T, d, [("17_A", w, a)], monkeypatch); assert loads == [w]
    loads.clear(); run(T, d, [("17_A", w, a)], monkeypatch); assert loads == []
    time.sleep(1.1); open(os.path.join(w, "model.safetensors"), "w").write("22")                # the same path, new weights (fetched again)
    loads.clear(); run(T, d, [("17_A", w, a)], monkeypatch); assert loads == [w]
    c = json.load(open(os.path.join(d, "cache", "17_A.json"))); del c["per"]; json.dump(c, open(os.path.join(d, "cache", "17_A.json"), "w"))
    loads.clear(); run(T, d, [("17_A", w, a)], monkeypatch); assert loads == [w]                 # the same key, "per" missing: measured
    c = json.load(open(os.path.join(d, "cache", "17_A.json"))); del c["info"]["measured_utc"]; json.dump(c, open(os.path.join(d, "cache", "17_A.json"), "w"))
    loads.clear(); run(T, d, [("17_A", w, a)], monkeypatch); assert loads == [w]                 # no measured_utc: measured
    open(os.path.join(d, "cache", "17_A.json"), "w").write("{broken")
    loads.clear(); run(T, d, [("17_A", w, a)], monkeypatch); assert loads == [w]                 # a bad cache file: measured
    os.utime(os.path.join(w, "model.safetensors"), (time.time() + 5, time.time() + 5))      # touched (ops-lead's case): measured
    loads.clear(); run(T, d, [("17_A", w, a)], monkeypatch); assert loads == [w]


def test_term_probe_sh_passes_the_cache():
    assert "--cache_dir $D/cache" in open(os.path.join(HERE, "term_probe.sh")).read()
