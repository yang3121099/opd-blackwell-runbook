"""[v2.5.9z15] unit tests of gen_eval's multiple-choice benches (gpqa, mmlupro1k): the rows (options shuffled / drawn with the fixed
seed, the gold letter, the domain as level), the letter grading (the last \\boxed{}, else "answer is (X)", else unparsed), the
WAIT_INPUTS exit when the #fetch copy is missing, and unparsed_rate. Reading MMLU-Pro's parquet needs pyarrow (the node has it);
here its rows are built from a DataFrame."""
import os
import tempfile
import types

import pandas as pd
import pytest

import gen_eval as G


def _gpqa_df(n=5):
    return pd.DataFrame([{"Question": f"Q{i}?", "Correct Answer": f"right{i}", "Incorrect Answer 1": f"w1_{i}", "Incorrect Answer 2": f"w2_{i}",
                          "Incorrect Answer 3": f"w3_{i}", "High-level domain": ["Physics", "Chemistry", "Biology"][i % 3]} for i in range(n)])


def test_gpqa_rows_shuffle_fixed_and_gold_letter():
    a, b = G.gpqa_rows(_gpqa_df()), G.gpqa_rows(_gpqa_df())
    assert [r["problem"] for r in a] == [r["problem"] for r in b]                 # the same seed, the same order
    for i, r in enumerate(a):
        line = next(l for l in r["problem"].splitlines() if l.startswith(f"{r['answer']}) "))
        assert line == f"{r['answer']}) right{i}" and r["letters"] == "ABCD" and r["pid"] == f"gpqa-{i}"
    assert {r["level"] for r in a} == {"Physics", "Chemistry", "Biology"}
    assert len({r["answer"] for r in G.gpqa_rows(_gpqa_df(40))}) == 4             # the gold is not always A


def test_mmlupro_rows_fixed_draw_and_options_as_given():
    df = pd.DataFrame([{"question_id": 100 + i, "question": f"q{i}", "options": ["a", "N/A", "c"] if i % 2 else list("abcdefghij"),
                        "answer": "C", "category": "law" if i % 2 else "math"} for i in range(3000)])
    a, b = G.mmlupro_rows(df), G.mmlupro_rows(df)
    assert len(a) == 1000 and [r["pid"] for r in a] == [r["pid"] for r in b] and a == sorted(a, key=lambda r: int(r["pid"].split("-")[1]))
    odd = next(r for r in a if int(r["pid"].split("-")[1]) % 2)
    assert odd["letters"] == "ABC" and "B) N/A" in odd["problem"] and odd["answer"] == "C"   # N/A kept: the gold letter indexes the list
    assert next(r for r in a if int(r["pid"].split("-")[1]) % 2 == 0)["letters"] == "ABCDEFGHIJ"
    assert len(G.mmlupro_rows(df.head(10))) == 10


@pytest.mark.parametrize("text,gold,ok,pred", [
    ("so \\boxed{B}", "B", True, "B"), ("\\boxed{(C)}", "C", True, "C"), ("\\boxed{\\text{D}}", "D", True, "D"),
    ("\\boxed{A. 42 m/s}", "A", True, "A"), ("\\boxed{\\frac{1}{2}} then the answer is (B)", "B", True, "B"),   # boxed not a letter: the regex
    ("I think the answer is (C) ... the answer is D", "D", True, "D"), ("\\boxed{F}", "B", False, None),        # F outside A-D: unparsed
    ("no answer here", "A", False, None), ("\\boxed{B} but the answer is (C)", "B", True, "B")])                # the box wins
def test_grade_letter(text, gold, ok, pred):
    assert G.grade_letter(text, gold, "ABCD") == (ok, pred)


def test_boxed_content_nested():
    assert G.boxed_content("x \\boxed{\\text{B}} y") == "\\text{B}" and G.boxed_content("\\boxed{a{b}c} \\boxed{D}") == "D"
    assert G.boxed_content("\\boxed{unclosed") is None and G.boxed_content("none") is None


def test_load_mc_waits_without_the_fetch_copy_and_reads_gpqa(monkeypatch):
    d = tempfile.mkdtemp(); monkeypatch.setenv("OPD_ROOT", d)
    with pytest.raises(SystemExit) as e:
        G.load_mc("gpqa")
    assert str(e.value).startswith("WAIT_INPUTS gpqa: Idavidrein__gpqa/gpqa_diamond.csv")
    os.makedirs(os.path.join(d, "data", "hf", "Idavidrein__gpqa")); _gpqa_df(7).to_csv(os.path.join(d, "data", "hf", "Idavidrein__gpqa", "gpqa_diamond.csv"), index=False)
    rows = G.load_bench("gpqa", limit=5)
    assert len(rows) == 5 and rows[0]["source"] == "Idavidrein/gpqa (diamond)"


def test_make_record_and_unparsed_rate():
    r = {"pid": "gpqa-0", "problem": "p", "answer": "B", "letters": "ABCD", "level": "Physics", "source": "s"}
    c = lambda t: types.SimpleNamespace(text=t, token_ids=[1, 2, 3], finish_reason="stop")
    recs = [G.make_record(r, 0, c("\\boxed{B}"), "no_think"), G.make_record(r, 1, c("hmm"), "no_think")]
    assert recs[0]["correct"] and recs[0]["pred"] == "B" and recs[0]["first_correct_pos"] is None and recs[1]["pred"] is None
    args = types.SimpleNamespace(model="m", bench="gpqa", mode="no_think", n=2, temperature=0.6, top_p=0.95, top_k=0, max_tokens=16384)
    m = G.compute_metrics(args, [r], recs, 1.0)
    assert m["avg_acc"] == 50.0 and m["unparsed_rate"] == 0.5 and m["by_level"]["Physics"]["acc"] == 50.0
    assert "unparsed_rate" not in G.compute_metrics(types.SimpleNamespace(**dict(vars(args), bench="math500")), [r], recs, 1.0)
