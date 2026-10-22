"""[v2.5.9z] gen_eval_vl.py's rule grader (no vLLM / images): Geometry3K as gen_eval grades MATH-500; MathVista multi-choice by text
or letter, numbers at the item's precision, text by normalised equality."""
import pytest

import gen_eval_vl as V


def mv(**kw):
    r = {"question_type": "free_form", "answer_type": "text", "precision": None, "choices": None, "answer": "x"}
    r.update(kw); return r


def test_geo3k_like_math500():
    pytest.importorskip("math_verify")
    assert V.grade("geo3k", "so \\boxed{8\\sqrt{3}}", {"answer": "8\\sqrt{3}"})[0]
    assert not V.grade("geo3k", "no box here: 12", {"answer": "12"})[0]


def test_mathvista_multi_choice_text_and_letter():
    row = mv(question_type="multi_choice", choices=["2", "4", "6", "8"], answer="6")
    assert V.grade_mathvista("\\boxed{6}", row)[0]
    assert V.grade_mathvista("\\boxed{C}", row)[0] and V.grade_mathvista("\\boxed{(C)}", row)[0]
    assert not V.grade_mathvista("\\boxed{B}", row)[0] and not V.grade_mathvista("\\boxed{Z}", row)[0]
    assert V.grade_mathvista("no box", row) == (False, None)


def test_mathvista_numbers_at_precision():
    assert V.grade_mathvista("\\boxed{3.14159}", mv(answer_type="float", precision=2, answer="3.14"))[0]
    assert not V.grade_mathvista("\\boxed{3.2}", mv(answer_type="float", precision=2, answer="3.14"))[0]
    assert V.grade_mathvista("\\boxed{1,200}", mv(answer_type="integer", answer="1200"))[0]
    assert V.grade_mathvista("\\boxed{12 cm}", mv(answer_type="integer", answer="12"))[0]
    assert not V.grade_mathvista("\\boxed{none}", mv(answer_type="integer", answer="12"))[0]


def test_boxed_nested():
    assert V.boxed("a \\boxed{\\frac{1}{2}} b \\boxed{x^{2}}") == "x^{2}" and V.boxed("none") is None and V.boxed("\\boxed{open") is None


def test_mathvista_text():
    pytest.importorskip("math_verify")
    assert V.grade_mathvista("\\boxed{\\text{Yes}}", mv(answer="yes"))[0]
    assert not V.grade_mathvista("\\boxed{no}", mv(answer="yes"))[0]


# [v2.5.9z33] MathVision / MathVerse: prompts from the rows, the reviewer's grading rule, by_level / unparsed_rate, images.zip
import os  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402


def _load(monkeypatch, bench, df):
    import pandas as pd
    root = tempfile.mkdtemp(); d, pat = V.DATA[bench]
    os.makedirs(os.path.join(root, d)); open(os.path.join(root, d, pat.replace("*", "00000-of-00001")), "w").close()
    monkeypatch.setattr(pd, "read_parquet", lambda f: df)
    return V.load(bench, root), root


def test_mathvision_rows_prompts(monkeypatch):
    import pandas as pd
    df = pd.DataFrame([{"id": "21", "question": "Which one?\n<image1>\n<image2>", "options": ["A", "B", "C", "D", "E"], "decoded_image": {"bytes": b"x", "path": None}, "image": "images/21.jpg", "answer": "C", "level": 1, "subject": "logic"},
                       {"id": "22", "question": "What time?\n<image1>", "options": ["10:00", "21:30"], "decoded_image": {"bytes": b"y", "path": None}, "image": "images/22.jpg", "answer": "B", "level": 2, "subject": "arithmetic"},
                       {"id": "4", "question": "How many digits?\n<image1>", "options": [], "decoded_image": {"bytes": b"z", "path": None}, "image": "images/4.jpg", "answer": "6", "level": 1, "subject": "counting"}])
    rows, root = _load(monkeypatch, "mathvision", df)
    assert rows[0]["text"] == "Which one?\nChoices: A, B, C, D, E" + V.G.MC_SUFFIX and rows[0]["letters"] == "ABCDE" and "<image" not in rows[0]["text"]
    assert rows[1]["text"] == "What time?\nChoices:\n(A) 10:00\n(B) 21:30" + V.G.MC_SUFFIX and rows[1]["options"] == ["10:00", "21:30"] and rows[1]["letters"] == "AB"
    assert rows[2]["text"] == "How many digits?" + V.G.PROMPT_SUFFIX and rows[2]["letters"] is None
    assert rows[0]["pid"] == "mathvision-21" and rows[0]["level"] == "level-1" and rows[0]["subject"] == "logic" and rows[0]["image"] == {"bytes": b"x", "path": None}
    assert rows[0]["image_zip"] == os.path.join(root, "data/hf/MathLLMs__MathVision", "images.zip")


def test_mathverse_rows_prompts(monkeypatch):
    import pandas as pd
    q = "As shown, find angle CED ()\nChoices:\nA:40°\nB:60°\nC:120°\nD:140°"
    df = pd.DataFrame([{"sample_index": "4", "problem_version": "Vision Dominant", "question": q, "question_for_eval": q, "question_type": "multi-choice", "answer": "D", "image": {"bytes": b"x", "path": None}},
                       {"sample_index": "5", "problem_version": "Vision Only", "question": "", "question_for_eval": q, "question_type": "multi-choice", "answer": "D", "image": {"bytes": b"x", "path": None}},
                       {"sample_index": "3001", "problem_version": "Text Dominant", "question": "Find the line.\n", "question_for_eval": "Find the line.\n", "question_type": "free-form", "answer": "$y-3=\\frac{2}{3}(x-8)$", "image": {"bytes": b"x", "path": None}}])
    rows, _ = _load(monkeypatch, "mathverse", df)
    assert rows[0]["text"] == q + V.G.MC_SUFFIX and rows[0]["letters"] == "ABCD" and rows[0]["options"] == ["40°", "60°", "120°", "140°"] and rows[0]["level"] == "Vision Dominant"
    assert rows[1]["text"] == V.VERSE_NO_TEXT + V.G.MC_SUFFIX and rows[1]["letters"] == "ABCD" and rows[1]["options"][3] == "140°"   # options from question_for_eval
    assert rows[2]["text"] == "Find the line." + V.G.PROMPT_SUFFIX and rows[2]["letters"] is None and rows[2]["answer"] == "y-3=\\frac{2}{3}(x-8)"


@pytest.mark.parametrize("text,ok,pred", [("\\boxed{D}", True, "D"), ("\\boxed{(D)}", True, "D"), ("\\boxed{D. 140°}", True, "D"), ("\\boxed{\\text{D}}", True, "D"),
                                          ("\\boxed{140°}", True, "D"), ("\\boxed{60°}", False, "B"), ("\\boxed{E}", False, None), ("\\boxed{about 90}", False, None),
                                          ("so the answer is (D)", True, "D"), ("\\boxed{about 90} the answer is (D)", False, None), ("nothing", False, None)])
def test_grade_vl_multiple_choice(text, ok, pred):
    row = {"answer": "D", "letters": "ABCD", "options": ["40°", "60°", "120°", "140°"]}
    assert V.grade("mathverse", text, row) == (ok, pred)


def test_grade_vl_free_form(monkeypatch):   # Math-Verify is gen_eval's grade (stubbed here: the Mac has no math_verify)
    monkeypatch.setattr(V.G, "grade", lambda text, gold: (V.boxed(text) == gold, V.boxed(text)))
    row = {"answer": "61", "letters": None}
    assert V.grade("mathvision", "\\boxed{61}", row) == (True, "61") and V.grade("mathvision", "\\boxed{62}", row) == (False, "62")
    assert V.grade("mathvision", "the answer is 61", row) == (False, None)   # no box: unparsed


def test_metrics_by_level_and_unparsed():
    rows = [{"pid": "a", "level": "level-1", "subject": "logic", "source": "s"}, {"pid": "b", "level": "level-2", "subject": "logic", "source": "s"}]
    recs = [{"pid": "a", "correct": True, "pred": "A", "n_tokens": 10, "finish_reason": "stop"}, {"pid": "b", "correct": False, "pred": None, "n_tokens": 20, "finish_reason": "length"}]
    args = types.SimpleNamespace(model="m", bench="mathvision", mode="no_think", n=1, temperature=0.0, top_p=0.95, top_k=0, max_tokens=16384)
    m = V.metrics(args, rows, recs, 1.0)
    assert m["avg_acc"] == 50.0 and m["unparsed_rate"] == 0.5 and m["protocol"] == "rule-graded, not the official LLM-extraction protocol"
    assert m["by_level"] == {"level-1": {"acc": 100.0, "n": 1}, "level-2": {"acc": 0.0, "n": 1}, "subject:logic": {"acc": 50.0, "n": 2}}
    assert "unparsed_rate" not in V.metrics(types.SimpleNamespace(**dict(vars(args), bench="mathvista")), rows, recs, 1.0)


def test_choices_of():
    assert V.choices_of("Q?\nChoices:\nA:40°\nB. 60°\n(C) 120°") == [("A", "40°"), ("B", "60°"), ("C", "120°")] and V.choices_of("no choices") == []


def test_image_of_from_zip():
    PIL = pytest.importorskip("PIL.Image")
    import io
    import zipfile
    d = tempfile.mkdtemp(); b = io.BytesIO(); PIL.new("RGB", (3, 2)).save(b, format="PNG")
    with zipfile.ZipFile(os.path.join(d, "images.zip"), "w") as z:
        z.writestr("images/21.png", b.getvalue())
    assert V.image_of({"bytes": None, "path": "images/21.png"}, os.path.join(d, "images.zip")).size == (3, 2)
    assert V.image_of("images/21.png", os.path.join(d, "images.zip")).size == (3, 2) and V.image_of({"bytes": b.getvalue()}).size == (3, 2)
