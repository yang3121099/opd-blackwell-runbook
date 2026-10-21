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
