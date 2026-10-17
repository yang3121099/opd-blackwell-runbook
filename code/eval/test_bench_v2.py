"""G2 for evaluation suite v2 (E11 amendment 2): OlympiadBench (OE_TO_maths_en_COMP) and Minerva Math load with the
expected sizes, every gold answer written as \\boxed{gold} grades correct with the EVAL-v1 grader, and a wrong boxed
number grades wrong. Needs network (HF datasets) and math_verify: run on a lab box (python -m pytest -q test_bench_v2.py).
Prints the gold answers the grader cannot self-match, so their share is known before any score is read."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_eval  # noqa: E402

SIZES = {"olympiadbench": 674, "minervamath": 272}


def self_match(name):
    rows = gen_eval.load_bench(name)
    bad = [r for r in rows if not gen_eval.grade(f"So the answer is \\boxed{{{r['answer']}}}.", r["answer"])[0]]
    return rows, bad


def test_sizes_and_fields():
    for name, n in SIZES.items():
        rows = gen_eval.load_bench(name)
        assert len(rows) == n, (name, len(rows))
        assert all(r["problem"] and r["answer"] for r in rows)
        assert len({r["pid"] for r in rows}) == n


def test_gold_self_match_and_wrong_answer():
    for name in SIZES:
        rows, bad = self_match(name)
        share = 1 - len(bad) / len(rows)
        print(f"{name}: gold self-match {100 * share:.1f}% ({len(bad)} of {len(rows)} not matched)")
        for r in bad[:15]:
            print("   unmatched gold:", r["pid"], repr(r["answer"])[:120])
        assert share >= 0.95, (name, share)
        wrong = sum(gen_eval.grade("So the answer is \\boxed{987654321}.", r["answer"])[0] for r in rows)
        assert wrong == 0, (name, wrong)
