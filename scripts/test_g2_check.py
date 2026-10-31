"""[v2.5.9z104a] g2_check.py: verify.sh's G2 accepts any number of passed tests and rejects a failure or an error.
Run: python -m pytest -q scripts/test_g2_check.py"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _check(tmp_path, text):
    p = tmp_path / "g2.txt"
    p.write_text(text)
    return subprocess.run([sys.executable, os.path.join(HERE, "g2_check.py"), str(p)]).returncode


def test_g2_accepts_any_count(tmp_path):
    assert _check(tmp_path, "..................... [100%]\n21 passed in 3.21s\n") == 0
    assert _check(tmp_path, "21 passed, 2 warnings in 3.21s\n") == 0
    assert _check(tmp_path, "16 passed in 2.02s\n") == 0


def test_g2_rejects_failed_or_error(tmp_path):
    for text in ["FAILED test_opd_variants.py::test_x - AssertionError\n20 passed, 1 failed in 3.40s\n",
                 "1 failed, 20 passed in 3.40s\n", "21 passed, 1 error in 3.40s\n",
                 "Interrupted: 1 error during collection\n1 error in 0.51s\n", "no tests ran in 0.01s\n", ""]:
        assert _check(tmp_path, text) == 1, text
    assert subprocess.run([sys.executable, os.path.join(HERE, "g2_check.py"), str(tmp_path / "missing.txt")]).returncode == 1
