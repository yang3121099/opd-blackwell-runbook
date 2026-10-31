"""[v2.5.9z104a] verify.sh step 1 (G2): PASS when pytest's summary names N passed and nothing failed or errored, whatever N is.
Until z104 the check was `grep "16 passed"`, but code/patches/test_opd_variants.py has had 21 tests since 10-19 (v2.5.7h / 7l),
so a new node failed G2 with every test passing (the H800's cold start 10-30, ops-lead 10-31 14:09 UTC).
python3 g2_check.py <file with pytest -q's last lines>: exit 0 PASS, 1 FAIL (no such summary, a failure or an error)."""
import re
import sys

PASSED = re.compile(r"\b[0-9]+ passed\b")
BAD = re.compile(r"\b[0-9]+ (failed|errors?)\b|\bno tests ran\b|\bInterrupted\b")


def ok(text):
    return bool(PASSED.search(text)) and not BAD.search(text)


if __name__ == "__main__":
    try:
        with open(sys.argv[1], errors="replace") as f:
            text = f.read()
    except (IndexError, OSError):
        sys.exit(1)
    sys.exit(0 if ok(text) else 1)
