"""[v2.5.9z104] profiles/*.env: node-class settings a cold start copies into one node's own control.env. Every line is a comment or
`export KEY=VALUE`, KEY a setenv key (control_rules.KEY_RE) and VALUE of control.py's VAL_RE without a path, so control.py's
read_env / setenv keep them. Run: python -m pytest -q scripts/test_profiles.py"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import control_rules as R  # noqa: E402

VAL_RE = re.compile(r"^[A-Za-z0-9_.,:/=+-]*$")   # control.py's VAL_RE (setenv's values)


def _kv(path):
    out = {}
    for i, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"export ([A-Z][A-Z0-9_]*)=(\S*)", line)
        assert m and R.KEY_RE.match(m.group(1)) and VAL_RE.match(m.group(2)) and "/" not in m.group(2), (path, i, line)
        assert m.group(1) not in out, (path, i, "key twice")
        out[m.group(1)] = m.group(2)
    return out


def test_every_profile_is_setenv_lines():
    files = sorted(glob.glob(os.path.join(REPO, "profiles", "*.env")))
    assert files
    for f in files:
        assert _kv(f)


def test_h800_keeps_the_b300_numerics_and_only_frees_memory():
    kv = _kv(os.path.join(REPO, "profiles", "h800_80g.env"))
    assert kv["VENV"] == "fa" and kv["FIXED_MICRO_BSZ"] == "4"   # as the B300 ran the arms: the same loss partition
    assert 0 < float(kv["VLLM_UTIL"]) <= 0.5 and kv["GC"] == "1"
    assert set(kv) == {"VENV", "FIXED_MICRO_BSZ", "VLLM_UTIL", "GC"}   # MAX_TOK_GPU untouched: a smaller budget would change the partition
