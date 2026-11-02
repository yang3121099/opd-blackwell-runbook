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


def test_80g_is_what_the_h800_ran_by_11_01_v259z126():
    """[v2.5.9z126] (Fable 11-02, next_machine_selfserve 2 item 2) the 80 GB class bootstrap.sh --profile 80g writes."""
    kv = _kv(os.path.join(REPO, "profiles", "80g.env"))
    assert kv == {"VENV": "fa", "FIXED_MICRO_BSZ": "4", "LOGP_MB": "1", "VLLM_UTIL": "0.35", "GC": "1", "STATUS_PUT_S": "120",
                  "PUBLIC_UPLOADS": "1"}
    for k, v in kv.items():
        R.parse({"cmd": f"setenv {k}={v}", "reason": "t"})   # every value is one setenv accepts


def test_rehearsal_stays_near_20_node_side_commits_an_hour_v259z126():
    """[v2.5.9z126] (item 8: <= 30 node-side HF commits an hour) status + pulse alone carry the control plane (receipts, events, procs
    inside the status commit), and no bulk upload or publish can start (headroom >= the budget)."""
    kv = _kv(os.path.join(REPO, "profiles", "rehearsal.env"))
    per_hour = 3600 / int(kv["STATUS_PUT_S"]) + 3600 / int(kv["PULSE_PUT_S"]) + 1   # + the hourly resync
    assert kv["CONTROL_BATCH"] == "1" and per_hour <= 20
    assert int(kv["BULK_HEADROOM"]) >= int(kv["HF_COMMIT_BUDGET"]) <= 30 and int(kv["CONTROL_ONLY_AT"]) < int(kv["HF_COMMIT_BUDGET"])
    for k, v in kv.items():
        R.parse({"cmd": f"setenv {k}={v}", "reason": "t"})


def test_setenv_control_batch_and_the_budget_v259z126():
    """[v2.5.9z126] z124c's canary (`setenv CONTROL_BATCH=1`) and z124b's budget were not setenv keys: rules 2.5.115."""
    import pytest
    for ok in ("CONTROL_BATCH=1", "CONTROL_BATCH=0", "CONTROL_BATCH=", "HF_COMMIT_BUDGET=30", "CONTROL_ONLY_AT=24", "BULK_HEADROOM=30"):
        R.parse({"cmd": f"setenv {ok}", "reason": "t"})
    for bad in ("CONTROL_BATCH=2", "HF_COMMIT_BUDGET=0", "CONTROL_ONLY_AT=abc", "BULK_HEADROOM=5000"):
        with pytest.raises(R.Reject):
            R.parse({"cmd": f"setenv {bad}", "reason": "t"})


def test_verify_smoke_on_the_main_venv_until_venv_fa_exists_v259z126(tmp_path):
    """[v2.5.9z126] a profile's VENV=fa before setup step 4 built venv-fa: verify.sh's smoke gets PROBE_ENV VENV= (which wins over
    control.env), and the profile's venv once venv-fa imports flash_attn."""
    import subprocess
    src = open(os.path.join(HERE, "verify.sh")).read()
    line = next(x for x in src.splitlines() if x.startswith('SE=""; '))
    assert 'PROBE_ENV="${PROBE_ENV:-} $SE" SMOKE=1' in src
    run = lambda env: subprocess.run(["bash", "-c", line + '\necho "[$SE]"'], env=dict(os.environ, OPD_ROOT=str(tmp_path), **env),
                                     capture_output=True, text=True).stdout.strip()
    assert run({"VENV": "fa"}) == "[VENV=]" and run({"VENV": ""}) == "[]"
    (tmp_path / "venv-fa" / "bin").mkdir(parents=True); py = tmp_path / "venv-fa" / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 0\n"); py.chmod(0o755)
    assert run({"VENV": "fa"}) == "[]"
    env_sh = open(os.path.join(HERE, "env.sh")).read()
    assert 'for _kv in ${PROBE_ENV:-}; do [[ $_kv =~ ^[A-Z][A-Z0-9_]*=[A-Za-z0-9_.,:/=+-]*$ ]] && export "$_kv"; done' in env_sh   # VENV= passes
