"""[v2.5.9z135] control.py's origin check with OPD_CONTROL_ORIGIN (the runbook's git mirror on Hugging Face, written into node.env by
bootstrap when RB_URL is on another host): the mirror is accepted under its own origin, GitHub is not, and without the variable the
GitHub origin stays the only one (real git remotes in a tmp clone)."""
import importlib
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402


@pytest.fixture
def clone(tmp_path, monkeypatch):
    d = tmp_path / "rb"; subprocess.run(["git", "init", "-q", str(d)], check=True)
    yield d
    monkeypatch.delenv("OPD_CONTROL_ORIGIN", raising=False); importlib.reload(C)   # the module as the other tests know it


def check(monkeypatch, d, origin, env):
    if env is None:
        monkeypatch.delenv("OPD_CONTROL_ORIGIN", raising=False)
    else:
        monkeypatch.setenv("OPD_CONTROL_ORIGIN", env)
    importlib.reload(C); monkeypatch.setattr(C, "RUNBOOK", str(d))
    subprocess.run(["git", "-C", str(d), "remote", "remove", "origin"], capture_output=True)
    subprocess.run(["git", "-C", str(d), "remote", "add", "origin", origin], check=True)
    return C.origin_ok()[0]


MIRROR = "huggingface.co/yang31210999/opd-blackwell-runbook"


def test_the_mirror_under_its_own_origin(clone, monkeypatch):
    assert check(monkeypatch, clone, "https://huggingface.co/yang31210999/opd-blackwell-runbook", MIRROR)
    assert check(monkeypatch, clone, "https://huggingface.co/yang31210999/opd-blackwell-runbook.git", MIRROR)
    assert not check(monkeypatch, clone, "https://github.com/yang3121099/opd-blackwell-runbook.git", MIRROR)


def test_github_alone_without_the_variable(clone, monkeypatch):
    assert check(monkeypatch, clone, "https://github.com/yang3121099/opd-blackwell-runbook.git", None)
    assert not check(monkeypatch, clone, "https://huggingface.co/yang31210999/opd-blackwell-runbook", None)


def test_bootstrap_writes_the_origin_control_py_normalises_to():
    """node.env's OPD_CONTROL_ORIGIN is RB_URL without scheme, .git and trailing slash: the form origin_ok compares."""
    src = open(os.path.join(HERE, "bootstrap.sh")).read()
    assert """OPD_CONTROL_ORIGIN=$(echo "${RB_URL%.git}" | sed -E 's#^https?://##; s#/$##')""" in src
    url = "https://huggingface.co/yang31210999/opd-blackwell-runbook.git"
    out = subprocess.run(["bash", "-c", f"""RB_URL={url}; echo "${{RB_URL%.git}}" | sed -E 's#^https?://##; s#/$##'"""],
                         capture_output=True, text=True).stdout.strip()
    assert out == MIRROR
