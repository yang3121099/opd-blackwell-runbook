"""[v2.5.8] Tests of fetch_model.py's per-path lock and overall cap. CPU only, no network: a stub huggingface_hub on
PYTHONPATH lists two files and "downloads" them after STUB_SLEEP seconds, writing each download to STUB_COUNT.
Run: python -m pytest -q scripts/test_fetch_lock.py"""
import fcntl
import json
import os
import subprocess
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STUB = '''
import os, time
class _E:
    def __init__(s, path, size): s.path, s.size, s.lfs = path, size, None
class HfApi:
    def __init__(self, endpoint=None, token=None): pass
    def list_repo_tree(self, repo, path_in_repo=None, recursive=True, repo_type=None):
        return [_E(path_in_repo + "/model.bin", 4), _E(path_in_repo + "/config.json", 2)]
def hf_hub_download(repo, filename, local_dir, endpoint=None, token=None, repo_type=None):
    time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
    with open(os.environ["STUB_COUNT"], "a") as f: f.write(filename + "\\n")
    p = os.path.join(local_dir, filename); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write("x" * (4 if filename.endswith(".bin") else 2))
'''


@pytest.fixture
def box(tmp_path):
    pkg = tmp_path / "stub" / "huggingface_hub"; pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(STUB)
    (pkg / "constants.py").write_text("HF_HUB_ENABLE_HF_TRANSFER = False\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "stub"), OPD_ROOT=str(tmp_path / "opd"), STUB_COUNT=str(tmp_path / "count"),
               HF_DOWNLOAD_ENDPOINTS="https://example.invalid", FETCH_TRIES="1", HF_HUB_ENABLE_HF_TRANSFER="0")
    return tmp_path, env


def run(env, local, **kw):
    e = dict(env, **{k: str(v) for k, v in kw.items()})
    return subprocess.Popen([sys.executable, os.path.join(HERE, "fetch_model.py"), "org/repo", "runs/T/global_step_1", local],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=e)


def result(p, timeout=60):
    out, _ = p.communicate(timeout=timeout)
    return json.loads(out.strip().splitlines()[-1])


def test_two_fetches_one_path_download_once(box):
    tmp, env = box
    local = str(tmp / "dl")
    a = run(env, local, STUB_SLEEP=2)
    time.sleep(1)
    b = run(env, local, STUB_SLEEP=2)
    ra, rb = result(a), result(b)
    assert ra["ok"] and rb["ok"]
    assert rb.get("cached") is True and rb.get("lock_wait_s", 0) >= 1
    assert sorted(open(tmp / "count").read().split()) == ["runs/T/global_step_1/config.json", "runs/T/global_step_1/model.bin"]


def test_overall_cap_ends_the_download(box):
    tmp, env = box
    t0 = time.time()
    r = result(run(env, str(tmp / "dl"), STUB_SLEEP=60, FETCH_TIMEOUT=3))
    assert not r["ok"] and r["errors"][0]["error"] == "FetchTimeout"
    assert time.time() - t0 < 30
    time.sleep(0.5)
    left = subprocess.run(["pgrep", "-f", "fetch_model.py --child org/repo"], capture_output=True, text=True).stdout.split()
    assert left == []


def test_lock_wait_is_bounded(box):
    tmp, env = box
    local = tmp / "dl"; local.mkdir()
    fd = os.open(str(local / ".runs__T__global_step_1.lock"), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        r = result(run(env, str(local), FETCH_LOCK_WAIT=1))
    finally:
        os.close(fd)
    assert not r["ok"] and r["errors"][0]["error"] == "LockTimeout"
    assert not (tmp / "count").exists()
