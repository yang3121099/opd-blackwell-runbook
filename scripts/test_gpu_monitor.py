"""[v3.1] gpu_monitor.py: the node's own utilisation record against the machine's 85 % rule - the window means, the 2-hour list (the
rule as we understand it) and the 30-minute early warning, windows that count only once they are full, the status file, and one pass
against a stub nvidia-smi. No GPU. Run: python -m pytest -q scripts/test_gpu_monitor.py"""
import importlib
import json
import os
import stat
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


@pytest.fixture
def M(tmp_path, monkeypatch):
    monkeypatch.setenv("OPD_ROOT", str(tmp_path / "root")); (tmp_path / "root" / "control").mkdir(parents=True)
    import gpu_monitor
    importlib.reload(gpu_monitor)
    monkeypatch.setattr(gpu_monitor, "owners", lambda: {0: "task", 1: "filler"})
    return gpu_monitor


def series(now, span, value, step=10):
    return [(now - span + i * step, value) for i in range(int(span / step) + 1)]


def test_window_mean_share_and_fullness(M):
    now = 100000.0
    q = series(now, 7200, 90.0)
    assert M.window(q, now, 60) == (90.0, True, None)
    mean, full, share = M.window(series(now, 3600, 100.0) + series(now, 1800, 60.0), now, 7200, 85.0)
    assert not full and 80 < mean < 90 and 0.6 < share < 0.7           # one hour of samples is not a full 2-hour window
    assert M.window([], now, 60) == (None, False, None)


def test_low_and_warn_lists(M):
    now = 200000.0
    qs = {0: series(now, 7200, 95.0),                                   # fine
          1: series(now, 7200, 80.0),                                   # under the floor for two hours: the rule
          2: series(now - 1800, 5400, 100.0) + series(now, 1800, 70.0), # dropped half an hour ago (2-hour mean still 92): the early warning only
          3: series(now, 600, 10.0)}                                    # ten minutes of data: nothing yet
    st = M.summarize(qs, now, 85.0)
    assert st["low2h"] == [1] and st["warn30m"] == [1, 2] and st["floor"] == 85.0
    assert st["gpus"]["0"]["on"] == "task" and st["gpus"]["1"]["on"] == "filler" and st["gpus"]["2"]["on"] == "none"
    assert st["gpus"]["1"]["u2h"] == 80.0 and st["gpus"]["1"]["share2h_at_floor"] == 0.0 and st["gpus"]["0"]["share2h_at_floor"] == 1.0
    assert st["gpus"]["3"]["u10m"] == 10.0 and 3 not in st["warn30m"]
    assert "/" not in json.dumps(st)                                    # numbers and three words: no path can be in it


def test_floor_from_control_env(M, tmp_path):
    assert M.floor() == 85.0
    (tmp_path / "root" / "control.env").write_text("export GPU_UTIL_FLOOR=90\n")
    assert M.floor() == 90.0
    (tmp_path / "root" / "control.env").write_text("export GPU_UTIL_FLOOR=5\n")
    assert M.floor() == 85.0


def test_once_against_a_stub_nvidia_smi(tmp_path):
    root = tmp_path / "root"; (root / "control").mkdir(parents=True); b = tmp_path / "bin"; b.mkdir()
    (b / "nvidia-smi").write_text("#!/bin/bash\nprintf '0, 97, 120000\\n1, 3, 12\\n'\n")
    (b / "nvidia-smi").chmod((b / "nvidia-smi").stat().st_mode | stat.S_IEXEC)
    r = subprocess.run([sys.executable, os.path.join(HERE, "gpu_monitor.py"), "once"], capture_output=True, text=True,
                       env=dict(os.environ, OPD_ROOT=str(root), PATH=f"{b}:{os.environ['PATH']}"))
    st = json.loads(r.stdout)
    assert st["gpus"]["0"]["u1m"] == 97.0 and st["gpus"]["1"]["u1m"] == 3.0 and st["low2h"] == [] and st["warn30m"] == []
    assert json.load(open(root / "control" / "gpu_monitor" / "state.json"))["gpus"].keys() == {"0", "1"}


def test_heartbeat_carries_it_and_the_loops_know_it():
    hb = open(os.path.join(HERE, "heartbeat.sh")).read(); rt = open(os.path.join(HERE, "run_tasks.sh")).read()
    assert '"gpu_util": gpu_util' in hb and 'control", "gpu_monitor", "state.json"' in hb
    assert rt.count("ensure_loop gpu_monitor") == 2 and rt.count("ensure_loop gpu_fill") == 2   # the frozen round and the full pass
    assert "gpu_monitor" in open(os.path.join(HERE, "loops.sh")).read()
