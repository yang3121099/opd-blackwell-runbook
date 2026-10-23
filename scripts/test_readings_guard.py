"""[v2.5.9z69] (Fable / ops-lead 10-23) the readings' disk guard: control.py readings_disk_low against READINGS_DISK_MIN_GB (1000,
control.env first), the heartbeat's readings_held from run_tasks.sh's readings_held.json, the pulse's readings_held, the setenv key."""
import json
import os
import sys
import tempfile
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402
import control_rules as R  # noqa: E402


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "control"))
    monkeypatch.setenv("OPD_ROOT", d); monkeypatch.setattr(C, "ROOT", d); monkeypatch.setattr(C, "CDIR", os.path.join(d, "control"))
    monkeypatch.delenv("READINGS_DISK_MIN_GB", raising=False)
    return d


def free_gb(monkeypatch, gb):
    monkeypatch.setattr(C.os, "statvfs", lambda p: types.SimpleNamespace(f_bavail=int(gb * 1e9 / 4096), f_frsize=4096))


def test_899_gb_holds_1001_starts(root, monkeypatch):
    free_gb(monkeypatch, 899); assert C.readings_disk_low() == "work"
    free_gb(monkeypatch, 1001); assert C.readings_disk_low() == ""


def test_the_line_from_control_env(root, monkeypatch):
    open(os.path.join(root, "control.env"), "w").write("export READINGS_DISK_MIN_GB=500\n")
    free_gb(monkeypatch, 899); assert C.readings_disk_low() == ""                         # setenv applies at once
    free_gb(monkeypatch, 499); assert C.readings_disk_low() == "work"


def test_running_readings_are_never_stopped():
    rt = open(os.path.join(HERE, "run_tasks.sh")).read()
    blk = rt[rt.index("readings_guard() {"):rt.index("deps_of() {")]
    assert "kill" not in blk and "pkill" not in blk                                        # it only skips starting new readings
    assert 'if [ -n "$READ_LOW" ] && reading_task $t; then RHELD="$RHELD $t"; continue; fi' in rt
    assert rt.index("running $t && continue") < rt.index('reading_task $t; then RHELD')       # a running task is passed over before


def test_heartbeat_readings_held(root):
    hb = open(os.path.join(HERE, "heartbeat.sh")).read()
    ns = {"os": os, "json": json, "time": time, "root": root}
    exec(hb[hb.index("def readings_held():"):hb.index("live = live_tasks()")], ns)
    p = os.path.join(root, "control", "readings_held.json")
    assert ns["readings_held"]() == 0
    json.dump({"low": "work", "held": ["EVAL-a", "EVAL-b"]}, open(p, "w")); assert ns["readings_held"]() == 2
    json.dump({"low": "", "held": []}, open(p, "w")); assert ns["readings_held"]() == 0
    json.dump({"low": "work", "held": ["EVAL-a"]}, open(p, "w")); os.utime(p, (time.time() - 1000, time.time() - 1000))
    assert ns["readings_held"]() == 0                                                     # stale: not reported
    assert '"readings_held": readings_held(),' in hb


def test_pulse_and_setenv(root):
    assert C.build_pulse({"tasks": {}, "readings_held": 3}, "")["readings_held"] == 3
    assert C.build_pulse({"tasks": {}}, "")["readings_held"] is None
    R.parse({"cmd": "setenv READINGS_DISK_MIN_GB=1000", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 83)


def test_readings_held_json_written_atomically(tmp_path):   # [ops-lead's A1 condition] a reader never sees a half-written file
    import re, subprocess
    rt = open(os.path.join(HERE, "run_tasks.sh")).read()
    i = rt.index("readings_held.json 2> /dev/null")
    line = rt[rt.rindex("python3 -c", 0, i):i]
    assert "os.replace(t, sys.argv[3])" in line and "open(sys.argv[3], 'w')" not in line
    code = re.search(r'python3 -c "(.*?)" \\', line).group(1)
    out = tmp_path / "readings_held.json"
    subprocess.run(["python3", "-c", code, "work", "EVAL-a EVAL-b", str(out)], check=True)
    assert json.load(open(out)) ["held"] == ["EVAL-a", "EVAL-b"] and not (tmp_path / "readings_held.json.tmp").exists()
