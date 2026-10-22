"""[v2.5.9z48] (ops-lead / the reviewer 10-22: the shared work disk) the disk guard's work line 1000 GB from control.env, the
control plane's fallback to the checkpoint disk below 200 GB free and back above 400 GB (no flapping in between, nothing
removed, writers go on through the symlinks), one disk_write_fail event per full episode, the new setenv keys, and rollouts
read through a symlinked directory as upload_sync / remote.py do."""
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402
import control_rules as R  # noqa: E402


@pytest.fixture
def disks(monkeypatch):
    root, ckpt = tempfile.mkdtemp(), tempfile.mkdtemp()
    for n in ("control", "logs"):
        os.makedirs(os.path.join(root, n, "events" if n == "control" else "x"))
    open(os.path.join(root, "control", "pulse.json"), "w").write("{}")
    open(os.path.join(root, "control", "events", "e1.json"), "w").write("1")
    open(os.path.join(root, "logs", "heartbeat.log"), "w").write("a\n")
    monkeypatch.setattr(C, "ROOT", root)
    ev = []
    monkeypatch.setattr(C, "event", lambda task, kind, detail="": ev.append(kind))
    return root, ckpt, ev


def run(root, ckpt, free, stamp="S"):
    return C.disk_fallback(root=root, ckpt=ckpt, free_fn=lambda p: free, stamp=stamp, check_dev=False)


def test_fallback_on_hold_off_no_flapping(disks):
    root, ckpt, ev = disks
    fb = os.path.join(ckpt, "opd_fallback")
    assert run(root, ckpt, 5000) == "off" and not os.path.islink(os.path.join(root, "control"))
    assert run(root, ckpt, 150, "T1") == "switched_on" and ev == ["disk_fallback_on"]
    for n in ("control", "logs"):
        p = os.path.join(root, n)
        assert os.path.islink(p) and os.path.realpath(p) == os.path.realpath(os.path.join(fb, n))
        assert os.path.isdir(p + ".pre_fallback_T1")                       # the old directory kept
    assert open(os.path.join(root, "control", "events", "e1.json")).read() == "1"   # the files came along
    open(os.path.join(root, "control", "pulse.json"), "w").write("new")      # a writer through the old path ...
    assert open(os.path.join(fb, "control", "pulse.json")).read() == "new"     # ... lands on the checkpoint disk
    assert run(root, ckpt, 300) == "on" and run(root, ckpt, 150) == "on"      # 200-400: no switch either way
    assert run(root, ckpt, 450, "T2") == "switched_off" and ev[-1] == "disk_fallback_off"
    for n in ("control", "logs"):
        assert not os.path.islink(os.path.join(root, n)) and os.path.isdir(os.path.join(fb, n + ".T2"))   # copy kept
    assert open(os.path.join(root, "control", "pulse.json")).read() == "new"  # written during the fallback, back home
    assert run(root, ckpt, 300) == "off" and len(ev) == 2


def test_disk_write_fail_once_per_episode(disks):
    root, ckpt, ev = disks
    run(root, ckpt, 0.5); run(root, ckpt, 0.4)
    assert ev.count("disk_write_fail") == 1 and ev.count("disk_fallback_on") == 1
    run(root, ckpt, 5); assert ev.count("disk_write_fail") == 1           # still under 10 GB: the episode goes on
    run(root, ckpt, 50); run(root, ckpt, 0.5)
    assert ev.count("disk_write_fail") == 2                               # a new episode after recovering past 10 GB


def test_lines_from_control_env(disks):
    root, ckpt, ev = disks
    open(os.path.join(root, "control.env"), "w").write("export CTRL_FALLBACK_LOW_GB=1000\nexport CTRL_FALLBACK_HIGH_GB=2000\n")
    assert run(root, ckpt, 900) == "switched_on" and run(root, ckpt, 1500) == "on" and run(root, ckpt, 2500) == "switched_off"


def test_same_disk_or_no_ckpt_is_na(disks):
    root, ckpt, ev = disks
    assert C.disk_fallback(root=root, ckpt=ckpt, free_fn=lambda p: 1) == "n/a"   # both temp dirs on one disk here
    assert C.disk_fallback(root=root, ckpt="", free_fn=lambda p: 1) == "n/a" and ev == []


def test_disk_low_work_line_1000_and_control_env(monkeypatch):
    root = tempfile.mkdtemp(); monkeypatch.setattr(C, "ROOT", root); monkeypatch.delenv("DISK_ALERT_GB", raising=False)
    monkeypatch.delenv("CKPT_ROOT", raising=False)
    free = C._free_gb(root)
    assert (C.disk_low() == "work") == (free < 1000)                       # the default line is 1000 GB
    open(os.path.join(root, "control.env"), "w").write(f"export DISK_ALERT_GB={int(free) + 10}\n")
    assert C.disk_low() == "work"
    open(os.path.join(root, "control.env"), "w").write("export DISK_ALERT_GB=0\n")
    monkeypatch.setenv("DISK_ALERT_GB", str(int(free) + 10))
    assert C.disk_low() == ""                                              # control.env wins over the environment


@pytest.mark.parametrize("cmd", ["setenv DISK_ALERT_GB=1000", "setenv CKPT_ALERT_GB=100", "setenv CTRL_FALLBACK_LOW_GB=200", "setenv CTRL_FALLBACK_HIGH_GB=400"])
def test_setenv_disk_keys(cmd):
    R.parse({"cmd": cmd, "reason": "t"})
    assert R.RULES_VERSION == "2.5.74"


def test_rollouts_through_a_symlink():   # what upload_sync.sh (ls $RUN/rollouts/*.jsonl) and remote.put (os.walk) read
    run_dir, ck = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(os.path.join(ck, "rollouts")); open(os.path.join(ck, "rollouts", "1.jsonl"), "w").write("{}\n")
    os.symlink(os.path.join(ck, "rollouts"), os.path.join(run_dir, "rollouts"))
    src = os.path.join(run_dir, "rollouts")
    assert os.path.isdir(src) and [f for _, _, fs in os.walk(src) for f in fs] == ["1.jsonl"]
    us = open(os.path.join(HERE, "upload_sync.sh")).read()
    assert "up $RUN/rollouts runs/$TASK/rollouts" in us   # the result repository's path is the same


def test_pulse_side_never_waits_on_the_lock(disks):   # [ops-lead's A1 condition] the pulse skips the round while another check holds the lock
    import fcntl
    import time
    root, ckpt, ev = disks
    fb = os.path.join(ckpt, "opd_fallback"); os.makedirs(fb, exist_ok=True)
    with open(os.path.join(fb, ".lock"), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)                                   # run_tasks.sh mid-move
        t0 = time.time()
        r = C.disk_fallback(root=root, ckpt=ckpt, free_fn=lambda p: 150, check_dev=False, blocking=False)
        assert r == "busy" and time.time() - t0 < 1 and not os.path.islink(os.path.join(root, "control")) and ev == []
    assert run(root, ckpt, 150) == "switched_on"                          # the lock free again: the next check acts
    src = open(os.path.join(HERE, "control.py")).read()
    assert "disk_fallback(blocking=False)" in src and src.count("disk_fallback(blocking=False)") == 1
