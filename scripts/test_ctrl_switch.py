"""[v2.5.9z71] (ops-lead 10-23 18:05 UTC) the planned switch of control/ and logs/ to the checkpoint disk: one atomic exchange
(control/ exists at every moment), the writes land there, a running task's open log is linked, the loops' files left to
the loops (ctrl_reopen, their .sha removed), the rollback (setenv CTRL_ON_CKPT=0), a refused exchange moves nothing, a
switch cut short is finished by the next check; new tasks' run directories on the checkpoint disk (place_run); the 300 GB
line of new tasks (tasks_disk_low); the new setenv keys; run_tasks.sh's wiring."""
import errno
import os
import sys
import tempfile
import threading
import time
import types

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
    for x in ("watchdog", "uploader", "gh_heartbeat", "heartbeat", "run_tasks", "weights_replay"):
        open(os.path.join(root, "control", x + ".sha"), "w").write("abc\n")
    open(os.path.join(root, "control", "watchdog.lock"), "w").write("")
    for f in ("heartbeat.log", "control.log", "E10b-X.log", "EVAL-y.log"):
        open(os.path.join(root, "logs", f), "w").write("a\n")
    monkeypatch.setattr(C, "ROOT", root)
    monkeypatch.delenv("CTRL_ON_CKPT", raising=False); monkeypatch.delenv("RUNS_ON_CKPT", raising=False)
    ev = []
    monkeypatch.setattr(C, "event", lambda task, kind, detail="": ev.append((kind, detail)))
    return root, ckpt, ev


def run(root, ckpt, free=5000, stamp="S1", held=()):
    return C.disk_fallback(root=root, ckpt=ckpt, free_fn=lambda p: free, stamp=stamp, check_dev=False,
                           held_fn=lambda prefix: {os.path.join(prefix, h) for h in held})


def kinds(ev):
    return [k for k, _ in ev]


def test_planned_switch_with_free_space(disks):   # CTRL_ON_CKPT defaults to 1: no free-space trigger needed
    root, ckpt, ev = disks
    assert run(root, ckpt) == "switched_on" and kinds(ev) == ["ctrl_switch_on"]
    fb = os.path.join(ckpt, "opd_fallback")
    for n in ("control", "logs"):
        p = os.path.join(root, n)
        assert os.path.islink(p) and os.path.realpath(p) == os.path.realpath(os.path.join(fb, n + ".S1"))
        assert os.path.isdir(p + ".pre_fallback_S1") and not os.path.islink(p + ".pre_fallback_S1")   # the old directory kept
        assert not os.path.lexists(p + ".z71_link")
    assert open(os.path.join(root, "control", "events", "e1.json")).read() == "1"
    open(os.path.join(root, "control", "pulse.json"), "w").write("new")                          # a writer through the path
    assert open(os.path.join(fb, "control.S1", "pulse.json")).read() == "new"                     # lands on idx1
    assert open(os.path.join(root, "control.pre_fallback_S1", "pulse.json")).read() == "{}"       # not on idx0
    assert run(root, ckpt) == "on" and run(root, ckpt, free=50) == "on" and len(ev) == 1          # no flapping, no event
    assert not [f for f in os.listdir(root) if f.startswith(".z71_xtest")]                       # the self-test left nothing


def test_the_loops_reopen(disks):
    root, ckpt, ev = disks
    run(root, ckpt)
    cd = os.path.join(root, "control")
    assert os.path.exists(os.path.join(cd, "ctrl_reopen")) and os.path.exists(os.path.join(cd, "ctrl_reopen_wr"))
    for x in ("watchdog", "uploader", "gh_heartbeat", "heartbeat"):                               # restart on the next ensure
        assert not os.path.exists(os.path.join(cd, x + ".sha"))
    for x in ("run_tasks", "weights_replay"):                                                     # reopen / between two uploads
        assert os.path.exists(os.path.join(cd, x + ".sha"))


def test_no_moment_without_control(disks):   # ③ the exchange: control/ (and a file in it) is there at every poll
    root, ckpt, ev = disks
    for i in range(300):
        open(os.path.join(root, "control", "events", f"f{i}.json"), "w").write("x" * 2000)
    gaps, stop = [], threading.Event()

    def watch():
        p = os.path.join(root, "control", "pulse.json")
        while not stop.is_set():
            if not os.path.exists(p):
                gaps.append(time.time())
    th = threading.Thread(target=watch); th.start()
    try:
        assert run(root, ckpt) == "switched_on"
    finally:
        stop.set(); th.join()
    assert gaps == []


def test_the_watch_sees_z48s_gap(disks, monkeypatch):   # the same watch against two renames (z48): it does see the gap
    root, ckpt, ev = disks

    def two_renames(a, b):
        t = b + ".tmp_gap"; os.rename(b, t); time.sleep(0.05); os.rename(a, b); os.rename(t, a)
    monkeypatch.setattr(C, "_exchange", two_renames)
    gaps, stop = [], threading.Event()

    def watch():
        while not stop.is_set():
            if not os.path.exists(os.path.join(root, "control", "pulse.json")):
                gaps.append(1)
    th = threading.Thread(target=watch); th.start()
    try:
        run(root, ckpt)
    finally:
        stop.set(); th.join()
    assert gaps


def test_gap_writes_come_along_and_later_writes_win(disks, monkeypatch):
    root, ckpt, ev = disks
    real = C._exchange
    calls = []

    def exch(a, b):
        if b.endswith("control") and not calls:
            open(os.path.join(b, "late.json"), "w").write("late")        # written after the copies, before the swap
            calls.append(1)
        real(a, b)
        if b.endswith("control") and len(calls) == 1:
            time.sleep(0.02); open(os.path.join(b, "pulse.json"), "w").write("after")   # after the swap, through the path
            calls.append(2)
    monkeypatch.setattr(C, "_exchange", exch)
    assert run(root, ckpt) == "switched_on"
    assert open(os.path.join(root, "control", "late.json")).read() == "late"
    assert open(os.path.join(root, "control", "pulse.json")).read() == "after"   # the gap copy never puts the older one back


def test_open_task_log_is_linked_loop_files_are_not(disks):
    root, ckpt, ev = disks
    held = ("E10b-X.log", "heartbeat.log", "control.log")
    assert run(root, ckpt, held=held) == "switched_on" and "1 open files linked" in ev[0][1]   # logs/E10b-X.log only
    lg = os.path.join(root, "logs")
    old = os.path.realpath(os.path.join(root, "logs.pre_fallback_S1"))
    t = os.path.join(os.path.realpath(lg), "E10b-X.log")
    assert os.path.islink(t) and os.readlink(t) == os.path.join(old, "E10b-X.log")
    open(os.path.join(old, "E10b-X.log"), "a").write("still writing\n")                       # the task's open fd
    assert open(os.path.join(lg, "E10b-X.log")).read().endswith("still writing\n")            # readers see it
    for f in ("heartbeat.log", "control.log", "EVAL-y.log"):                                    # loop / shared / not open
        assert not os.path.islink(os.path.join(os.path.realpath(lg), f))


def test_link_count_in_event(disks):
    root, ckpt, ev = disks
    run(root, ckpt, held=("E10b-X.log", "EVAL-y.log", "watchdog.lock"))
    assert "2 open files linked" in ev[0][1]


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="Linux /proc only (node-only test)")
def test_real_open_fd_found(disks):   # node-only: /proc/<pid>/fd
    root, ckpt, ev = disks
    fh = open(os.path.join(root, "logs", "E10b-X.log"), "a")
    import subprocess
    child = subprocess.Popen(["sleep", "30"], stdout=fh)
    try:
        assert C.disk_fallback(root=root, ckpt=ckpt, free_fn=lambda p: 5000, stamp="S1", check_dev=False) == "switched_on"
        assert os.path.islink(os.path.join(os.path.realpath(os.path.join(root, "logs")), "E10b-X.log"))
    finally:
        child.kill(); fh.close()


def test_rollback_setenv(disks):   # ③ setenv CTRL_ON_CKPT=0 brings them back, with what was written meanwhile
    root, ckpt, ev = disks
    run(root, ckpt)
    open(os.path.join(root, "control", "pulse.json"), "w").write("during")
    open(os.path.join(root, "control.env"), "w").write("export CTRL_ON_CKPT=0\n")
    assert run(root, ckpt, free=5000, stamp="S2") == "switched_off" and kinds(ev)[-1] == "disk_fallback_off"
    for n in ("control", "logs"):
        p = os.path.join(root, n)
        assert os.path.isdir(p) and not os.path.islink(p) and not os.path.lexists(p + ".z71_back")
        assert os.path.isdir(os.path.join(ckpt, "opd_fallback", n + ".S1"))                    # the copy stays
    assert open(os.path.join(root, "control", "pulse.json")).read() == "during"
    assert run(root, ckpt, free=5000) == "off"
    assert run(root, ckpt, free=150, stamp="S3") == "switched_on" and kinds(ev)[-1] == "disk_fallback_on"   # z48's line again
    assert os.path.realpath(os.path.join(root, "control")) == os.path.realpath(os.path.join(ckpt, "opd_fallback", "control.S3"))


def test_refused_exchange_moves_nothing(disks, monkeypatch):   # [v2.5.9z71b] an error other than "no exchange here"
    root, ckpt, ev = disks

    def bad_exchange(a, b):
        raise OSError(errno.EPERM, "Operation not permitted")
    monkeypatch.setattr(C, "_exchange", bad_exchange)
    assert run(root, ckpt) == "refused" and run(root, ckpt) == "refused"
    assert kinds(ev) == ["ctrl_switch_refused"]                                                # once per episode
    for n in ("control", "logs"):
        assert os.path.isdir(os.path.join(root, n)) and not os.path.islink(os.path.join(root, n))
    assert sorted(os.listdir(root)) == ["control", "logs"]                                     # nothing left behind
    assert open(os.path.join(root, "control", "pulse.json")).read() == "{}"


def test_cut_short_is_finished(disks, monkeypatch):   # control switched, logs refused: the next check does logs only
    root, ckpt, ev = disks
    real, n = C._exchange, []

    def flaky(a, b):
        n.append(b)
        if b.endswith("/logs"):
            raise OSError(errno.EIO, "I/O error")
        real(a, b)
    monkeypatch.setattr(C, "_exchange", flaky)
    assert run(root, ckpt) == "refused"
    assert os.path.islink(os.path.join(root, "control")) and not os.path.islink(os.path.join(root, "logs"))
    assert os.path.exists(os.path.join(root, "control", "ctrl_reopen"))                        # the loops reopen control/ files
    monkeypatch.setattr(C, "_exchange", real)
    assert run(root, ckpt, stamp="S2") == "switched_on"
    assert os.path.islink(os.path.join(root, "logs")) and os.path.realpath(os.path.join(root, "control")).endswith("control.S1")


def test_exchange_itself(tmp_path):
    d, ln, q = tmp_path / "d", tmp_path / "l", tmp_path / "q"
    d.mkdir(); q.mkdir(); (d / "f").write_text("1"); os.symlink(q, ln)
    C._exchange(str(ln), str(d))
    assert os.path.islink(d) and (ln / "f").read_text() == "1"
    with pytest.raises(OSError):
        C._exchange(str(tmp_path / "missing"), str(d))


# ---------------- place_run: new tasks' run directories on the checkpoint disk
@pytest.fixture
def runs(monkeypatch):
    root, ckpt = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "runs"))
    monkeypatch.setattr(C, "ROOT", root); monkeypatch.delenv("RUNS_ON_CKPT", raising=False)
    return root, ckpt


def place(root, ckpt, t, stamp="S1", busy=False):
    return C.place_run(t, root=root, ckpt=ckpt, check_dev=False, stamp=stamp, busy=busy)


def test_new_run_dir_on_ckpt(runs):
    root, ckpt = runs
    assert place(root, ckpt, "EVAL-a") == "linked_new"
    p = os.path.join(root, "runs", "EVAL-a")
    assert os.path.islink(p) and os.path.realpath(p) == os.path.realpath(os.path.join(ckpt, "runs_live", "EVAL-a.S1"))
    open(os.path.join(p, "x.jsonl"), "w").write("{}\n")
    assert os.path.exists(os.path.join(ckpt, "runs_live", "EVAL-a.S1", "x.jsonl"))
    assert place(root, ckpt, "EVAL-a", "S2") == "linked"                                       # a retry keeps its directory
    assert os.path.isdir(p) and [f for _, _, fs in os.walk(p) for f in fs] == ["x.jsonl"]      # os.walk from the link (uploads)
    assert sorted(os.listdir(os.path.join(root, "runs"))) == ["EVAL-a"]


def test_marker_only_dir_moved(runs):
    root, ckpt = runs
    p = os.path.join(root, "runs", "E10b-Y"); os.makedirs(p)
    open(os.path.join(p, ".prefetch_ok"), "w").write("1"); os.symlink("/nonexistent/ckpt", os.path.join(p, "ckpt"))
    assert place(root, ckpt, "E10b-Y") == "linked_moved"
    assert os.path.islink(p) and open(os.path.join(p, ".prefetch_ok")).read() == "1" and os.readlink(os.path.join(p, "ckpt")) == "/nonexistent/ckpt"
    assert os.path.isdir(os.path.join(root, "runs_pre_link", "E10b-Y.S1"))                     # kept, nothing removed
    assert sorted(os.listdir(os.path.join(root, "runs"))) == ["E10b-Y"]


def test_kept_when_big_with_subdirs_running_off_or_bad(runs):
    root, ckpt = runs
    p = os.path.join(root, "runs", "EVAL-b"); os.makedirs(os.path.join(p, "up"))
    assert place(root, ckpt, "EVAL-b") == "kept" and not os.path.islink(p)                    # an earlier attempt's outputs
    p = os.path.join(root, "runs", "EVAL-c"); os.makedirs(p); open(os.path.join(p, "out.jsonl"), "w").write("x" * 3000)
    open(os.path.join(root, "control.env"), "w").write("export RUNS_MOVE_MAX_MB=0.001\n")
    assert place(root, ckpt, "EVAL-c") == "kept"
    assert place(root, ckpt, "EVAL-d", busy=True) == "running" and not os.path.lexists(os.path.join(root, "runs", "EVAL-d"))
    assert place(root, ckpt, "../x") == "n/a" and place(root, "", "EVAL-e") == "n/a"
    open(os.path.join(root, "control.env"), "w").write("export RUNS_ON_CKPT=0\n")
    assert place(root, ckpt, "EVAL-f") == "off" and not os.path.lexists(os.path.join(root, "runs", "EVAL-f"))


def test_same_disk_is_na(runs):
    root, ckpt = runs
    assert C.place_run("EVAL-g", root=root, ckpt=ckpt) == "n/a"                               # both temp dirs on one disk


def test_result_queue_paths_through_the_link(runs, monkeypatch):   # z65's hold_ok / result_stage on a linked run directory
    import weights_replay as W
    root, ckpt = runs
    place(root, ckpt, "EVAL-h")
    monkeypatch.setenv("OPD_ROOT", root); monkeypatch.setenv("CKPT_ROOT", ckpt)
    hold = os.path.join(root, "runs", "EVAL-h")
    assert W.hold_ok(hold, "runs/EVAL-h")
    st = W.result_stage(hold, "runs/EVAL-h")
    assert os.path.realpath(st).startswith(os.path.realpath(os.path.join(ckpt, "runs_live")))   # staged on the same disk


# ---------------- the 300 GB line
def free_gb(monkeypatch, gb):
    monkeypatch.setattr(C, "_free_gb", lambda p: gb)


def test_tasks_line(runs, monkeypatch):
    root, ckpt = runs
    for v in ("TASKS_DISK_MIN_GB", "TASKS_DISK_CRIT_GB"):
        monkeypatch.delenv(v, raising=False)
    free_gb(monkeypatch, 301); assert C.tasks_disk_low() == ""
    free_gb(monkeypatch, 299); assert C.tasks_disk_low() == "work"
    free_gb(monkeypatch, 99); assert C.tasks_disk_low() == "work critical"
    open(os.path.join(root, "control.env"), "w").write("export TASKS_DISK_MIN_GB=500\nexport TASKS_DISK_CRIT_GB=50\n")
    free_gb(monkeypatch, 400); assert C.tasks_disk_low() == "work"
    free_gb(monkeypatch, 99); assert C.tasks_disk_low() == "work"


def test_setenv_keys_and_pulse(runs):
    for k in ("CTRL_ON_CKPT=0", "RUNS_ON_CKPT=1", "RUNS_MOVE_MAX_MB=64", "TASKS_DISK_MIN_GB=300", "TASKS_DISK_CRIT_GB=100"):
        R.parse({"cmd": "setenv " + k, "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 84)
    root, _ = runs
    os.makedirs(os.path.join(root, "control")); open(os.path.join(root, "control", "tasks_guard"), "w").write("work\n")
    p = C.build_pulse({"tasks": {}}, "")
    assert p["tasks_guard"] == "work" and p["ctrl_on_ckpt"] is False


# ---------------- run_tasks.sh's wiring
RT = open(os.path.join(HERE, "run_tasks.sh")).read()


def test_wiring():
    loop = RT[RT.index("while true; do\n  date -u +%s > $CDIR/loop_run_tasks"):]
    assert loop.index("ctrl_reopen   # [v2.5.9z71]") < loop.index("ensure_watchdog;")           # reopen before the ensures
    assert loop.index("tasks_guard   #") < loop.index("&& prefetch_pass") < loop.index("&& fetch_pass") < loop.index("&& git_pass")
    for x in ("prefetch_pass", "fetch_pass", "git_pass"):   # [v2.5.9z71c] below either work-disk line
        assert f'[ -z "$ALL_LOW$READ_LOW" ] && {x}' in loop
    assert loop.index("readings_guard   #") < loop.index("&& prefetch_pass")
    i = loop.index('if [ -n "$ALL_LOW" ] && ! runs_on_ckpt $t; then AHELD="$AHELD $t"; continue; fi')   # [v2.5.9z71c]
    assert loop.index("running $t && continue") < i < loop.index("if cpu_task $t; then")          # the CPU lane too
    assert '! verify_busy && [ -z "$ALL_LOW" ]; then' in loop                                    # ad-hoc requests wait
    assert "(setsid nohup nice -n 10 python $RUNBOOK/scripts/control.py disk_fallback >> $OPD_ROOT/logs/control.log" in loop
    la = RT[RT.index("launch() {"):RT.index("# deps of a task id")]
    assert la.index("control.py place_run $2") < la.index("mkdir -p $OPD_ROOT/runs/$2")
    blk = RT[RT.index("tasks_guard() {"):RT.index("deps_of() {")]
    assert "kill" not in blk.replace("killed", "")                                              # nothing running is stopped



# ---------------- [v2.5.9z71b] no exchange on the work disk (the node 10-23 19:01 UTC: EINVAL): two renames back to back
def no_exchange(a, b):
    raise OSError(errno.EINVAL, "Invalid argument")


def test_einval_switches_by_two_renames(disks, monkeypatch):
    root, ckpt, ev = disks
    monkeypatch.setattr(C, "_exchange", no_exchange)
    assert run(root, ckpt, held=("E10b-X.log",)) == "switched_on" and kinds(ev) == ["ctrl_switch_on"]
    assert "two renames each (no exchange on the work disk), gaps " in ev[0][1] and "1 open files linked" in ev[0][1]
    fb = os.path.join(ckpt, "opd_fallback")
    for n in ("control", "logs"):
        p = os.path.join(root, n)
        assert os.path.islink(p) and os.path.realpath(p) == os.path.realpath(os.path.join(fb, n + ".S1"))
        assert os.path.isdir(p + ".pre_fallback_S1") and not os.path.lexists(p + ".z71_link")
    open(os.path.join(root, "control", "pulse.json"), "w").write("new")
    assert open(os.path.join(fb, "control.S1", "pulse.json")).read() == "new"
    assert os.path.exists(os.path.join(root, "control", "ctrl_reopen"))
    assert not [f for f in os.listdir(root) if f.startswith(".z71_xtest")]
    open(os.path.join(root, "control.env"), "w").write("export CTRL_ON_CKPT=0\n")                 # the rollback the same way
    assert run(root, ckpt, stamp="S2") == "switched_off" and "two renames each" in ev[-1][1]
    for n in ("control", "logs"):
        assert not os.path.islink(os.path.join(root, n)) and not os.path.lexists(os.path.join(root, n + ".z71_off"))
    assert open(os.path.join(root, "control", "pulse.json")).read() == "new"


def test_a_writer_inside_the_gap(disks, monkeypatch):   # a mkdir -p control/ between the two renames: set aside, merged
    root, ckpt, ev = disks
    monkeypatch.setattr(C, "_exchange", no_exchange)
    real, done = os.rename, []

    def rename(a, b):
        real(a, b)
        if b.endswith(".pre_fallback_S1") and b.startswith(os.path.join(root, "control")) and not done:
            os.makedirs(os.path.join(root, "control", "events")); open(os.path.join(root, "control", "events", "gap.json"), "w").write("g")
            done.append(1)
    monkeypatch.setattr(C.os, "rename", rename)
    assert run(root, ckpt) == "switched_on" and done
    assert os.path.islink(os.path.join(root, "control"))
    assert open(os.path.join(root, "control", "events", "gap.json")).read() == "g"               # merged into the new one
    assert open(os.path.join(root, "control", "events", "e1.json")).read() == "1"
    assert os.path.isdir(os.path.join(root, "control.gap_S1"))                                    # kept, nothing removed


def test_second_rename_fails_puts_the_old_back(disks, monkeypatch):
    root, ckpt, ev = disks
    monkeypatch.setattr(C, "_exchange", no_exchange)
    real = os.rename

    def rename(a, b):
        if a.endswith(".z71_link"):
            raise OSError(errno.EIO, "I/O error")
        real(a, b)
    monkeypatch.setattr(C.os, "rename", rename)
    assert run(root, ckpt) == "refused" and kinds(ev) == ["ctrl_switch_refused"] and "I/O error" in ev[0][1]
    for n in ("control", "logs"):
        assert os.path.isdir(os.path.join(root, n)) and not os.path.islink(os.path.join(root, n))
        assert not os.path.lexists(os.path.join(root, n + ".z71_link"))
    assert open(os.path.join(root, "control", "pulse.json")).read() == "{}"


def test_other_errors_still_refuse(disks, monkeypatch):
    root, ckpt, ev = disks
    monkeypatch.setattr(C, "_exchange", lambda a, b: (_ for _ in ()).throw(OSError(errno.EACCES, "Permission denied")))
    assert run(root, ckpt) == "refused" and "Permission denied" in ev[0][1]
    assert not os.path.islink(os.path.join(root, "control"))


def test_place_run_marker_dir_without_exchange(runs, monkeypatch):
    root, ckpt = runs
    monkeypatch.setattr(C, "_exchange", no_exchange)
    p = os.path.join(root, "runs", "EVAL-m"); os.makedirs(p); open(os.path.join(p, ".prefetch_ok"), "w").write("1")
    assert place(root, ckpt, "EVAL-m") == "linked_moved"
    assert os.path.islink(p) and open(os.path.join(p, ".prefetch_ok")).read() == "1"
    assert os.path.isdir(os.path.join(root, "runs_pre_link", "EVAL-m.S1"))



# ---------------- [v2.5.9z71c] (Fable 10-24) which tasks the work disk's lines still hold
def test_runs_ckpt_and_run_placeable(runs):
    root, ckpt = runs
    assert C.runs_ckpt(root=root, ckpt=ckpt, check_dev=False) == "1" and C.runs_ckpt(root=root, ckpt=ckpt) == "0"   # same disk here
    assert C.runs_ckpt(root=root, ckpt="", check_dev=False) == "0"
    open(os.path.join(root, "control.env"), "w").write("export RUNS_ON_CKPT=0\n")
    assert C.runs_ckpt(root=root, ckpt=ckpt, check_dev=False) == "0"
    os.remove(os.path.join(root, "control.env"))
    assert C.run_placeable("EVAL-new", root=root)                                       # no directory yet: place_run links it
    p = os.path.join(root, "runs", "EVAL-m"); os.makedirs(p); open(os.path.join(p, ".prefetch_ok"), "w").write("1")
    assert C.run_placeable("EVAL-m", root=root)                                         # markers only: moved
    os.makedirs(os.path.join(p, "up")); assert not C.run_placeable("EVAL-m", root=root)   # an earlier attempt's outputs: kept
    os.symlink(ckpt, os.path.join(root, "runs", "EVAL-l")); assert C.run_placeable("EVAL-l", root=root)
