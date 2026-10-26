"""[v2.5.9z88c] (10-26 06:49Z: heartbeat.sh's live_tasks counted CORPUS-RECORD's sandboxed task.sh as live tasks and its loop
hung; ops-lead / Fable: every scan, a stub each) a process `task.sh 90 X --dryrun-sandbox` is invisible to the node's process
scans: heartbeat live_tasks, control.py running_pids / proc_table, run_tasks.sh running / cpu_busy, upload_sync.sh's
still-running check, dr_sync.sh's training check, watchdog.sh's pids_of. The pgrep-based ones run anywhere; the /proc readers
(and `pgrep -a`, Linux only) need DRYRUN_NODE_TESTS=1 on the node. Only sandbox-shaped stubs are started: no stub ever looks like
a real task to the node's own loops. The deadman scan of v2.6 S5 gets its case there."""
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.dirname(HERE)
LINUX_NODE = os.path.isdir("/proc/self") and os.environ.get("DRYRUN_NODE_TESTS") == "1"
node_only = pytest.mark.skipif(not LINUX_NODE, reason="node only (DRYRUN_NODE_TESTS=1, Linux /proc)")
NAME = f"SBXTEST-{os.getpid()}"


@pytest.fixture
def stub():
    """a sandbox-shaped task.sh: <tmp>/scripts/task.sh 90 SBXTEST-<pid> --dryrun-sandbox, its own session, killed after"""
    tmp = tempfile.mkdtemp(prefix="sbx.")
    sh = os.path.join(tmp, "scripts", "task.sh"); os.makedirs(os.path.dirname(sh))
    open(sh, "w").write("sleep 60\n")
    p = subprocess.Popen(["bash", sh, "90", NAME, "--dryrun-sandbox"], start_new_session=True,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    yield p
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass
    shutil.rmtree(tmp, ignore_errors=True)


def _fn(path, name):   # a one-line shell function of a runbook script, as written there
    return re.search(rf"^{name}\(\) \{{.*?; \}}", open(os.path.join(RB, "scripts", path)).read(), re.M).group(0)


def test_sandbox_cmd_predicate():
    sys.path.insert(0, HERE)
    import control_rules  # noqa: F401  (control.py imports it)
    src = open(os.path.join(HERE, "control.py")).read()
    ns = {"re": re}
    exec(src[src.index('SANDBOX_MARK = '):src.index("OUR_ROOTS = ")], ns)
    f = ns["sandbox_cmd"]
    assert f("bash /x/scripts/task.sh 90 E10b-P5 --dryrun-sandbox") and f("bash /x/scripts/task.sh 90,91 E12d-P5")
    assert not f("bash /x/scripts/task.sh 3 E10b-P5") and not f("bash /x/scripts/task.sh 1,2 E12d-P5") and not f("bash task.sh cpu X")


def test_run_tasks_running_and_cpu_busy_do_not_see_it(stub):
    for script in (f"{_fn('run_tasks.sh', 'running')}\nrunning {NAME}", f"CPU_MAX=1\n{_fn('run_tasks.sh', 'cpu_busy')}\ncpu_busy"):
        assert subprocess.run(["bash", "-c", script]).returncode != 0, script.split("\n")[-1]


def test_positive_control_running_sees_an_unmarked_stub():   # the check is not vacuous: without the marker running() sees it
    tmp = tempfile.mkdtemp(prefix="sbx.")
    sh = os.path.join(tmp, "scripts", "task.sh"); os.makedirs(os.path.dirname(sh)); open(sh, "w").write("sleep 60\n")
    p = subprocess.Popen(["bash", sh, "90", NAME + "-U"], start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # GPU 90: still no real task to the node's loops
    try:
        time.sleep(0.5)
        assert subprocess.run(["bash", "-c", f"{_fn('run_tasks.sh', 'running')}\nrunning {NAME}-U"]).returncode == 0
    finally:
        os.killpg(p.pid, signal.SIGKILL); shutil.rmtree(tmp, ignore_errors=True)


def test_watchdog_pids_of_does_not_see_it(stub):
    for loop in ("run_tasks", "heartbeat", "uploader", "watchdog"):
        out = subprocess.run(["bash", "-c", f"{_fn('watchdog.sh', 'pids_of')}\npids_of {loop}"], capture_output=True, text=True).stdout
        assert str(stub.pid) not in out.split(), loop


@node_only
def test_upload_sync_and_dr_sync_checks_do_not_see_it(stub):
    us = open(os.path.join(HERE, "upload_sync.sh")).read()
    expr = re.search(r'\[ -z "(\$\(pgrep -af "scripts/task.*?--dryrun-sandbox\))" \]', us).group(1)
    assert subprocess.run(["bash", "-c", f'TASK={NAME}; [ -z "{expr}" ]']).returncode == 0
    ds = open(os.path.join(HERE, "dr_sync.sh")).read()
    assert '| grep -v dryrun_shim)" ]' in ds and "pgrep -af \"experiment_name=$TASK( |$)\"" in ds


@node_only
def test_heartbeat_live_tasks_does_not_see_it(stub):
    src = open(os.path.join(HERE, "heartbeat.sh")).read()
    ns = {"re": re, "os": os, "time": time}
    exec(src[src.index("def gpu_arg"):src.index("def kind(")], ns)
    assert NAME not in ns["live_tasks"]()


@node_only
def test_control_running_pids_and_proc_table_do_not_see_it(stub):
    sys.path.insert(0, HERE)
    import control
    assert control.running_pids(NAME) == []
    assert not any(r.get("task") == NAME for r in control.proc_table())


@node_only
def test_corpus_selfcheck_passes_on_this_node():
    sys.path.insert(0, HERE)
    import dryrun_json as J
    base = tempfile.mkdtemp(prefix="dryrun_json.selfcheck.")
    try:
        assert J.scanner_selfcheck(RB, base) == ("ok", [])
    finally:
        shutil.rmtree(base, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ [v2.5.9z88d] script-name scans
RAW = re.compile(r"(?<![_A-Za-z])pgrep -f \"(run_queue_eval|scripts/probe_net2|fetch_model|scripts/git_fetch|scripts/weights_replay|dr_syn|venv-vl)"
                 r"|ps -eo args= \| grep -o \"(run_queue_eval|fetch_model)")


def test_z88d_script_name_scans_use_ours_pgrep():   # static: none of the listed scans calls pgrep / ps directly any more
    for f in ("run_tasks.sh", "dr.sh", "venv_vl_nvrtc.sh"):
        bad = [l for l in open(os.path.join(HERE, f)).read().split("\n") if RAW.search(l) and not l.lstrip().startswith("#")]
        assert not bad, (f, bad)
    src = open(os.path.join(HERE, "control.py")).read()
    raw = [m.group(0) for m in re.finditer(r'\["pgrep", "-a?f", [^\]]*\]', src)]
    assert all("task.sh" in x or '"-f", t]' in x or "pat]" in x for x in raw if "*args" not in x), raw   # running_pids (anchored), rm-ours (paths)


def test_z88d_ours_pgrep_shell_positive_control(tmp_path):   # without DRYRUN_SB a matching process is seen (Mac and node alike)
    sh = tmp_path / "scripts" / "probe_net2.sh"; sh.parent.mkdir(); sh.write_text("sleep 60\n")
    env = {k: v for k, v in os.environ.items() if k != "DRYRUN_SB"}
    p = subprocess.Popen(["bash", str(sh), "90"], start_new_session=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        fn = re.search(r"^ours_pgrep\(\) \{.*?; \}$", open(os.path.join(HERE, "env.sh")).read(), re.M).group(0)
        out = subprocess.run(["bash", "-c", fn + f'\nours_pgrep -f "{tmp_path}/scripts/probe_net2\\.sh"'], capture_output=True, text=True).stdout
        assert str(p.pid) in out.split()
    finally:
        os.killpg(p.pid, signal.SIGKILL)


@node_only
def test_z88d_ours_pgrep_hides_a_sandbox_process(tmp_path):
    sh = tmp_path / "scripts" / "probe_net2.sh"; sh.parent.mkdir(); sh.write_text("sleep 60\n")
    p = subprocess.Popen(["bash", str(sh), "90"], start_new_session=True, env=dict(os.environ, DRYRUN_SB=str(tmp_path)),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(0.5)
        fn = re.search(r"^ours_pgrep\(\) \{.*?; \}$", open(os.path.join(HERE, "env.sh")).read(), re.M).group(0)
        clean = {k: v for k, v in os.environ.items() if k != "DRYRUN_SB"}
        out = subprocess.run(["bash", "-c", fn + f'\nours_pgrep -f "{tmp_path}/scripts/probe_net2\\.sh"'], capture_output=True, text=True, env=clean).stdout
        assert str(p.pid) not in out.split()
        sys.path.insert(0, HERE)
        import control
        assert not [l for l in control.ours_pgrep("-f", f"{tmp_path}/scripts/probe_net2[.]sh") if l.split()[0] == str(p.pid)]
    finally:
        os.killpg(p.pid, signal.SIGKILL)
