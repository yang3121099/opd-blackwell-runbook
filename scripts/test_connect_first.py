"""[v2.5.9z106] connect-first (Fable 11-01; ops-lead's A1 11-01 16:11 UTC): the new control commands env-probe / setup / verify /
tail / restart-loops / dispatch, the status and pulse fields, and the control plane without torch.
Run: python -m pytest -q scripts/test_connect_first.py"""
import importlib
import json
import os
import socket
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control_rules as R  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    rb, root = tmp_path / "rb", tmp_path / "root"
    (rb / "scripts").mkdir(parents=True); (rb / "control").mkdir(); (root / "logs").mkdir(parents=True)
    monkeypatch.setenv("OPD_ROOT", str(root)); monkeypatch.setenv("RUNBOOK", str(rb)); monkeypatch.setenv("OPD_NODE", "testnode")
    monkeypatch.setenv("CONTROL_DIRECT", "0")
    import control
    importlib.reload(control)
    os.makedirs(control.CDIR, exist_ok=True)
    monkeypatch.setattr(control, "queue_put", lambda src, dst, **k: True)
    return types.SimpleNamespace(c=control, rb=rb, root=root)


@pytest.mark.parametrize("cmd", ["env-probe", "setup 2", "setup 7", "setup all", "setup stop", "verify", "tail heartbeat.log",
                                 "tail heartbeat.log 200", "tail sub/x.log 5", "restart-loops", "dispatch on", "dispatch off"])
def test_new_verbs_accepted(cmd):
    R.parse({"cmd": cmd, "reason": "t"})


@pytest.mark.parametrize("cmd", ["setup 1", "setup 8", "setup", "setup x", "tail ../x", "tail /etc/passwd", "tail a/../../b",
                                 "tail x.log 201", "tail x.log 0", "tail x.log;id", "dispatch maybe", "restart-loops now", "env-probe x"])
def test_new_verbs_refused(cmd):
    with pytest.raises(R.Reject):
        R.parse({"cmd": cmd, "reason": "t"})


def test_new_verbs_lifetimes():
    assert R.VERBS["tail"][2] == R.VERBS["env-probe"][2] == R.H24   # read only
    assert R.VERBS["setup"][2] == R.VERBS["verify"][2] == R.VERBS["restart-loops"][2] == R.VERBS["dispatch"][2] == R.H2


def test_tail_confined_to_logs(env, tmp_path):
    logs = env.root / "logs"
    (logs / "a.log").write_text("".join(f"line {i}\n" for i in range(300)))
    (tmp_path / "secret.txt").write_text("outside\n")
    os.symlink(tmp_path / "secret.txt", logs / "link.log")
    st, msg, _ = env.c.v_tail(["a.log", "3"], 11)
    assert st == "ok" and open(os.path.join(env.c.CDIR, "req_11", "tail.txt")).read() == "line 297\nline 298\nline 299\n"
    assert env.c.v_tail(["link.log"], 12)[0] == "rejected"   # a link out of logs/
    assert env.c.v_tail(["missing.log"], 13)[0] == "rejected"
    st, _, _ = env.c.v_tail(["a.log"], 14)
    assert len(open(os.path.join(env.c.CDIR, "req_14", "tail.txt")).read().splitlines()) == 50   # default n


def test_tail_redaction(env):
    host = socket.gethostname().split(".")[0]
    text = (f"opened {env.root}/runs/x/log and {env.rb}/scripts/a.py\n"
            "token hf_" + "Ab12" * 10 + " short hf_abcd1234XYZ\n"
            "proxy http://user:pa55@proxy.example:3128 and 10.1.2.3:8080, fe80:0:0:0:1:2:3:4\n"
            f"host {host} ready\n")
    (env.root / "logs" / "r.log").write_text(text)
    env.c.v_tail(["r.log", "10"], 21)
    out = open(os.path.join(env.c.CDIR, "req_21", "tail.txt")).read()
    assert str(env.root) not in out and str(env.rb) not in out and "<opd>/runs/x/log" in out and "<runbook>/scripts/a.py" in out
    nested = env.root / "runbook"   # [rehearsal 11-01] a runbook path that starts with the opd root's path
    (env.root / "logs" / "n.log").write_text(f"x {env.root}/run {env.root}book/y\n")
    assert "hf_Ab12" not in out and "hf_abcd" not in out and "user:pa55" not in out and "//***:***@" in out
    assert "10.1.2.3" not in out and "<ip>" in out and "fe80:0:0:0" not in out
    if len(host) > 2:
        assert f"host {host} " not in out and "<host>" in out


def test_setup_state_running_then_stop(env):
    log = env.root / "logs" / "setup_step_2.log"
    log.write_text("".join(f"step line {i} {env.root}/x\n" for i in range(10)))
    env.c.setup_state_write("2", "overnight", "running")
    s1 = json.load(open(os.path.join(env.c.CDIR, "setup_state.json")))
    assert s1["state"] == "running" and s1["started_utc"] and s1["last_lines"] == []
    env.c.setup_state_write("2", "overnight", "STOP", str(log))
    s2 = json.load(open(os.path.join(env.c.CDIR, "setup_state.json")))
    assert s2["state"] == "STOP" and s2["started_utc"] == s1["started_utc"] and len(s2["last_lines"]) == 5
    assert all(str(env.root) not in x for x in s2["last_lines"]) and s2["last_lines"][-1].startswith("step line 9 <opd>/x")
    assert os.path.exists(os.path.join(env.c.CDIR, "heartbeat_now"))
    st = env.c.status()
    assert st["setup_state"]["state"] == "STOP" and st["dispatch_frozen"] is False and "env_probe" in st and "loops" in st


def test_setup_refused_without_bootstrap(env):
    assert env.c.v_setup(["all"], 31)[0] == "rejected"


def test_verify_state(env):
    log = env.root / "logs" / "verify_41.log"
    log.write_text("== 1. G2\nG2 PASS (21 passed)\nVLLM PASS\nSMOKE FAIL (exit 1, steps with loss 0)\nUPLOAD_CHECK PASS (60s)\n")
    env.c.verify_state(str(log), 41)
    v = json.load(open(os.path.join(env.c.CDIR, "verify_state.json")))
    assert v["result"] == {"G2": "PASS", "VLLM": "PASS", "SMOKE": "FAIL", "UPLOAD_CHECK": "PASS", "CONTROL": None} and v["seq"] == 41
    assert env.c.status()["verify"]["result"]["SMOKE"] == "FAIL"


def test_dispatch_needs_the_main_venv(env):
    (env.root / "venv-ctl").mkdir()
    assert env.c.v_dispatch(["off"], 51)[0] == "ok" and os.path.exists(os.path.join(env.c.CDIR, "dispatch_frozen"))
    assert env.c.v_dispatch(["on"], 52)[0] == "rejected" and os.path.exists(os.path.join(env.c.CDIR, "dispatch_frozen"))
    open(os.path.join(env.c.CDIR, "venv_ready"), "w").write("x\n")
    # [v2.5.9z127] and the canary on this node: PROBE-TAIL's report PASS (verify's result does not matter)
    st, msg, _ = env.c.v_dispatch(["on"], 53)
    assert st == "rejected" and "canary PROBE-TAIL" in msg and os.path.exists(os.path.join(env.c.CDIR, "dispatch_frozen"))
    os.makedirs(env.root / "runs" / "PROBE-TAIL"); rep = env.root / "runs" / "PROBE-TAIL" / "report.json"
    rep.write_text('{"result": "FAIL"}'); assert env.c.v_dispatch(["on"], 54)[0] == "rejected"
    rep.write_text('{"result": "PASS"}')
    json.dump({"result": {"G2": "FAIL"}, "ok": False}, open(os.path.join(env.c.CDIR, "verify_state.json"), "w"))
    assert env.c.v_dispatch(["on"], 55)[0] == "ok" and not os.path.exists(os.path.join(env.c.CDIR, "dispatch_frozen"))


def test_pulse_fields_pass_the_leak_check(env, monkeypatch):
    env.c.setup_state_write("4", "models and venvs", "running")
    open(os.path.join(env.c.CDIR, "dispatch_frozen"), "w").write("x\n")
    sp = env.root / "status.json"; sp.write_text("{}")
    monkeypatch.setattr(env.c, "queue_put", lambda *a, **k: True)
    p = env.c.pulse(str(sp), upload=False)
    assert p is not None and p["online"] is True and p["setup"] == {"step": "4", "state": "running"} and p["dispatch_frozen"] is True
    assert env.c.leak_check(p) == []


def test_restart_loops_waits_for_its_receipt(env, monkeypatch):
    calls = []
    monkeypatch.setattr(env.c.subprocess, "Popen", lambda args, **k: calls.append((args, k)))
    st, msg, _ = env.c.v_restart_loops([], 61)
    sh = calls[0][0][2]
    assert st == "accepted" and calls[0][1].get("start_new_session") is True
    assert "applied_seq" in sh and "-ge 61" in sh and sh.endswith("scripts/loops.sh restart both")
    assert sh.index("control[.]py apply") < sh.index("loops.sh restart")   # the receipt's upload finishes first (rehearsal 11-01)


def test_control_plane_without_torch(env):
    """design session ②: the status, the pulse and the commands' state import nothing of the training stack."""
    code = ("import sys; sys.modules['torch'] = None; sys.modules['vllm'] = None; sys.modules['verl'] = None\n"
            f"sys.path.insert(0, {HERE!r}); import control as c\n"
            "s = c.status(); assert 'setup_state' in s and 'loops' in s; print('OK')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=dict(os.environ))
    assert r.stdout.strip() == "OK", r.stderr[-500:]


def test_env_probe_writes_no_path_or_proxy_value(env, monkeypatch, tmp_path):
    import env_probe as E
    importlib.reload(E)
    monkeypatch.setattr(E, "http", lambda url, proxy: "200")
    monkeypatch.setenv("https_proxy", "http://user:secret@proxy.example:3128")
    monkeypatch.setenv("OLD_BASE_CANDS", f"{tmp_path}:/nonexistent/x")
    o = E.probe()
    blob = json.dumps(o)
    assert o["proxy_set"] is True and "secret" not in blob and "proxy.example" not in blob and str(tmp_path) not in blob
    assert o["old_base"] == {"candidates": 2, "found": 1, "with_venv": 0}


def test_prefetch_skips_paused_readings(tmp_path):
    """[v2.5.9z106] prefetch_pass: a paused #eval task's model is not fetched (Fable / ops-lead 11-01 17:19 UTC)."""
    src = open(os.path.join(HERE, "run_tasks.sh")).read()
    i = src.index("prefetch_pass() {"); j = src.index("  done; }", i) + len("  done; }")
    root = tmp_path / "root"; (root / "runs" / "EVAL-a").mkdir(parents=True); (root / "runs" / "EVAL-b").mkdir(parents=True)
    (root / "runs" / "EVAL-b" / ".paused").write_text("seq 0\n")
    q = tmp_path / "queue.txt"; q.write_text("#eval EVAL-a math500 Qwen/x\n#eval EVAL-b math500 Qwen/y\n")
    rec = tmp_path / "started"
    stub = (f"OPD_ROOT={root}; Q={q}; RUNBOOK={tmp_path}\n"
            "paused() { [ -f $OPD_ROOT/runs/$1/.paused ]; }; inputs_ok() { return 1; }; deps_of() { :; }\n"
            "ours_pargs() { :; }; ours_pgrep() { :; }\n"
            f"setsid() {{ echo \"$*\" | grep -o 'prefetch [A-Za-z0-9-]*' >> {rec}; }}\n")
    subprocess.run(["bash", "-c", stub + src[i:j] + "\nPREFETCH_PAR=3 prefetch_pass; sleep 0.2"], check=True, capture_output=True, text=True)
    assert open(rec).read().split() == ["prefetch", "EVAL-a"]


def test_commands_apply_within_a_long_pass():
    """[v2.5.9z106] the pass checks pull_due before each task (the H800's 40-min passes held its commands back)."""
    src = open(os.path.join(HERE, "run_tasks.sh")).read()
    loop = src[src.index('    for spec in $(for s in $rest;'):src.index('  done < <(stage_lines "$Q")')]
    assert loop.index("pull_due && { last_pull=$(date +%s); pull_and_control; }") < loop.index("if done_ $t || dead $t || skipped $t")
    r = subprocess.run(["bash", "-c", src[src.index("pull_due() {"):src.index("\n", src.index("pull_due() {"))] +
                        "\nlast_pull=$(( $(date +%s) - 61 )); pull_due && echo due; last_pull=$(date +%s); pull_due || echo wait; DRY=1; last_pull=0; pull_due || echo dry"],
                       capture_output=True, text=True)
    assert r.stdout.split() == ["due", "wait", "dry"], r.stdout + r.stderr


def test_liveness_within_a_long_pass():
    """[v2.5.9z106b] (H800 11-01: run_tasks restarted by the watchdog every ~11 min, the run requests never served) the pass writes
    the watchdog's liveness file before each task, ahead of the pull and the done check."""
    src = open(os.path.join(HERE, "run_tasks.sh")).read()
    loop = src[src.index('    for spec in $(for s in $rest;'):src.index('  done < <(stage_lines "$Q")')]
    i = loop.index("date -u +%s > $CDIR/loop_run_tasks")
    assert i < loop.index("pull_due && { last_pull=$(date +%s); pull_and_control; }") < loop.index("if done_ $t || dead $t || skipped $t")
    wd = open(os.path.join(HERE, "watchdog.sh")).read()
    assert "f=$CDIR/loop_$L" in wd                                      # the file the watchdog reads


def test_no_literal_backslash_line_ends():
    """[v2.5.9z106b] (since z69: readings_held.json's writer ended its line with a literal '\\\\', so python got no arguments, an
    IndexError every pass, and the next line ran as a command of its own) a shell line ends with one backslash, never two."""
    for name in ("run_tasks.sh", "heartbeat.sh", "watchdog.sh", "loops.sh", "upload_sync.sh", "task.sh"):
        for i, line in enumerate(open(os.path.join(HERE, name)).read().splitlines(), 1):
            assert not line.endswith("\\\\"), f"{name}:{i}"

