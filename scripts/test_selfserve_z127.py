"""[v2.5.9z127] (Fable 11-02, next_machine_selfserve_20261102 2 items 3 and 6) verify reports, it does not stop: `dispatch on` needs the
main venv and the canary on this node, verify's four training checks hold only the rows that say `#needs verify <task>`, and a
connect-first overnight.sh records verify and goes on; `run FETCH-<name>` / `run FETCH-all` let the frozen round pull the queue's
#fetch lines on the control venv (no GPU, never task.sh). Stubs only: no network, no GPU, nothing started but stub scripts.
Run: python -m pytest -q scripts/test_selfserve_z127.py"""
import ast
import importlib
import json
import os
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control_rules as R  # noqa: E402

QUEUE = """version: 2
s1 open A B
#fetch Qwen/Qwen3-8B
#fetch dataset:someone/somedata
#fetch result:runs/E10e-P5tail/global_step_50/actor/huggingface
#fetch result:runs/E10b-P5/global_step_150/actor/huggingface
#fetch junk/notamodel
#needs verify A
"""
NAMES = {"Qwen3-8B", "someone__somedata", "result_E10e-P5tail_s50", "result_E10b-P5_s150"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    rb, root = tmp_path / "rb", tmp_path / "root"
    (rb / "scripts").mkdir(parents=True); (rb / "tasks").mkdir(); (root / "logs").mkdir(parents=True)
    (rb / "tasks" / "queue.txt").write_text(QUEUE)
    monkeypatch.setenv("OPD_ROOT", str(root)); monkeypatch.setenv("RUNBOOK", str(rb)); monkeypatch.setenv("OPD_NODE", "testnode")
    monkeypatch.setenv("CONTROL_DIRECT", "0")
    import control
    importlib.reload(control)
    os.makedirs(control.CDIR, exist_ok=True)
    monkeypatch.setattr(control, "queue_put", lambda src, dst, **k: True)
    return types.SimpleNamespace(c=control, rb=rb, root=root, tmp=tmp_path)


def _bash_fn(src, name):
    i = src.index(f"\n{name}() {{") + 1; depth = 0
    for j in range(i, len(src)):
        depth += {"{": 1, "}": -1}.get(src[j], 0)
        if depth == 0 and src[j] == "}":
            return src[i:j + 1]
    raise AssertionError(name)


def _fetch_harness(env, body):
    """run_tasks.sh's fetch_pass (and `body`) with a stub python: fetch_model.py records its --task, remote.py repo public answers."""
    bin_ = env.tmp / "bin"; bin_.mkdir(exist_ok=True)
    (bin_ / "python").write_text('#!/bin/bash\ncase "$1" in *fetch_model.py) for a; do [ "$prev" = --task ] && echo "$a" >> "$CALLS"; prev=$a; done;;\n'
                                 '  *remote.py) echo someone/pub;; *) exec python3 "$@";; esac\n')
    (bin_ / "setsid").write_text('#!/bin/sh\nexec "$@"\n')
    for f in ("python", "setsid"):
        os.chmod(bin_ / f, 0o755)
    src = open(os.path.join(HERE, "run_tasks.sh")).read()
    fns = "\n".join(_bash_fn(src, f) for f in ("cenv", "fetch_pass"))
    script = (f"set -u\n{fns}\nours_pargs() {{ :; }}\nours_pgrep() {{ return 0; }}\nfetch_progress() {{ :; }}\n{body}\n"
              'for i in $(seq 30); do [ -s "$CALLS" ] && break; sleep 0.1; done; sleep 0.3\n')
    calls = env.tmp / "calls"; calls.write_text("")
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, OPD_ROOT=str(env.root), RUNBOOK=str(env.rb), CDIR=str(env.root / "control"),
                                CKPT_ROOT=str(env.tmp / "ck"), Q=str(env.rb / "tasks" / "queue.txt"), CALLS=str(calls),
                                PREFETCH_PAR="9", PATH=f"{bin_}:{os.environ['PATH']}"))
    return {x.removeprefix("FETCH-") for x in calls.read_text().split()}, r


def test_fetch_names_are_fetch_pass_s_names(env):
    assert set(env.c.fetch_names()) == NAMES
    got, r = _fetch_harness(env, "fetch_pass")
    assert got == NAMES, r.stdout + r.stderr   # the same names in python and in bash, junk skipped by both


def test_run_fetch_allows_named_pulls(env):
    st, msg, _ = env.c.v_run(["FETCH-result_E10e-P5tail_s50"], 7)
    assert st == "ok" and open(os.path.join(env.c.CDIR, "fetch_allow")).read() == "result_E10e-P5tail_s50\n"
    assert env.c.v_run(["FETCH-result_E10e-P5tail_s50"], 8)[0] == "ok"
    assert open(os.path.join(env.c.CDIR, "fetch_allow")).read().count("result_E10e") == 1   # once
    st, msg, _ = env.c.v_run(["FETCH-nosuch"], 9)
    assert st == "rejected" and "no #fetch line pulled as FETCH-nosuch" in msg and "Qwen3-8B" in msg
    assert env.c.v_run(["FETCH-all"], 10)[0] == "ok"
    assert set(open(os.path.join(env.c.CDIR, "fetch_allow")).read().split()) == NAMES
    R.parse({"cmd": "run FETCH-result_E10e-P5tail_s50", "reason": "t"}); R.parse({"cmd": "run FETCH-all", "reason": "t"})
    assert not os.path.exists(os.path.join(env.c.CDIR, "requests"))   # no GPU request queued


def test_allow_pulls_only_the_allowed(env):
    open(os.path.join(env.c.CDIR, "fetch_allow"), "w").write("Qwen3-8B\nresult_E10b-P5_s150\n")
    got, r = _fetch_harness(env, "fetch_pass allow")
    assert got == {"Qwen3-8B", "result_E10b-P5_s150"}, r.stdout + r.stderr


def test_the_frozen_round_pulls_before_the_main_venv(env):
    """a bootstrap node: venv-ctl, no venv_ready, dispatch frozen. With control/fetch_allow the frozen round pulls those names, before
    it returns for the missing main venv; without it, nothing."""
    (env.root / "venv-ctl" / "bin").mkdir(parents=True); (env.root / "venv-ctl" / "bin" / "activate").write_text("")
    src = open(os.path.join(HERE, "run_tasks.sh")).read()
    fr = _bash_fn(src, "frozen_round")
    assert fr.index("fetch_pass allow") < fr.index("[ ! -f $CDIR/venv_ready ] && return")
    body = (fr + "\nensure_watchdog() { :; }\nensure_uploader() { :; }\nensure_loop() { :; }\nverify_busy() { return 0; }\n"
            "last_pull=0\nPULL=0\nRUNBOOK=$OPD_ROOT\nmkdir -p $OPD_ROOT/scripts; echo 'exit 0' > $OPD_ROOT/scripts/loops.sh\nfrozen_round")
    got, _ = _fetch_harness(env, body)
    assert got == set()
    open(os.path.join(env.c.CDIR, "fetch_allow"), "w").write("result_E10e-P5tail_s50\n")
    got, r = _fetch_harness(env, body)
    assert got == {"result_E10e-P5tail_s50"}, r.stdout + r.stderr


def test_fetch_runs_on_the_control_venv():
    """fetch_model.py and what it imports need the standard library and huggingface_hub alone (requirements.ctl.txt), hf_transfer only
    inside a try (it is off by default)."""
    std = {"argparse", "fcntl", "hashlib", "json", "os", "re", "signal", "subprocess", "sys", "threading", "time", "urllib", "shutil",
           "tempfile", "glob", "socket", "random", "errno", "math", "io", "collections", "functools", "contextlib", "pathlib", "datetime"}
    ours = {"par_fetch", "par_range", "huggingface_hub", "hf_transfer", "remote"}   # remote.py: the control plane, on venv-ctl since z106
    for f in ("fetch_model.py", "par_fetch.py", "par_range.py"):
        mods = set()
        for n in ast.walk(ast.parse(open(os.path.join(HERE, f)).read())):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                mods.add(n.module.split(".")[0])
        assert mods <= std | ours, (f, mods - std - ours)
    assert "huggingface-hub==" in open(os.path.join(os.path.dirname(HERE), "requirements.ctl.txt")).read()
    src = open(os.path.join(HERE, "fetch_model.py")).read()
    assert "    try:" in src[src.index("import hf_transfer") - 200:src.index("import hf_transfer")]


def test_verify_state_ok_is_the_four_training_checks(env):
    log = env.root / "logs" / "verify.log"
    log.write_text("G2 PASS (21 passed)\nVLLM PASS\nSMOKE PASS (300s)\nUPLOAD_CHECK PASS (10s)\nCONTROL FAIL\n")
    env.c.verify_state(str(log), 0)
    v = json.load(open(os.path.join(env.c.CDIR, "verify_state.json")))
    assert v["ok"] is True and v["result"]["CONTROL"] == "FAIL"   # the control self-test is reported, not required
    log.write_text("G2 PASS\nVLLM PASS\nSMOKE FAIL\nUPLOAD_CHECK PASS\n")
    env.c.verify_state(str(log), 3)
    assert json.load(open(os.path.join(env.c.CDIR, "verify_state.json")))["ok"] is False
    assert env.c.status()["verify"]["ok"] is False


def test_needs_verify_holds_only_its_rows(env):
    src = open(os.path.join(HERE, "run_tasks.sh")).read()
    fn = _bash_fn(src, "needs_verify_ok")
    assert "needs_verify_ok $t || continue" in src
    run = lambda t: subprocess.run(["bash", "-c", f"{fn}\nneeds_verify_ok {t}"], env=dict(
        os.environ, Q=str(env.rb / "tasks" / "queue.txt"), CDIR=str(env.root / "control"))).returncode
    assert run("A") == 1 and run("B") == 0 and run("AB") == 0   # no verify yet: A waits, B (and a longer name) do not
    json.dump({"ok": False}, open(os.path.join(env.c.CDIR, "verify_state.json"), "w")); assert run("A") == 1
    json.dump({"ok": True}, open(os.path.join(env.c.CDIR, "verify_state.json"), "w")); assert run("A") == 0


@pytest.mark.parametrize("ctl,want", [(True, "STEP 7 verify reported, not a gate"), (False, "FAIL 7 verify")])
def test_overnight_records_verify_and_goes_on_when_connect_first(tmp_path, ctl, want):
    src = open(os.path.join(HERE, "overnight.sh")).read()
    a = src.index("[ -d $OPD_ROOT/venv-ctl ] && python3 scripts/control.py verify_state"); b = src.index("\n", src.index("else fail 7", a))
    root = tmp_path / "root"; (root / "logs").mkdir(parents=True); (tmp_path / "scripts").mkdir()
    (root / "logs" / "verify.log").write_text("G2 PASS\nVLLM FAIL\nSMOKE PASS\n")
    (tmp_path / "scripts" / "control.py").write_text("import sys; open(sys.argv[0] + '.called', 'w').write(' '.join(sys.argv[1:]))\n")
    if ctl:
        (root / "venv-ctl").mkdir()
    script = 'step() { echo "STEP $1 $2"; }\nfail() { echo "FAIL $1 $2"; exit 1; }\n' + src[a:b] + "\n"
    r = subprocess.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True, env=dict(os.environ, OPD_ROOT=str(root)))
    assert want in r.stdout, r.stdout + r.stderr
    called = tmp_path / "scripts" / "control.py.called"
    assert called.exists() == ctl and (not ctl or called.read_text().startswith("verify_state "))


def test_cold_start_health_reports_verify_on_a_bootstrap_node():
    src = open(os.path.join(HERE, "cold_start.sh")).read()
    i = src.index('V2="grep -q'); block = src[i:src.index("\n", src.index("else chk \"verify.sh", i))]
    assert "if [ $BOOT = 1 ]" in block and "INFO verify.sh" in block and "bad=1" not in block.split("else chk")[0]


def test_prefetch_par_is_read_on_every_pass_v259z131(env):
    """[v2.5.9z131] (ops-lead 11-03: 3 pulls x 32 connections > the node's ~64) `setenv PREFETCH_PAR=1` reaches the running run_tasks
    loop: fetch_pass reads it from control.env on every pass (the loop sources control.env only when it starts)."""
    open(os.path.join(env.c.CDIR, "fetch_allow"), "w").write("Qwen3-8B\nresult_E10b-P5_s150\n")
    (env.root / "control.env").write_text("export PREFETCH_PAR=1\n")
    got, r = _fetch_harness(env, "fetch_pass allow")
    assert len(got) == 1, (got, r.stdout + r.stderr)
    (env.root / "control.env").write_text("export PREFETCH_PAR=2\n")
    got, r = _fetch_harness(env, "fetch_pass allow")
    assert len(got) == 2


def test_dispatch_on_waits_for_venv_fa_under_vfa_v259z131(env):
    """[v2.5.9z131] (ops-lead 11-03) the 80g profile's VENV=fa: `dispatch on` also needs venv-fa to import flash_attn."""
    (env.root / "venv-ctl").mkdir(); open(os.path.join(env.c.CDIR, "venv_ready"), "w").write("x\n")
    (env.root / "runs" / "PROBE-TAIL").mkdir(parents=True); (env.root / "runs" / "PROBE-TAIL" / "report.json").write_text('{"result": "PASS"}')
    (env.root / "control.env").write_text("export VENV=fa\n")
    st, msg, _ = env.c.v_dispatch(["on"], 70)
    assert st == "rejected" and "venv-fa" in msg
    (env.root / "venv-fa" / "bin").mkdir(parents=True); py = env.root / "venv-fa" / "bin" / "python"
    py.write_text("#!/bin/sh\nexit 1\n"); py.chmod(0o755)
    assert env.c.v_dispatch(["on"], 71)[0] == "rejected"   # there, but no flash_attn
    py.write_text("#!/bin/sh\nexit 0\n")
    assert env.c.v_dispatch(["on"], 72)[0] == "ok"
    (env.root / "control.env").write_text("export VENV=\n"); py.write_text("#!/bin/sh\nexit 1\n")
    open(os.path.join(env.c.CDIR, "dispatch_frozen"), "w").write("x\n")
    assert env.c.v_dispatch(["on"], 73)[0] == "ok"   # VENV empty: the main venv alone


def test_dispatch_on_a_hung_flash_attn_import_is_a_refusal_v259z132(env, monkeypatch):
    """[v2.5.9z132] (ops-lead 11-03 05:5x UTC) venv-fa's import check times out (120 s): rejected, no exception out of apply."""
    import subprocess as sp
    (env.root / "venv-ctl").mkdir(); open(os.path.join(env.c.CDIR, "venv_ready"), "w").write("x\n")
    (env.root / "runs" / "PROBE-TAIL").mkdir(parents=True); (env.root / "runs" / "PROBE-TAIL" / "report.json").write_text('{"result": "PASS"}')
    (env.root / "control.env").write_text("export VENV=fa\n")
    (env.root / "venv-fa" / "bin").mkdir(parents=True); (env.root / "venv-fa" / "bin" / "python").write_text("#!/bin/sh\nexit 0\n")
    def hang(*a, **k):
        raise sp.TimeoutExpired(a[0], k.get("timeout"))
    monkeypatch.setattr(env.c.subprocess, "run", hang)
    st, msg, _ = env.c.v_dispatch(["on"], 80)
    assert st == "rejected" and "venv-fa" in msg
