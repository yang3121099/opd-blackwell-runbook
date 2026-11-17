"""[v3.0] (Fable 11-17) one queue file per node: the verb `queue <name>|none` writes control/queue_name (env.sh turns it into
QUEUE_FILE = tasks/queue_<name>.txt), refuses a name the pulled runbook has no file for, and restarts the loops; the repository's
tasks/queue.txt opens no stage, so a node without a queue of its own starts nothing. Stubs only: no network, nothing started.
Run: python -m pytest -q scripts/test_queue_verb.py"""
import importlib
import os
import re
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import control_rules as R  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    rb, root = tmp_path / "rb", tmp_path / "root"
    (rb / "scripts").mkdir(parents=True); (rb / "tasks").mkdir(); (root / "logs").mkdir(parents=True)
    (rb / "tasks" / "queue.txt").write_text("version: 2\n")
    (rb / "tasks" / "queue_n1.txt").write_text("version: 2\ns1 open PROBE-TAIL\n")
    monkeypatch.setenv("OPD_ROOT", str(root)); monkeypatch.setenv("RUNBOOK", str(rb)); monkeypatch.setenv("OPD_NODE", "testnode")
    monkeypatch.setenv("CONTROL_DIRECT", "0")
    import control
    importlib.reload(control)
    os.makedirs(control.CDIR, exist_ok=True)
    calls = []
    monkeypatch.setattr(control, "v_restart_loops", lambda a, seq, flags=None, entry=None: calls.append(seq) or ("accepted", "", {}))
    return types.SimpleNamespace(c=control, rb=rb, root=root, calls=calls)


def test_rules_accept_and_refuse():
    for cmd in ("queue n1", "queue none", "queue borrow8_a"):
        assert R.parse({"seq": 1, "cmd": cmd, "reason": "r"})[0] == "queue"
    for cmd in ("queue", "queue a b", "queue ../x", "queue a-b", "queue " + "x" * 33, "queue a.txt"):
        with pytest.raises(R.Reject):
            R.parse({"seq": 1, "cmd": cmd, "reason": "r"})
    assert R.VERBS["queue"] == (1, 1, R.H24) and tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 116)
    assert f"rules {R.RULES_VERSION}" in open(os.path.join(RB, "control", "CHANGES.md")).read()


def test_queue_writes_the_name_and_restarts_the_loops(env):
    qn = os.path.join(env.c.CDIR, "queue_name")
    st, msg, _ = env.c.v_queue(["nosuch"], 7)
    assert st == "rejected" and "queue_nosuch.txt" in msg and not os.path.exists(qn) and env.calls == []
    st, msg, _ = env.c.v_queue(["n1"], 8)
    assert st == "accepted" and open(qn).read() == "n1\n" and env.calls == [8] and "queue_n1.txt" in msg
    st, msg, _ = env.c.v_queue(["none"], 9)
    assert st == "accepted" and not os.path.exists(qn) and env.calls == [8, 9] and "opens nothing" in msg
    assert env.c.v_queue(["none"], 10)[0] == "accepted"   # nothing to remove: still fine


def test_env_sh_reads_the_name(env):
    """the real env.sh: QUEUE_FILE follows control/queue_name, and is unset without it"""
    def qf():
        r = subprocess.run(["bash", "-c", f'source "{RB}/scripts/env.sh" > /dev/null 2>&1; echo "Q=${{QUEUE_FILE:-}}"'], capture_output=True, text=True,
                           env=dict(os.environ, OPD_ROOT=str(env.root), RUNBOOK=str(env.rb), HOME=str(env.root), CKPT_ROOT=str(env.root / "ck")))
        return r.stdout.strip().splitlines()[-1]
    assert qf() == "Q="
    env.c.v_queue(["n1"], 11)
    assert qf() == f"Q={env.rb}/tasks/queue_n1.txt"


def test_the_default_queue_opens_nothing():
    """tasks/queue.txt of this repository: no stage line at all (only comments and the version)"""
    lines = [l for l in open(os.path.join(RB, "tasks", "queue.txt")).read().splitlines() if l.strip() and not l.startswith("#")]
    assert lines == ["version: 2"], lines
    import queue_profiles as QP   # every tasks/queue_<name>.txt a node could be given is a generated profile (row sources: tasks/rows_*)
    for q in os.listdir(os.path.join(RB, "tasks")):
        assert not q.startswith("queue") or q == "queue.txt" or q[6:-4] in QP.PROFILES, f"tasks/{q}: not a queue a node can be given"


def test_the_borrowed_h200_queues():
    """[v3.0] h2a: the canary and Q-107's gate, the two 3K arms behind the gate on 4 GPUs each, one reading task per registered point
    behind its arm (reading + the 16K forced remedy), then the zero-training readings as an `auto` stage; h2b: nothing h2a trains."""
    import queue_profiles as QP
    a = open(os.path.join(RB, "tasks", "queue_h2a.txt")).read(); b = open(os.path.join(RB, "tasks", "queue_h2b.txt")).read()
    st = {l.split()[0]: l.split()[1:] for l in a.splitlines() if l and not l.startswith("#") and len(l.split()) > 2}
    assert list(st) == ["h2g", "q107", "q107r", "q107z", "q107z6"] and [v[0] for v in st.values()] == ["open", "open", "open", "auto", "auto"]
    assert st["q107"][1:] == ["E12b-P0think-3k:PROBE-E12B-P0THINK-3K", "E12b-P0think-3k-s44:PROBE-E12B-P0THINK-3K"]
    for t in ("PROBE-E12B-P0THINK-3K", "E12b-P0think-3k", "E12b-P0think-3k-s44"):
        assert f"#gpus {t} 4\n" in a
    for task, arm, step in (("EVAL-q107-3k-s50", "E12b-P0think-3k", 50), ("EVAL-q107-3k-s44-s50", "E12b-P0think-3k-s44", 50),
                            ("EVAL-q107-3k-s25", "E12b-P0think-3k", 25), ("EVAL-q107-3k-s44-s25", "E12b-P0think-3k-s44", 25)):
        assert f"{task}:{arm}" in st["q107r"]
        m = f"result:runs/{arm}/global_step_{step}/actor/huggingface"
        ev = [l for l in a.splitlines() if l.startswith(f"#eval {task} ")]; fo = [l for l in a.splitlines() if l.startswith(f"#forced {task} ")]
        assert len(ev) == 1 and ev[0].split()[3:] == [m, "math500", "4", "0.6", "16384", "think"], ev
        assert len(fo) == 1 and fo[0].split()[2:] == [ev[0].split()[2], m, "think", "budgets=16384"], fo
    tb = {x.split(":")[0] for l in b.splitlines() if l and not l.startswith("#") and len(l.split()) > 2 for x in l.split()[2:]}
    assert tb == {"PROBE-TAIL", "EVAL-q107-4k-s50", "EVAL-q107-4k-s25", "EVAL-q107-6k-s25"} and not any(t.startswith("E1") for t in tb)
    for q in (a, b):   # the direct 6K arm's s25 (owed): its weights are fetched ahead as the 4K arm's
        assert "#fetch result:runs/E12b-P0think-6k/global_step_25/actor/huggingface\n" in q and "#eval EVAL-q107-6k-s25 " in q
    assert QP.main(["--check"]) == 0
