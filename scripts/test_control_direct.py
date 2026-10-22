"""[v2.5.9z53] (Fable 10-22 14:1x UTC, ops-lead A1) the status, the private pulse and the receipts go up directly (never through
the outbox), bounded by CONTROL_PUT_TIMEOUT, with z52's throttles; one event per failing episode; the logs verb reads only the
file's tail; the diagnostic tails; PROBE-TERM4's guard for a sample without a final box."""
import json
import os
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402
import control_rules as R  # noqa: E402


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp()
    open(os.path.join(d, "hf_repos.txt"), "w").write("public=me/pub\nprivate=me/priv\n")
    os.makedirs(os.path.join(d, "control")); os.makedirs(os.path.join(d, "logs"))
    monkeypatch.setenv("OPD_ROOT", d)
    for k in ("CONTROL_DIRECT", "CONTROL_PUT_TIMEOUT", "DIAG_TAILS_S", "REMOTE_PUT_TIMEOUT", "STATUS_PUT_S", "PULSE_PUT_S", "OUTBOX"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(C, "ROOT", d); monkeypatch.setattr(C, "CDIR", os.path.join(d, "control"))
    open(C._p("loop_uploader"), "w").write(str(int(time.time())))   # a live uploader: before z53 these went into the outbox
    return d


def src(root, name):
    p = os.path.join(root, name); open(p, "w").write("{}"); return p


def outbox():
    d = C._outbox()
    return [json.load(open(os.path.join(d, n)))["dst"] for n in sorted(os.listdir(d)) if n.endswith(".json")] if os.path.isdir(d) else []


@pytest.fixture
def puts(monkeypatch):
    calls, ev = [], []
    def put(s, d):
        calls.append((d, os.environ.get("REMOTE_PUT_TIMEOUT"))); return not d.endswith("fail.json")
    monkeypatch.setattr(C, "remote_put", put)
    monkeypatch.setattr(C, "event", lambda task, kind, detail="": ev.append((kind, detail)))
    return calls, ev


def test_status_pulse_receipts_direct_with_60s_and_throttled(root, puts):
    calls, ev = puts
    for d in ("status/n.json", "pulse/n.json", "status/n/replies/482.json", "status/n/replies/482"):
        assert C.queue_put(src(root, d.replace("/", "_")), d)
    assert [c[0] for c in calls] == ["status/n.json", "pulse/n.json", "status/n/replies/482.json", "status/n/replies/482"]
    assert {c[1] for c in calls} == {"60"} and os.environ.get("REMOTE_PUT_TIMEOUT") is None and outbox() == []
    C.queue_put(src(root, "s2"), "status/n.json"); C.queue_put(src(root, "r2"), "status/n/replies/483.json")
    assert [c[0] for c in calls[4:]] == ["status/n/replies/483.json"]          # the status every 5 min (z52); receipts never held
    C.queue_put(src(root, "e"), "status/n/events/e.json"); C.queue_put(src(root, "p"), "status/n/procs.json")
    assert outbox() == ["status/n/events/e.json", "status/n/procs.json"]        # events and procs: the outbox as before


def test_timeout_from_control_env_and_switch_off(root, puts):
    calls, _ = puts
    open(os.path.join(root, "control.env"), "w").write("export CONTROL_PUT_TIMEOUT=45\n")
    C.queue_put(src(root, "r"), "status/n/replies/1.json"); assert calls[-1] == ("status/n/replies/1.json", "45")
    open(os.path.join(root, "control.env"), "w").write("export CONTROL_DIRECT=0\n")
    C.queue_put(src(root, "r"), "status/n/replies/2.json"); assert outbox() == ["status/n/replies/2.json"] and len(calls) == 1
    os.environ["REMOTE_PUT_TIMEOUT"] = "180"
    try:
        open(os.path.join(root, "control.env"), "w").write(""); C.queue_put(src(root, "r"), "status/n/replies/3.json")
        assert calls[-1][1] == "60" and os.environ["REMOTE_PUT_TIMEOUT"] == "180"   # restored after the call
    finally:
        del os.environ["REMOTE_PUT_TIMEOUT"]


def test_one_event_per_failing_episode(root, puts):
    calls, ev = puts
    for i in range(3):
        assert C.queue_put(src(root, "r"), f"status/n/replies/{i}fail.json") is False
    assert [k for k, _ in ev] == ["control_direct_failed"] and open(os.path.join(root, "logs", "control_timeouts.log")).read().count("direct put failed") == 3
    C.queue_put(src(root, "r"), "status/n/replies/9.json")
    assert [k for k, _ in ev] == ["control_direct_failed", "control_direct_recovered"] and "after 3 failed" in ev[-1][1]
    C.queue_put(src(root, "r"), "status/n/replies/10.json"); assert len(ev) == 2


def test_logs_verb_reads_only_the_tail(root):
    p = os.path.join(root, "logs", "heartbeat.log")
    with open(p, "w") as f:
        for i in range(400000):
            f.write(f"HEARTBEAT line {i}\n")
    assert os.path.getsize(p) > 6 << 20
    t = C._tail_lines(p, 60)
    assert t.splitlines() == [f"HEARTBEAT line {i}" for i in range(399940, 400000)]
    assert C._tail_lines(p, 10, max_bytes=50).splitlines()[-1] == "HEARTBEAT line 399999"   # fewer lines when the window is short
    src = open(os.path.join(HERE, "control.py")).read()
    assert 'redact(_tail_lines(src, n))' in src and '_read(src).splitlines(True)[-n:]' not in src


def test_diag_tails_every_interval_direct(root, puts, monkeypatch):
    calls, _ = puts
    open(os.path.join(root, "logs", "uploader.log"), "w").write("".join(f"pass {i}\n" for i in range(500)))
    monkeypatch.setenv("CONTROL_DIRECT", "0"); C.queue_put(src(root, "st"), "status/n.json")   # one item waiting in the outbox
    assert C.diag_tails() is True and calls[-1] == (f"logs/{C.node_id()}/diag", "60")
    d = C._p("diag")
    assert open(os.path.join(d, "uploader.log.tail")).read().splitlines() == [f"pass {i}" for i in range(300, 500)]
    s = json.load(open(os.path.join(d, "control_state.json")))
    assert s["outbox_items"] == 1 and s["outbox_head"][0]["dst"] == "status/n.json" and s["outbox_head"][0]["src_exists"]
    assert C.diag_tails() is None and len(calls) == 1                          # within DIAG_TAILS_S (1800 s)
    assert C.diag_tails(force=True) is True and len(calls) == 2
    monkeypatch.setenv("DIAG_TAILS_S", "0"); open(C._p("diag_tails.t"), "w").write("0")
    assert C.diag_tails() is None                                              # 0: off


def test_setenv_keys():
    for k, v in (("CONTROL_DIRECT", "0"), ("CONTROL_PUT_TIMEOUT", "60"), ("DIAG_TAILS_S", "1800")):
        R.parse({"cmd": f"setenv {k}={v}", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 77)


def test_term4_guard_for_a_sample_without_a_final_box():   # torch-free: the source; the node runs PROBE-TERM4 again
    s = open(os.path.join(HERE, "..", "code", "analysis", "term_probe.py")).read()
    i = s.index('t4 = term4_positions(pieces, ids) if a.term4 else None')
    assert 'cnt["term4_no_box"] += 1' in s[i:i + 400] and "if t4 is not None:" in s[i:i + 600]
    assert s.index("if t4 is not None:", i) < s.index("t4[\"k_fb\"] + 1", i)
