"""[v2.5.9z124c] (ops-lead 11-02 17:19 UTC on notes/z124c_design_20261102.md) the control plane's private items in one commit per status
round, on CONTROL_BATCH=1: receipts / events / procs go to the control spool (one uploader per file), the status round carries them
(caps, procs only when due), a failed batch keeps them, two non-429 failures fall back to the status alone until it succeeds, a 429
does not count; CONTROL_BATCH=0 keeps the old paths. control.remote_put is stubbed: no network, no subprocess."""
import importlib
import json
import os
import sys
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def C(tmp_path, monkeypatch):
    root, rb = tmp_path / "root", tmp_path / "rb"
    root.mkdir(); (rb / "scripts").mkdir(parents=True)
    monkeypatch.setenv("OPD_ROOT", str(root)); monkeypatch.setenv("RUNBOOK", str(rb)); monkeypatch.setenv("OPD_NODE", "testnode")
    monkeypatch.setenv("HF_REPO", "someone/results"); monkeypatch.setenv("UPLOAD_BACKEND", "hf"); monkeypatch.setenv("OUTBOX", "0")
    monkeypatch.setenv("STATUS_PUT_S", "0"); monkeypatch.setenv("PROCS_PUT_S", "0"); monkeypatch.setenv("CONTROL_BATCH", "1")
    sys.path.insert(0, HERE)
    import control
    control = importlib.reload(control)
    os.makedirs(control.CDIR, exist_ok=True)
    calls, result = [], {"ok": True}

    def fake_put(src, dst, kind=None):
        calls.append(src if isinstance(src, list) else [[src, dst]])
        return result["ok"]
    monkeypatch.setattr(control, "remote_put", fake_put)
    monkeypatch.setattr(control, "_backoff_left", lambda: 0)
    return types.SimpleNamespace(c=control, calls=calls, result=result, root=root, monkeypatch=monkeypatch)


def _status(C):
    p = os.path.join(C.root, "status_now.json"); open(p, "w").write("{}")
    return C.c.queue_put(p, "status/testnode.json")


def _procs(C):
    p = C.c._p("procs.json"); open(p, "w").write('{"rows": []}')
    return C.c.queue_put(p, "status/testnode/procs.json")


def _dsts(batch):
    return [d for _, d in batch]


def test_items_wait_in_the_spool_and_go_up_with_the_status(C):
    C.c.receipt(5, {"cmd": "resume X", "reason": "t"}, "ok", "done")
    C.c.event("node", "something", "detail")
    _procs(C)
    assert C.calls == []                                                          # nothing went up on its own
    assert len(os.listdir(C.c._p("ctrl_spool"))) == 3
    assert _status(C) is True
    assert len(C.calls) == 1                                                      # one commit
    d = _dsts(C.calls[0])
    assert d[0] == "status/testnode.json" and "status/testnode/replies/5.json" in d and "status/testnode/procs.json" in d
    assert any(x.startswith("status/testnode/events/") for x in d)
    assert C.c._spool_items() == []                                              # sent: the spool is empty
    _status(C)
    assert _dsts(C.calls[1]) == ["status/testnode.json"]                          # each receipt went up once


def test_procs_newest_only_and_only_when_due(C):
    _procs(C); _procs(C)
    assert len([n for n in os.listdir(C.c._p("ctrl_spool")) if n.startswith("procs")]) == 1
    C.monkeypatch.setenv("PROCS_PUT_S", "600"); C.c._stamp([f"procs:status/testnode/procs.json"])
    _status(C)
    assert "status/testnode/procs.json" not in _dsts(C.calls[0])                   # not due: it waits
    assert any(it["dst"].endswith("procs.json") for _, it in C.c._spool_items())


def test_caps_keep_the_rest_for_the_next_round(C):
    C.monkeypatch.setenv("CTRL_BATCH_FILES", "3")
    for s in range(1, 6):
        C.c.receipt(s, {"cmd": "x", "reason": "t"}, "ok", "d")
    _status(C)
    assert len(C.calls[0]) == 3 and C.calls[0][0][1] == "status/testnode.json"
    assert len(C.c._spool_items()) == 3
    _status(C); assert len(C.calls[1]) == 3; _status(C); assert len(C.calls[2]) == 2


def test_two_failed_batches_fall_back_to_the_status_alone(C):
    C.c.receipt(7, {"cmd": "x", "reason": "t"}, "ok", "d")
    C.result["ok"] = False
    _status(C); _status(C)                                                        # two failed batches, not a 429
    assert all(len(b) >= 2 and "status/testnode/replies/7.json" in _dsts(b) for b in C.calls[:2])   # (failure events join the spool too)
    ev = [n for n in os.listdir(C.c._p("events")) if "control_batch_degraded" in n]
    assert len(ev) == 1
    _status(C)
    assert _dsts(C.calls[2]) == ["status/testnode.json"]                          # the status alone, the spool kept
    C.result["ok"] = True
    _status(C)                                                                    # the status alone succeeds: batches again
    assert _dsts(C.calls[3]) == ["status/testnode.json"]
    assert any("control_batch_recovered" in n for n in os.listdir(C.c._p("events")))
    _status(C)
    assert "status/testnode/replies/7.json" in _dsts(C.calls[4])                  # the kept receipt goes up now


def test_a_429_does_not_count_as_a_failed_batch(C):
    C.c.receipt(8, {"cmd": "x", "reason": "t"}, "ok", "d")
    C.result["ok"] = False; C.monkeypatch.setattr(C.c, "_backoff_left", lambda: 120)
    _status(C); _status(C); _status(C)
    assert all(len(b) >= 2 and "status/testnode/replies/8.json" in _dsts(b) for b in C.calls)   # still batches: a backoff is not a bad file
    assert not any("control_batch_degraded" in n for n in os.listdir(C.c._p("events")))


def test_switch_off_keeps_the_old_paths(C):
    C.monkeypatch.setenv("CONTROL_BATCH", "0")
    C.c.receipt(9, {"cmd": "x", "reason": "t"}, "ok", "d")
    assert C.calls == [[[C.c._p("replies", "9.json"), "status/testnode/replies/9.json"]]]   # a direct put as before
    assert not os.path.isdir(C.c._p("ctrl_spool")) or not os.listdir(C.c._p("ctrl_spool"))
    _status(C)
    assert C.calls[-1] == [[os.path.join(C.root, "status_now.json"), "status/testnode.json"]]


def test_heartbeat_reports_commits_last_hour():
    hb = open(os.path.join(HERE, "heartbeat.sh")).read()
    assert '"commits_last_hour": commits_last_hour(),' in hb and "def commits_last_hour():" in hb
