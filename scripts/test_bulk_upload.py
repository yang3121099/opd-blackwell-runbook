"""[v2.5.9z124] (ops-lead 11-02 16:14 UTC, Fable 16:2x UTC) the control plane before bulk uploads: bulk uploads one at a time, small ones
never wait for them; in a 429 backoff the node status is tried once per STATUS_TRY_S and nothing else is; control_blind_since set /
cleared; bulk uploads wait while the node is blind; the lock is released after a failure; publish-redacted's put_public_unchanged
respects the backoff. No network: remote._put and remote.hf are stubbed."""
import importlib
import json
import os
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


@pytest.fixture
def R(tmp_path, monkeypatch):
    monkeypatch.setenv("OPD_ROOT", str(tmp_path))
    for k, v in (("BULK_MIN_MB", "0.001"), ("BULK_POLL_S", "0.05"), ("STATUS_TRY_S", "300"), ("BULK_YIELD_MAX_S", "30"), ("BULK_LOCK_WAIT_S", "30")):
        monkeypatch.setenv(k, v)
    (tmp_path / "control").mkdir()
    import remote
    remote = importlib.reload(remote)
    remote.calls = []
    monkeypatch.setattr(remote, "public_paused", lambda: False)
    return remote


def _file(tmp_path, name, n):
    p = tmp_path / name; p.write_bytes(b"x" * n); return str(p)


def _slow_put(R, secs=0.3, fail=None):
    def put(src, dst, kind=None):
        R.calls.append(("start", dst, time.time()))
        if fail:
            raise fail
        time.sleep(secs)
        R.calls.append(("end", dst, time.time()))
    return put


def test_two_bulk_uploads_run_one_after_the_other(R, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_put", _slow_put(R))
    a, b = _file(tmp_path, "a.jsonl", 5000), _file(tmp_path, "b.jsonl", 5000)
    ts = [threading.Thread(target=R.put, args=(a, "runs/T/a.jsonl")), threading.Thread(target=R.put, args=(b, "runs/T/b.jsonl"))]
    [t.start() for t in ts]; [t.join() for t in ts]
    ev = sorted(R.calls, key=lambda c: c[2])
    assert [e[0] for e in ev] == ["start", "end", "start", "end"]                 # never two at once
    assert not os.path.exists(tmp_path / "control" / "control_blind_since")


def test_a_small_upload_does_not_wait_for_the_lock(R, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_put", _slow_put(R, 0.0))
    small = _file(tmp_path, "s.json", 10)                                          # below BULK_MIN_MB (1000 bytes here)
    with R._BulkLock(10 ** 9):                                                     # a bulk upload holds the lock
        t0 = time.time(); R.put(small, "status/node.json")
        assert time.time() - t0 < 1.0 and R.calls[0][1] == "status/node.json"


def test_in_a_backoff_only_the_status_is_tried_once_per_window(R, tmp_path, monkeypatch):
    class E429(Exception):
        pass
    e = E429("429 Too Many Requests")
    monkeypatch.setattr(R, "_put", _slow_put(R, 0.0, fail=e))
    json.dump({"level": 1, "wait_s": 60, "until": time.time() + 60}, open(tmp_path / "control" / "hf_backoff.json", "w"))
    st = _file(tmp_path, "st.json", 10)
    with pytest.raises(E429):
        R.put(st, "status/node.json")                                              # the one try of this window reaches HF
    assert len(R.calls) == 1 and os.path.exists(tmp_path / "control" / "control_blind_since")
    with pytest.raises(R.Backoff):
        R.put(st, "status/node.json")                                              # within STATUS_TRY_S: no request
    with pytest.raises(R.Backoff):
        R.put(st, "pulse/node.json")                                               # the pulse never gets the try
    with pytest.raises(R.Backoff):
        R.put(st, "status/node/replies/7.json")                                    # nor a receipt
    assert len(R.calls) == 1
    assert json.load(open(tmp_path / "control" / "hf_backoff.json"))["level"] == 2  # the failed try counted as a 429


def test_a_status_put_that_succeeds_ends_the_backoff_and_the_blindness(R, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_put", _slow_put(R, 0.0))
    json.dump({"level": 3, "wait_s": 240, "until": time.time() + 240}, open(tmp_path / "control" / "hf_backoff.json", "w"))
    (tmp_path / "control" / "control_blind_since").write_text("t\n")
    R.put(_file(tmp_path, "st.json", 10), "status/node.json")
    assert not os.path.exists(tmp_path / "control" / "hf_backoff.json") and not os.path.exists(tmp_path / "control" / "control_blind_since")


def test_a_bulk_upload_waits_while_the_node_is_blind(R, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_put", _slow_put(R, 0.0)); monkeypatch.setenv("BULK_YIELD_MAX_S", "0.4")
    (tmp_path / "control" / "control_blind_since").write_text("t\n")
    t0 = time.time(); R.put(_file(tmp_path, "big.jsonl", 5000), "runs/T/big.jsonl")
    assert time.time() - t0 >= 0.4 and R.calls                                     # it waited the cap, then went up
    R.calls.clear(); (tmp_path / "control" / "control_blind_since").unlink()
    t0 = time.time(); R.put(_file(tmp_path, "big2.jsonl", 5000), "runs/T/big2.jsonl")
    assert time.time() - t0 < 0.3                                                  # not blind: no wait


def test_the_lock_is_released_after_a_failure_and_a_long_wait_times_out(R, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "_put", _slow_put(R, 0.0, fail=RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        R.put(_file(tmp_path, "big.jsonl", 5000), "runs/T/big.jsonl")
    monkeypatch.setattr(R, "_put", _slow_put(R, 0.0))
    R.put(_file(tmp_path, "big2.jsonl", 5000), "runs/T/big2.jsonl")                 # the failed one did not keep the lock
    monkeypatch.setenv("BULK_LOCK_WAIT_S", "0.2")
    with R._BulkLock(10 ** 9):
        with pytest.raises(TimeoutError):
            R.put(_file(tmp_path, "big3.jsonl", 5000), "runs/T/big3.jsonl")


def test_put_public_unchanged_respects_the_backoff(R, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "hf", lambda kind: (_ for _ in ()).throw(AssertionError("no request in a backoff")))
    json.dump({"level": 1, "wait_s": 60, "until": time.time() + 60}, open(tmp_path / "control" / "hf_backoff.json", "w"))
    with pytest.raises(R.Backoff):
        R.put_public_unchanged([(_file(tmp_path, "r.jsonl", 10), "runs/T/redacted/r.jsonl")], "m")


def test_a_bulk_commit_takes_one_lane(R, monkeypatch):
    seen = []

    class Api:
        def create_commit(self, repo_id=None, repo_type=None, operations=None, commit_message=None, num_threads=5):
            seen.append(num_threads)
    R._commit(Api(), 10 ** 9, repo_id="r", operations=[], commit_message="m")       # bulk: BULK_THREADS (1)
    R._commit(Api(), 10, repo_id="r", operations=[], commit_message="m")            # small: the hub's default
    monkeypatch.setenv("BULK_THREADS", "2"); R._commit(Api(), 10 ** 9, repo_id="r", operations=[], commit_message="m")
    assert seen == [1, 5, 2]

    class OldApi:                                                                   # a hub without num_threads: not passed
        def create_commit(self, repo_id=None, repo_type=None, operations=None, commit_message=None):
            seen.append("old")
    R._commit(OldApi(), 10 ** 9, repo_id="r", operations=[], commit_message="m")
    assert seen[-1] == "old"


def _transfers(tmp_path, n_recent, n_old=0, n_failed=0):
    now = time.time()
    rows = [{"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 60)), "dir": "up", "dst": "runs/T", "ok": True}] * n_recent
    rows += [{"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 4000)), "dir": "up", "dst": "runs/T", "ok": True}] * n_old
    rows += [{"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 60)), "dir": "up", "dst": "runs/T", "ok": False}] * n_failed
    rows += [{"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 60)), "dir": "up", "dst": "runs/T", "ok": True, "deferred": "x"}] * 5
    rows += [{"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now - 60)), "dir": "down", "dst": "runs/T", "ok": True}] * 5
    open(tmp_path / "control" / "transfers.jsonl", "w").write("first line is skipped as possibly partial\n" + "\n".join(json.dumps(r) for r in rows) + "\n")


def test_the_hourly_commit_budget_v259z124b(R, tmp_path, monkeypatch):   # [v2.5.9z124b] Fable's 9th item
    monkeypatch.setattr(R, "_put", _slow_put(R, 0.0))
    _transfers(tmp_path, 79, n_old=50, n_failed=30)
    assert R.commits_last_hour() == 79                                              # old, failed, deferred, downloads not counted
    R.put(_file(tmp_path, "small.json", 10), "runs/T/small.json")                   # 79 < 80: data still goes
    _transfers(tmp_path, 79)
    R.put(_file(tmp_path, "big.jsonl", 5000), "runs/T/big.jsonl")                   # 100 - 79 = 21 > 20: a bulk upload goes
    _transfers(tmp_path, 80)
    with pytest.raises(R.Budget):
        R.put(_file(tmp_path, "small2.json", 10), "runs/T/small2.json")            # at 80: control plane only
    R.put(_file(tmp_path, "st.json", 10), "status/node.json")                      # the control plane is never refused
    R.put(_file(tmp_path, "pl.json", 10), "pulse/node.json")
    assert issubclass(R.Budget, R.Backoff)                                           # the callers' retry paths take it
    monkeypatch.setenv("HF_COMMIT_BUDGET", "90"); _transfers(tmp_path, 70)
    with pytest.raises(R.Budget):
        R.put(_file(tmp_path, "big2.jsonl", 5000), "runs/T/big2.jsonl")            # 90 - 70 = 20: no room for a bulk upload
    monkeypatch.setattr(R, "hf", lambda kind: (_ for _ in ()).throw(AssertionError("no request over budget")))
    with pytest.raises(R.Budget):
        R.put_public_unchanged([(_file(tmp_path, "r.jsonl", 10), "runs/T/redacted/r.jsonl")], "m")   # a publish needs the headroom
