"""[v2.5.9z58] control/state.json unreadable (10-22 ~11:17 UTC: one object plus extra data) took down every reader - the status
builder, apply, diag_tails - and state() could not rewrite it. Now: a tolerant read recovers the complete objects (state.json: the
newest apply, never backwards), saves the bad file, rewrites it clean and sends one event per file; writes are atomic; status(),
alerts(), diag and apply() work on a corrupt file, and apply replays nothing."""
import glob
import json
import os
import re
import sys
import tempfile

import pytest

from test_control import env  # noqa: F401  (shared fixture)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402

BAD = '{"alerts": ["stalled:EVAL-x"], "pending": 0, "last_apply_utc": "2026-10-22T11:16:00Z"}ply_utc": "2026-10-22T11:17:00Z"}'


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "control")); os.makedirs(os.path.join(d, "logs"))
    monkeypatch.setattr(C, "ROOT", d); monkeypatch.setattr(C, "CDIR", os.path.join(d, "control")); monkeypatch.setenv("OPD_ROOT", d)
    ev = []
    monkeypatch.setattr(C, "event", lambda task, kind, detail="": ev.append((kind, detail)))
    return d, ev


def test_the_10_22_file_is_read_repaired_and_kept(root):
    d, ev = root
    open(C._p("state.json"), "w").write(BAD)
    with pytest.raises(ValueError):
        json.loads(BAD)                                                   # what every reader hit
    s = C._state_read()
    assert s == {"alerts": ["stalled:EVAL-x"], "pending": 0, "last_apply_utc": "2026-10-22T11:16:00Z"}
    assert json.load(open(C._p("state.json"))) == s                       # rewritten clean
    bad = glob.glob(C._p("state.json.bad.*")); assert len(bad) == 1 and open(bad[0]).read() == BAD
    assert [k for k, _ in ev] == ["state_json_repaired"]
    C._state_read(); assert len(ev) == 1                                  # clean now: no second event


def test_garbage_and_empty(root):
    d, ev = root
    open(C._p("state.json"), "w").write("\x00\x00garbage")
    assert C._state_read() == {} and json.load(open(C._p("state.json"))) == {}
    open(C._p("state.json"), "w").write("")
    assert C._state_read() == {} and len(ev) == 1
    for i in range(7):
        open(C._p("state.json"), "w").write("[1]x"); C._state_read()
    assert len(glob.glob(C._p("state.json.bad.*"))) <= 5                    # at most 5 copies kept


def test_state_writes_atomically_and_works_on_a_corrupt_file(root):
    d, ev = root
    open(C._p("state.json"), "w").write(BAD)
    C.state(pending=3)
    s = json.load(open(C._p("state.json")))
    assert s["pending"] == 3 and s["alerts"] == ["stalled:EVAL-x"]
    assert not [f for f in os.listdir(C._p()) if f.endswith(".tmp")]


def test_status_alerts_and_diag_on_a_corrupt_file(root, monkeypatch):
    d, ev = root
    open(C._p("state.json"), "w").write(BAD)
    st = C.status()                                                       # the heartbeat's status builder calls this
    assert st["pending"] == 0 and st["alerts"] == ["stalled:EVAL-x"]
    open(C._p("state.json"), "w").write(BAD)
    sp = os.path.join(d, "st.json"); json.dump({"disk_alert": True, "disk_free_gb": 1}, open(sp, "w"))
    monkeypatch.setattr(C, "reserve_low", lambda: None); monkeypatch.setattr(C, "running_pids", lambda: [])
    assert "disk_alert" in C.alerts(sp)
    open(C._p("state.json"), "w").write(BAD)
    monkeypatch.setattr(C, "_direct_put", lambda s, dd: True)
    assert C.diag_tails(force=True) is True


def test_no_reader_left_on_the_old_pattern():
    src = open(os.path.join(HERE, "control.py")).read()
    assert 'json.loads(_read(_p("state.json")' not in src and src.count("_state_read()") >= 8
    assert not re.search(r'json\.loads\(_read\(_p\("[a-z_]+\.json"\)', src)          # every control-plane JSON read is tolerant
    assert not re.search(r'json\.dump\([^\n]*open\(_p\("(state|setenv_times|assets_cache|reserve)\.json"\), "w"\)', src)
    hb = open(os.path.join(HERE, "heartbeat.sh")).read()
    assert '"held": jload(' in hb and '"our_du": jload(' in hb and '"stack": jload(' in hb
    rt = open(os.path.join(HERE, "run_tasks.sh")).read()
    assert "last_pull.json.tmp && mv -f" in rt and "held.json.tmp && mv -f" in rt


def test_two_whole_objects_the_newer_apply_wins_never_backwards(root):   # [ops-lead's A1 condition]
    d, ev = root
    old = {"pending": 2, "last_apply_utc": "2026-10-22T11:10:00Z", "rejected_entries": ["a"], "alerts": []}
    new = {"pending": 0, "last_apply_utc": "2026-10-22T11:16:00Z", "rejected_entries": ["b"], "alerts": ["x"]}
    for raw in (json.dumps(new) + json.dumps(old), json.dumps(old) + "\n" + json.dumps(new) + 'tail": 1}'):
        open(C._p("state.json"), "w").write(raw)
        s = C._state_read()
        assert s["last_apply_utc"] == "2026-10-22T11:16:00Z" and s["pending"] == 0 and s["rejected_entries"] == ["a", "b"]
    open(C._p("applied_seq"), "w").write("487")
    open(C._p("state.json"), "w").write(json.dumps(dict(old, applied_seq=480)) + json.dumps(dict(new, applied_seq=481)))
    assert C._state_read()["applied_seq"] == 487                          # never below control/applied_seq


def test_a_nested_object_in_a_torn_tail_is_not_taken(root):
    d, ev = root
    open(C._p("state.json"), "w").write('{"pending": 1, "last_apply_utc": "T1"}ly_utc": "T9", "x": {"last_apply_utc": "T99"}}')
    assert C._state_read() == {"pending": 1, "last_apply_utc": "T1"}


def test_other_files_repaired_once_each(root):
    d, ev = root
    open(C._p("reserve.json"), "w").write('{"low_since": 5, "ready": 1}{"ready": 2}junk')
    open(C._p("setenv_times.json"), "w").write('{"DR": 1}}')
    assert C._jread("reserve.json", {}) == {"low_since": 5, "ready": 2} and C._jread("setenv_times.json", {}) == {"DR": 1}
    assert sorted(x.split(":")[0] for k, x in ev) == ["reserve.json", "setenv_times.json"]
    C._jread("reserve.json", {}); assert len(ev) == 2                     # clean after the repair
    assert json.load(open(C._p("reserve.json"))) == {"low_since": 5, "ready": 2}
    assert C._jread("missing.json", {"a": 1}) == {"a": 1}


def test_apply_replays_nothing_with_a_corrupt_state(env):   # the commands are guarded by control/applied_seq, not by state.json
    from test_control_v25 import write, iso
    open(os.path.join(env.c.CDIR, "applied_seq"), "w").write("1")
    open(os.path.join(env.c.CDIR, "state.json"), "w").write(BAD)
    write(env, [{"seq": 1, "cmd": "pause T", "reason": "r", "utc": iso(0)}, {"seq": 2, "cmd": "pause U", "reason": "r", "utc": iso(0)}])
    env.c.apply()
    assert open(os.path.join(env.c.CDIR, "applied_seq")).read().strip() == "2"
    assert not os.path.exists(os.path.join(env.c.CDIR, "replies", "1.json")) and os.path.exists(os.path.join(env.c.CDIR, "replies", "2.json"))
    json.load(open(os.path.join(env.c.CDIR, "state.json")))               # clean again
