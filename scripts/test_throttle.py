"""[v2.5.9z52] (ops-lead / Fable 10-22 21:0x Beijing: the HF account answered 429 from 12:38 UTC) the 429 backoff's waits
(60 s doubling to 900 s, no request while it lasts, ended by a success), the outbox throttles (one try per class and
interval, the newest content), the events of a pass in one commit without losing any, the direct path's throttle, the pulse's
public mirror, fetch_model.py skipping our repositories during the backoff, and remote.py putmany as one commit."""
import json
import os
import subprocess
import sys
import tempfile
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402
import control_rules as R  # noqa: E402
import fetch_model as F  # noqa: E402
import remote as RM  # noqa: E402


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp()
    open(os.path.join(d, "hf_repos.txt"), "w").write("public=me/pub\nprivate=me/priv\n")
    os.makedirs(os.path.join(d, "control"))
    monkeypatch.setenv("OPD_ROOT", d); monkeypatch.delenv("PUBLIC_UPLOADS", raising=False)
    for k in ("STATUS_PUT_S", "PROCS_PUT_S", "EVENTS_PUT_S", "RESYNC_PUT_S", "PULSE_PUT_S", "PULSE_MIRROR_S", "PULSE_PUBLIC_MIRROR", "OUTBOX"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(C, "ROOT", d); monkeypatch.setattr(C, "CDIR", os.path.join(d, "control"))
    return d


@pytest.fixture(autouse=True)
def through_the_outbox(monkeypatch):   # [v2.5.9z53] these tests are the outbox's: the control plane's items through it as before
    monkeypatch.setenv("CONTROL_DIRECT", "0")
    monkeypatch.setenv("DIAG_TAILS_S", "0")   # and no diagnostic upload in the pass


def alive():
    open(C._p("loop_uploader"), "w").write(str(int(time.time())))


def items():
    d = C._outbox()
    return [json.load(open(os.path.join(d, n))) for n in sorted(os.listdir(d)) if n.endswith(".json") and not n.startswith(".")]


def src(root, name, body="{}"):
    p = os.path.join(root, name.replace("/", "_")); open(p, "w").write(body); return p


def age(key, s):   # the stamp of a class moved s seconds back
    st = C._stamps(); st[key] = st.get(key, time.time()) - s
    json.dump(st, open(C._p("put_stamps.json"), "w"))


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(C, "remote_put", lambda s, d, kind=None: (calls.append((s, d, kind)), True)[1])
    return calls


def test_backoff_waits_double_to_the_cap_and_a_success_ends_it(root):
    waits = [RM.note_429() for _ in range(7)]
    assert waits == [60, 120, 240, 480, 900, 900, 900]
    assert [l.split("\t")[1:] for l in open(os.path.join(root, "control", "hf_429.log")).read().splitlines()][:2] == [["1", "60"], ["2", "120"]]
    assert 899 < RM.backoff_left() <= 900
    RM.note_ok()
    assert RM.backoff_left() == 0 and RM.note_429() == 60   # the next 429 starts at 60 s again


def test_no_request_during_the_backoff_and_a_429_starts_it(root, monkeypatch):
    calls = []
    monkeypatch.setattr(RM, "_put", lambda s, d: calls.append(d))
    f = src(root, "a.json")
    RM.put(f, "status/n/replies/1.json"); assert calls == ["status/n/replies/1.json"]
    def r429(s, d):
        raise Exception("429 Client Error: Too Many Requests for url: https://huggingface.co/api/models/x")
    monkeypatch.setattr(RM, "_put", r429)
    with pytest.raises(Exception):
        RM.put(f, "status/n/replies/2.json")
    assert 59 < RM.backoff_left() <= 60
    monkeypatch.setattr(RM, "_put", lambda s, d: calls.append(d))
    with pytest.raises(RM.Backoff):
        RM.put(f, "status/n/replies/3.json")
    assert calls == ["status/n/replies/1.json"]          # nothing tried while it lasts
    e = Exception("x"); e.response = types.SimpleNamespace(status_code=429)
    assert RM.is_429(e) and not RM.is_429(Exception("500 Server Error"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "remote.py"), "put", f, "status/n.json"], capture_output=True, text=True,
                       env={k: v for k, v in dict(os.environ, OPD_ROOT=root).items() if k != "HF_TOKEN"})
    assert r.returncode == 1 and "put failed: Backoff" in r.stderr
    assert subprocess.run([sys.executable, os.path.join(HERE, "remote.py"), "backoff"], capture_output=True, text=True,
                          env=dict(os.environ, OPD_ROOT=root)).stdout.strip() in ("59", "58")


def test_a_held_weight_needs_no_request_during_the_backoff(root, monkeypatch):
    RM.note_429(); ck = tempfile.mkdtemp(); monkeypatch.setenv("CKPT_ROOT", ck)
    w = os.path.join(root, "w"); os.makedirs(w)
    with open(os.path.join(w, "model.safetensors"), "wb") as f:
        f.truncate(101 * 10 ** 6)
    RM.put(w, "runs/E1/global_step_50/actor/huggingface")   # private first: held on the checkpoint disk, no exception
    assert os.path.exists(os.path.join(ck, "weights_pending", "runs/E1/global_step_50/actor/huggingface", "model.safetensors"))


def test_status_procs_pulse_once_per_interval_with_the_newest_content(root, sent):
    alive()
    st, pr, pu, rc = src(root, "st"), src(root, "pr"), src(root, "pu"), src(root, "rc")
    for s, d in ((st, "status/n.json"), (pr, "status/n/procs.json"), (pu, "pulse/n.json"), (rc, "status/n/replies/1.json")):
        C.queue_put(s, d)
    assert C.ship_outbox(60) == (4, 0, 0) and len(sent) == 4
    for s, d in ((st, "status/n.json"), (pr, "status/n/procs.json"), (pu, "pulse/n.json"), (rc, "status/n/replies/2.json")):
        C.queue_put(s, d)
    C.queue_put(st, "status/n.json")                                       # coalesced: one item
    assert C.ship_outbox(60) == (1, 0, 0) and sent[-1][1] == "status/n/replies/2.json"   # a receipt is never held
    assert sorted(i["dst"] for i in items()) == ["pulse/n.json", "status/n.json", "status/n/procs.json"]
    age("status:status/n.json", 301); age("pulse:pulse/n.json", 301); age("procs:status/n/procs.json", 301)
    assert C.ship_outbox(60) == (2, 0, 0) and [i["dst"] for i in items()] == ["status/n/procs.json"]   # procs: 600 s
    age("procs:status/n/procs.json", 300)
    assert C.ship_outbox(60) == (1, 0, 0) and items() == []
    assert [d for _, d, _ in sent].count("status/n.json") == 2


def test_intervals_from_control_env_and_zero_turns_a_throttle_off(root, sent):
    alive(); st = src(root, "st")
    open(os.path.join(root, "control.env"), "w").write("export STATUS_PUT_S=0\n")
    for _ in range(3):
        C.queue_put(st, "status/n.json"); C.ship_outbox(60)
    assert len(sent) == 3
    R.parse({"cmd": "setenv STATUS_PUT_S=300", "reason": "t"}); R.parse({"cmd": "setenv RUNS_PUT_S=600", "reason": "t"})
    R.parse({"cmd": "setenv PULSE_PUBLIC_MIRROR=1", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 76)


def test_events_one_commit_none_lost(root, monkeypatch):
    alive(); calls = []
    fail = [False]
    def put(s, d, kind=None):
        calls.append(s if isinstance(s, list) else d); return not (isinstance(s, list) and fail[0])
    monkeypatch.setattr(C, "remote_put", put)
    for i in range(5):
        C.queue_put(src(root, f"e{i}"), f"status/n/events/e{i}.json")
    C.queue_put(src(root, "r1"), "status/n/replies/1.json")
    assert C.ship_outbox(60) == (6, 0, 0) and items() == []
    assert calls[0] == [[os.path.join(root, f"e{i}"), f"status/n/events/e{i}.json"] for i in range(5)] and calls[1] == "status/n/replies/1.json"
    for i in range(5, 8):
        C.queue_put(src(root, f"e{i}"), f"status/n/events/e{i}.json")
    assert C.ship_outbox(60) == (0, 0, 0) and len(items()) == 3 and len(calls) == 2   # within EVENTS_PUT_S: they wait
    age("events", 301); fail[0] = True
    assert C.ship_outbox(60) == (0, 0, 3) and [i["fails"] for i in items()] == [1, 1, 1]   # a failed commit keeps all three
    age("events", 301); fail[0] = False
    assert C.ship_outbox(60) == (3, 0, 0) and items() == [] and [d for _, d in calls[-1]] == [f"status/n/events/e{i}.json" for i in range(5, 8)]


def test_a_single_event_goes_as_an_ordinary_put(root, sent):
    alive(); C.queue_put(src(root, "e"), "status/n/events/e.json")
    assert C.ship_outbox(60) == (1, 0, 0) and sent == [(os.path.join(root, "e"), "status/n/events/e.json", None)]


def test_the_backoff_stops_the_pass(root, sent):
    alive(); C.queue_put(src(root, "rc"), "status/n/replies/1.json"); RM.note_429()
    assert C.ship_outbox(60) == (0, 0, 0) and sent == [] and len(items()) == 1
    RM.note_ok()
    assert C.ship_outbox(60) == (1, 0, 0)


def test_heartbeat_resync_of_replies_and_events_hourly(root, sent):
    alive(); d = os.path.join(root, "control", "replies"); os.makedirs(d, exist_ok=True)
    C.queue_put(d, "status/n/replies"); C.ship_outbox(60)
    C.queue_put(d, "status/n/replies"); C.ship_outbox(60)
    assert [x[1] for x in sent] == ["status/n/replies"] and len(items()) == 1
    age("resync:status/n/replies", 3601)
    assert C.ship_outbox(60) == (1, 0, 0)


def test_direct_path_throttles_latest_state_only(root, sent):   # no live uploader: queue_put uploads itself
    st = src(root, "st")
    assert C.queue_put(st, "status/n.json") and C.queue_put(st, "status/n.json")
    assert C.queue_put(src(root, "e1"), "status/n/events/e1.json") and C.queue_put(src(root, "e2"), "status/n/events/e2.json")
    assert [x[1] for x in sent] == ["status/n.json", "status/n/events/e1.json", "status/n/events/e2.json"]   # an event is never skipped


def test_pulse_public_mirror_every_30_minutes(root, monkeypatch, sent):
    alive(); monkeypatch.setattr(C, "_split_mode", lambda: True)
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="", returncode=0))
    st = os.path.join(root, "st.json"); open(st, "w").write('{"utc": "x", "tasks": {}}')
    C.pulse(st)
    assert [(i["dst"], i.get("kind")) for i in items()] == [(f"pulse/{C.node_id()}.json", None)]   # the mirror is off by default
    open(os.path.join(root, "control.env"), "w").write("export PULSE_PUBLIC_MIRROR=1\n")
    C.pulse(st)
    assert sorted((i["dst"], i.get("kind") or "") for i in items()) == [(f"pulse/{C.node_id()}.json", ""), (f"pulse/{C.node_id()}.json", "public")]
    C.ship_outbox(60)
    assert sorted(x[2] or "" for x in sent) == ["", "public"] and C._read(C._p("public_mirror_utc")).strip()
    C.pulse(st); age(f"pulse:pulse/{C.node_id()}.json", 301); C.ship_outbox(60)
    assert [x[2] for x in sent[2:]] == [None] and [i.get("kind") for i in items()] == ["public"]   # the mirror waits 1800 s
    open(os.path.join(root, "control.env"), "w").write("export PULSE_PUBLIC_MIRROR=1\nexport PUBLIC_UPLOADS=1\n")
    for n in os.listdir(C._outbox()):
        os.remove(os.path.join(C._outbox(), n))
    C.pulse(st)
    assert [i.get("kind") for i in items()] == [None]   # split again: the pulse itself goes to the public repository, no mirror


def test_fetch_skips_our_repositories_during_the_backoff(root, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(F, "locked_fetch", lambda repo, p, l: (calls.append(repo), {"ok": True})[1])
    monkeypatch.setattr(sys, "argv", ["fetch_model.py", "me/priv,me/pub", "runs/E1/global_step_1/actor/huggingface", tempfile.mkdtemp()])
    RM.note_429()
    with pytest.raises(SystemExit) as e:
        F.main()
    r = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert e.value.code == 1 and calls == [] and {x["error"] for x in r["errors"]} == {"Backoff429"}
    RM.note_ok()
    monkeypatch.setattr(F, "locked_fetch", lambda repo, p, l: (calls.append(repo), {"ok": False, "errors": [
        {"endpoint": "https://huggingface.co", "attempt": 1, "error": "HfHubHTTPError", "detail": "429 Client Error: Too Many Requests for url"}]})[1])
    with pytest.raises(SystemExit):
        F.main()
    assert calls == ["me/priv"] and RM.backoff_left() > 0   # the 429 starts the backoff; the public repository is not tried
    monkeypatch.setattr(sys, "argv", ["fetch_model.py", "Qwen/Qwen3-1.7B", "", tempfile.mkdtemp()]); calls.clear()
    monkeypatch.setattr(F, "locked_fetch", lambda repo, p, l: (calls.append(repo), {"ok": True})[1])
    with pytest.raises(SystemExit) as e:
        F.main()
    assert e.value.code == 0 and calls == ["Qwen/Qwen3-1.7B"]   # a model from elsewhere is not held


def test_putmany_is_one_commit(root, monkeypatch):
    commits = []
    api = types.SimpleNamespace(create_commit=lambda **k: commits.append(k))
    monkeypatch.setattr(RM, "hf", lambda kind="public": (api, "me/priv" if kind == "private" else "me/pub"))
    monkeypatch.setattr(RM, "private_ok", lambda a, r: True)
    pairs = [[src(root, f"e{i}"), f"status/n/events/e{i}.json"] for i in range(4)] + [[os.path.join(root, "gone"), "status/n/events/g.json"]]
    lf = os.path.join(root, "l.json"); json.dump(pairs, open(lf, "w"))
    RM.put_many(lf)
    assert len(commits) == 1 and commits[0]["repo_id"] == "me/priv" and [o.path_in_repo for o in commits[0]["operations"]] == [f"status/n/events/e{i}.json" for i in range(4)]
    RM.note_429()
    with pytest.raises(RM.Backoff):
        RM.put_many(lf)
    assert len(commits) == 1
