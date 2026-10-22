"""[v2.5.9z62] (ops-lead A1 10-23) weights_replay.py: the held weights to the PUBLIC repository under their old path, the weight
files only, one at a time, final and read steps first, a size-bounded timeout, nothing during the 429 backoff or while a
control-plane item waits, struck only after get_paths_info shows the sizes, the held copy never deleted, one event per batch.
The upload (remote.py put ... public) and the repository are stubbed; no network."""
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
import remote as RM  # noqa: E402
import weights_replay as W  # noqa: E402

DST = "runs/E1/global_step_50/actor/huggingface"


@pytest.fixture
def node(monkeypatch):
    d = tempfile.mkdtemp(); ck = tempfile.mkdtemp(); rb = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "control")); os.makedirs(os.path.join(rb, "tasks"))
    open(os.path.join(d, "hf_repos.txt"), "w").write("public=me/pub\nprivate=me/priv\n")
    open(os.path.join(rb, "tasks", "queue.txt"), "w").write("#eval EVAL-x a result:runs/E2/global_step_25/actor/huggingface math500 4 0.6\n")
    monkeypatch.setenv("OPD_ROOT", d); monkeypatch.setenv("CKPT_ROOT", ck); monkeypatch.setenv("RUNBOOK", rb)
    for k in ("WEIGHTS_REPLAY", "WEIGHTS_REPLAY_MBPS", "WEIGHTS_REPLAY_MAX_S", "WEIGHTS_REPLAY_BATCH", "REPLAY_OUTBOX_MAX_AGE", "WEIGHTS_REPLAY_COOL_S"):
        monkeypatch.delenv(k, raising=False)
    calls, events, remote = [], [], {}
    def run(cmd, **k):
        if cmd[1].endswith("control.py"):
            events.append(cmd[-1]); return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        stage = cmd[3]; calls.append({"cmd": cmd[2:], "files": sorted(os.listdir(stage)), "timeout": k.get("timeout"),
                                      "ino": {f: os.stat(os.path.join(stage, f)).st_ino for f in os.listdir(stage)}})
        if node.mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, k.get("timeout"))
        if node.mode == "fail":
            return types.SimpleNamespace(returncode=1, stdout="", stderr="hf put failed: HfHubHTTPError")
        for f in os.listdir(stage):
            remote[f"{cmd[4]}/{f}"] = os.path.getsize(os.path.join(stage, f)) + (1 if node.mode == "short" and f.endswith(".safetensors") else 0)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    node.mode = "ok"
    monkeypatch.setattr(W.subprocess, "run", run)
    api = types.SimpleNamespace(get_paths_info=lambda repo, paths: [types.SimpleNamespace(path=p, size=remote[p]) for p in paths if p in remote])
    monkeypatch.setattr(RM, "hf", lambda kind="public": (api, "me/pub" if kind == "public" else "me/priv"))
    return types.SimpleNamespace(root=d, ck=ck, calls=calls, events=events, remote=remote, set=lambda m: setattr(node, "mode", m))


def hold(n, dst, mb=3, extra=True, utc="2026-10-22T12:00:00Z"):
    h = os.path.join(n.ck, "weights_pending", dst); os.makedirs(h, exist_ok=True)
    with open(os.path.join(h, "model.safetensors"), "wb") as f:
        f.truncate(mb * 10 ** 6)
    open(os.path.join(h, "config.json"), "w").write("{}"); open(os.path.join(h, "tokenizer.json"), "w").write("{}")
    if extra:
        open(os.path.join(h, "train.log"), "w").write("not a weight file")
    with open(os.path.join(n.root, "control", "weights_pending.txt"), "a") as f:
        f.write(f"{utc}\t{dst}\t{h}\t{mb * 10 ** 6 + 4}\n")
    return h


def lines(n, name="weights_pending.txt"):
    p = os.path.join(n.root, "control", name)
    return open(p).read().splitlines() if os.path.exists(p) else []


def test_public_destination_weight_files_only_struck_after_verify_held_copy_kept(node):
    h = hold(node, DST)
    r = W.once()
    assert r.startswith(f"REPLAY_OK {DST}")
    c = node.calls[0]
    assert c["cmd"][0] == "put" and c["cmd"][2:] == [DST, "public"]                      # the public side, the path it always had
    assert c["files"] == ["config.json", "model.safetensors", "tokenizer.json"]            # no log
    assert lines(node) == [] and lines(node, "weights_replayed.txt")[0].split("\t")[1] == DST
    assert os.path.getsize(os.path.join(h, "model.safetensors")) == 3 * 10 ** 6 and os.path.exists(os.path.join(h, "train.log"))   # not deleted
    assert not [x for x in os.listdir(os.path.dirname(h)) if x.startswith(".stage_")]     # the staging links gone
    assert W.once() == "REPLAY_IDLE"


def test_timeout_by_size_and_not_struck(node, monkeypatch):
    monkeypatch.setenv("WEIGHTS_REPLAY_COOL_S", "0")   # no cooling down here
    hold(node, DST, mb=30); node.set("timeout")
    r = W.once()
    assert r.startswith(f"REPLAY_FAIL {DST} timed out after") and lines(node) and not lines(node, "weights_replayed.txt")
    assert node.calls[0]["timeout"] == int(30.000004 / 0.5 * 1.5 + 600)                   # 0.5 MB/s x 1.5 + 600 s
    open(os.path.join(node.root, "control.env"), "w").write("export WEIGHTS_REPLAY_MAX_S=100\n")
    W.once(); assert node.calls[1]["timeout"] == 100                                       # the cap


def test_429_backoff_tries_nothing(node):
    hold(node, DST); RM.note_429()
    assert W.once().startswith("REPLAY_WAIT the 429 backoff") and node.calls == [] and lines(node)
    RM.note_ok()
    assert W.once().startswith("REPLAY_OK")


def test_verify_fails_or_upload_fails_not_struck(node, monkeypatch):
    monkeypatch.setenv("WEIGHTS_REPLAY_COOL_S", "0")
    hold(node, DST); node.set("short")
    r = W.once()
    assert "not verified: model.safetensors size" in r and len(lines(node)) == 1 and not lines(node, "weights_replayed.txt")
    node.set("fail")
    assert W.once().startswith(f"REPLAY_FAIL {DST} hf put failed") and len(lines(node)) == 1


def test_order_final_and_read_steps_first_older_first(node):
    hold(node, "runs/E2/global_step_10/actor/huggingface", utc="2026-10-22T10:00:00Z")
    hold(node, "runs/E2/global_step_20/actor/huggingface", utc="2026-10-22T11:00:00Z")
    hold(node, "runs/E2/global_step_25/actor/huggingface", utc="2026-10-22T13:00:00Z")   # a reading uses it
    hold(node, "runs/E2/global_step_50/actor/huggingface", utc="2026-10-22T14:00:00Z")   # the final step
    for _ in range(4):
        W.once()
    assert [c["cmd"][2] for c in node.calls] == ["runs/E2/global_step_25/actor/huggingface", "runs/E2/global_step_50/actor/huggingface",
                                                 "runs/E2/global_step_10/actor/huggingface", "runs/E2/global_step_20/actor/huggingface"]


def test_the_control_plane_first(node):
    hold(node, DST); d = os.path.join(node.root, "control", "outbox"); os.makedirs(d)
    json.dump({"src": "/x", "dst": "status/n.json"}, open(os.path.join(d, f"{int((time.time() - 700) * 1e9):020d}__k.json"), "w"))
    assert W.once().startswith("REPLAY_WAIT a control-plane item waits") and node.calls == []
    os.remove(os.path.join(d, os.listdir(d)[0]))
    json.dump({"src": "/x", "dst": "runs/E1/a.csv"}, open(os.path.join(d, f"{int((time.time() - 9000) * 1e9):020d}__k.json"), "w"))
    assert W.once().startswith("REPLAY_OK")                                                # an old result item does not hold it


def test_one_event_per_batch(node):
    for i in range(7):
        hold(node, f"runs/E1/global_step_{i + 1}/actor/huggingface", mb=1, utc=f"2026-10-22T1{i}:00:00Z")
    for _ in range(7):
        W.once()
    assert len(node.events) == 2 and "5 item(s)" in node.events[0] and "2 left" in node.events[0] and "2 item(s)" in node.events[1] and "0 left" in node.events[1]


def test_off_bad_lines_and_the_append_lock(node, monkeypatch):
    open(os.path.join(node.root, "control.env"), "w").write("export WEIGHTS_REPLAY=0\n")
    hold(node, DST); assert W.once() == "REPLAY_OFF"
    open(os.path.join(node.root, "control.env"), "w").write("")
    with open(os.path.join(node.root, "control", "weights_pending.txt"), "a") as f:
        f.write("garbage line\n")
    assert W.once().startswith("REPLAY_OK") and lines(node) == ["garbage line"]            # a line it cannot read is left as it is
    src = open(os.path.join(HERE, "remote.py")).read()
    i = src.index("def defer_weight")
    assert "weights_pending.lock" in src[i:src.index("def put(", i)] and "fcntl.flock" in src[i:src.index("def put(", i)]
    assert "ensure_loop weights_replay" in open(os.path.join(HERE, "run_tasks.sh")).read()


def test_a_failed_item_cools_down_and_the_next_one_goes(node):   # [ops-lead's A1 condition] no head-of-line blocking
    a = "runs/E1/global_step_50/actor/huggingface"; b = "runs/E3/global_step_50/actor/huggingface"
    hold(node, a, utc="2026-10-22T10:00:00Z"); hold(node, b, utc="2026-10-22T11:00:00Z")
    node.set("fail"); assert W.once().startswith(f"REPLAY_FAIL {a}")
    node.set("ok"); assert W.once().startswith(f"REPLAY_OK {b}")                                 # the next round: the second item
    assert W.once() == "REPLAY_WAIT all 1 item(s) cooling down after failures" and [c["cmd"][2] for c in node.calls] == [a, b]
    st = json.load(open(os.path.join(node.root, "control", "weights_replay.json")))
    assert st["fails"][a]["n"] == 1 and b not in st["fails"]
    def again(n_expected_wait):
        st = json.load(open(os.path.join(node.root, "control", "weights_replay.json")))
        assert abs(W.cooling(W.state(), a, st["fails"][a]["last"]) - n_expected_wait) < 1
        st["fails"][a]["last"] -= n_expected_wait + 1; json.dump(st, open(os.path.join(node.root, "control", "weights_replay.json"), "w"))
    node.set("fail")
    again(1800); W.once(); again(3600); W.once()                                                # 30 min, then 60 min
    assert [e for e in node.events if "3 failed tries" in e] and len([e for e in node.events if "3 failed tries" in e]) == 1
    again(7200); W.once(); again(14400); W.once(); again(21600)                                 # ... at most 6 h
    assert len([e for e in node.events if "3 failed tries" in e]) == 1                          # one stuck event
    node.set("ok"); assert W.once().startswith(f"REPLAY_OK {a}")
    assert a not in json.load(open(os.path.join(node.root, "control", "weights_replay.json")))["fails"]


def test_hold_outside_weights_pending_refused(node):   # [ops-lead's A1 condition] the path guard
    out = tempfile.mkdtemp(); open(os.path.join(out, "model.safetensors"), "w").write("x")
    with open(os.path.join(node.root, "control", "weights_pending.txt"), "a") as f:
        f.write(f"2026-10-22T09:00:00Z\t{DST}\t{out}\t1\n")
    r = W.once()
    assert "the hold path is not under $CKPT_ROOT/weights_pending" in r and node.calls == []
    link = os.path.join(node.ck, "weights_pending", "runs", "E9"); os.makedirs(os.path.dirname(link), exist_ok=True); os.symlink(out, link)
    with open(os.path.join(node.root, "control", "weights_pending.txt"), "w") as f:
        f.write(f"2026-10-22T09:00:00Z\truns/E9/global_step_1/actor/huggingface\t{link}\t1\n")
    assert "not under" in W.once() and node.calls == []                                          # a link out of it: refused too


def test_weights_linked_other_files_copied(node):   # [ops-lead's A1 condition] nothing the upload does reaches the hold
    h = hold(node, DST); W.once()
    ino = node.calls[0]["ino"]
    assert ino["model.safetensors"] == os.stat(os.path.join(h, "model.safetensors")).st_ino
    assert ino["config.json"] != os.stat(os.path.join(h, "config.json")).st_ino and ino["tokenizer.json"] != os.stat(os.path.join(h, "tokenizer.json")).st_ino

