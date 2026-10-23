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
        stage = cmd[3]
        if os.path.isfile(stage):   # [v2.5.9z65] one file of a result sent on its own
            calls.append({"cmd": cmd[2:], "files": [os.path.basename(stage)], "timeout": k.get("timeout"), "ino": {}})
            if node.mode in ("timeout", "fail") or os.path.basename(stage) in node.failing:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="hf put failed: ConnectionError")
            remote[cmd[4]] = os.path.getsize(stage); return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        calls.append({"cmd": cmd[2:], "files": sorted(os.listdir(stage)), "timeout": k.get("timeout"),
                      "ino": {f: os.stat(os.path.join(stage, f)).st_ino for f in os.listdir(stage)}})
        if node.mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, k.get("timeout"))
        if node.mode == "fail":
            return types.SimpleNamespace(returncode=1, stdout="", stderr="hf put failed: HfHubHTTPError")
        for r, _, fs in os.walk(stage):
            for f in fs:
                rel = os.path.relpath(os.path.join(r, f), stage).replace(os.sep, "/")
                remote[f"{cmd[4]}/{rel}"] = os.path.getsize(os.path.join(r, f)) + (1 if node.mode == "short" and f.endswith(".safetensors") else 0)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    node.mode = "ok"; node.failing = set()
    monkeypatch.setattr(W.subprocess, "run", run)
    api = types.SimpleNamespace(get_paths_info=lambda repo, paths: [types.SimpleNamespace(path=p, size=remote[p]) for p in paths if p in remote])
    monkeypatch.setattr(RM, "hf", lambda kind="public": (api, "me/pub" if kind == "public" else "me/priv"))
    return types.SimpleNamespace(root=d, ck=ck, calls=calls, events=events, remote=remote, set=lambda m: setattr(node, "mode", m),
                                 failing=node.failing)


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


def held_result(n, dst, utc="2026-10-22T09:00:00Z", log_mb=6):
    h = os.path.join(n.ck, "weights_pending", dst); os.makedirs(os.path.join(h, "sub"), exist_ok=True)
    open(os.path.join(h, "x_math500_no_think_t0.6.jsonl"), "w").write("{}\n" * 10)
    open(os.path.join(h, "x.metrics.json"), "w").write("{}")
    open(os.path.join(h, "sub", "pair.json"), "w").write("{}")
    with open(os.path.join(h, "gen_eval.log"), "wb") as f:
        f.write(b"x" * (log_mb * 10 ** 6))
    with open(os.path.join(n.root, "control", "weights_pending.txt"), "a") as f:
        f.write(f"{utc}\t{dst}\t{h}\t123\n")
    return h


@pytest.fixture
def priv(node, monkeypatch):   # the private repository's view for verify_private
    api = types.SimpleNamespace(get_paths_info=lambda repo, paths: [types.SimpleNamespace(path=p, size=node.remote[p]) for p in paths if p in node.remote])
    monkeypatch.setattr(RM, "hf", lambda kind="public": (api, "me/priv" if kind == "private" else "me/pub"))
    return node


def test_held_results_go_private_first_and_while_paused(priv, monkeypatch):   # [v2.5.9z64] ops-lead's ruling 20:45 UTC
    n = priv
    open(os.path.join(n.root, "control.env"), "w").write("export WEIGHTS_REPLAY=0\n")   # the weights paused ...
    hold(n, DST, utc="2026-10-22T08:00:00Z")
    h = held_result(n, "runs/EVAL-budA-P5mix90")
    st = {"items": 0, "bytes": 0, "fails": {"runs/EVAL-budA-P5mix90": {"n": 3, "last": time.time(), "why": "not a weight destination"}}}
    json.dump(st, open(os.path.join(n.root, "control", "weights_replay.json"), "w"))     # z62's refusals do not hold it
    r = W.once()
    assert r.startswith("REPLAY_OK runs/EVAL-budA-P5mix90 (result)")                     # ... the result still goes
    c = n.calls[0]
    assert c["cmd"][2:] == ["runs/EVAL-budA-P5mix90"]                                     # no "public": the private repository
    assert sorted(c["files"]) == ["gen_eval.log.tail.gz", "sub", "x.metrics.json", "x_math500_no_think_t0.6.jsonl"]
    assert [l.split("\t")[1] for l in lines(n)] == [DST]                                  # struck; the weight stays
    assert any("results_replayed" in e or "runs/EVAL-budA-P5mix90" in e for e in n.events)
    assert os.path.getsize(os.path.join(h, "gen_eval.log")) == 6 * 10 ** 6                # the held copy untouched
    assert W.once() == "REPLAY_OFF"                                                        # then the paused weights


def test_log_tail_gz_at_most_5mb(priv):
    import gzip
    n = priv; held_result(n, "runs/EVAL-x", log_mb=7)
    W.once()
    p = [k for k in n.remote if k.endswith("gen_eval.log.tail.gz")][0]
    assert n.remote[p] < 5 * 10 ** 6 + 100   # gzip of 5 MB of "x" is tiny; never the 7 MB log


def test_result_not_verified_stays_listed(priv, monkeypatch):
    n = priv; held_result(n, "runs/EVAL-y"); n.set("short")
    real = W.verify_private
    monkeypatch.setattr(W, "verify_private", lambda dst, staged: (False, "x.metrics.json: None != 2"))
    assert "not verified" in W.once() and [l.split("\t")[1] for l in lines(n)] == ["runs/EVAL-y"]


def test_result_timeout_at_a_single_connection_rate(priv):   # [v2.5.9z64a] 268 MB at 0.4 MB/s timed out at ~77 % on 10-22
    n = priv; h = held_result(n, "runs/EVAL-z")
    with open(os.path.join(h, "x_math500_no_think_t0.6.jsonl"), "r+b") as f:
        f.truncate(268 * 10 ** 6)
    W.once()
    assert n.calls[0]["files"] == ["x_math500_no_think_t0.6.jsonl"]                        # [v2.5.9z65] over 100 MB: the big file alone
    assert n.calls[0]["timeout"] == int(min(21600, 268e6 / 1e6 / 0.08 * 1.5 + 600))
    assert n.calls[0]["timeout"] > 268 / 0.13 * 1.2                                        # the observed 0.09-0.13 MB/s fits


# ---- [v2.5.9z65] the results queue: a task's own directory, file by file, the marker, the 2 h event ----------------------------
def task_dir(n, task, big=(150, 60), small=True):
    d = os.path.join(n.root, "runs", task); os.makedirs(d, exist_ok=True)
    for i, mb in enumerate(big):
        with open(os.path.join(d, f"r{i}_math500_no_think_t0.6.jsonl"), "wb") as f:
            f.truncate(mb * 10 ** 6)
    if small:
        open(os.path.join(d, f"r0.metrics.json"), "w").write("{}"); open(os.path.join(d, "gen_eval.log"), "w").write("ok\n")
    return d


def test_pend_lists_the_task_dir_and_marks_it(priv):
    n = priv; d = task_dir(n, "EVAL-q", big=(1,))
    size = RM.pend_result(d, "runs/EVAL-q")
    assert [l.split("\t")[1:] for l in lines(n)] == [["runs/EVAL-q", os.path.realpath(d), str(size)]]
    assert open(os.path.join(d, ".upload_pending")).read().split()[1] == "runs/EVAL-q"
    RM.pend_result(d, "runs/EVAL-q")
    assert len(lines(n)) == 1                                                             # the same destination: replaced
    other = task_dir(n, "EVAL-r", big=(1,))
    for src, dst in ((other, "runs/EVAL-q"), (d, "runs/EVAL-q/sub"), (d, "ckpts/x"), ("/tmp", "runs/EVAL-q")):
        with pytest.raises(ValueError):
            RM.pend_result(src, dst)                                                      # only runs/<task> from its own directory


def test_split_file_by_file_largest_first_skipping_present(priv):
    n = priv; d = task_dir(n, "EVAL-s"); RM.pend_result(d, "runs/EVAL-s")
    n.remote["runs/EVAL-s/r1_math500_no_think_t0.6.jsonl"] = 60 * 10 ** 6                # already up from an earlier try
    r = W.once()
    assert r.startswith("REPLAY_OK runs/EVAL-s (result)"), r
    assert [c["files"] for c in n.calls] == [["r0_math500_no_think_t0.6.jsonl"], ["gen_eval.log", "r0.metrics.json"]]
    assert n.calls[0]["cmd"][2] == "runs/EVAL-s/r0_math500_no_think_t0.6.jsonl"          # its own commit at its path
    assert n.calls[0]["timeout"] == int(150 / 0.08 * 1.5 + 600)
    assert lines(n) == [] and not os.path.exists(os.path.join(d, ".upload_pending"))      # struck, the marker gone
    assert os.path.getsize(os.path.join(d, "r0_math500_no_think_t0.6.jsonl")) == 150 * 10 ** 6   # the local files untouched
    assert not any(k.endswith(".upload_pending") for k in n.remote)                       # the marker never goes up
    assert not [x for x in os.listdir(os.path.dirname(d)) if x.startswith(".stage_")]                  # [A1 condition] nothing new under runs/
    assert os.listdir(os.path.join(n.root, "control", "stage")) == []                                # staged in control/stage, then removed


def test_a_failed_part_keeps_the_parts_before_it(priv, monkeypatch):
    monkeypatch.setenv("WEIGHTS_REPLAY_COOL_S", "0")
    n = priv; d = task_dir(n, "EVAL-t"); RM.pend_result(d, "runs/EVAL-t")
    n.failing.add("r1_math500_no_think_t0.6.jsonl")
    r = W.once()
    assert r.startswith("REPLAY_FAIL result:runs/EVAL-t r1_math500_no_think_t0.6.jsonl:"), r
    assert "runs/EVAL-t/r0_math500_no_think_t0.6.jsonl" in n.remote and len(lines(n)) == 1   # the 150 MB part stays up
    assert os.path.exists(os.path.join(d, ".upload_pending"))
    n.failing.clear(); n.calls.clear()
    assert W.once().startswith("REPLAY_OK runs/EVAL-t")
    assert [c["files"] for c in n.calls][0] == ["r1_math500_no_think_t0.6.jsonl"]          # the next try sends only what is missing


def test_a_small_result_goes_in_one_commit_and_a_subdir_hold_is_accepted(priv):
    n = priv; d = task_dir(n, "PROBE-TERMX", big=(), small=True); up = os.path.join(d, "up"); os.makedirs(up)
    open(os.path.join(up, "report.json"), "w").write("{}")
    RM.pend_result(up, "runs/PROBE-TERMX")
    assert W.once().startswith("REPLAY_OK runs/PROBE-TERMX") and [c["files"] for c in n.calls] == [["report.json"]]
    assert not os.path.exists(os.path.join(d, ".upload_pending"))


def test_a_hold_outside_the_task_dir_is_refused(priv, monkeypatch):
    n = priv; elsewhere = tempfile.mkdtemp(); open(os.path.join(elsewhere, "a.json"), "w").write("{}")
    with open(os.path.join(n.root, "control", "weights_pending.txt"), "a") as f:
        f.write(f"2026-10-23T01:00:00Z\truns/EVAL-u\t{elsewhere}\t2\n")
    r = W.once()
    assert "neither under $CKPT_ROOT/weights_pending nor the task's own directory" in r and not n.calls


def test_one_waiting_event_after_two_hours(priv):
    n = priv; d = task_dir(n, "EVAL-w", big=(1,)); RM.pend_result(d, "runs/EVAL-w")
    items = W.pending(); t = __import__("calendar").timegm(time.strptime(items[0][0], "%Y-%m-%dT%H:%M:%SZ"))
    W.waiting_events(items, now=t + 3600)
    assert not any("result_upload_waiting" in e for e in n.events)
    W.waiting_events(items, now=t + 7300); W.waiting_events(items, now=t + 9000)
    assert sum("result_upload_waiting" in e or "runs/EVAL-w" in e for e in n.events) == 1


def test_a_task_dir_result_is_staged_under_control_not_runs(priv, monkeypatch):   # [v2.5.9z65, ops-lead's A1 condition]
    n = priv; d = task_dir(n, "EVAL-v", big=(1,)); RM.pend_result(d, "runs/EVAL-v")
    seen = []
    real = W.stage_result
    monkeypatch.setattr(W, "stage_result", lambda hold, stage: (seen.append(stage), real(hold, stage))[1])
    assert W.once().startswith("REPLAY_OK runs/EVAL-v")
    assert seen and seen[0].startswith(os.path.join(n.root, "control", "stage") + os.sep)
    assert not [x for x in os.listdir(os.path.join(n.root, "runs")) if x.startswith(".")]
    assert W.result_stage(os.path.join(n.ck, "weights_pending", "runs", "EVAL-x"), "runs/EVAL-x").startswith(os.path.join(n.ck, "weights_pending", "runs"))   # z64's holds as before
