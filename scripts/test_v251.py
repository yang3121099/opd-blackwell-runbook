"""Unit tests of runbook v2.5.1: two repositories (remote.py routing, host names out of file names, the private
repository must be private), the new set keys, the path-free pulse and its leak check, receipts that keep cmd/reason.
CPU only, no network or credentials (huggingface_hub is replaced). Run: python -m pytest -q scripts/test_v251.py"""
import importlib
import json
import os
import re
import time
import socket
import subprocess
import sys
import types

import pytest

from test_control import env, write_cmds  # noqa: F401  (shared fixture)
import control_rules as R

HERE = os.path.dirname(os.path.abspath(__file__))


class FakeApi:
    """Records uploads per repository; model_info(repo).private from PRIVATE."""
    calls, PRIVATE, commits = [], {}, []

    def __init__(self, *a, **k):
        pass

    def upload_file(self, repo_id, path_or_fileobj, path_in_repo, **k):
        FakeApi.calls.append((repo_id, path_in_repo))

    def upload_folder(self, repo_id, folder_path, path_in_repo, **k):
        FakeApi.calls.append((repo_id, path_in_repo + "/"))

    def create_commit(self, repo_id, operations, commit_message, **k):   # v2.5.5c: public uploads
        FakeApi.calls.append((repo_id, commit_message)); FakeApi.commits.append(list(operations))

    def model_info(self, repo):
        if repo not in FakeApi.PRIVATE:
            raise FileNotFoundError(repo)
        return types.SimpleNamespace(private=FakeApi.PRIVATE[repo])

    def whoami(self):
        return {"name": "x"}

    def create_repo(self, *a, **k):
        pass


@pytest.fixture
def rem(tmp_path, monkeypatch):
    monkeypatch.setenv("OPD_ROOT", str(tmp_path)); monkeypatch.setenv("HF_REPO", "me/old"); monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setenv("OPD_NODE", "abcd1234"); monkeypatch.setenv("UPLOAD_BACKEND", "hf")
    monkeypatch.setenv("PUBLIC_UPLOADS", "1")   # [v2.5.9z50] these are the split's tests (the default is now private first: test_private_first.py)
    monkeypatch.setattr(socket, "gethostname", lambda: "gpu-host-77.dc9.example-corp.com")
    monkeypatch.setattr(socket, "getfqdn", lambda: "gpu-host-77.dc9.example-corp.com")
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        HfApi=FakeApi, CommitOperationAdd=lambda path_in_repo, path_or_fileobj: types.SimpleNamespace(path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj)))
    FakeApi.calls, FakeApi.PRIVATE, FakeApi.commits = [], {"me/pub": False, "me/ops": True, "me/old": False}, []
    sys.path.insert(0, HERE)
    import remote
    importlib.reload(remote)
    f = tmp_path / "x.json"; f.write_text("{}")
    return types.SimpleNamespace(r=remote, root=tmp_path, f=str(f))


def test_no_file_is_v25(rem):
    assert rem.r.repos() == ("me/old", "me/old") and not rem.r.split_mode()
    rem.r.put(rem.f, "status/gpu-host-77.json"); rem.r.put(rem.f, "runs/E10b-P2/metrics.jsonl")
    assert [c[0] for c in FakeApi.calls] == ["me/old", "me/old"]


def test_split_routing_and_rename(rem):
    (rem.root / "hf_repos.txt").write_text("public=me/pub\nprivate=me/ops\n")
    for dst in ("status/gpu-host-77.json", "status/compare_gpu-host-77.json", "logs/gpu-host-77.dc9.example-corp.com/verify.log",
                "status/abcd1234/replies/3.json", "runs/E10b-P2/global_step_150/actor/huggingface", "ckpts/x/y", "pulse/abcd1234.json"):
        rem.r.put(rem.f, dst)
    got = dict((p, r) for r, p in FakeApi.calls)
    assert got == {"status/abcd1234.json": "me/ops", "status/compare_abcd1234.json": "me/ops", "logs/abcd1234/verify.log": "me/ops",
                   "status/abcd1234/replies/3.json": "me/ops", "runs/E10b-P2/global_step_150/actor/huggingface": "me/pub",
                   "ckpts/x/y": "me/pub", "pulse/abcd1234.json": "me/pub"}
    assert not any("gpu-host" in p or "example-corp" in p for p in got)


def test_private_refused_when_public(rem):
    (rem.root / "hf_repos.txt").write_text("public=me/pub\nprivate=me/pub\n")
    with pytest.raises(PermissionError):
        rem.r.put(rem.f, "status/abcd1234.json")
    rem.r.put(rem.f, "runs/E10b-P2/metrics.jsonl")   # public-class files still go
    assert FakeApi.calls == [("me/pub", "runs/E10b-P2/metrics.jsonl")]


def test_only_private_set_keeps_weights_on_old(rem):
    (rem.root / "hf_repos.txt").write_text("private=me/ops\n")
    rem.r.put(rem.f, "runs/E10b-P2/metrics.jsonl"); rem.r.put(rem.f, "status/abcd1234.json")
    assert FakeApi.calls == [("me/old", "runs/E10b-P2/metrics.jsonl"), ("me/ops", "status/abcd1234.json")]


@pytest.mark.parametrize("cmd", ["set HF_REPO_PUBLIC=me/opd-lab-b300", "set HF_REPO_PRIVATE=me/opd-lab-b300-ops", "set HF_REPO_PUBLIC=",
                                 "set HEARTBEAT_SEC=120", "set HEARTBEAT_SEC=60"])
def test_rules_accept_new_keys(cmd):
    R.parse({"cmd": cmd, "reason": "t"})


@pytest.mark.parametrize("cmd", ["set HF_REPO_PUBLIC=noslash", "set HF_REPO_PUBLIC=a/b/c", "set HEARTBEAT_SEC=30", "set HEARTBEAT_SEC=9999",
                                 "set HEARTBEAT_SEC=2m", "set HF_REPO=me/x"])
def test_rules_refuse_new_keys(cmd):
    with pytest.raises(R.Reject):
        R.parse({"cmd": cmd, "reason": "t"})


def test_set_repo_and_heartbeat_sec(env, monkeypatch):
    FakeApi.PRIVATE = {"me/pub": False, "me/ops": True}
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    c = env.c
    st, msg, _ = c.v_set(["HF_REPO_PRIVATE=me/pub"], 1)
    assert st == "rejected" and "public" in msg                      # a public repository cannot be the private one
    st, msg, _ = c.v_set(["HF_REPO_PUBLIC=me/missing"], 2)
    assert st == "rejected" and "not reachable" in msg
    assert c.v_set(["HF_REPO_PRIVATE=me/ops"], 3)[0] == "ok" and c.v_set(["HF_REPO_PUBLIC=me/pub"], 4)[0] == "ok"
    assert (env.root / "hf_repos.txt").read_text() == "private=me/ops\npublic=me/pub\n"
    assert c.v_set(["HF_REPO_PUBLIC="], 5)[0] == "ok" and (env.root / "hf_repos.txt").read_text() == "private=me/ops\n"
    assert c.v_set(["HF_REPO_PRIVATE="], 6)[0] == "ok" and not (env.root / "hf_repos.txt").exists()
    assert c.v_set(["HEARTBEAT_SEC=120"], 7)[0] == "ok" and open(c._p("heartbeat_sec")).read().strip() == "120"


STATUS = {"host": "gpu-host-77", "utc": "2026-10-17T13:01:26Z", "queue_version": 2, "runbook_head": "938604f", "runbook_version": "v2.5.1",
          "disk_free_gb": 4826.1, "disk_alert": False, "ckpt_root": "/mnt/shared9/someuser/code/opd-ckpt", "ckpt_disk_free_gb": 6209.1,
          "ckpt_roots": [{"base": "/mnt/shared9/someuser/code", "free_gb": 6209}],
          "tasks": {"E10b-P2": {"state": "running", "step": 59, "total_steps": 150, "val_math500": 0.79, "truncation": 0.2,
                                "step_time_mean5_s": 135.7, "eta_min": 196, "attempts_failed": 0,
                                "log_tail": ["/mnt/shared9/someuser/code/opd-run/runs/E10b-P2 something"]},
                    "E11b-P0": {"state": "done", "step": 0}},
          "control": {"applied_seq": 8, "pending": 0, "last_pull_utc": "2026-10-17T13:00:00Z",
                      "alerts": ["stalled:E10b-P3", "ckpt_root_alert:/mnt/shared9/someuser/code", "disk_alert"]}}
GPU = "0, 17, 162465, 275040\n1, 100, 121681, 275040\n"


def test_pulse_has_no_paths_or_names(env, monkeypatch):
    monkeypatch.setenv("HOME", "/mnt/shared9/someuser"); monkeypatch.setenv("OPD_ROOT", str(env.root))
    monkeypatch.setattr(socket, "gethostname", lambda: "gpu-host-77.dc9.example-corp.com")
    p = env.c.build_pulse(STATUS, GPU)
    text = json.dumps(p)
    assert "/" not in text and "someuser" not in text and "gpu-host" not in text and "example-corp" not in text
    assert p["tasks"]["E10b-P2"] == {"state": "running", "step": 59, "total_steps": 150, "val": 0.79, "trunc": 0.2, "s_per_step": 135.7,
                                     "eta_min": 196, "attempts_failed": 0, "gpu": None, "gpus": [], "kind": None, "started_utc": None}
    assert p["gpus"][1] == {"idx": 1, "util": 100.0, "mem_used": 121681.0, "mem_total": 275040.0}
    assert p["alerts"] == ["ckpt_root_alert", "disk_alert", "stalled:E10b-P3"] and p["node_id"] == "testnode"
    assert env.c.leak_check(p) == [] and len(text) < 2048


def test_leak_check_catches(env, monkeypatch):
    monkeypatch.setenv("HOME", "/mnt/shared9/someuser")
    monkeypatch.setattr(socket, "gethostname", lambda: "gpu-host-77.dc9.example-corp.com")
    toks = env.c._forbidden_tokens()
    assert {"someuser", "shared9", "example"} <= toks
    for bad in ("someuser", "x@y", "a/b", "node.dc9.example-corp.com", "host.internal", "run by SomeUser"):
        assert env.c.leak_check({"k": bad}, toks), bad
    for ok in ("E10b-P5-s43", "running", "2026-10-17T13:01:26Z", "v2.5.1", "stalled:E10b-P3", "938604f"):
        assert env.c.leak_check({"k": ok}, toks) == [], ok


def test_pulse_marks_padding_tasks(env, monkeypatch):   # [v2.5.9q] pad: true reaches the public pulse, only when true
    import copy
    st = copy.deepcopy(STATUS)
    st["tasks"]["EVAL-fill"] = {"state": "queued", "step": 0, "pad": True}
    st["tasks"]["EVAL-odd"] = {"state": "queued", "step": 0, "pad": "yes"}   # anything but true is dropped
    monkeypatch.setenv("OPD_ROOT", str(env.root))
    p = env.c.build_pulse(st, GPU)
    assert p["tasks"]["EVAL-fill"]["pad"] is True and "pad" not in p["tasks"]["EVAL-odd"] and "pad" not in p["tasks"]["E10b-P2"]


def test_pulse_blocked_by(env, monkeypatch):   # [v2.5.9z3] a blocked task's gate reaches the public pulse, a task name only
    import copy
    st = copy.deepcopy(STATUS)
    st["tasks"]["E10b-P5x"] = {"state": "blocked", "step": 0, "blocked_by": "PROBE-STOPMERGE"}
    st["tasks"]["E10b-P5y"] = {"state": "blocked", "step": 0, "blocked_by": "PROBE-X /etc/passwd"}   # not a task name: dropped
    monkeypatch.setenv("OPD_ROOT", str(env.root))
    p = env.c.build_pulse(st, GPU)
    assert p["tasks"]["E10b-P5x"]["state"] == "blocked" and p["tasks"]["E10b-P5x"]["blocked_by"] == "PROBE-STOPMERGE"
    assert "blocked_by" not in p["tasks"]["E10b-P5y"] and "blocked_by" not in p["tasks"]["E10b-P2"]


def test_pulse_refused_and_not_uploaded(env, monkeypatch):
    monkeypatch.setenv("HOME", "/home/running")   # a forbidden word that the pulse contains (the state "running")
    st = env.root / "status.json"; st.write_text(json.dumps(STATUS))
    monkeypatch.setattr(env.c.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=GPU, returncode=0))
    assert env.c.pulse(str(st)) is None and not os.path.exists(env.c._p("pulse.json"))


def test_pulse_uploaded_only_when_split(env, monkeypatch):
    st = env.root / "status.json"; st.write_text(json.dumps(STATUS))
    sent = []
    monkeypatch.setattr(env.c, "remote_put", lambda src, dst: sent.append(dst) or True)
    monkeypatch.setattr(env.c.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=GPU, returncode=0))
    monkeypatch.setattr(env.c, "_split_mode", lambda: False)
    assert env.c.pulse(str(st)) and sent == []
    monkeypatch.setattr(env.c, "_split_mode", lambda: True)
    assert env.c.pulse(str(st)) and sent == ["pulse/testnode.json"]


def test_receipt_keeps_cmd(env):
    c = env.c
    c.receipt(5, {"cmd": "run E10b-P5-s43", "reason": "fill GPU7"}, "accepted", "queued")
    r = c.receipt(5, None, "accepted", "started on gpu 7")
    assert r["cmd"] == "run E10b-P5-s43" and r["reason"] == "fill GPU7" and r["detail"] == "started on gpu 7"


@pytest.mark.parametrize("cmd", ["set FIXED_MICRO_BSZ=4", "setenv FIXED_MICRO_BSZ=16", "set FIXED_MICRO_BSZ="])
def test_rules_accept_fixed_micro(cmd):
    R.parse({"cmd": cmd, "reason": "t"})


@pytest.mark.parametrize("cmd", ["set FIXED_MICRO_BSZ=0", "set FIXED_MICRO_BSZ=17", "setenv FIXED_MICRO_BSZ=four"])
def test_rules_refuse_fixed_micro(cmd):
    with pytest.raises(R.Reject):
        R.parse({"cmd": cmd, "reason": "t"})


def test_pulse_gpu_mem_alert(env):
    p = env.c.build_pulse(STATUS, "0, 17, 162465, 275040\n2, 0, 255686, 275040\n")
    assert "gpu_mem_high:2" in p["alerts"] and "gpu_mem_high:0" not in p["alerts"] and env.c.leak_check(p) == []


def test_run_arm_fixed_micro_branch(tmp_path):
    """The batching lines of run_arm.sh: FIXED_MICRO_BSZ set -> fixed micro-batches everywhere; unset -> token budget; no flash-attn -> chunked entropy."""
    src = open(os.path.join(HERE, "run_arm.sh")).read()
    i = src.index('if [ -n "${FIXED_MICRO_BSZ:-}" ]'); j = src.index("\n", src.index('ENT=""', i))
    block = src[i:j]
    import subprocess
    run = lambda env: subprocess.run(["bash", "-c", block + '\necho "$BATCH|$LOGP_MB|$ENT"'], env=dict(os.environ, **env), capture_output=True, text=True).stdout.strip()
    a = run({"FIXED_MICRO_BSZ": "4", "RMPAD": "False", "TOK_DEFAULT": "6144"})
    assert a == "actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4|4|actor_rollout_ref.actor.entropy_from_logits_with_chunking=True", a
    b = run({"FIXED_MICRO_BSZ": "", "RMPAD": "True", "TOK_DEFAULT": "16384"})
    assert b == "actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384|8|", b


@pytest.mark.parametrize("cmd", ["setenv VENV=fa", "set VENV=fa", "setenv VENV=", "run FA-PROBE"])
def test_rules_accept_v253(cmd):
    R.parse({"cmd": cmd, "reason": "t"})


@pytest.mark.parametrize("cmd", ["setenv VENV=/tmp/x", "setenv VENV=venv-fa", "setenv VENV=FA"])
def test_rules_refuse_v253(cmd):
    with pytest.raises(R.Reject):
        R.parse({"cmd": cmd, "reason": "t"})


def test_run_accepts_fa_probe(env):
    st, msg, _ = env.c.v_run(["FA-PROBE"], 21)
    assert st == "accepted", msg
    assert env.c.v_run(["FA-PROBE2"], 22)[0] == "rejected"


def test_run_accepts_v258a(env):   # [v2.5.8a] E12b-P5self, the 0.6B ladder E12c (P0 / P5 only) and its gate PROBE-E12C
    for i, t in enumerate(("E12b-P5self", "E12b-P5self-s43", "E10b-P5self-s43", "E10d-P5self", "PROBE-P5SELF-4B", "E12c-P0", "E12c-P5", "PROBE-E12C", "E12b-P5-kl", "PROBE-E12B-KL", "E12b-P5-stop", "PROBE-E12B-STOP", "E12b-P5-8k", "E12b-P5-16k", "PROBE-E12B-8K", "PROBE-E12B-16K", "E10d-X1", "E10d-X2", "PROBE-X2507-X1", "PROBE-X2507-X2", "E10d-X4", "PROBE-X2507-X4", "E10b-P5lenS", "E10b-P5lenL", "E10b-P5lenR", "PROBE-LENSEL", "PROBE-2GPU", "PROBE-TOOL", "PROBE-NET", "E12b-P5-stop2", "PROBE-E12B-STOP2", "PROBE-NET2", "PROBE-ROLLOUT-U60", "PROBE-ROLLOUT-U85", "E10b-P5stopmerge", "PROBE-STOPMERGE", "PROBE-STOPMERGE-UNIT", "PROBE-WDIFF", "E10d-P5-t32", "PROBE-T32", "E10b-P5tail", "PROBE-TAIL", "E10c-P5-polish", "PROBE-POLISH", "E10c-P5self", "PROBE-P5SELF-06B", "PROBE-TERM4-06B", "E12b-P5-r3", "PROBE-E12B-R3", "E12b-P5-16k-kl", "BUILD-VENV-VL", "PROBE-VLLM-VL", "E10b-P5route", "E10b-P5mix", "PROBE-DIFF17", "PROBE-DIFF17-4B", "PROBE-ROUTE", "READINGS-EXTRA", "PROBE-VLM", "E15-VLM-P5")):   # [v2.5.8b] kl; [v2.5.8c] stop; [v2.5.8h] 8k / 16k
        st, msg, _ = env.c.v_run([t], 30 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E12c-P2", "E12c-P5self", "E12b-P2self", "PROBE-E12D", "E12b-P2-kl", "E12b-P2-stop", "PROBE-E12B-FOO", "E10d-P2self", "E10b-P5self-s44", "E12b-P5-32k", "PROBE-E12B-32K", "E10d-X3", "PROBE-X2507-X3", "E10d-X5", "PROBE-X2507-X5", "E10b-P5lenX", "E10c-P5lenS-fix4", "PROBE-LENSEL2", "E12b-P5-stop3", "E12b-P2-stop2")):
        assert env.c.v_run([t], 40 + i)[0] == "rejected", t


def test_run_accepts_v259z73(env):   # [v2.5.9z73] (Q-061) the budget arms and their gate
    for i, t in enumerate(("E10b-P5bud2k", "E10b-P5bud3k", "PROBE-TAIL-BUD")):
        st, msg, _ = env.c.v_run([t], 80 + i)
        assert st == "accepted", (t, msg)
    st, msg, _ = env.c.v_run(["PROBE-DIFFPROBE"], 85)   # [v2.5.9z74] (Q-062)
    assert st == "accepted", msg
    st, msg, _ = env.c.v_run(["PROBE-PSTAR"], 86)   # [v2.5.9z75] (Q-064)
    assert st == "accepted", msg
    for i, t in enumerate(("E10d-P5self-s43", "E12d-P5-8k-s43")):   # [v2.5.9z77] (Q-070 / Q-071)
        st, msg, _ = env.c.v_run([t], 95 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E10b-P5selfroute", "E10c-P5tail-c50", "PROBE-SELFROUTE")):   # [v2.5.9z82] (Q-072 / Q-055b)
        st, msg, _ = env.c.v_run([t], 97 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E10b-P5-dense", "E10b-P5-akl", "PROBE-DENSE")):   # [v2.5.9z76] (Q-065)
        st, msg, _ = env.c.v_run([t], 87 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E10b-P5bud4k", "PROBE-TAIL-BUD2", "PROBE-DIFFPROBE2")):
        assert env.c.v_run([t], 90 + i)[0] == "rejected", t


def test_run_e12_fixed_micro_branch():
    """v2.5.4a: run_e12.sh's batching lines read the same FIXED_MICRO_BSZ as run_arm.sh."""
    src = open(os.path.join(HERE, "run_e12.sh")).read()
    i = src.index('if [ -n "${FIXED_MICRO_BSZ:-}" ]'); j = src.index("\n", src.index('ENT=""', i))
    import subprocess
    run = lambda env: subprocess.run(["bash", "-c", src[i:j] + '\necho "$BATCH|$LOGP_MB|$ENT"'], env=dict(os.environ, **env), capture_output=True, text=True).stdout.strip()
    assert run({"FIXED_MICRO_BSZ": "4", "RMPAD": "False", "TOK": "6144"}) == \
        "actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4|4|actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
    assert run({"FIXED_MICRO_BSZ": "", "RMPAD": "False", "TOK": "6144"}).startswith("actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144|8|")
    assert "$BATCH $ENT" in src and "log_prob_micro_batch_size_per_gpu=$LOGP_MB" in src and "use_dynamic_bsz=True \\" not in src


def test_run_accepts_eval_tasks(env):
    assert env.c.v_run(["EVAL-bridge"], 31)[0] == "accepted"
    assert env.c.v_run(["EVAL-"], 32)[0] == "rejected"


def _fake_proc(root, pid, cmd, ppid, env):
    d = root / str(pid); d.mkdir(parents=True)
    (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
    (d / "status").write_text(f"Name:\tx\nPPid:\t{ppid}\n")
    (d / "environ").write_bytes("\0".join(f"{k}={v}" for k, v in env.items()).encode())


def test_reap_only_our_orphans(env, monkeypatch, tmp_path):
    c = env.c; pr = tmp_path / "proc"; R0 = c.ROOT
    _fake_proc(pr, 101, "VLLM::EngineCore", 1, {"OPD_ROOT": R0, "CUDA_VISIBLE_DEVICES": "4"})                    # reap
    _fake_proc(pr, 102, "ray::WorkerDict.actor_rollout_update_actor", 1, {"OPD_ROOT": R0, "CUDA_VISIBLE_DEVICES": "0", "RAY_TMPDIR": "/tmp/ray_gpu4"})  # reap
    _fake_proc(pr, 103, "VLLM::EngineCore", 555, {"OPD_ROOT": R0, "CUDA_VISIBLE_DEVICES": "4"})                  # parent alive: keep
    _fake_proc(pr, 104, "VLLM::EngineCore", 1, {"OPD_ROOT": R0, "CUDA_VISIBLE_DEVICES": "5"})                    # other GPU: keep
    _fake_proc(pr, 105, "VLLM::EngineCore", 1, {"OPD_ROOT": "/elsewhere", "CUDA_VISIBLE_DEVICES": "4"})          # not ours: keep
    _fake_proc(pr, 106, "bash scripts/upload_sync.sh runs/E10b-P2", 1, {"OPD_ROOT": R0, "CUDA_VISIBLE_DEVICES": "4"})  # not an engine: keep
    killed, events, mem = [], [], iter([251000.0, 120.0, 120.0])
    monkeypatch.setattr(c.os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(c, "_alive", lambda pid: False)
    monkeypatch.setattr(c, "_gpu_used_mib", lambda g: next(mem, 120.0))
    monkeypatch.setattr(c, "event", lambda task, kind, detail="": events.append((task, kind, detail)))
    got = c.reap(4, "E11b-P6", wait_s=1, proc_root=str(pr), grace_s=0)
    assert sorted(got) == [101, 102], got
    assert sorted(p for p, _ in killed) == [101, 102] and all(s == c.signal.SIGTERM for _, s in killed)
    assert [e[1] for e in events] == ["orphan_reaped", "orphan_reaped"] and "gpu_mem_before_mib=251000.0" in events[0][2]


def test_reap_gpu_list(env, monkeypatch, tmp_path):   # [v2.5.9a] a #gpus task's orphans are reaped on each of its GPUs, not on gpu 12
    c = env.c; pr = tmp_path / "proc"; R0 = c.ROOT
    _fake_proc(pr, 111, "VLLM::EngineCore", 1, {"OPD_ROOT": R0, "CUDA_VISIBLE_DEVICES": "1,2", "RAY_TMPDIR": "/tmp/ray_gpu1_2"})
    _fake_proc(pr, 112, "ray::WorkerDict", 1, {"OPD_ROOT": R0, "CUDA_VISIBLE_DEVICES": "0", "RAY_TMPDIR": "/tmp/ray_gpu1_2"})
    _fake_proc(pr, 113, "VLLM::EngineCore", 1, {"OPD_ROOT": R0, "CUDA_VISIBLE_DEVICES": "12", "RAY_TMPDIR": "/tmp/ray_gpu12"})
    monkeypatch.setattr(c.os, "kill", lambda pid, sig: None); monkeypatch.setattr(c, "_alive", lambda pid: False)
    monkeypatch.setattr(c, "_gpu_used_mib", lambda g: 100.0); monkeypatch.setattr(c, "event", lambda *a, **k: None)
    assert sorted(c.reap(1, "t", wait_s=1, proc_root=str(pr), grace_s=0)) == [111, 112]
    assert sorted(c.reap(2, "t", wait_s=1, proc_root=str(pr), grace_s=0)) == [111, 112]
    assert c.reap(12, "t", wait_s=1, proc_root=str(pr), grace_s=0) == [113]


def test_heartbeat_gpu_arg():   # [v2.5.9a] task.sh's GPU argument in the pulse: gpu = the first, gpus = all
    import os as _os, re as _re
    src = open(_os.path.join(_os.path.dirname(__file__), "heartbeat.sh")).read()
    m = _re.search(r"^def gpu_arg\(s\):.*?^    return None\n", src, _re.S | _re.M); assert m
    ns = {"re": _re}; exec(m.group(0), ns); g = ns["gpu_arg"]
    assert g("3") == (3, [3]) and g("1,2") == (1, [1, 2]) and g("cpu") == ("cpu", []) and g("x") is None and g("1;2") is None
    assert 't["gpus"] = lv[2] if lv else []' in src


def test_reap_clobbered_environ(env, monkeypatch, tmp_path):
    """v2.5.4c: vLLM / Ray titles overwrite the environment area; ours by the mapped venv, the GPU by nvidia-smi."""
    import os as _os
    c = env.c; pr = tmp_path / "proc"
    real_venv = tmp_path / "disk2" / "venv"; real_venv.mkdir(parents=True)
    vlink = _os.path.join(c.ROOT, "venv")
    if not _os.path.lexists(vlink):
        _os.symlink(real_venv, vlink)                                           # LINK_VENV: venv on another disk
    lib = lambda root: f"7f00-7f01 r-xp 0 08:01 1 {root}/lib/python3.10/site-packages/torch/lib/libtorch_cuda.so\n"
    def fake(pid, maps):
        _fake_proc(pr, pid, "VLLM::EngineCore", 1, {})
        (pr / str(pid) / "environ").write_bytes(b"VLLM::EngineCore" + b"\0" * 40)   # what setproctitle leaves
        (pr / str(pid) / "maps").write_text(maps)
    fake(201, lib(_os.path.realpath(real_venv)))           # ours (through the link), on gpu 4: reap
    fake(202, lib("/opt/someone/venv"))                    # another venv: keep
    fake(203, lib(_os.path.realpath(real_venv)))           # ours but on gpu 5: keep
    fake(204, lib(_os.path.realpath(c.ROOT) + "2/venv"))   # a sibling directory that only shares the prefix: keep
    monkeypatch.setattr(c, "_gpu_apps", lambda: {201: (4, 248000), 202: (4, 1000), 203: (5, 248000), 204: (4, 1000)})
    killed = []
    monkeypatch.setattr(c.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(c, "_alive", lambda pid: False)
    monkeypatch.setattr(c, "_gpu_used_mib", lambda g: 100.0)
    monkeypatch.setattr(c, "event", lambda *a, **k: None)
    assert c.reap(4, "gpu4", wait_s=1, proc_root=str(pr), grace_s=0) == [201] and killed == [201]


def test_env_sh_sets_spt_noenv():
    import os as _os
    assert "export SPT_NOENV=${SPT_NOENV:-1}" in open(_os.path.join(_os.path.dirname(__file__), "env.sh")).read()


def test_pulse_eval_gpu_and_orphan_alerts(env):
    import json as _j
    rows = [{"gpu": 0, "task": "E11b-P2", "ours": True, "gpu_mem_mb": 249784},
            {"gpu": 4, "task": None, "ours": False, "gpu_mem_mb": 249784},
            {"gpu": 1, "task": "E10b-P3", "ours": True, "gpu_mem_mb": 250000}]
    open(env.c._p("procs.json"), "w").write(_j.dumps({"rows": rows}))
    p = env.c.build_pulse(STATUS, "0, 89, 250000, 275040\n1, 90, 250000, 275040\n4, 0, 249784, 275040\n")
    assert "gpu_mem_high:0" not in p["alerts"]            # an evaluation GPU is not flagged
    assert "gpu_mem_high:1" in p["alerts"]                # a training GPU close to full is
    assert "gpu_orphan:4" in p["alerts"] and "gpu_mem_high:4" in p["alerts"]


@pytest.mark.parametrize("cmd,ok", [("reap 4", True), ("reap 0", True), ("reap 7", True), ("reap 8", False), ("reap 16", False), ("reap x", False), ("reap", False)])
def test_rules_reap(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_v_reap_calls_reap(env, monkeypatch):
    monkeypatch.setattr(env.c, "reap", lambda gpu, task=None, wait_s=0: [101, 102] if gpu == 4 else [])
    monkeypatch.setattr(env.c, "proc_table", lambda: [{"gpu": 3, "ours": True, "task": "E10b-P5"}, {"gpu": 4, "ours": False, "task": None}])
    monkeypatch.setattr(env.c.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="".join(f"GPU {i}: B300\n" for i in range(8)), returncode=0))
    st, msg, _ = env.c.v_reap(["3"], 40)
    assert st == "rejected" and "E10b-P5" in msg                     # never next to a live training
    assert env.c.v_reap(["9"], 43)[0] == "rejected"                    # no such GPU
    assert env.c.v_reap(["4"], 41)[0] == "ok" and "[101, 102]" in env.c.v_reap(["4"], 41)[1]
    assert "no orphaned" in env.c.v_reap(["5"], 42)[1]


def test_eval_max_tokens():   # v2.5.7g: the 7th field of a queue #eval line
    R.check_eval_args(["Qwen/Qwen3-1.7B", "math500", "4", "0.6", "32768"]); R.check_eval_args(["Qwen/Qwen3-1.7B", "math500", "4", "0.6"])
    for bad in ("12000", "65536", "2k", ""):
        with pytest.raises(R.Reject):
            R.check_eval_args(["Qwen/Qwen3-1.7B", "math500", "4", "0.6", bad])


@pytest.mark.parametrize("cmd,ok", [("eval Qwen/Qwen3-1.7B math500 4 0.6", True)])
def test_eval_verb_unchanged(cmd, ok):   # the eval verb keeps 2-4 arguments (no budget there)
    R.parse({"cmd": cmd, "reason": "t"})
    with pytest.raises(R.Reject):
        R.parse({"cmd": cmd + " 32768", "reason": "t"})


@pytest.mark.parametrize("cmd,ok", [("rerun FA-PROBE", True), ("rerun EVAL-bridge", True), ("rerun E10b-P2", True), ("rerun SFT-P1", True), ("rerun E10c-P4lowlr-fix4", True), ("rerun PROBE-TERM", True), ("rerun E10c-P4lowlr-dyn6144", True), ("rerun E12c-P0", True), ("rerun PROBE-E12C", True), ("rerun E12b-P5self", True), ("rerun PROBE-E12B-KL", True), ("rerun E12b-P5-kl", True), ("rerun PROBE-E12B-STOP", True), ("rerun PROBE-P5SELF-4B", True), ("rerun PROBE-X2507-X1", True), ("rerun PROBE-X2507-X4", True), ("rerun PROBE-LENSEL", True), ("rerun PROBE-2GPU", True), ("rerun PROBE-TOOL", True), ("rerun PROBE-NET", True), ("rerun PROBE-NET2", True), ("rerun PROBE-ROLLOUT-U75", True), ("rerun PROBE-STOPMERGE", True), ("rerun PROBE-STOPMERGE-UNIT", True), ("rerun PROBE-WDIFF", True), ("rerun PROBE-WDIFFX", False), ("rerun PROBE-T32", True), ("rerun PROBE-T33", False), ("rerun E10d-P5-t32", True), ("rerun PROBE-STOPMERGE-UNITX", False), ("rerun PROBE-TAIL", True), ("rerun PROBE-POLISH", True), ("rerun PROBE-ROLLOUT-U95", False), ("rerun PROBE-E12B-STOP2", True), ("rerun PROBE-X2507-X3", False), ("rerun PROBE-P5SELF-06B", True), ("rerun PROBE-E12B-R3", True), ("rerun PROBE-E12B-R4", False), ("rerun E12b-P5-r3", True), ("rerun E12b-P5-16k-kl", True), ("rerun BUILD-VENV-VL", True), ("rerun BUILD-VENV-IFEVAL", True), ("rerun PROBE-DIFF17", True), ("rerun READINGS-EXTRA", True), ("rerun PROBE-VLM", True), ("rerun E15-VLM-P5", True), ("rerun E15-VLM-P6", False), ("rerun PROBE-DIFF17-4B", True), ("rerun PROBE-ROUTE", True), ("rerun PROBE-ROUTE-4B", False), ("rerun E10b-P5route", True), ("rerun PROBE-VLLM-VL", True), ("rerun PROBE-VLLM", False), ("rerun PROBE-E12B-16K-KL", False), ("rerun PROBE-TERM4-06B", True), ("rerun PROBE-TERM4-06", False), ("rerun PROBE-P5SELF-06", False), ("rerun E10c-P5self", True), ("rerun E10d-P5self", True), ("rerun E12b-P5-stop", True), ("rerun /etc", False),
                                    ("rerun EVAL-a;b", False), ("rerun", False), ("rerun FA-PROBE x", False)])
def test_rules_rerun(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_v_rerun(env, monkeypatch):
    c = env.c; d = os.path.join(c.ROOT, "runs", "FA-PROBE"); os.makedirs(d, exist_ok=True)
    monkeypatch.setattr(c, "running_pids", lambda t=None: [])
    assert c.v_rerun(["FA-PROBE"], 50)[0] == "rejected"                    # neither done, failed nor skipped
    for f in (".done", "report.json", ".failed"):
        open(os.path.join(d, f), "w").write("x")
    st, msg, _ = c.v_rerun(["FA-PROBE"], 51)
    assert st == "ok" and not os.path.exists(os.path.join(d, ".done")) and os.path.exists(os.path.join(d, ".done.prev_51"))
    assert os.path.exists(os.path.join(d, "report.json.prev_51")) and not os.path.exists(os.path.join(d, ".failed"))
    open(os.path.join(d, ".done"), "w").write("x")
    monkeypatch.setattr(c, "running_pids", lambda t=None: [123])
    assert c.v_rerun(["FA-PROBE"], 52)[0] == "rejected"                    # running


def test_v_rerun_failed_gate(env, monkeypatch):   # [v2.5.9z79] a finished gate whose report says FAIL is rerun, listed or not
    c = env.c; monkeypatch.setattr(c, "running_pids", lambda t=None: [])
    def done(t, report):
        d = os.path.join(c.ROOT, "runs", t); os.makedirs(d, exist_ok=True); open(os.path.join(d, ".done"), "w").write("x")
        if report is not None:
            json.dump(report, open(os.path.join(d, "report.json"), "w"))
        return d
    d = done("PROBE-8B", {"result": "FAIL"})                                # not on v_rerun's list (seq 515 was rejected)
    st, msg, _ = c.v_rerun(["PROBE-8B"], 80)
    assert st == "ok" and os.path.exists(os.path.join(d, "report.json.prev_80")) and not os.path.exists(os.path.join(d, ".done"))
    done("PROBE-8B", {"result": "PASS"})
    assert c.v_rerun(["PROBE-8B"], 81)[0] == "rejected"                    # a passed gate off the list stays done
    done("PROBE-P5-8K", None)
    assert c.v_rerun(["PROBE-P5-8K"], 82)[0] == "rejected"                 # no report: as before
    done("E10b-P5", {"result": "FAIL"})
    assert c.v_rerun(["E10b-P5"], 83)[0] == "rejected"                     # a finished training is never rerun
    d = done("PROBE-TAIL-BUD", {"result": "FAIL"})
    assert c.v_rerun(["PROBE-TAIL-BUD"], 84)[0] == "ok"                    # a listed gate as before


def test_pulse_task_gpu_kind_started(env):
    st = {"tasks": {"FA-PROBE": {"state": "running", "gpu": 6, "kind": "probe", "started_utc": "2026-10-17T17:28:20Z"},
                    "E10b-P2": {"state": "done", "gpu": None, "kind": "train", "started_utc": None},
                    "EVAL-x": {"state": "running", "gpu": "6; rm", "kind": "/root", "started_utc": "17:28 on host-1"}}}
    p = env.c.build_pulse(st, "6, 50, 1000, 275040\n")
    assert p["tasks"]["FA-PROBE"]["gpu"] == 6 and p["tasks"]["FA-PROBE"]["kind"] == "probe"
    assert p["tasks"]["FA-PROBE"]["started_utc"] == "2026-10-17T17:28:20Z" and p["tasks"]["E10b-P2"]["gpu"] is None
    assert p["tasks"]["EVAL-x"]["gpu"] is None and p["tasks"]["EVAL-x"]["kind"] is None and p["tasks"]["EVAL-x"]["started_utc"] is None


def test_put_records_transfer(rem, monkeypatch):
    big = rem.root / "w.bin"; big.write_bytes(b"x" * 1000)
    rem.r.put(str(big), "runs/E10b-P2/w.bin")
    recs = [json.loads(l) for l in open(rem.root / "control" / "transfers.jsonl")]
    assert recs[-1]["dir"] == "up" and recs[-1]["bytes"] == 1000 and recs[-1]["ok"] is True and recs[-1]["mb_s"] is None   # < 100 MB
    def boom(src, dst):
        raise OSError("network")
    monkeypatch.setattr(rem.r, "_put", boom)
    with pytest.raises(OSError):
        rem.r.put(str(big), "runs/E10b-P2/w.bin")
    assert json.loads(open(rem.root / "control" / "transfers.jsonl").read().splitlines()[-1])["ok"] is False


def test_record_transfer_rate(rem):
    rem.r.record_transfer({"dir": "down", "bytes": 1_519_183_690, "seconds": 192.1})
    assert json.loads(open(rem.root / "control" / "transfers.jsonl").read().splitlines()[-1])["mb_s"] == 7.9


def test_public_upload_is_scrubbed(rem, monkeypatch, tmp_path):
    """v2.5.5c: text files bound for the public repository lose this machine's paths, user and host names."""
    import control
    (rem.root / "hf_repos.txt").write_text("public=me/pub\nprivate=me/ops\n")
    other = tmp_path / "disk2" / "someuser" / "code" / "opd-run"; (other / "venv").mkdir(parents=True)
    os.symlink(other / "venv", rem.root / "venv")                                   # LINK_VENV: the venv on another disk
    ck = tmp_path / "ckpt"; ck.mkdir(); monkeypatch.setenv("CKPT_ROOT", str(ck))
    monkeypatch.setattr(control, "_forbidden_tokens", lambda: {"someuser", "gpuhost77"})
    d = tmp_path / "up"; d.mkdir()
    (d / "train.log").write_text(f"model {rem.root}/runs/E10b-P2/ckpt\nvenv {other}/venv/lib/python3.10/x.py\n"
                                 f"mount /mnt/someuser/data/y and someuser@gpuhost77 done\n")
    (d / "m.metrics.json").write_text(json.dumps({"model": f"{ck}/adhoc_models/runs/E12-P5c05/global_step_40", "avg_acc": 74.1}))
    (d / "model.safetensors").write_bytes(b"\x00weights")
    (d / ".failed").write_text(f"attempts: 1\n  File \"/mnt/someuser/venv/lib/x.py\", line 3\n")      # v2.5.6a: a dotfile is text
    (d / "ray.log").write_text("Started a local Ray instance at 10.9.8.7:6379; node 192.168.3.4; version 2.49.1.0; ok 8.8.8.8\n")
    (d / "gen.jsonl").write_text('{"text": "1.2.1.2 then 10.1.1.1"}\n')
    monkeypatch.setattr(rem.r, "own_ips", lambda: {"8.8.8.8"})
    rem.r.put(str(d), "runs/EVAL-bridge")
    ops = {o.path_in_repo: o.path_or_fileobj for o in FakeApi.commits[-1]}
    log, met = ops["runs/EVAL-bridge/train.log"].decode(), ops["runs/EVAL-bridge/m.metrics.json"].decode()
    assert "$OPD_ROOT/runs/E10b-P2/ckpt" in log and "$OPD_ROOT/venv/lib/python3.10/x.py" in log
    assert "<path>" in log and "<redacted>@<redacted>" in log and "someuser" not in log and "gpuhost77" not in log
    assert json.loads(met) == {"model": "$CKPT_ROOT/adhoc_models/runs/E12-P5c05/global_step_40", "avg_acc": 74.1}
    assert ops["runs/EVAL-bridge/model.safetensors"] == str(d / "model.safetensors")    # weights as they are
    assert b"someuser" not in ops["runs/EVAL-bridge/.failed"] and b"<path>" in ops["runs/EVAL-bridge/.failed"]
    ray = ops["runs/EVAL-bridge/ray.log"].decode()                                        # v2.5.6g: addresses
    assert "10.9.8.7" not in ray and "192.168.3.4" not in ray and "8.8.8.8" not in ray and "<ip>:6379" in ray and "2.49.1.0" in ray
    assert ops["runs/EVAL-bridge/gen.jsonl"] == b'{"text": "1.2.1.2 then 10.1.1.1"}\n'     # generation outputs untouched
    assert rem.r.is_text(".inputs_ok", 10) and rem.r.is_text("train.log", 10**9) and not rem.r.is_text("model.safetensors", 10)
    assert not rem.r.is_text(".big", 2 << 20)
    rem.r.put(str(d / "train.log"), "logs/abcd1234/x.log")                              # private: unchanged route
    assert FakeApi.calls[-1] == ("me/ops", "logs/abcd1234/x.log")


def test_public_model_files_unchanged(rem, monkeypatch, tmp_path):
    """v2.5.6k: a model directory's vocabulary goes up byte for byte, its JSON configs only lose path values (B300
    2026-10-18: HOME -> "~" broke the vocabulary entry equal to HOME; EVAL-e12b-P0 / -P2 failed on the tokenizer)."""
    import control
    (rem.root / "hf_repos.txt").write_text("public=me/pub\nprivate=me/ops\n")
    home = tmp_path / "root"; home.mkdir(); monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(control, "_forbidden_tokens", lambda: {"someuser", "share"})
    d = tmp_path / "bf16"; d.mkdir()
    vocab = json.dumps({str(home): 1, "~": 2, "share": 3, "/mnt/someuser/x": 4})
    (d / "vocab.json").write_text(vocab); (d / "tokenizer.json").write_text(vocab); (d / "merges.txt").write_text("sh are\n")
    cfg = {"architectures": ["Qwen3ForCausalLM"], "_name_or_path": f"{rem.root}/runs/E12b-P0/ckpt", "note": "share this"}
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "generation_config.json").write_text('{"do_sample": true}')
    (d / "model-00001-of-00002.safetensors").write_bytes(b"\x00w")
    rem.r.put(str(d), "runs/E12b-P0/global_step_100/actor/huggingface")
    ops = {o.path_in_repo.rsplit("/", 1)[1]: o.path_or_fileobj for o in FakeApi.commits[-1]}
    for f in ("vocab.json", "tokenizer.json", "merges.txt", "generation_config.json"):
        assert ops[f] == str(d / f)                                  # the file itself, unchanged
    c = json.loads(ops["config.json"].decode())
    assert c["_name_or_path"] == "$OPD_ROOT/runs/E12b-P0/ckpt" and c["note"] == "share this" and c["architectures"] == ["Qwen3ForCausalLM"]
    assert rem.r.is_model_file("", "runs/X/global_step_1/actor/huggingface/vocab.json") and not rem.r.is_model_file("", "runs/X/train.log")


@pytest.mark.parametrize("cmd,ok", [("republish E10b-P2", True), ("republish EVAL-bridge", True), ("republish a/b", False), ("republish", False)])
def test_rules_republish(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_do_republish_only_published_text_files(env, monkeypatch):
    c = env.c; d = os.path.join(c.ROOT, "runs", "E10b-P2"); hf = os.path.join(d, "global_step_150", "actor", "huggingface")
    os.makedirs(hf); os.makedirs(os.path.join(d, "ckpt"))
    for f, body in ((".failed", "attempts: 1"), ("train.log", "log"), ("global_step_150/actor/huggingface/config.json", "{}"),
                    ("global_step_150/actor/huggingface/model.safetensors", "w"), ("ckpt/state.json", "{}"), ("upload.log", "u")):
        open(os.path.join(d, f), "w").write(body)
    published = ["runs/E10b-P2/.failed", "runs/E10b-P2/train.log", "runs/E10b-P2/global_step_150/actor/huggingface/config.json",
                 "runs/E10b-P2/global_step_150/actor/huggingface/model.safetensors", "runs/E10b-P2/ckpt/state.json",
                 "runs/E10b-P2/gone.log", "runs/E10b-P3/train.log"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda token=None: types.SimpleNamespace(list_repo_files=lambda repo: published)))
    monkeypatch.setenv("HF_REPO", "me/pub")
    staged, got = [], {}
    def put(src, dst):
        staged.extend(sorted(os.path.relpath(os.path.join(r, f), src) for r, _, fs in os.walk(src) for f in fs)); got["dst"] = dst; return True
    monkeypatch.setattr(c, "remote_put", put)
    monkeypatch.setattr(c, "receipt", lambda seq, e, st, detail, **k: got.update(st=st, detail=detail, **k))
    monkeypatch.setattr(c, "event", lambda *a, **k: None)
    c.do_republish(60, "E10b-P2")
    assert staged == [".failed", "global_step_150/actor/huggingface/config.json", "train.log"] and got["dst"] == "runs/E10b-P2"   # no weights, no ckpt/, no unpublished upload.log
    assert got["st"] == "ok" and got["missing"] == ["gone.log"]


def test_do_republish_model_files_from_ckpt(env, monkeypatch, tmp_path):
    """v2.5.6k: E12b keeps no weights after the upload, but ckpt/global_step_N/actor/huggingface keeps the tokenizer and
    configs: republish takes them from there, and drops this node's fetched copies so the next reading fetches again."""
    c = env.c; d = os.path.join(c.ROOT, "runs", "E12b-P0"); hf = os.path.join(d, "ckpt", "global_step_100", "actor", "huggingface")
    os.makedirs(hf)
    for f in ("tokenizer.json", "vocab.json", "config.json"):
        open(os.path.join(hf, f), "w").write("{}")
    ck = tmp_path / "ck"; got_dir = ck / "adhoc_models" / "result" / "runs" / "E12b-P0" / "global_step_100" / "actor" / "huggingface"
    got_dir.mkdir(parents=True)
    for f in ("tokenizer.json", ".fetch_ok", "model-00001-of-00002.safetensors"):
        (got_dir / f).write_text("x")
    monkeypatch.setenv("CKPT_ROOT", str(ck))
    pre = "runs/E12b-P0/global_step_100/actor/huggingface/"
    published = [pre + f for f in ("tokenizer.json", "vocab.json", "config.json", "model-00001-of-00002.safetensors", "merges.txt")]
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda token=None: types.SimpleNamespace(list_repo_files=lambda repo: published)))
    monkeypatch.setenv("HF_REPO", "me/pub")
    staged, got = [], {}
    def put(src, dst):
        staged.extend(sorted(os.path.relpath(os.path.join(r, f), src) for r, _, fs in os.walk(src) for f in fs)); return True
    monkeypatch.setattr(c, "remote_put", put)
    monkeypatch.setattr(c, "receipt", lambda seq, e, st, detail, **k: got.update(st=st, detail=detail, **k))
    monkeypatch.setattr(c, "event", lambda *a, **k: None)
    c.do_republish(61, "E12b-P0")
    assert staged == [pre[len("runs/E12b-P0/"):] + f for f in ("config.json", "tokenizer.json", "vocab.json")]
    assert got["st"] == "ok" and got["missing"] == ["global_step_100/actor/huggingface/merges.txt"]
    assert sorted(os.listdir(got_dir)) == ["model-00001-of-00002.safetensors"]   # stale tokenizer and .fetch_ok gone, weights kept



def test_v_rerun_failed_and_skipped(env, monkeypatch):
    c = env.c; monkeypatch.setattr(c, "running_pids", lambda t=None: [])
    sft, e12 = os.path.join(c.ROOT, "runs", "SFT-P1"), os.path.join(c.ROOT, "runs", "E12b-P1")
    os.makedirs(sft); os.makedirs(e12)
    open(os.path.join(sft, ".failed"), "w").write("attempts: 3\n"); open(os.path.join(e12, ".skipped"), "w").write("dep\n")
    assert c.v_rerun(["SFT-P1"], 70)[0] == "ok" and os.path.exists(os.path.join(sft, ".failed.prev_70")) and not os.path.exists(os.path.join(sft, ".failed"))
    assert c.v_rerun(["E12b-P1"], 71)[0] == "ok" and not os.path.exists(os.path.join(e12, ".skipped"))
    open(os.path.join(sft, ".done"), "w").write("x")
    assert c.v_rerun(["SFT-P1"], 72)[0] == "rejected"                      # a finished training is never rerun


@pytest.mark.parametrize("cmd,ok", [("publish E10b-P2", True), ("publish a/b", False), ("leakscan", True), ("leakscan x", False)])
def test_rules_publish_leakscan(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


@pytest.mark.parametrize("cmd,ok", [("fa-build", True), ("fa-build now", False)])
def test_rules_fa_build(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_v_fa_build_needs_venv_fa(env, monkeypatch):
    """v2.5.6m: fa-build is refused without FA-PROBE's venv-fa and while a build runs; otherwise started in the background."""
    c = env.c; started = []
    monkeypatch.setattr(c.subprocess, "Popen", lambda a, **k: started.append(a))
    assert c.v_fa_build([], 90)[0] == "rejected" and not started
    os.makedirs(os.path.join(c.ROOT, "venv-fa"))
    monkeypatch.setattr(c.subprocess, "run", lambda a, **k: types.SimpleNamespace(returncode=0, stdout="4242\n"))   # pgrep finds a build (its pid)
    assert c.v_fa_build([], 91)[0] == "rejected" and not started
    monkeypatch.setattr(c.subprocess, "run", lambda a, **k: types.SimpleNamespace(returncode=1, stdout=""))
    assert c.v_fa_build([], 92)[0] == "accepted" and started[0][-2:] == ["do_fa_build", "92"]


@pytest.mark.parametrize("cmd,ok", [("scrubdiff", True), ("scrubdiff fix", True), ("scrubdiff now", False), ("scrubdiff fix x", False)])
def test_rules_scrubdiff(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_gen_scrub_keeps_bare_home(rem):
    """v2.5.6n: in generation outputs a root is replaced only as a path prefix; logs keep the full replacement."""
    roots = [("/home/u1", "~"), ("/data/x/opd", "$OPD_ROOT")]
    t = '{"text": "so a/home/u1 = 3, see /home/u1/notes and /data/x/opd/runs"}'
    assert rem.r.scrub_text(t, roots, set(), None, gen=True) == '{"text": "so a/home/u1 = 3, see ~/notes and $OPD_ROOT/runs"}'
    assert rem.r.scrub_text("cd /home/u1", roots, set()) == "cd ~"


def test_scrub_changes_counts_without_content(env):
    c = env.c; roots = [("/home/u1", "~")]
    orig = "\n".join([json.dumps({"pid": 3, "sample": 1, "text": "x /home/u1 y"}), json.dumps({"pid": 4, "sample": 0, "text": "ok"}),
                      json.dumps({"pid": 5, "sample": 2, "text": "path /home/u1/a and someuser"})])
    pub = "\n".join([json.dumps({"pid": 3, "sample": 1, "text": "x ~ y"}), json.dumps({"pid": 4, "sample": 0, "text": "ok"}),
                     json.dumps({"pid": 5, "sample": 2, "text": "path ~/a and <redacted>"})])
    r = c.scrub_changes(orig, pub, {"someuser"}, roots)
    assert r["lines"] == 3 and r["changed"] == 2 and r["sensitive"] == 1 and r["kinds"] == {"~": 2, "<redacted>": 1}
    assert r["ids"] == [[3, 1], [5, 2]] and "someuser" not in json.dumps(r) and "/home/u1" not in json.dumps(r)


def test_do_scrubdiff_fix_only_clean_files(env, monkeypatch, tmp_path):
    c = env.c; d = os.path.join(c.ROOT, "runs", "EVAL-x"); os.makedirs(d)
    files = {"a.jsonl": ('{"pid": 1, "sample": 0, "t": "a/home/u1 b"}\n', '{"pid": 1, "sample": 0, "t": "a~ b"}\n'),       # artifact
             "b.jsonl": ('{"pid": 2, "sample": 0, "t": "someuser"}\n', '{"pid": 2, "sample": 0, "t": "<redacted>"}\n'),   # sensitive
             "c.jsonl": ('{"pid": 3}\n', '{"pid": 3}\n')}                                                                   # unchanged
    pubdir = tmp_path / "pub"; pubdir.mkdir()
    for f, (o, p) in files.items():
        open(os.path.join(d, f), "w").write(o); (pubdir / f).write_text(p)
    names = [f"runs/EVAL-x/{f}" for f in files] + ["runs/EVAL-y/gone.jsonl", "runs/EVAL-x/train.log"]
    hub = types.SimpleNamespace(HfApi=lambda token=None: types.SimpleNamespace(repo_info=lambda repo: types.SimpleNamespace(
        sha="abc123def4567", siblings=[types.SimpleNamespace(rfilename=n) for n in names])),
        hf_hub_download=lambda repo, f, revision=None, cache_dir=None: str(pubdir / os.path.basename(f)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setenv("HF_REPO", "me/pub"); monkeypatch.setenv("HOME", "/home/u1")
    monkeypatch.setattr(c, "_forbidden_tokens", lambda: {"someuser"})
    sys.path.insert(0, os.path.join(c.RUNBOOK, "scripts")); import remote
    ups, got = [], {}
    monkeypatch.setattr(remote, "put_public_unchanged", lambda pairs, msg: ups.extend(pairs))
    monkeypatch.setattr(c, "receipt", lambda seq, e, st, detail, **k: got.update(st=st, detail=detail, **k))
    monkeypatch.setattr(c, "event", lambda *a, **k: None)
    c.do_scrubdiff(97)
    assert got["st"] == "ok" and set(got["files"]) == {"runs/EVAL-x/a.jsonl", "runs/EVAL-x/b.jsonl"} and got["clean"] == ["runs/EVAL-x/a.jsonl"]
    assert got["missing"] == ["runs/EVAL-y/gone.jsonl"] and not ups and got["totals"]["~"] == 1 and got["totals"]["<redacted>"] == 1
    c.do_scrubdiff(98, fix=True)
    assert ups == [(os.path.join(c.ROOT, "runs/EVAL-x/a.jsonl"), "runs/EVAL-x/a.jsonl")] and "re-uploaded unchanged" in got["detail"]


@pytest.mark.parametrize("cmd,ok", [("setenv FUSED=1", True), ("setenv FUSED=", True), ("setenv GC=0", True), ("setenv GC=2", False),
                                    ("setenv FUSED=yes", False), ("rerun SPEED-PROBE", True)])
def test_rules_speed_probe(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_speed_probe_report(tmp_path):
    """v2.5.6o: report.json from the probe's files: log-prob checks, rc, peak, steps 2-3 means and ratios to base."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import speed_probe as S
    d = tmp_path
    (d / "lp_student.json").write_text(json.dumps({"pass": True, "max_abs_diff": 0.004})); (d / "lp_teacher.json").write_text(json.dumps({"pass": True}))
    for v, st, rc, oom in (("base", 130, 0, False), ("fused", 120, 0, False), ("gc0", 100, 1, True), ("fused_gc0", 95, 0, False)):
        (d / f"{v}.rc").write_text(f"{rc}\n"); (d / f"{v}.mem.csv").write_text("1,1000\n2,150000\n3,90000\n"); (d / v).mkdir()
        log = "".join(f"(TaskRunner pid=1) step:{i} - timing_s/gen:{40 + i} - timing_s/update_actor:{st - 80} - timing_s/step:{st + (30 if i == 1 else 0)} - actor/pg_loss:0.0{i} - response_length/mean:2200\n" for i in (1, 2, 3))
        (d / v / "train.log").write_text(log + ("torch.OutOfMemoryError: CUDA out of memory\n" if oom else ""))
    r = S.report(str(d))
    assert r["variants"]["base"]["mean_s23"]["step"] == 130 and r["variants"]["fused"]["ratio_to_base"]["step"] == round(130 / 120, 3)
    assert r["variants"]["base"]["mem_peak_mib"] == 150000 and r["variants"]["base"]["steps"][0]["actor/pg_loss"] == 0.01
    assert r["fused_pass"] and not r["gc0_pass"] and r["fused_gc0_pass"] and r["variants"]["gc0"]["oom"]


def test_do_publish_maps_weights_to_local_bf16(env, monkeypatch):
    c = env.c; d = os.path.join(c.ROOT, "runs", "E10b-P2"); bf = os.path.join(d, "ckpt", "global_step_150", "actor", "huggingface_bf16")
    os.makedirs(bf)
    for f, body in (("train.log", "log"), ("config.yaml", "c")): open(os.path.join(d, f), "w").write(body)
    for f in ("model-00001-of-00002.safetensors", "config.json"): open(os.path.join(bf, f), "w").write("w")
    old = ["runs/E10b-P2/train.log", "runs/E10b-P2/config.yaml", "runs/E10b-P2/gone.json",
           "runs/E10b-P2/global_step_150/actor/huggingface/model-00001-of-00002.safetensors",
           "runs/E10b-P2/global_step_150/actor/huggingface/config.json", "runs/E10b-P2/global_step_150/actor/x.pt", "runs/E10b-P3/train.log"]
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=lambda token=None: types.SimpleNamespace(list_repo_files=lambda repo: old)))
    staged, got = [], {}
    def put(src, dst):
        staged.extend(sorted(os.path.relpath(os.path.join(r, f), src) for r, _, fs in os.walk(src) for f in fs)); got["dst"] = dst; return True
    monkeypatch.setattr(c, "remote_put", put)
    monkeypatch.setattr(c, "receipt", lambda seq, e, st, detail, **k: got.update(st=st, detail=detail, **k))
    monkeypatch.setattr(c, "event", lambda *a, **k: None)
    c.do_publish(80, "E10b-P2")
    assert staged == ["config.yaml", "global_step_150/actor/huggingface/config.json",
                      "global_step_150/actor/huggingface/model-00001-of-00002.safetensors", "train.log"] and got["dst"] == "runs/E10b-P2"
    assert got["st"] == "ok" and sorted(got["missing"]) == ["global_step_150/actor/x.pt", "gone.json"]


def test_leak_counts(env):
    c = env.c
    t = "path /home/someuser/x and 10.9.8.7 and hf_" + "a" * 32
    assert c.leak_counts(t, {"someuser"}, set(), False) == 3
    assert c.leak_counts(t, {"someuser"}, set(), True) == 2          # generation outputs: no address check
    assert c.leak_counts("clean text 1.2.1.2", {"someuser"}, set(), False) == 0
    assert c.leak_counts('{"Ġsomeuser": 5, "x": "hf_' + "b" * 32 + '"}', {"someuser"}, set(), False, vocab=True) == 1   # v2.5.6k: tokens only


# ---------- v2.5.7: disaster recovery, du / rm-ours / dr-keep ----------
@pytest.mark.parametrize("cmd,ok", [("setenv DRILL=1", True), ("setenv DRILL=2", False), ("setenv REMOTE_FAULT=hang_once", True),
                                    ("setenv REMOTE_FAULT=hang_once_hard", True), ("setenv REMOTE_FAULT=", True), ("setenv REMOTE_FAULT=rm", False)])
def test_rules_drill(cmd, ok):   # v2.5.7zb (rules 2.5.29)
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_drill_fault_once(tmp_path, monkeypatch):   # v2.5.7zb: only with DRILL=1, once (marker), read from control.env
    import importlib.util, threading
    spec = importlib.util.spec_from_file_location("remote_t", os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote.py"))
    rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)
    monkeypatch.setenv("OPD_ROOT", str(tmp_path)); slept = []
    monkeypatch.setattr(rm.time, "sleep", lambda s: slept.append(s))
    S = "status/14b9aaef.json"
    rm._drill_fault(S); assert slept == []                                 # no drill: nothing
    (tmp_path / "control.env").write_text("export DRILL=1\nexport REMOTE_FAULT=hang_once\n")
    for other in ("status/14b9aaef/replies/9.json", "status/14b9aaef/events/x.json", "runs/E12d-P5/global_step_75", "pulse/14b9aaef.json"):
        rm._drill_fault(other)
    assert slept == [] and not (tmp_path / "control" / "drill_hang_once.done").exists()   # v2.5.7zc: only the status upload
    rm._drill_fault(S); assert slept == [7200] and (tmp_path / "control" / "drill_hang_once.done").exists()
    rm._drill_fault(S); assert slept == [7200]                             # once only


@pytest.mark.parametrize("cmd,ok", [("du opd_root", True), ("du old_base runs", True), ("du /etc", False), ("du dr ../x", False),
                                    ("du ckpt", True), ("du ckpt E12d-P0", True), ("rm-ours ckpt E12d-P0", False),
                                    ("rm-ours old_base runs/E10b-P2", True), ("rm-ours opd_root runs", False), ("rm-ours dr", False),
                                    ("rm-ours old_base /abs", False), ("dr-keep SFT-P1", True), ("dr-keep a/b", False),
                                    ("setenv DR=0", True), ("setenv DR=2", False), ("setenv DR_EVERY=25", True), ("setenv DR_EVERY=5", False),
                                    ("setenv DR_MBPS=300", True), ("setenv DR_MBPS=5000", False), ("setenv E12B_GC=1", True), ("setenv E12B_GC=y", False)])
def test_rules_dr(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def _old_base(env, tmp_path):
    old = tmp_path / "disk2" / "opd-run"; (old / "venv" / "bin").mkdir(parents=True); (old / "runs" / "E10b-P2").mkdir(parents=True)
    (old / "runs" / "E10b-P2" / "a.bin").write_bytes(b"x" * 1000); (old / "models").mkdir()
    os.symlink(old / "venv", env.root / "venv")      # the venv in use lives in the old base (LINK_VENV)
    return old


def _quiet(c, monkeypatch, got):
    monkeypatch.setattr(c, "running_pids", lambda t=None: [])
    monkeypatch.setattr(c, "receipt", lambda seq, e, st, detail, **k: got.update(st=st, detail=detail, **k))
    monkeypatch.setattr(c, "event", lambda *a, **k: None)


def test_our_prefixes_and_protection(env, tmp_path, monkeypatch):
    c = env.c; old = _old_base(env, tmp_path); _quiet(c, monkeypatch, {})
    p = c.our_prefixes()
    assert p["old_base"] == os.path.realpath(old) and p["dr"] == os.path.realpath(str(tmp_path / "opd-dr"))
    assert c.resolve_ours("old_base", "runs/E10b-P2", deleting=True)[0] == os.path.realpath(old / "runs" / "E10b-P2")
    for rel, why in (("venv", "protected"), ("venv/bin", "protected"), (".", "relative"), ("", "relative"), ("runs/../venv", "relative"), ("nope", "no such")):
        assert why in (c.resolve_ours("old_base", rel, deleting=True)[1] or ""), rel
    assert "dr or old_base" in c.resolve_ours("opd_root", "runs", deleting=True)[1]
    os.symlink(tmp_path, old / "escape")
    assert "leaves" in c.resolve_ours("old_base", "escape", deleting=True)[1]
    assert c.resolve_ours("old_base", "")[0] == os.path.realpath(old)            # du may size the whole prefix
    ck = tmp_path / "ck"; (ck / "E12d-P0").mkdir(parents=True); monkeypatch.setenv("CKPT_ROOT", str(ck))   # v2.5.7y
    assert c.our_prefixes()["ckpt"] == os.path.realpath(ck) and c.resolve_ours("ckpt", "E12d-P0")[0] == os.path.realpath(ck / "E12d-P0")
    assert "dr or old_base" in c.resolve_ours("ckpt", "E12d-P0", deleting=True)[1]


def test_do_du_and_rm_ours(env, tmp_path, monkeypatch):
    c = env.c; old = _old_base(env, tmp_path); got = {}; _quiet(c, monkeypatch, got)
    c.do_du(110, "old_base", "runs")
    assert got["st"] == "ok" and got["children_gb"] == {"E10b-P2": 0.0} and "fs_free_gb" in got
    c.do_rm_ours(111, "old_base", "runs/E10b-P2")
    assert got["st"] == "ok" and got["freed_gb"] == 0.0 and not (old / "runs" / "E10b-P2").exists() and (old / "venv").exists()
    c.do_rm_ours(112, "old_base", "venv")
    assert got["st"] == "failed" and "protected" in got["detail"] and (old / "venv").exists()


def test_dr_report_and_keep(env, tmp_path, monkeypatch):
    c = env.c; got = {}; _quiet(c, monkeypatch, got)
    d = tmp_path / "opd-dr" / "dr" / "E10b-P2" / "global_step_25" / "actor"; d.mkdir(parents=True); (d / "m.pt").write_bytes(b"x" * 2000)
    (tmp_path / "opd-dr" / "dr" / "E10b-P2" / "latest").write_text("25\n")
    r = c.dr_report()
    assert r["exists"] and r["tasks"]["E10b-P2"]["step"] == "25" and r["cap_gb"] == 1500 and not r["over_cap"]   # [v2.5.8] was 900
    hf = env.root / "runs" / "SFT-P1" / "ckpt" / "global_step_283" / "huggingface"; hf.mkdir(parents=True); (hf / "model.safetensors").write_bytes(b"w")
    (env.root / "runs" / "SFT-P1" / "global_step_1").mkdir(); (env.root / "runs" / "SFT-P1" / "final_step").write_text("283\n")
    started, real = [], c.subprocess.Popen
    c.subprocess.Popen = lambda a, **k: started.append(a)
    try:
        assert c.v_dr_keep(["SFT-P1"], 113)[0] == "accepted" and started[0][-2:] == ["113", "SFT-P1"]
    finally:
        c.subprocess.Popen = real
    c.do_dr_keep(113, "SFT-P1")
    k = tmp_path / "opd-dr" / "keep" / "SFT-P1"
    assert got["st"] == "ok" and (k / "ckpt" / "global_step_283" / "huggingface" / "model.safetensors").exists() and (k / "final_step").exists()
    assert c.v_dr_keep(["E10b-none"], 114)[0] == "rejected"


def test_dr_keep_auto(env, tmp_path, monkeypatch):   # [v2.5.8] upload_sync: final + uploaded steps kept, others not
    c = env.c; evs = []
    monkeypatch.setattr(c, "event", lambda t, k, d="": evs.append((t, k, d)))
    ck = env.root / "runs" / "E12d-X" / "ckpt"
    for n, sub, up in ((25, "huggingface", "not"), (50, "huggingface_bf16", "yes"), (100, "huggingface", "yes")):
        h = ck / f"global_step_{n}" / "actor" / sub; h.mkdir(parents=True); (h / "model.safetensors").write_bytes(b"w")
        (ck / f"global_step_{n}" / ".uploaded").write_text("")
        if up == "not":
            (ck / f"global_step_{n}" / ".not_uploaded").write_text("")
    (env.root / "runs" / "E12d-X" / "final_step").write_text("100\n")
    assert c.dr_keep_auto("E12d-X") == 2
    k = tmp_path / "opd-dr" / "keep" / "E12d-X" / "ckpt"
    assert (k / "global_step_100" / "actor" / "huggingface" / "model.safetensors").exists()
    assert (k / "global_step_50" / "actor" / "huggingface_bf16" / "model.safetensors").exists() and not (k / "global_step_25").exists()
    assert [e[1] for e in evs] == ["dr_kept", "dr_kept"]
    monkeypatch.setenv("DR_CAP_GB", "0"); evs.clear()
    assert c.dr_keep_auto("E12d-X") == 0 and evs[0][1] == "dr_cap"
    monkeypatch.setenv("DR", "0"); assert c.dr_keep_auto("E12d-X") == 0


@pytest.mark.parametrize("cmd,ok", [("persist-env", True), ("persist-env now", False), ("rebuild-drill", True), ("rebuild-drill x", False)])
def test_rules_persist(cmd, ok):
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_do_script_receipt(env, monkeypatch):
    """v2.5.7a: persist-env / rebuild-drill run their script and write the receipt from its JSON result."""
    c = env.c; got = {}; _quiet(c, monkeypatch, got)
    os.makedirs(os.path.join(c.ROOT, "logs"), exist_ok=True)
    s = os.path.join(c.RUNBOOK, "scripts", "rebuild_drill.sh")
    open(s, "w").write('echo \'{"steps": {"clone": {"seconds": 3, "ok": true}}, "seconds_total": "3", "pass": "true"}\' > $OPD_ROOT/logs/rebuild_drill.json\n')
    c.do_script(120, "rebuild_drill.sh", "rebuild_drill.json")
    assert got["st"] == "ok" and got["result"]["seconds_total"] == "3" and "clone" in got["detail"]
    open(s, "w").write('echo \'{"result": "FAIL", "reason": "venv copy failed"}\' > $OPD_ROOT/logs/rebuild_drill.json\n')
    c.do_script(121, "rebuild_drill.sh", "rebuild_drill.json")
    assert got["st"] == "failed" and "venv copy failed" in got["detail"]


def test_pulse_v259c_disks_held_gpus(env):   # [v2.5.9c] disk totals and our usage, a held GPU, a task's GPU list
    st = {"utc": "2026-10-21T08:00:00Z", "disk_free_gb": 20328.8, "disk_total_gb": 2251799.8, "disk_free_bytes": 20328812345678,
          "ckpt_disk_free_gb": 5024.1, "ckpt_disk_total_gb": 7696.6,
          "our_du": {"utc": "2026-10-21T07:45:00Z", "work_gb": 612.3, "ckpt_gb": 1100.1, "secs": 120, "timed_out": False},
          "held": {"task": "PROBE-2GPU", "gpus": [7], "since_utc": "2026-10-21T08:24:43Z"},
          "tasks": {"E12b-P5-16k": {"state": "running", "gpu": 2, "gpus": [2, 3], "kind": "train"},
                    "EVAL-x": {"state": "running", "gpu": 1, "gpus": [1, "x", 99]}}}
    p = env.c.build_pulse(st, "2, 90, 1000, 275040\n7, 0, 0, 275040\n")
    d0, d1 = p["disks"]
    assert (d0["free_gb"], d0["total_gb"], d0["free_bytes"], d0["our_gb"], d0["our_gb_utc"]) == (20328.8, 2251799.8, 20328812345678, 612.3, "2026-10-21T07:45:00Z")
    assert (d1["free_gb"], d1["total_gb"], d1["our_gb"]) == (5024.1, 7696.6, 1100.1)
    assert [g.get("held_for") for g in p["gpus"]] == [None, "PROBE-2GPU"]
    assert p["tasks"]["E12b-P5-16k"]["gpus"] == [2, 3] and p["tasks"]["EVAL-x"]["gpus"] == [1]
    p2 = env.c.build_pulse({"tasks": {}, "held": {"task": "bad;name", "gpus": [7]}}, "7, 0, 0, 275040\n")   # no our_du, a bad hold
    assert p2["disks"][0]["our_gb"] is None and "held_for" not in p2["gpus"][0]
    assert not env.c.leak_check(p)


def _hb_progress():   # [v2.5.9s] heartbeat.sh's progress(), from its embedded Python (between _hms and last)
    src = open(os.path.join(HERE, "heartbeat.sh")).read()
    i = src.index("def _hms(x):"); j = src.index("def last(pat, text):")
    g = {"re": __import__("re"), "time": __import__("time")}
    exec(src[i:j], g)
    return g["progress"]


Q9S = "version: 2\ne open EVAL-p\n#eval EVAL-p t1 m math500 4 0.6\n#eval EVAL-p t2 m aime24 16 0.6\n#eval EVAL-p t3 m aime25 16 0.6\n"


def test_progress_from_a_vllm_tqdm_line():
    prog = _hb_progress()
    lines = ["EVAL_DONE 10:00:00 t1", "EVAL_START 10:00:01 t2",
             "Processed prompts:  45%|\x1b[32m####5     \x1b[0m| 900/2000 [10:00<12:12,  1.50it/s, est. speed input: 100.0 toks/s]"]
    p = prog("EVAL-p", lines, Q9S)
    assert (p["done"], p["total"], p["unit"], p["rate"], p["eta_s"], p["part"]) == (900, 2000, "prompts", 1.5, 732, "2/3 t2")
    assert p["tqdm"].startswith("Processed prompts:") and "\x1b" not in p["tqdm"] and len(p["tqdm"]) <= 200 and p["utc"].endswith("Z")


def test_progress_seconds_per_item_and_without_tqdm():
    prog = _hb_progress()
    p = prog("EVAL-p", ["EVAL_START 10:00:01 t3", "x | 3/16 [00:40<, 13.33s/it]"], Q9S)
    assert (p["rate"], p["eta_s"], p["part"]) == (round(1 / 13.33, 3), round(13 * 13.33), "3/3 t3")   # no eta in the line: from the rate
    p = prog("EVAL-p", ["EVAL_START 10:00:01 t1", "LOCAL_FIRST result:runs/X ..."], Q9S)
    assert set(p) == {"part", "utc"} and p["part"] == "1/3 t1"
    assert prog("EVAL-p", ["loading weights"], Q9S) is None and prog("EVAL-zz", ["EVAL_START 1 tq"], Q9S) is None


def test_pulse_progress_public_form(env, monkeypatch):
    import copy
    st = copy.deepcopy(STATUS)
    pr = {"done": 900, "total": 2000, "unit": "prompts", "rate": 1.5, "eta_s": 732, "part": "2/3 ext32k4B_X4_s50_aime24_T0.6",
          "tqdm": "Processed prompts:  45%| 900/2000 [10:00<12:12,  1.50it/s]", "utc": "2026-10-21T16:00:00Z"}
    st["tasks"]["EVAL-a"] = {"state": "running", "step": 0, "progress": pr}
    st["tasks"]["EVAL-b"] = {"state": "running", "step": 0, "progress": dict(pr, tqdm="loading /mnt/shared9/someuser/m.bin", unit="BAD UNIT")}
    monkeypatch.setenv("OPD_ROOT", str(env.root))
    base = len(json.dumps(env.c.build_pulse(STATUS, GPU)))
    p = env.c.build_pulse(st, GPU)
    a, b = p["tasks"]["EVAL-a"]["progress"], p["tasks"]["EVAL-b"]["progress"]
    assert a["part"] == "2 of 3 ext32k4B_X4_s50_aime24_T0.6" and a["tqdm"] == "Processed prompts:  45%| 900 of 2000 [10:00<12:12,  1.50it per s]"
    assert (a["done"], a["total"], a["rate"], a["eta_s"], a["unit"]) == (900, 2000, 1.5, 732, "prompts")
    assert "tqdm" not in b and "unit" not in b and b["done"] == 900   # a path in the line: dropped; the numbers stay
    assert env.c.leak_check(p) == [] and len(json.dumps(p)) - base < 900   # two tasks with progress add < 900 bytes


def test_pulse_progress_keeps_vllm_speed_suffix(env, monkeypatch):   # [v2.5.9t] "toks/s" no longer drops the line
    import copy
    st = copy.deepcopy(STATUS)
    line = "Processed prompts:  45%| 900/2000 [10:00<12:12,  1.50it/s, est. speed input: 100.0 toks/s, output: 50.0 tokens/s]"
    st["tasks"]["EVAL-a"] = {"state": "running", "step": 0, "progress": {"done": 900, "tqdm": line}}
    st["tasks"]["EVAL-b"] = {"state": "running", "step": 0, "progress": {"done": 1, "tqdm": "speed 3.0 toks/s from a/b/c"}}
    monkeypatch.setenv("OPD_ROOT", str(env.root))
    p = env.c.build_pulse(st, GPU)
    assert p["tasks"]["EVAL-a"]["progress"]["tqdm"].endswith("input: 100.0 toks per s, output: 50.0 tokens per s]")
    assert "tqdm" not in p["tasks"]["EVAL-b"]["progress"] and env.c.leak_check(p) == []   # other slashes still drop it



def test_image_benches_v259z():   # [2.5.58] geo3k / mathvista are evaluation benches (gen_eval_vl.py); others still refused
    import control_rules as R
    R.check_eval_args(["local:models/Qwen3-VL-2B-Instruct", "geo3k", "4", "0.6", "16384", "no_think"])
    R.check_eval_args(["result:runs/E15-VLM-P5/global_step_50/actor/huggingface", "mathvista", "4", "0.6", "16384", "no_think"])
    with pytest.raises(R.Reject):
        R.check_eval_args(["local:models/Qwen3-VL-2B-Instruct", "chartqa", "4", "0.6", "16384", "no_think"])
    R.check_eval_args(["local:models/Qwen3-VL-2B-Instruct", "mathvision", "1", "0", "16384", "no_think"])   # [2.5.66]
    R.check_eval_args(["result:runs/E15-VLM-P5/global_step_50/actor/huggingface", "mathverse", "1", "0", "16384", "no_think"])


def test_every_queue_eval_bench_is_accepted_v259z39():   # [2.5.66] 10-22: 9z33's mathvision / mathverse rows were refused on the node
    import control_rules as R
    q = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tasks", "queue.txt")
    bad = sorted({f[4] for f in (l.split() for l in open(q)) if len(f) >= 5 and f[0] == "#eval" and f[4] not in R.BENCHES})
    assert not bad, f"#eval benches the node refuses: {bad} (add them to control_rules.BENCHES)"


def _task_sh_names():
    """[v2.5.9z11] the names task.sh dispatches: its explicit case labels, and per family (E10b-* ...) the regex it checks."""
    src = open(os.path.join(HERE, "task.sh")).read()
    names, fams = [], {}
    for m in re.finditer(r"^  ([A-Za-z0-9|._*-]+)\)(.*)$", src, re.M):
        lab, rest = m.group(1), m.group(2)
        if lab.endswith("-*"):
            r = re.search(r"\[\[ \$T =~ (\^\S+\$) \]\]", rest)
            fams[lab[:-2]] = r.group(1) if r else None
        elif lab != "*":   # the default: an unknown task
            names += lab.split("|")
    return names, fams


def test_task_sh_names_accepted_by_run_and_rerun(env):   # [v2.5.9z11] ops-lead 10-21: an allowlist entry was missed twice
    names, _ = _task_sh_names()
    assert len(names) > 40 and "PROBE-STOPMERGE-UNIT" in names and "PROBE-T32" in names
    no_run, no_rerun = [], []
    for i, t in enumerate(names):
        if t in R.QUEUE_ONLY:
            continue
        if env.c.v_run([t], 900 + i)[0] != "accepted":
            no_run.append(t)
        try:
            R.parse({"cmd": f"rerun {t}", "reason": "t"})
        except R.Reject:
            no_rerun.append(t)
    assert not no_run and not no_rerun, f"not accepted by run: {no_run}; by rerun: {no_rerun} (add them, or to control_rules.QUEUE_ONLY)"
    assert all(t in names for t in R.QUEUE_ONLY), "QUEUE_ONLY lists a name task.sh does not have"


def test_task_sh_family_regexes_match_v_run():   # [v2.5.9z11] E10b-* / E10d-* ...: the same regex in task.sh and in control.py's v_run
    _, fams = _task_sh_names()
    src = open(os.path.join(HERE, "control.py")).read()
    vr = src[src.index("def v_run("):src.index("def v_rerun(")]
    assert {"E10b", "E10c", "E10d", "E11b", "E12b", "E12c", "E12d", "EVAL"} <= set(fams)
    for fam, rx in fams.items():
        assert rx and f"{fam}-*) [[ $T =~ {rx} ]]" in vr, (fam, rx)


# ---------- [v2.5.9z12] queue reruns (#at / #after), the disk guard, the reserve ----------
def _q(env, text):
    q = env.root / "queue.txt"; q.write_text("version: 2\n" + text); return str(q)


def _done(env, t, mtime=None):
    d = env.root / "runs" / t; d.mkdir(parents=True, exist_ok=True); f = d / ".done"; f.write_text("")
    if mtime is not None:
        os.utime(f, (mtime, mtime))
    return f


def _events(env, kind):
    d = env.root / "control" / "events"
    return [json.load(open(d / n)) for n in sorted(os.listdir(d)) if n.endswith(f"_{kind}.json")] if d.exists() else []


def test_at_rerun_once_a_day_inside_its_window(env, monkeypatch):
    import calendar as cal
    q = _q(env, "#at 03:00 rerun PROBE-TERM4\n#at 03:00 rerun E10b-P5\n#at 25:00 rerun PROBE-TERM4\n#at 03:00 rerun EVAL-q\n")
    day = cal.timegm((2026, 10, 22, 3, 5, 0, 0, 0, 0))
    _done(env, "PROBE-TERM4"); _done(env, "E10b-P5")
    assert env.c.auto_reruns(q, now=day - 600) == []                       # 02:55: not yet
    out = env.c.auto_reruns(q, now=day)                                     # 03:05
    got = {(t, k) for t, k, _ in out}
    assert ("PROBE-TERM4", "at_rerun_ok") in got and not (env.root / "runs" / "PROBE-TERM4" / ".done").exists()
    assert (env.root / "runs" / "PROBE-TERM4" / ".done.prev_at20261022T0300").exists()
    assert ("E10b-P5", "at_rerun_rejected") in got                          # a finished training is never rerun
    assert ("EVAL-q", "at_rerun_rejected") in got                           # still queued: nothing to do that day
    assert len(out) == 3 and _events(env, "at_rerun_ok")                    # 25:00 is not a time
    _done(env, "PROBE-TERM4")
    assert env.c.auto_reruns(q, now=day + 600) == []                        # once a day
    assert env.c.auto_reruns(q, now=day + 86400 + 3 * 3600) == []           # the next day at 06:05: outside the 120-min window
    out = env.c.auto_reruns(q, now=day + 86400)                             # the next day at 03:05
    assert ("PROBE-TERM4", "at_rerun_ok") in {(t, k) for t, k, _ in out}


def test_at_rerun_of_a_running_task_waits_inside_the_window(env, monkeypatch):
    import calendar as cal
    q = _q(env, "#at 03:00 rerun PROBE-TERM4\n"); now = cal.timegm((2026, 10, 22, 3, 1, 0, 0, 0, 0)); _done(env, "PROBE-TERM4")
    monkeypatch.setattr(env.c, "running_pids", lambda t=None: [1] if t == "PROBE-TERM4" else [])
    assert env.c.auto_reruns(q, now=now) == []                              # running: not marked, tried again
    monkeypatch.setattr(env.c, "running_pids", lambda t=None: [])
    assert [k for _, k, _ in env.c.auto_reruns(q, now=now + 300)] == ["at_rerun_ok"]


def test_after_rerun_when_a_reading_lands_after_the_probe(env):
    q = _q(env, "#after EVAL-a rerun PROBE-TERM4\n#after EVAL-b rerun PROBE-TERM4\n#after EVAL-a rerun E10b-P5\n#after EVAL-a rerun NOT-A-TASK\n")
    t0 = time.time() - 1000
    _done(env, "PROBE-TERM4", t0); _done(env, "EVAL-b", t0 - 100); _done(env, "E10b-P5", t0)
    assert env.c.auto_reruns(q) == []                                       # EVAL-a not done; EVAL-b older than the probe
    _done(env, "EVAL-a", t0 + 100)
    out = {(t, k) for t, k, _ in env.c.auto_reruns(q)}
    assert ("PROBE-TERM4", "after_rerun_ok") in out and ("E10b-P5", "after_rerun_rejected") in out
    assert ("NOT-A-TASK", "after_rerun_refused") not in out                 # no .done for it: nothing to compare
    assert env.c.auto_reruns(q) == []                                       # the probe's .done moved; the refusal not repeated
    _done(env, "PROBE-TERM4", t0 + 200)                                     # the probe ran again: newer than both readings
    assert env.c.auto_reruns(q) == []
    _done(env, "EVAL-b", t0 + 300)                                          # another reading lands
    assert [(t, k) for t, k, _ in env.c.auto_reruns(q)] == [("PROBE-TERM4", "after_rerun_ok")]


def test_after_rerun_refused_by_the_rules(env):
    q = _q(env, "#after EVAL-a rerun E99-x\n"); t0 = time.time() - 100
    _done(env, "E99-x", t0); _done(env, "EVAL-a", t0 + 10)
    out = env.c.auto_reruns(q)
    assert [(t, k) for t, k, _ in out] == [("E99-x", "after_rerun_refused")] and (env.root / "runs" / "E99-x" / ".done").exists()
    assert env.c.auto_reruns(q) == []                                       # not repeated for the same reading


def test_disk_low(env, monkeypatch, tmp_path):
    monkeypatch.setenv("DISK_ALERT_GB", "0"); monkeypatch.setenv("CKPT_ROOT", str(tmp_path)); monkeypatch.setenv("CKPT_ALERT_GB", "0")
    assert env.c.disk_low() == ""
    monkeypatch.setenv("DISK_ALERT_GB", "1e12")
    assert env.c.disk_low() == "work"
    monkeypatch.setenv("CKPT_ALERT_GB", "1e12")
    assert env.c.disk_low() == "work ckpt"
    monkeypatch.setenv("CKPT_ROOT", str(tmp_path / "missing"))
    assert env.c.disk_low() == "work"


def test_reserve_note_and_alert(env):
    t0 = 1_800_000_000
    r = env.c.reserve_note(1, 2, ["EVAL-a"], [], now=t0)
    assert r["need"] == 4 and r["low_since"] == t0
    assert env.c.reserve_low(now=t0 + 300) is None                         # low for 5 min: not yet
    assert env.c.reserve_low(now=t0 + 660)["ready"] == 1                    # 11 min
    assert env.c.reserve_note(2, 1, ["EVAL-a", "EVAL-b"], [], now=t0 + 700)["low_since"] == t0   # still low (2 < 3): kept
    assert env.c.reserve_note(5, 1, [], [], now=t0 + 760)["low_since"] is None and env.c.reserve_low(now=t0 + 1400) is None
    env.c.reserve_note(0, 3, [], ["E10b-P5"], now=t0)
    assert env.c.reserve_low(now=t0 + 2000) is None                         # stale (> 15 min): no alert


def test_alerts_carry_reserve_low(env, tmp_path):
    now = time.time()
    json.dump({"utc": "x", "t": now, "ready": 0, "free_gpus": 2, "need": 4, "ready_tasks": [], "held_by_disk": ["E10b-P5"],
               "low_since": now - 900}, open(env.c._p("reserve.json"), "w"))
    s = tmp_path / "status.json"; s.write_text("{}")
    assert "reserve_low" in env.c.alerts(str(s))
    ev = _events(env, "reserve_low")
    assert ev and json.loads(ev[0]["detail"]) == {"ready": 0, "free_gpus": 2, "need": 4, "ready_tasks": [], "held_by_disk": ["E10b-P5"]}
    json.dump({"t": now, "low_since": None}, open(env.c._p("reserve.json"), "w"))
    assert "reserve_low" not in env.c.alerts(str(s)) and _events(env, "reserve_low_cleared")


@pytest.mark.parametrize("kv,ok", [("GPUS_HOLD_MIN=45", True), ("GPUS_SOON_MIN=1440", True), ("GPUS_SOON_MIN=", True), ("GPUS_HOLD_MIN=0", False),
                                   ("GPUS_HOLD_MIN=1441", False), ("GPUS_SOON_MIN=abc", False), ("GPUS_FOO=1", False)])
def test_setenv_gpu_hold_v2512a(kv, ok):   # [2.5.61] the hold's minutes
    if ok:
        R.parse({"cmd": f"setenv {kv}", "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": f"setenv {kv}", "reason": "t"})


def test_run_tasks_reads_the_hold_from_control_env_v2512a(tmp_path):   # [v2.5.9z12a] a setenv applies without a loop restart
    src = open(os.path.join(HERE, "run_tasks.sh")).read()
    fn = next(l for l in src.splitlines() if l.startswith("cenv() {"))
    (tmp_path / "control.env").write_text("# x\nexport GPUS_HOLD_MIN=45\nexport VENV=fa\nexport GPUS_SOON_MIN=x9\n")
    import subprocess as sp
    out = sp.run(["bash", "-c", f'OPD_ROOT={tmp_path}; {fn}; echo "$(cenv GPUS_HOLD_MIN 30) $(cenv GPUS_SOON_MIN 30) $(cenv CPU_MAX 1)"'],
                 capture_output=True, text=True).stdout.split()
    assert out == ["45", "30", "1"]                         # a non-integer is ignored (the default)
    assert "$(cenv GPUS_HOLD_MIN ${GPUS_HOLD_MIN:-30})" in src and "$(cenv GPUS_SOON_MIN ${GPUS_SOON_MIN:-30})" in src


@pytest.mark.parametrize("soon,eta,held", [("1440", None, True), ("", None, False), ("30", None, False), ("30", 12, True), ("30", 45, False), ("1440", 900, True)])
def test_take_gpus_holds_with_soon_1440_without_eta_v2512a(tmp_path, soon, eta, held):   # [v2.5.9z12a] evaluations have no ETA
    src = os.path.join(HERE, "run_tasks.sh")
    (tmp_path / "logs").mkdir(); (tmp_path / "control").mkdir(); (tmp_path / "runs").mkdir()
    json.dump({"tasks": {"EVAL-x": {"state": "running", "eta_min": eta}, "Y": {"state": "done", "eta_min": 1}}}, open(tmp_path / "logs" / "status_n.json", "w"))
    (tmp_path / "control.env").write_text(f"export GPUS_SOON_MIN={soon}\n" if soon else "# none\n")
    (tmp_path / "q").write_text("#gpus A 2\n")
    import subprocess as sp
    script = (f'Q={tmp_path}/q OPD_ROOT={tmp_path}; unset GPUS_SOON_MIN GPUS_HOLD_MIN\n'
              f'eval "$(sed -n \'/^gpus_of() {{/p\' {src})"; eval "$(sed -n \'/^cenv() {{/p\' {src})"\n'
              f'eval "$(sed -n \'/^soon_free() {{/,/^}}$/p\' {src})"; eval "$(sed -n \'/^take_gpus() {{/,/; }}$/p\' {src})"\n'
              'FREE=(4); take_gpus A > /dev/null; echo "${#FREE[@]}"')
    out = sp.run(["bash", "-c", script], capture_output=True, text=True)
    assert out.stdout.strip() == ("0" if held else "1"), (out.stdout, out.stderr)


def test_pulse_assets_ready_pulling_and_leak_free(env):   # [v2.5.9z13] the monitor page's assets: repo ids as owner__name, no path
    r = env.root
    m = r / "models" / "Qwen3-32B"; m.mkdir(parents=True)
    (m / "a.safetensors").write_bytes(b"x" * 10)
    json.dump({"repo": "Qwen/Qwen3-32B", "kind": "model", "prefix": "", "files": [["a.safetensors", 1500000000, "0" * 64], ["b", 500000000, "1" * 64]],
               "utc": "2026-10-21T20:27:54Z"}, open(m / ".fetch_ok", "w"))
    h = r / "data" / "hf" / "AI4Math__MathVista"; h.mkdir(parents=True)
    json.dump({"repo": "AI4Math/MathVista", "kind": "dataset", "files": [["x.parquet", 1754088814, "2" * 64]], "utc": "2026-10-21T20:37:13Z"}, open(h / ".fetch_ok", "w"))
    g = r / "data" / "git" / "xirui-li__ClawEnvKit"; g.mkdir(parents=True); (g / "f.py").write_bytes(b"y" * 2221858)
    (g / ".fetch_ok").write_text("e7003442030e15c5e85f4c5f28a5e54774c6a21a\n")
    p = r / "data" / "hf" / "MathLLMs__MathVision"; p.mkdir(parents=True); (p / "test.parquet.incomplete").write_bytes(b"z")
    e = r / "models" / "Qwen3-14B"; e.mkdir(); (e / ".fetch_ok").write_text("")              # an old empty marker: a ready model, size by walk
    (r / "data" / "hf" / "bad name").mkdir()                                                  # not a repo-shaped name: left out
    a = env.c.assets_view()
    assert a["Qwen__Qwen3-32B"] == {"state": "已就位", "at": "2026-10-21T20:27:54Z 2.00 GB"}
    assert a["AI4Math__MathVista"] == {"state": "已就位", "at": "2026-10-21T20:37:13Z 1.75 GB"}
    assert a["xirui-li__ClawEnvKit"]["state"] == "已就位" and a["xirui-li__ClawEnvKit"]["at"].endswith(" 0.00 GB")
    assert a["MathLLMs__MathVision"] == {"state": "拉取中", "at": None} and a["Qwen3-14B"]["state"] == "已就位"
    assert all("/" not in k and "bad" not in k for k in a)
    p0 = env.c.build_pulse({}, "")
    assert p0["assets"] == a and env.c.leak_check(p0["assets"]) == []


def test_pulse_survives_assets_failure_once_event(env, monkeypatch):   # [v2.5.9z13] (ops-lead's A1 condition a)
    def boom():
        raise OSError("listdir failed")
    monkeypatch.setattr(env.c, "assets_view", boom)
    p = env.c.build_pulse({}, "")
    assert p["assets"] is None and p["assets_dropped"] is None and "gpus" in p and env.c.leak_check(p) == []
    env.c.build_pulse({}, "")                                                 # still failing: no second event
    ev = lambda k: [n for n in os.listdir(env.c._p("events")) if n.endswith(f"_{k}.json")] if os.path.isdir(env.c._p("events")) else []
    assert len(ev("assets_view_failed")) == 1
    monkeypatch.setattr(env.c, "assets_view", lambda: {"Qwen__Qwen3-32B": {"state": "已就位", "at": "2026-10-21T20:27:54Z 65.54 GB"}})
    p = env.c.build_pulse({}, "")
    assert p["assets"] == {"Qwen__Qwen3-32B": {"state": "已就位", "at": "2026-10-21T20:27:54Z 65.54 GB"}} and len(ev("assets_view_ok")) == 1


def test_pulse_drops_a_leaking_asset_entry(env, monkeypatch):   # [v2.5.9z13] (ops-lead's A1 condition b)
    tok = sorted(env.c._forbidden_tokens())[0]
    bad = f"x-{tok}"
    assert env.c.leak_check({bad: {}})
    monkeypatch.setattr(env.c, "assets_view", lambda: {"AI4Math__MathVista": {"state": "已就位", "at": "t 1.75 GB"}, bad: {"state": "拉取中", "at": None}})
    p = env.c.build_pulse({}, "")
    assert p["assets"] == {"AI4Math__MathVista": {"state": "已就位", "at": "t 1.75 GB"}} and p["assets_dropped"] == 1 and env.c.leak_check(p) == []


@pytest.mark.parametrize("args,ok", [(["Qwen/Qwen3-4B", "mmlupro1k", "1", "0", "4096", "no_think"], True), (["Qwen/Qwen3-4B", "gpqa", "4", "0.6", "16384", "think"], True),
                                     (["Qwen/Qwen3-4B", "gpqa", "4", "0.5", "16384", "think"], False), (["Qwen/Qwen3-4B", "ifeval", "1", "0", "4096", "no_think"], True)])   # [2.5.92] ifeval
def test_eval_args_mc_benches_and_greedy_v259z15(args, ok):   # [2.5.62]
    if ok:
        R.check_eval_args(args)
    else:
        with pytest.raises(R.Reject):
            R.check_eval_args(args)


def test_run_queue_eval_greedy_and_mc_wait_v259z15():
    src = open(os.path.join(HERE, "run_queue_eval.sh")).read()
    assert '0) SAMP="--temperature 0 --top_p 1.0 --top_k 0";;' in src
    assert 'case $BENCH in gpqa) DS=Idavidrein__gpqa;; mmlupro1k) DS=TIGER-Lab__MMLU-Pro;; ifeval) DS=google__IFEval;; *) DS="";; esac' in src   # [v2.5.9z16]
    assert 'echo "WAIT_INPUTS $T: $DS not in (no .fetch_ok)"; exit 1' in src


def test_setenv_pt32_max_hours_v259z40():   # [2.5.67] PROBE-T32's HOLD limit, set after the reviewer's yes
    import control_rules as R
    R.parse({"cmd": "setenv PT32_MAX_HOURS=30", "reason": "t"})
    with pytest.raises(R.Reject):
        R.parse({"cmd": "setenv PT32_MAX_HOUR=30", "reason": "t"})


def _eval_outputs(lines):
    """{<task>/<tag>_<bench>_<mode>_t<T>.jsonl} of the #eval rows, as gen_eval.py / gen_eval_vl.py name them."""
    out = set()
    for f in (l.split() for l in lines):
        if len(f) >= 7 and f[0] == "#eval":
            mode = f[9] if len(f) > 9 else (f[8] if len(f) > 8 and not f[8].isdigit() else "no_think")
            out.add(f"{f[1]}/{f[2]}_{f[4]}_{mode}_t{float(f[6])}.jsonl")
    return out


def test_term_rows_name_an_eval_output_v259z38():   # [v2.5.9z38] the reviewer 10-22: 9z38's TERM4 rows named e10b_P5mix7575_s50_...
    q = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tasks", "queue.txt")
    lines = open(q).read().splitlines()
    outs, evals = _eval_outputs(lines), {l.split()[1] for l in lines if l.startswith("#eval ")}
    bad = [f[-1] for f in (l.split() for l in lines) if len(f) >= 5 and f[0] == "#term" and f[-1].endswith(".jsonl")
           and f[-1].split("/")[0] in evals and f[-1] not in outs]
    assert not bad, f"#term rows naming no #eval output: {bad}"


def test_run_and_rerun_accept_v259z43(env):   # [v2.5.9z43] (Q-058) E12b-P5-prog / -prog-up, their gate PROBE-PROG (rules 2.5.70)
    for i, t in enumerate(("E12b-P5-prog", "E12b-P5-prog-up", "PROBE-PROG")):
        st, msg, _ = env.c.v_run([t], 80 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E12b-P5-prog-down", "E12b-P5-progup", "PROBE-PROG2", "E12d-P5-prog", "E10b-P5-prog")):
        assert env.c.v_run([t], 90 + i)[0] == "rejected", t
    R.parse({"cmd": "rerun PROBE-PROG", "reason": "t"})
    with pytest.raises(R.Reject):
        R.parse({"cmd": "rerun PROBE-PROGX", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 70)


def test_prog_wiring_v259z43():   # [v2.5.9z43] task.sh routes the three names, PROBE-PROG takes the CPU lane, the uploader restarts at s25
    import subprocess
    task = open(os.path.join(HERE, "task.sh")).read(); rt = open(os.path.join(HERE, "run_tasks.sh")).read()
    assert "  E12b-P5-prog) bash $RUNBOOK/scripts/run_e12.sh $G $T;;" in task and "  E12b-P5-prog-up) bash $RUNBOOK/scripts/run_e12.sh $G $T;;" in task
    assert "  PROBE-PROG) bash $RUNBOOK/scripts/probe_prog.sh;;" in task
    line = next(x for x in rt.splitlines() if x.startswith("cpu_task() {"))
    out = subprocess.run(["bash", "-c", line + "\nfor t in PROBE-PROG PROBE-MIXP E12b-P5-prog EVAL-x; do cpu_task $t && printf \"$t \"; done"], capture_output=True, text=True).stdout
    assert out == "PROBE-PROG PROBE-MIXP "
    assert "        E12b-P5-prog|E12b-P5-prog-up) env KEEP_BF16=0 UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-25} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh" in rt


def test_queue_prog_stage_before_diff4b_v259z43():   # [v2.5.9z43] ops-lead: Q-058 before diff4b, after b259z41o; every task of the stage has rows or a branch
    q = open(os.path.join(HERE, "..", "tasks", "queue.txt")).read().splitlines()
    names = [x.split()[0] for x in q if x and not x.startswith("#") and len(x.split()) > 2 and x.split()[1] == "open"]
    # [10-23 ops-lead: the scheduler's reorder, a09e09e] Q-058 and Q-066 ahead of the 32K OOD readings, diff4b still after them
    # [10-23 22:0x UTC, the scheduler's v79] Q-066 ahead of the 8k-s44 readings and Q-058's readings (its arms already run)
    assert names.index("b259z60") < names.index("b259z41s") < names.index("b259z43p") < names.index("b259z43") < names.index("b259z41o") < names.index("diff4b")
    stage = next(x for x in q if x.startswith("b259z43 ")).split()[2:]
    evals = {x.split()[1] for x in q if x.startswith("#eval ")}
    for t in (s.split(":")[0] for s in stage):
        assert t in evals or t in ("E12b-P5-prog", "E12b-P5-prog-up"), t


def test_run_and_rerun_accept_v259z45(env):   # [v2.5.9z45] (Q-063) E10b-P5route-inv and its gate PROBE-ROUTE-INV (rules 2.5.71)
    for i, t in enumerate(("E10b-P5route-inv", "PROBE-ROUTE-INV")):
        st, msg, _ = env.c.v_run([t], 110 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E10b-P5route-inv2", "E10b-P5mix-inv", "PROBE-ROUTE-INVX", "E10d-P5route-inv")):
        assert env.c.v_run([t], 120 + i)[0] == "rejected", t
    R.parse({"cmd": "rerun PROBE-ROUTE-INV", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 71)


def test_run_and_rerun_accept_v259z82(env):   # [v2.5.9z82] (Q-072 / Q-055b) E10b-P5selfroute, E10c-P5tail-c50, PROBE-SELFROUTE (rules 2.5.90)
    for i, t in enumerate(("E10b-P5selfroute2", "E10c-P5tail-c60", "PROBE-SELFROUTE2", "E10d-P5selfroute")):
        assert env.c.v_run([t], 130 + i)[0] == "rejected", t
    for t in ("PROBE-SELFROUTE", "E10b-P5selfroute", "E10c-P5tail-c50"):
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 90)


def test_run_and_rerun_accept_v259z83(env):   # [v2.5.9z83] (Q-073) E10b-P5softmix50 and its gate PROBE-SOFTMIX (rules 2.5.91)
    for i, t in enumerate(("E10b-P5softmix50", "PROBE-SOFTMIX")):
        st, msg, _ = env.c.v_run([t], 140 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E10b-P5softmix60", "PROBE-SOFTMIX2", "E10d-P5softmix50")):
        assert env.c.v_run([t], 150 + i)[0] == "rejected", t
    R.parse({"cmd": "rerun PROBE-SOFTMIX", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 91)


def test_run_and_rerun_accept_v26_s0(env):   # [v2.6 S0] CORPUS-RECORD (rules 2.5.95): run, rerun after done; the CPU lane
    st, msg, _ = env.c.v_run(["CORPUS-RECORD"], 250)
    assert st == "accepted", msg
    R.parse({"cmd": "rerun CORPUS-RECORD", "reason": "t"})
    assert env.c.v_run(["CORPUS-RECORDS"], 251)[0] == "rejected"
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 95)
    assert " CORPUS-RECORD) return 0;;" in open(os.path.join(os.path.dirname(__file__), "run_tasks.sh")).read()


def test_run_and_rerun_accept_v259z85(env):   # [v2.5.9z85] (Q-079 / Q-072b / Q-077) the AKL series, selfroute-c100, the 0.6B window arms and their gates (rules 2.5.94)
    names = ("E10b-P5-akl2", "E10b-P5-fkl", "E10b-P5-dense-k256", "E10b-P5-akl-mu03", "E10b-P5-akl-mu07", "E10d-P5-akl", "E10b-P5self-akl",
             "E10b-P5selfroute-c100", "E12c-P5-8k", "E12c-P5-16k", "PROBE-DENSE-4B", "PROBE-E12C-8K", "PROBE-E12C-16K")
    for i, t in enumerate(names):
        st, msg, _ = env.c.v_run([t], 220 + i)
        assert st == "accepted", (t, msg)
    for t in ("PROBE-DENSE-4B", "PROBE-E12C-8K", "PROBE-E12C-16K"):
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    for i, t in enumerate(("E10b-P5-akl3", "E10b-P5-fkl2", "E12c-P5-32k", "PROBE-DENSE-8B", "E10d-P5-fkl")):
        assert env.c.v_run([t], 240 + i)[0] == "rejected", t
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 94)


def test_run_and_rerun_accept_v259z84(env):   # [v2.5.9z84] (Q-076) the 16K window's second starts (no rules change: the E12 families)
    for i, t in enumerate(("E12b-P5-16k-s44", "E12d-P5-16k-s43")):
        st, msg, _ = env.c.v_run([t], 200 + i)
        assert st == "accepted", (t, msg)
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    for i, t in enumerate(("E12b-P5-16k-s45", "E12d-P5-16k-s44")):
        assert env.c.v_run([t], 210 + i)[0] == "rejected", t


def test_run_and_rerun_accept_v259z84a(env):   # [v2.5.9z84a] (Q-074 / Q-075) E12b-P5tail-8k, E10d-P5tail and PROBE-TAIL-4B (rules 2.5.93)
    for i, t in enumerate(("E12b-P5tail-8k", "E10d-P5tail", "PROBE-TAIL-4B")):
        st, msg, _ = env.c.v_run([t], 180 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E12b-P5tail-32k", "E10d-P5tail-s44", "PROBE-TAIL-16B")):   # [v2.5.9z92] -16k / -s43 exist since z92   # [v2.5.9z91] PROBE-TAIL-8B exists since z91
        assert env.c.v_run([t], 190 + i)[0] == "rejected", t
    R.parse({"cmd": "rerun PROBE-TAIL-4B", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 93)


def test_task_sh_named_lines_not_shadowed():   # [v2.5.9z84b] E12b-P5tail-8k sat below E12b-*, whose name check exits 2 before any output
    """Every literal name on a branch of task.sh's main case reaches its own line: no earlier glob branch takes it (bash's own matching)."""
    import subprocess
    d = os.path.dirname(os.path.abspath(__file__))
    L = open(os.path.join(d, "task.sh")).read().split("\n")
    start = next(i for i, l in enumerate(L) if re.match(r'^case "?\$T"? in', l))
    end = next(i for i in range(start + 1, len(L)) if L[i] == "esac")
    pats = [(i + 1, m.group(1)) for i in range(start + 1, end) for m in [re.match(r"^  ([^ )#][^)]*)\)", L[i])] if m]
    names = [(ln, a) for ln, p in pats for a in p.split("|") if not re.search(r"[*?\[]", a)]
    assert len(pats) > 50 and len(names) > 50
    sh = "for n in " + " ".join(a for _, a in names) + '; do case $n in ' + " ".join(f"{p}) echo $n {ln};;" for ln, p in pats) + " esac; done"
    got = dict(l.split() for l in subprocess.run(["bash", "-c", sh], capture_output=True, text=True, check=True).stdout.split("\n") if l)
    bad = [(a, ln, got.get(a)) for ln, a in names if got.get(a) != str(ln)]
    assert not bad, bad   # (name, its line, the line bash takes)


def test_run_and_rerun_accept_v259z46(env):   # [v2.5.9z46] (Q-057) E16-X5 / -X5e and their gate PROBE-X5 (rules 2.5.72)
    for i, t in enumerate(("E16-X5", "E16-X5e", "PROBE-X5")):
        st, msg, _ = env.c.v_run([t], 130 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E16-X6", "E16-X5f", "PROBE-X5e", "E16-P5")):
        assert env.c.v_run([t], 140 + i)[0] == "rejected", t
    for c in ("rerun PROBE-X5", "rerun E16-X5", "rerun E16-X5e"):
        R.parse({"cmd": c, "reason": "t"})
    with pytest.raises(R.Reject):
        R.parse({"cmd": "rerun E16-X5f", "reason": "t"})
    R.check_eval_args(["local:models/Qwen2.5-Math-1.5B-Instruct", "math500", "4", "0.6", "3072"])   # the student's 4096 positions
    with pytest.raises(R.Reject):
        R.check_eval_args(["local:models/Qwen2.5-Math-1.5B-Instruct", "math500", "4", "0.6", "3000"])
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 72)


def test_run_and_rerun_accept_v259z47(env):   # [v2.5.9z47] E10b-P5mix60 (rules 2.5.73)
    st, msg, _ = env.c.v_run(["E10b-P5mix60"], 150)
    assert st == "accepted", msg
    for i, t in enumerate(("E10b-P5mix65", "E10b-P5mix60-s43", "E10b-P5mix6")):
        assert env.c.v_run([t], 160 + i)[0] == "rejected", t
    R.parse({"cmd": "rerun E10b-P5mix60", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 73)


def test_run_and_rerun_accept_v259z66(env):   # [v2.5.9z66] (Q-059) the two 8K cells and their gates (rules 2.5.81)
    for i, t in enumerate(("E10b-P5-8k", "E10b-P5smerge-8k", "PROBE-P5-8K", "PROBE-SMERGE-8K")):
        st, msg, _ = env.c.v_run([t], 170 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E10b-P5-16k", "E10b-P5smerge-4k", "E10b-P5-8k-s43", "PROBE-P5-16K")):
        assert env.c.v_run([t], 180 + i)[0] == "rejected", t
    for t in ("E10b-P5-8k", "E10b-P5smerge-8k", "PROBE-P5-8K", "PROBE-SMERGE-8K"):
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 81)


def test_run_and_rerun_accept_v259z68(env):   # [v2.5.9z68] (Q-068) the 8B self-distillation and its gate (rules 2.5.82)
    for i, t in enumerate(("E10e-P5self", "PROBE-8B")):
        st, msg, _ = env.c.v_run([t], 190 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E10e-P5", "E10e-P5self-s43", "PROBE-8B-2")):
        assert env.c.v_run([t], 195 + i)[0] == "rejected", t
    for t in ("E10e-P5self", "PROBE-8B"):
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 82)
    here = os.path.dirname(os.path.abspath(__file__))
    assert "Qwen/Qwen3-8B) JUDGE=$TEACHER;;" in open(os.path.join(here, "run_queue_eval.sh")).read()     # the 8B base judges TWR
    rt = open(os.path.join(here, "run_tasks.sh")).read()
    assert "training_task() { case $1 in E10[bcde]-*|" in rt and rt.count("E1(0[bcde]|2[bcde]|6)-") == 2   # trained alike: disk guard, restart
    assert 'r"E1(0[bcde]|2[bcde]|6)-"' in open(os.path.join(here, "heartbeat.sh")).read()               # kind train
    q = open(os.path.join(os.path.dirname(here), "tasks", "queue.txt")).read()
    assert "\n#gpus E10e-P5self 4\n" in q and "\n#gpus PROBE-8B 4\n" in q


def test_run_and_rerun_accept_v259z70(env):   # [v2.5.9z70] (Q-055) the 0.6B tail-swap arm
    st, msg, _ = env.c.v_run(["E10c-P5tail"], 200)
    assert st == "accepted", msg
    for i, t in enumerate(("E10c-P5tail-s43", "E10c-P5tail-fix4", "E10c-P5tails")):
        assert env.c.v_run([t], 205 + i)[0] == "rejected", t
    R.parse({"cmd": "rerun E10c-P5tail", "reason": "t"})
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "task.sh")).read()
    assert src.index("  E10c-P5tail) bash $RUNBOOK/scripts/run_arm.sh $G $T;;") < src.index("  E10c-*)")


def test_e12d_p5_16k_run_and_rerun_accepted():   # [v2.5.9z67] (Q-069)
    import re
    for c in ("rerun E12d-P5-16k", "run E12d-P5-16k"):
        R.parse({"cmd": c, "reason": "t"})
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "task.sh")).read()
    assert src.index("  E12d-P5-16k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;") < src.index("  E12d-*)")
    q = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks", "queue.txt")).read()
    assert re.search(r"^#gpus E12d-P5-16k 2$", q, re.M)                                     # two GPUs, as the 1.7B 16K arms


def test_e12d_p5_8k_run_and_rerun_accepted():   # [v2.5.9z60] (ops-lead's A1 condition: Runming is away 10-23, a night failure must be rerunnable)
    import re
    for c in ("rerun E12d-P5-8k", "run E12d-P5-8k"):
        R.parse({"cmd": c, "reason": "t"})
    assert re.search(r"E12\[cde\]-\[A-Za-z0-9\._-\]\+", open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "control_rules.py")).read())
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "task.sh")).read()
    assert src.index("  E12d-P5-8k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;") < src.index("  E12d-*)")


STUDENTS16 = {f"EVAL-gm-{x}" for x in ("17P5", "17P5self", "17P5rl", "17P5kl", "17P5l8k", "4BP5", "4BP5self", "4BP5rl", "4BX4", "06P5", "06P5rl")}


def test_ifeval_rules_and_rows_v259z16(env):   # [v2.5.9z16] (rules 2.5.92) bench ifeval, BUILD-VENV-IFEVAL, the 18 b259z15 rows
    st, msg, _ = env.c.v_run(["BUILD-VENV-IFEVAL"], 200)
    assert st == "accepted", msg
    assert env.c.v_run(["BUILD-VENV-IFEVAL2"], 201)[0] == "rejected"
    R.parse({"cmd": "rerun BUILD-VENV-IFEVAL", "reason": "t"})
    for v in ("1", "0", ""):
        R.parse({"cmd": f"setenv VENV_IFEVAL_REBUILD={v}", "reason": "t"})
    with pytest.raises(R.Reject):
        R.parse({"cmd": "setenv VENV_IFEVAL_REBUILD=2", "reason": "t"})
    for args in (["result:runs/E10b-P5/global_step_150/actor/huggingface", "ifeval", "1", "0", "4096", "no_think"], ["Qwen/Qwen3-4B", "ifeval", "1", "0", "16384", "think"]):
        R.check_eval_args(args)
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 92)
    assert re.search(r"^\S+ rules 2\.5\.92 \(bench ifeval", open(os.path.join(HERE, "..", "control", "CHANGES.md")).read(), re.M)   # [10-25] renumbered
    L = open(os.path.join(HERE, "..", "tasks", "queue.txt")).read().splitlines()
    i = next(k for k, l in enumerate(L) if l.startswith("b259z15  open "))
    tasks = L[i].split()[2:]
    j = next(k for k, l in enumerate(L) if l.startswith("b259z16  open "))   # [10-25] ops-lead's condition 2: after b259z83, before fill17
    z83 = next(k for k, l in enumerate(L) if l.startswith("b259z83 "))   # [v2.5.9z84a] later formal stages may sit between: after b259z83, right before fill17
    assert len(tasks) == 16   # [v2.5.9z97] 18 -> 16: EVAL-gm-4BX4 / -4Bi2507 dropped (research_program 3h 6)
    assert L[j] == "b259z16  open   BUILD-VENV-IFEVAL" and z83 < j and L[j + 1].startswith("fill17 ")
    assert " b259z16 b259z15" in next(l for l in L if l.startswith("#pad "))   # [10-25] Fable's A1: padding
    rows = [(k, l.split()) for k, l in enumerate(L) if l.startswith("#eval EVAL-gm-")]
    for t in tasks:
        mine = [(k, f) for k, f in rows if f[1] == t]
        ife = [(k, f) for k, f in mine if f[4] == "ifeval"]
        gp = [k for k, f in mine if f[4] == "gpqa"]
        mm = [f for k, f in mine if f[4] == "mmlupro1k"][0]
        think = mm[8] == "think"
        student = t in STUDENTS16   # [10-25] Fable's A1: the 11 student checkpoints also at 16K, right after their 4K row
        assert len(ife) == (2 if student else 1) and len(gp) == 1 and ife[0][0] == gp[0] + 1, t           # the ifeval row(s), right after the gpqa row
        if student:
            g = ife[1][1]
            assert ife[1][0] == ife[0][0] + 1 and g[2] == ife[0][1][2].replace("_4k", "_16k") and g[3] == ife[0][1][3], g   # the same checkpoint
            assert g[4:9] == ["ifeval", "1", "0", "16384", "no_think"], g
            R.check_eval_args(g[3:9])
        f = ife[0][1]
        R.check_eval_args(f[3:9])
        assert f[3] == mm[3] and f[5:8] == ["1", "0", "16384" if think else "4096"] and f[8] == mm[8], f
        assert f[2] == mm[2].replace("_mmlupro1k_T0_", "_ifeval_T0_") and f[2].endswith("_16k" if think else "_4k"), f
    assert sum(1 for t in tasks if [f for k, f in rows if f[1] == t and f[4] == "mmlupro1k"][0][8] == "think") == 3
    after = [l.split() for l in L if l.startswith("#after BUILD-VENV-IFEVAL rerun ")]
    assert [a[3] for a in after] == tasks and not [l for l in L if l.startswith("#eval EVAL-gm16-") and " ifeval " in l]
    rt, ts = open(os.path.join(HERE, "run_tasks.sh")).read(), open(os.path.join(HERE, "task.sh")).read()
    assert "|BUILD-VENV-IFEVAL) return 0;;" in rt and "  BUILD-VENV-IFEVAL) bash $RUNBOOK/scripts/build_venv_ifeval.sh;;" in ts
    assert '"BUILD-VENV-IFEVAL"' in open(os.path.join(HERE, "heartbeat.sh")).read()


@pytest.mark.parametrize("cmd,ok", [("pend-weight SFT-P1 283", True), ("pend-weight E10b-P2 150", True), ("pend-weight SFT-P1", False),
                                    ("pend-weight SFT-P1 x", False), ("pend-weight a/b 1", False), ("pend-weight SFT-P1 0", False)])
def test_rules_pend_weight(cmd, ok):   # [v2.5.9z88b] rules 2.5.96
    if ok:
        R.parse({"cmd": cmd, "reason": "t"})
    else:
        with pytest.raises(R.Reject):
            R.parse({"cmd": cmd, "reason": "t"})


def test_v_pend_weight_sources_and_refusals(env, monkeypatch):   # [v2.5.9z88b] SFT-P1's final_bf16, an arm's bf16 copy, never fp32
    c = env.c; got = []
    monkeypatch.setitem(sys.modules, "remote", types.SimpleNamespace(defer_weight=lambda src, dst: got.append((src, dst)) or 7))
    monkeypatch.setattr(c, "running_pids", lambda t: [])
    sft = os.path.join(c.ROOT, "runs", "SFT-P1"); os.makedirs(os.path.join(sft, "final_bf16"))
    open(os.path.join(sft, "final_step"), "w").write("283\n")
    open(os.path.join(sft, "final_bf16", "model.safetensors"), "w").write("w" * 10)
    st, msg, extra = c.v_pend_weight(["SFT-P1", "283"], 90)
    assert st == "ok" and got[-1] == (os.path.join(sft, "final_bf16"), "runs/SFT-P1/global_step_283/actor/huggingface") and extra["bytes"] == 10
    assert c.v_pend_weight(["SFT-P1", "100"], 91)[0] == "rejected"                    # not the final step: no bf16 source
    arm = os.path.join(c.ROOT, "runs", "E10b-P9"); fp32 = os.path.join(arm, "ckpt", "global_step_50", "actor", "huggingface")
    os.makedirs(fp32); open(os.path.join(fp32, "model.safetensors"), "w").write("w")
    assert c.v_pend_weight(["E10b-P9", "50"], 92)[0] == "rejected"                    # fp32 only: never queued
    bf = fp32 + "_bf16"; os.makedirs(bf); open(os.path.join(bf, "model.safetensors"), "w").write("ww")
    assert c.v_pend_weight(["E10b-P9", "50"], 93)[0] == "ok" and got[-1] == (bf, "runs/E10b-P9/global_step_50/actor/huggingface")
    monkeypatch.setattr(c, "running_pids", lambda t: [123])
    assert c.v_pend_weight(["E10b-P9", "50"], 94)[0] == "rejected" and len(got) == 2  # running: refused
    assert c.v_pend_weight(["E10b-X", "50"], 95)[0] == "rejected"                     # no run dir


def test_run_and_rerun_accept_v259z88f(env):   # [v2.5.9z88f] HEARTBEAT-UNSTICK (rules 2.5.97): run, rerun after done; the CPU lane
    st, msg, _ = env.c.v_run(["HEARTBEAT-UNSTICK"], 260)
    assert st == "accepted", msg
    R.parse({"cmd": "rerun HEARTBEAT-UNSTICK", "reason": "t"})
    assert env.c.v_run(["HEARTBEAT-UNSTICKS"], 261)[0] == "rejected"
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 97)
    assert " HEARTBEAT-UNSTICK) return 0;;" in open(os.path.join(os.path.dirname(__file__), "run_tasks.sh")).read()


@pytest.mark.parametrize("sandboxed,marker,want", [(True, True, "WAIT"), (False, True, "PASS"), (False, False, "PASS")])
def test_heartbeat_unstick(tmp_path, sandboxed, marker, want):   # [v2.5.9z88f] a fake /proc: the marker goes only once no sandbox is left
    rb = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    opd = tmp_path / "opd"; (opd / "control").mkdir(parents=True); proc = tmp_path / "proc"
    for pid, env in (("11", b"PATH=/bin\0HOME=/h\0"), ("12", b"DRYRUN_SB=/sb\0PATH=/bin\0" if sandboxed else b"PATH=/bin\0")):
        (proc / pid).mkdir(parents=True); (proc / pid / "environ").write_bytes(env)
    if marker:
        (opd / "control" / "watchdog_heartbeat.gave_up").write_text("x")
    (opd / "control" / "loop_heartbeat").write_text("0")
    e = {k: v for k, v in os.environ.items() if not k.startswith("DRYRUN_")}
    e.update(OPD_ROOT=str(opd), PROC_ROOT=str(proc), CKPT_ROOT=str(tmp_path / "ck"), HOME=str(tmp_path))
    r = subprocess.run(["bash", os.path.join(rb, "scripts", "heartbeat_unstick.sh")], env=e, capture_output=True, text=True, timeout=60)
    rep = json.load(open(opd / "runs" / "HEARTBEAT-UNSTICK" / "report.json"))
    assert rep["result"] == want and (opd / "runs" / "HEARTBEAT-UNSTICK" / ".done").exists(), (r.stdout, r.stderr)
    assert (opd / "control" / "watchdog_heartbeat.gave_up").exists() == (marker and sandboxed)


def test_run_and_rerun_accept_v259z90(env):   # [v2.5.9z90] (Q-074b) E12d-P5tail-8k: run, rerun when failed; task.sh above E12d-*
    st, msg, _ = env.c.v_run(["E12d-P5tail-8k"], 270)
    assert st == "accepted", msg
    R.parse({"cmd": "rerun E12d-P5tail-8k", "reason": "t"})
    ts = open(os.path.join(os.path.dirname(__file__), "task.sh")).read()
    assert ts.index("  E12d-P5tail-8k) ") < ts.index("  E12d-*) ")


def test_run_and_rerun_accept_v259z91(env):   # [v2.5.9z91] (Q-082 / Q-075b / Q-074c / Q-081) four arms and two four-GPU gates (rules 2.5.98)
    names = ("E10b-P5-t32", "E10e-P5tail", "E12b-P5tail-8k-s43", "E12e-P5self-8k", "PROBE-E12E-8K", "PROBE-TAIL-8B")
    for i, t in enumerate(names):
        st, msg, _ = env.c.v_run([t], 280 + i)
        assert st == "accepted", (t, msg)
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 98)
    here = os.path.dirname(os.path.abspath(__file__))
    ts = open(os.path.join(here, "task.sh")).read()
    assert ts.index("  E12b-P5tail-8k-s43) ") < ts.index("  E12b-*) ") and ts.index("  E10b-P5-t32) ") < ts.index("  E10b-*) ")
    assert "|E12[bcde]-|PROBE-2GPU$|PROBE-8B$|PROBE-E12E-8K$|PROBE-E12E-16K$|PROBE-TAIL-8B$)" in ts   # [v2.5.9z97] PROBE-E12E-16K too      # the four-GPU tasks take a GPU list
    rt = open(os.path.join(here, "run_tasks.sh")).read()
    assert "training_task() { case $1 in E10[bcde]-*|E12[bcde]-*|" in rt                 # trained alike: disk guard, restart
    assert "case $TASK in E12[bcde]-*) stats 0;; esac" in open(os.path.join(here, "upload_sync.sh")).read()
    q = open(os.path.join(os.path.dirname(here), "tasks", "queue.txt")).read()
    for g in ("PROBE-E12E-8K", "E12e-P5self-8k", "PROBE-TAIL-8B", "E10e-P5tail"):
        assert f"\n#gpus {g} 4\n" in q, g


def test_run_and_rerun_accept_v259z92(env):   # [v2.5.9z92] (Q-074d / Q-075c / Q-074f / Q-074e) the second offline batch, rules unchanged
    for i, t in enumerate(("E12b-P5tail-16k", "E10d-P5tail-s43", "E12c-P5tail-8k", "E12e-P5tail-8k")):
        st, msg, _ = env.c.v_run([t], 290 + i)
        assert st == "accepted", (t, msg)
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    here = os.path.dirname(os.path.abspath(__file__))
    ts = open(os.path.join(here, "task.sh")).read()
    assert ts.index("  E12b-P5tail-16k) ") < ts.index("  E12b-*) ") and ts.index("  E12c-P5tail-8k) ") < ts.index("  E12c-*) ")
    assert ts.index("  E10d-P5tail-s43) ") < ts.index("  E10d-*) ")
    q = open(os.path.join(os.path.dirname(here), "tasks", "queue.txt")).read()
    assert "\n#gpus E12b-P5tail-16k 4\n" in q and "\n#gpus E12e-P5tail-8k 4\n" in q
    st = [l for l in q.split("\n") if l.startswith(("b259z91 ", "b259z92 ", "b259z85 "))]
    assert [l.split()[0] for l in st] == ["b259z91", "b259z92", "b259z85"]   # the scheduler's order: z91, z92, then z85
    items = st[1].split()[2:]
    arms = [x for x in items if not x.startswith("EVAL-")]
    assert items[:len(arms)] == arms   # readings after every arm


def test_run_and_rerun_accept_v259z89(env):   # [v2.5.9z89] (Q-080) the VLM retry arms, their gates and the CPU-lane diagnostic (rules 2.5.99)
    names = ("E15-VLM-P2lo", "E15-VLM-P5tail-lo", "PROBE-VLM-COMPLETE", "PROBE-VLM-TAIL", "DIAG-VLM-TRUNC")
    for i, t in enumerate(names):
        st, msg, _ = env.c.v_run([t], 310 + i)
        assert st == "accepted", (t, msg)
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 99)
    here = os.path.dirname(os.path.abspath(__file__))
    ts = open(os.path.join(here, "task.sh")).read()
    assert "  E15-VLM-P2lo|E15-VLM-P5tail-lo) bash $RUNBOOK/scripts/run_vlm.sh $G $T;;" in ts
    assert "PTL_PROBE=PROBE-VLM-TAIL PTL_ARM=E15-VLM-P5tail-lo PTL_RUNNER=run_vlm.sh" in ts and "PTL_FINISHED=1 bash" in ts
    assert "DIAG-VLM-TRUNC) return 0;;" in open(os.path.join(here, "run_tasks.sh")).read()           # the CPU lane
    q = open(os.path.join(os.path.dirname(here), "tasks", "queue.txt")).read()
    st = [l.split()[0] for l in q.split("\n") if l.startswith(("b259z92 ", "b259z93 ", "b259z89 ", "b259z92b "))]
    assert st == ["b259z92", "b259z93", "b259z89", "b259z92b"]                                         # the scheduler 23:3x UTC: after z93, before z92b
    items = next(l for l in q.split("\n") if l.startswith("b259z89 ")).split()[2:]
    arms = [x for x in items if not x.startswith("EVAL-")]
    assert items[:len(arms)] == arms                                                                   # readings after the arms


def test_run_and_rerun_accept_v259z97(env):   # [v2.5.9z97] (Q-074i / Q-074j; research_program 3h) two chain arms, a 16K gate, rows dropped
    for i, t in enumerate(("E12d-P5tail-8k-s43", "E12e-P5tail-16k", "PROBE-E12E-16K")):
        st, msg, _ = env.c.v_run([t], 330 + i)
        assert st == "accepted", msg
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    assert R.RULES_VERSION == "2.5.101"
    here = os.path.dirname(os.path.abspath(__file__))
    ts = open(os.path.join(here, "task.sh")).read()
    assert ts.index("  E12d-P5tail-8k-s43) ") < ts.index("  E12d-*) ")
    assert "PROBE-E12E-16K$" in ts.split("\n")[9]                                                   # a #gpus list is allowed
    pe = open(os.path.join(here, "probe_e12var.sh")).read()
    assert "E12e:16k) ;;" in pe and "[ $L:$VAR = E12e:16k ] && ARM=E12e-P5tail-16k" in pe
    q = open(os.path.join(os.path.dirname(here), "tasks", "queue.txt")).read()
    for g in ("#gpus E12d-P5tail-8k-s43 2", "#gpus E12e-P5tail-16k 4", "#gpus PROBE-E12E-16K 4"):
        assert f"\n{g}\n" in q
    lines = q.split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("b259z93 "))   # the scheduler v135: after b259z93, before b259z89
    assert lines[i + 3].startswith("b259z89 ")
    assert lines[i + 1].startswith("b259z97  open   E12d-P5tail-8k-s43:PROBE-E12B-8K+E10d-P5tail-s43 ")
    assert lines[i + 2].startswith("# [held until Q-074d H1 holds") and " b259z97j open   PROBE-E12E-16K:" in lines[i + 2]
    assert not any(l.startswith("b259z97j") for l in lines)                                            # Q-074j stays held
    assert "E12e-P5tail-16k:PROBE-E12E-16K+PROBE-E12B-16K+E10e-P5tail" in lines[i + 2]
    assert "vlm32-p2lo" not in q and "vlm32-p5tail" not in q                                           # the VLM 32K rows (3h 4)
    assert "EVAL-gm-4BX4" not in q and "EVAL-gm-4Bi2507" not in q                                      # fill: X arms out (3h 6)
    assert "EVAL-vlm32-s50-mathverse" in q and "EVAL-gm-4BP5self" in q                                 # Q-051's 32K and the kept gm rows stay


def test_run_and_rerun_accept_v259z95(env):   # [v2.5.9z95] (Q-074h) the 4B 16K chain arm: its own stage right after b259z92 (the scheduler)
    st, msg, _ = env.c.v_run(["E12d-P5tail-16k"], 320)
    assert st == "accepted", msg
    R.parse({"cmd": "rerun E12d-P5tail-16k", "reason": "t"})
    here = os.path.dirname(os.path.abspath(__file__))
    ts = open(os.path.join(here, "task.sh")).read()
    assert ts.index("  E12d-P5tail-16k) ") < ts.index("  E12d-*) ")
    q = open(os.path.join(os.path.dirname(here), "tasks", "queue.txt")).read()
    assert "\n#gpus E12d-P5tail-16k 4\n" in q
    st = [l.split()[0] for l in q.split("\n") if l.startswith(("b259z92 ", "b259z95 ", "b259z93 "))]
    assert st == ["b259z92", "b259z95", "b259z93"]


def test_run_and_rerun_accept_v259z96a(env):   # [v2.5.9z96a] VERIFY-PRIV-LFS: the CPU lane, read-only (rules 2.5.100)
    st, msg, _ = env.c.v_run(["VERIFY-PRIV-LFS"], 330)
    assert st == "accepted", msg
    R.parse({"cmd": "rerun VERIFY-PRIV-LFS", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 100)
    here = os.path.dirname(os.path.abspath(__file__))
    assert "VERIFY-PRIV-LFS) return 0;;" in open(os.path.join(here, "run_tasks.sh")).read()
    assert "  VERIFY-PRIV-LFS) bash $RUNBOOK/scripts/verify_priv_lfs.sh;;" in open(os.path.join(here, "task.sh")).read()


def test_run_and_rerun_accept_v259z93(env):   # [v2.5.9z93] (Q-074g) the third start's tail arm and its 8K arm, rules unchanged
    for i, t in enumerate(("E10b-P5tail-s44", "E12b-P5tail-8k-s44")):
        st, msg, _ = env.c.v_run([t], 300 + i)
        assert st == "accepted", (t, msg)
        R.parse({"cmd": f"rerun {t}", "reason": "t"})
    here = os.path.dirname(os.path.abspath(__file__))
    ts = open(os.path.join(here, "task.sh")).read()
    assert ts.index("  E10b-P5tail-s44) ") < ts.index("  E10b-*) ") and ts.index("  E12b-P5tail-8k-s44) ") < ts.index("  E12b-*) ")
    q = open(os.path.join(os.path.dirname(here), "tasks", "queue.txt")).read()
    st = [l.split()[0] for l in q.split("\n") if l.startswith(("b259z92 ", "b259z93 ", "b259z92b ", "b259z85 "))]
    assert st == ["b259z92", "b259z93", "b259z92b", "b259z85"]   # Fable's order: Q-074g before the 32K rows
