"""Unit tests of runbook v2.5.1: two repositories (remote.py routing, host names out of file names, the private
repository must be private), the new set keys, the path-free pulse and its leak check, receipts that keep cmd/reason.
CPU only, no network or credentials (huggingface_hub is replaced). Run: python -m pytest -q scripts/test_v251.py"""
import importlib
import json
import os
import socket
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
    for i, t in enumerate(("E12b-P5self", "E12b-P5self-s43", "E10b-P5self-s43", "E10d-P5self", "PROBE-P5SELF-4B", "E12c-P0", "E12c-P5", "PROBE-E12C", "E12b-P5-kl", "PROBE-E12B-KL", "E12b-P5-stop", "PROBE-E12B-STOP", "E12b-P5-8k", "E12b-P5-16k", "PROBE-E12B-8K", "PROBE-E12B-16K", "E10d-X1", "E10d-X2", "PROBE-X2507-X1", "PROBE-X2507-X2", "E10d-X4", "PROBE-X2507-X4", "E10b-P5lenS", "E10b-P5lenL", "E10b-P5lenR", "PROBE-LENSEL", "PROBE-2GPU", "PROBE-TOOL", "PROBE-NET", "E12b-P5-stop2", "PROBE-E12B-STOP2", "PROBE-NET2", "PROBE-ROLLOUT-U60", "PROBE-ROLLOUT-U85", "E10b-P5stopmerge", "PROBE-STOPMERGE", "E10b-P5tail", "PROBE-TAIL", "E10c-P5-polish", "PROBE-POLISH", "E10c-P5self", "PROBE-P5SELF-06B", "PROBE-TERM4-06B", "E12b-P5-r3", "PROBE-E12B-R3", "E12b-P5-16k-kl", "BUILD-VENV-VL", "PROBE-VLLM-VL")):   # [v2.5.8b] kl; [v2.5.8c] stop; [v2.5.8h] 8k / 16k
        st, msg, _ = env.c.v_run([t], 30 + i)
        assert st == "accepted", (t, msg)
    for i, t in enumerate(("E12c-P2", "E12c-P5self", "E12b-P2self", "PROBE-E12D", "E12b-P2-kl", "E12b-P2-stop", "PROBE-E12B-FOO", "E10d-P2self", "E10b-P5self-s44", "E12b-P5-32k", "PROBE-E12B-32K", "E10d-X3", "PROBE-X2507-X3", "E10d-X5", "PROBE-X2507-X5", "E10b-P5lenX", "E10c-P5lenS-fix4", "PROBE-LENSEL2", "E12b-P5-stop3", "E12b-P2-stop2")):
        assert env.c.v_run([t], 40 + i)[0] == "rejected", t


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


@pytest.mark.parametrize("cmd,ok", [("rerun FA-PROBE", True), ("rerun EVAL-bridge", True), ("rerun E10b-P2", True), ("rerun SFT-P1", True), ("rerun E10c-P4lowlr-fix4", True), ("rerun PROBE-TERM", True), ("rerun E10c-P4lowlr-dyn6144", True), ("rerun E12c-P0", True), ("rerun PROBE-E12C", True), ("rerun E12b-P5self", True), ("rerun PROBE-E12B-KL", True), ("rerun E12b-P5-kl", True), ("rerun PROBE-E12B-STOP", True), ("rerun PROBE-P5SELF-4B", True), ("rerun PROBE-X2507-X1", True), ("rerun PROBE-X2507-X4", True), ("rerun PROBE-LENSEL", True), ("rerun PROBE-2GPU", True), ("rerun PROBE-TOOL", True), ("rerun PROBE-NET", True), ("rerun PROBE-NET2", True), ("rerun PROBE-ROLLOUT-U75", True), ("rerun PROBE-STOPMERGE", True), ("rerun PROBE-TAIL", True), ("rerun PROBE-POLISH", True), ("rerun PROBE-ROLLOUT-U95", False), ("rerun PROBE-E12B-STOP2", True), ("rerun PROBE-X2507-X3", False), ("rerun PROBE-P5SELF-06B", True), ("rerun PROBE-E12B-R3", True), ("rerun PROBE-E12B-R4", False), ("rerun E12b-P5-r3", True), ("rerun E12b-P5-16k-kl", True), ("rerun BUILD-VENV-VL", True), ("rerun PROBE-VLLM-VL", True), ("rerun PROBE-VLLM", False), ("rerun PROBE-E12B-16K-KL", False), ("rerun PROBE-TERM4-06B", True), ("rerun PROBE-TERM4-06", False), ("rerun PROBE-P5SELF-06", False), ("rerun E10c-P5self", True), ("rerun E10d-P5self", True), ("rerun E12b-P5-stop", True), ("rerun /etc", False),
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
    monkeypatch.setattr(c.subprocess, "run", lambda a, **k: types.SimpleNamespace(returncode=0))     # pgrep finds a build
    assert c.v_fa_build([], 91)[0] == "rejected" and not started
    monkeypatch.setattr(c.subprocess, "run", lambda a, **k: types.SimpleNamespace(returncode=1))
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

