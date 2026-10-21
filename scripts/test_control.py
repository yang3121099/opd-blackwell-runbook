"""Unit tests of scripts/control.py (ops-1; CPU, Linux; no GPU, network or credentials).
Run: python -m pytest -q scripts/test_control.py"""
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = "hf_" + "A1b2C3d4" * 5


@pytest.fixture
def env(tmp_path, monkeypatch):
    rb, root, remote = tmp_path / "rb", tmp_path / "root", tmp_path / "remote"
    (rb / "scripts").mkdir(parents=True); (rb / "control").mkdir(); root.mkdir(); remote.mkdir()
    (rb / "scripts" / "remote.py").write_text(
        "import shutil, sys, os\nsrc, dst = sys.argv[2], os.path.join(%r, sys.argv[3])\n"
        "shutil.copytree(src, dst) if os.path.isdir(src) else (os.makedirs(os.path.dirname(dst), exist_ok=True), shutil.copy(src, dst))\n" % str(remote))
    (rb / "scripts" / "verify.sh").write_text("sleep 1\n")
    subprocess.run(["git", "init", "-q", str(rb)], check=True)
    subprocess.run(["git", "-C", str(rb), "remote", "add", "origin", "https://github.com/yang3121099/opd-blackwell-runbook.git"], check=True)
    monkeypatch.setenv("OPD_ROOT", str(root)); monkeypatch.setenv("RUNBOOK", str(rb)); monkeypatch.setenv("OPD_NODE", "testnode")
    monkeypatch.setenv("HF_REPO", "someone/results"); monkeypatch.setenv("UPLOAD_BACKEND", "hf")
    sys.path.insert(0, HERE)
    import control
    importlib.reload(control)
    os.makedirs(control.CDIR, exist_ok=True)
    return types.SimpleNamespace(c=control, rb=rb, root=root, remote=remote)


def write_cmds(e, cmds, node="testnode"):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())   # v2.5: reason and utc on every entry (ctl.py writes both)
    lines = [f"node: {node}", "commands:"] + [f"  - {{seq: {s}, cmd: {json.dumps(c)}, reason: test, utc: '{now}'}}" for s, c in cmds]
    (e.rb / "control" / "testnode.yaml").write_text("\n".join(lines) + "\n")


def results(e):
    p = os.path.join(e.c.CDIR, "log.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def test_redact_patterns():
    r = __import__("control").redact
    assert r(f"x {TOKEN} y") == "x hf_*** y"
    assert r("key ms-1234abcd-12ab-34cd-56ef-1234567890ab") == "key ms-***"
    assert r("API_KEY=" + "Zx9" * 15) == "API_KEY=***"
    p = "TRITON_PTXAS_PATH=/a/very/long/path/that/is/more/than/forty/characters/bin/ptxas"
    assert r(p) == p   # paths are not secrets


def test_apply_once_and_in_order(env):
    write_cmds(env, [(1, "pause T1"), (2, "prio T1 5"), (3, "setenv VLLM_USE_FLASHINFER_SAMPLER=0")])
    env.c.apply(); env.c.apply()
    res = results(env)
    assert [r["seq"] for r in res] == [1, 2, 3] and all(r["status"] == "ok" for r in res)
    assert open(os.path.join(env.c.CDIR, "applied_seq")).read() == "3"
    assert os.path.exists(env.root / "runs" / "T1" / ".paused") and (env.root / "runs" / "T1" / ".prio").read_text().strip() == "5"
    assert "export VLLM_USE_FLASHINFER_SAMPLER=0" in (env.root / "control.env").read_text()
    write_cmds(env, [(1, "pause T1"), (2, "prio T1 5"), (3, "setenv VLLM_USE_FLASHINFER_SAMPLER=0"), (4, "resume T1"),
                     (5, "setenv VLLM_USE_FLASHINFER_SAMPLER=")])
    env.c.apply()
    assert [r["seq"] for r in results(env)] == [1, 2, 3, 4, 5]
    assert not os.path.exists(env.root / "runs" / "T1" / ".paused")
    assert "VLLM_USE_FLASHINFER_SAMPLER" not in (env.root / "control.env").read_text()


def test_seq_must_increase(env):
    write_cmds(env, [(1, "pause A"), (3, "pause B"), (2, "pause C"), (3, "pause D"), (4, "pause E")])
    env.c.apply()
    st = {r["seq"]: r["status"] for r in results(env) if r["status"] == "ok"}
    assert st == {1: "ok", 3: "ok", 4: "ok"}
    assert sum(r["status"] == "rejected" for r in results(env)) == 2
    assert not os.path.exists(env.root / "runs" / "C" / ".paused") and not os.path.exists(env.root / "runs" / "D" / ".paused")


def test_node_and_origin_mismatch(env):
    write_cmds(env, [(1, "pause X")], node="othernode")
    env.c.apply()
    assert results(env) == [] and env.c.status()["node_mismatch"] == "othernode"
    write_cmds(env, [(1, "pause X")])
    subprocess.run(["git", "-C", str(env.rb), "remote", "set-url", "origin", "https://github.com/evil/fork.git"], check=True)
    env.c.apply()
    assert results(env) == [] and env.c.status()["origin_mismatch"].endswith("evil/fork.git")


def test_setenv_pull_settings(env):   # [2.5.50] the parallel pulls' connections and day cap (Runming 10-21 23:3x Beijing)
    for cmd in ("setenv FETCH_CONNS_NIGHT=64", "setenv FETCH_CONNS_DAY=64", "setenv FETCH_DAY_MAX_BYTES=1e13", "setenv VENV_VL_REBUILD=1"):   # [2.5.55] VENV_VL_REBUILD
        status, msg, _ = env.c.run_one(cmd, 1)
        assert status == "ok", (cmd, status, msg)
    txt = (env.root / "control.env").read_text()
    assert "FETCH_CONNS_NIGHT=64" in txt and "FETCH_CONNS_DAY=64" in txt and "FETCH_DAY_MAX_BYTES=1e13" in txt


@pytest.mark.parametrize("cmd", ["reboot", "pause T; rm -rf /", "pause $(id)", "pause ../x", "pause", "prio T x",
                                 "setenv PATH=/tmp", "setenv LD_PRELOAD=/x.so", "setenv HF_TOKEN=abc", "setenv VLLM_X=a b",
                                 "setenv FETCH_CONNS_DAY=500", "setenv FETCH_CONNS_NIGHT=x", "setenv FETCH_DAY_MAX_BYTES=lots",   # [2.5.50]
                                 "setenv VENV_VL_REBUILD=yes",   # [2.5.55]
                                 "snapshot /etc", "snapshot ../../", "restore T runs/OTHER/global_step_20", "kill T extra"])
def test_rejected(env, cmd):
    status, msg, _ = env.c.run_one(cmd, 1)
    assert status == "rejected", (cmd, status, msg)
    assert not (env.root / "control.env").exists()


def test_snapshot_confined_and_redacted(env):
    logs = env.root / "logs"; logs.mkdir()
    (logs / "a.log").write_text(f"step 1\ntoken {TOKEN}\n")
    (logs / "hf_token.txt").write_text("secret")
    (logs / "big.log").write_text("x" * (6 << 20))
    (logs / "w.safetensors").write_text("weights")
    os.symlink("/etc/hostname", logs / "outside.log")
    status, msg, _ = env.c.run_one("snapshot logs", 7)
    assert status == "ok", msg
    up = env.remote / "logs" / "testnode" / "requests" / "7"
    assert (up / "logs" / "a.log").read_text() == "step 1\ntoken hf_***\n"
    man = json.load(open(up / "MANIFEST.json")); skipped = dict(man["skipped"])
    assert skipped["logs/hf_token.txt"] == "secret-like name" and skipped["logs/big.log"] == "larger than 5 MB"
    assert skipped["logs/w.safetensors"] == "weights/data" and skipped["logs/outside.log"] == "link outside $OPD_ROOT"
    os.symlink("/etc", env.root / "etc_link")
    assert env.c.run_one("snapshot etc_link", 8)[0] == "rejected"


def _fake_hf(monkeypatch, files):
    class F:
        def __init__(self, path, data, lfs):
            self.path, self.size = path, len(data)
            self.lfs = {"sha256": hashlib.sha256(data).hexdigest()} if lfs else None
            self.blob_id = None if lfs else hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    store = {p: d for p, d, _ in files}
    meta = [F(p, d, l) for p, d, l in files]
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = lambda: types.SimpleNamespace(list_repo_tree=lambda repo, path_in_repo, recursive, expand: meta)

    def dl(repo, path, local_dir):
        out = os.path.join(local_dir, path); os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(store[path]); return out
    mod.hf_hub_download = dl
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return store


def test_restore_verified_resume_state(env, monkeypatch):
    base = "runs/T/global_step_20/actor/"
    _fake_hf(monkeypatch, [(base + "huggingface/model.safetensors", b"w" * 100, True), (base + "huggingface/config.json", b"{}", False),
                           (base + "optim_world_size_1_rank_0.pt", b"o" * 50, True)])
    status, msg, _ = env.c.run_one("restore T runs/T/global_step_20", 3)
    assert status == "ok", msg
    d = env.root / "runs" / "T" / "ckpt"
    assert (d / "global_step_20" / "actor" / "huggingface" / "config.json").read_text() == "{}"
    assert (d / "latest_checkpointed_iteration.txt").read_text() == "20"


def test_restore_weights_only_and_mismatch(env, monkeypatch):
    base = "runs/T/global_step_40/actor/"
    store = _fake_hf(monkeypatch, [(base + "huggingface/model.safetensors", b"w" * 100, True)])
    assert env.c.run_one("restore T runs/T/global_step_40", 4)[0] == "ok"
    assert not (env.root / "runs" / "T" / "ckpt" / "latest_checkpointed_iteration.txt").exists()   # no resume state
    store[base + "huggingface/model.safetensors"] = b"x" * 100       # corrupted download, same size
    status, msg, _ = env.c.run_one("restore T runs/T/global_step_40", 5)
    assert status == "failed" and "sha256" in msg
    assert not os.path.exists(os.path.join(env.c.CDIR, "restore_5"))
    assert (env.root / "runs" / "T" / "ckpt" / "global_step_40" / "actor" / "huggingface" / "model.safetensors").read_bytes() == b"w" * 100


KILL_OK = os.environ.get("OPD_ALLOW_KILL_TEST") == "1"   # only on a machine with no training (ops-1 amendment 2)


def _fake_task(tmp_path, name):
    bindir = tmp_path / "bin"; bindir.mkdir(exist_ok=True)
    # a child that escapes into its own session, like a Ray daemon, with /tmp/ray_gpu0 in its command line
    (bindir / "task.sh").write_text("setsid bash -c 'exec -a \"raylet --session-dir=/tmp/ray_gpu0/x\" sleep 300' &\nsleep 300 &\nwait\n")
    return subprocess.Popen(["bash", str(bindir / "task.sh"), "0", name], start_new_session=True)


def _pids_with(text):
    r = subprocess.run(["pgrep", "-f", text], capture_output=True, text=True)
    return {int(x) for x in r.stdout.split()}


@pytest.mark.skipif(not KILL_OK, reason="kills processes: set OPD_ALLOW_KILL_TEST=1 on a machine without training")
def test_kill_and_running_guards(env, tmp_path):
    decoy = subprocess.Popen(["bash", "-c", 'exec -a "raylet --session-dir=/tmp/ray_gpu0/decoy" sleep 300'], start_new_session=True)
    p = _fake_task(tmp_path, "KT"); time.sleep(1)
    escaped = _pids_with("ray_gpu0/[x]") 
    assert env.c.run_one("rerun_verify", 9)[0] == "rejected"                      # a task is running
    assert env.c.run_one("restore KT runs/KT/global_step_20", 10)[0] == "rejected"
    status, msg, extra = env.c.run_one("kill KT", 11)
    assert status == "ok" and "gpu_clear" in extra, msg
    p.wait(timeout=40); time.sleep(1)
    assert not env.c.running_pids("KT")
    assert escaped and not (_pids_with("ray_gpu0/[x]") & escaped), "the task's escaped daemon survived"
    assert decoy.poll() is None, "a process outside the task's tree was killed"
    decoy.kill()
    assert (env.root / "runs" / "KT" / ".paused").exists() and (env.root / "runs" / "KT" / ".killed").exists()
    env.c.run_one("resume KT", 12)
    assert not (env.root / "runs" / "KT" / ".killed").exists()


def test_suspect_and_status(env):
    write_cmds(env, [(1, "setenv VLLM_ATTENTION_BACKEND=FLASH_ATTN")]); env.c.apply()
    d = env.root / "runs" / "E12b-P2"; d.mkdir(parents=True)
    (d / ".failed").write_text(f"attempts: 1\nlast: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\nEngineCore failed to start.\n")
    s = env.c.status()
    assert s["node"] == "testnode" and s["applied_seq"] == 1 and s["control_env"] == {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}
    assert s["control_env_suspect"] and s["control_env_suspect"][0]["task"] == "E12b-P2"
    assert s["recent"][-1]["status"] == "ok"
