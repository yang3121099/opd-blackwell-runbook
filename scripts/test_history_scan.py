"""test_history_scan.py: the search for earlier installs on a fake machine (two shared mounts, a user's code directory)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import history_scan as hs


def mk(root, files):
    for f, size in files.items():
        p = os.path.join(root, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b"x" * size)


def install(r, models=("Qwen3-1.7B",), venv=True, weights=("E10b-P5/global_step_150",)):
    mk(r, {"paths.env": 10, "control/x": 1, "runs/t/log": 1, "data/a.parquet": 5})
    for m in models:
        mk(r, {f"models/{m}/config.json": 10, f"models/{m}/model.safetensors": 1000001})
    mk(r, {"models/Qwen3-8B/config.json": 10})   # incomplete: no weights
    if venv:
        mk(r, {"venv/bin/python": 1})
    for w in weights:
        mk(r, {f"adhoc_models/result/runs/{w}/actor/huggingface/config.json": 1})


def machine(tmp_path):
    a, b = str(tmp_path / "share_a"), str(tmp_path / "share_b")
    install(os.path.join(a, "alice", "code", "opd-run"))
    install(os.path.join(b, "alice", "code", "opd-run"), models=("Qwen3-1.7B", "Qwen3-4B"), venv=False, weights=())
    install(os.path.join(a, "bob", "deep", "er", "still", "deeper", "opd-run"))            # deeper than the search goes
    mk(a, {"alice/code/train.py": 1, "alice/code/models/big/x": 1, "alice/code/proj/opd-notes/readme": 1, "carol/opd-run/readme": 1})
    return a, b, [(a, "ceph"), (b, "ceph")]


def test_found_through_the_filler_path_and_on_the_other_mount(tmp_path, monkeypatch):
    a, b, mts = machine(tmp_path)
    monkeypatch.setattr(hs, "ROOT", str(tmp_path / "local" / "opd" / "run"))
    rep = hs.scan(mts, [os.path.join(a, "alice", "code", "train.py")], {})
    paths = [r["path"] for r in rep["roots"]]
    assert sorted(paths) == sorted(os.path.realpath(os.path.join(m, "alice", "code", "opd-run")) for m in (a, b)) and not rep["truncated"]
    ra = next(r for r in rep["roots"] if r["path"].startswith(os.path.realpath(a)))
    assert ra["models"] == ["Qwen3-1.7B"] and ra["venvs"] == ["venv"] and ra["weights"] == 1 and ra["data"] == 1
    s = hs.summary(rep)
    assert s["roots"] == 2 and s["models"] == ["Qwen3-1.7B", "Qwen3-4B"] and s["weights"] == 1


def test_own_root_and_plain_directories_are_not_installs(tmp_path, monkeypatch):
    a, b, mts = machine(tmp_path)
    mine = os.path.join(a, "alice", "code", "opd-run")
    monkeypatch.setattr(hs, "ROOT", mine)
    rep = hs.scan(mts, [os.path.join(a, "alice")], {})
    assert [r["path"] for r in rep["roots"]] == [os.path.realpath(os.path.join(b, "alice", "code", "opd-run"))]


def test_no_hint_searches_the_network_mounts_and_budget_cuts(tmp_path, monkeypatch):
    a, b, mts = machine(tmp_path)
    monkeypatch.setattr(hs, "ROOT", str(tmp_path / "local"))
    rep = hs.scan(mts, [], {})
    assert len(rep["roots"]) == 2 and rep["hints"] == 0
    rep = hs.scan(mts, [], {}, budget=1)
    assert rep["truncated"] and rep["dirs_listed"] <= 1


def test_written_files_are_private_and_merge_what_the_node_was_told(tmp_path, monkeypatch):
    a, b, mts = machine(tmp_path)
    root = str(tmp_path / "local" / "opd" / "run")
    monkeypatch.setattr(hs, "ROOT", root)
    told = os.path.join(a, "carol")
    rep = hs.scan(mts, [os.path.join(a, "alice", "code")], {"OLD_BASE_CANDS": told})
    hs.write(rep)
    env = open(os.path.join(root, "control", "history_secret.env")).read()
    assert env.startswith("export OLD_BASE_CANDS=" + os.path.realpath(told) + ":") and env.count("opd-run") == 2
    assert oct(os.stat(os.path.join(root, "control", "history_secret.json")).st_mode & 0o777) == "0o600"
    assert json.load(open(os.path.join(root, "control", "history_secret.json")))["roots"]


def test_hints_from_a_stored_filler_command(tmp_path, monkeypatch):
    a, b, mts = machine(tmp_path)
    root = str(tmp_path / "local" / "opd" / "run")
    mk(root, {"control/x": 1})
    open(os.path.join(root, "control", "gpu_fill_secret.cmd"), "w").write(f"python {a}/alice/code/train.py --duration_hours 888\n")
    monkeypatch.setattr(hs, "ROOT", root)
    assert os.path.join(a, "alice", "code", "train.py") in hs.hint_paths({}, [])
    assert os.path.join(b, "alice") in hs.hint_dirs(hs.hint_paths({}, []), mts)


def test_store_is_beside_a_root_on_the_mount_with_most_room(tmp_path, monkeypatch):
    a, b, mts = machine(tmp_path)
    monkeypatch.setattr(hs, "ROOT", str(tmp_path / "local" / "opd" / "run"))
    rep = hs.scan(mts, [os.path.join(a, "alice", "code")], {"HISTORY_STORE_MIN_GB": "0"})
    assert rep["store"] in (os.path.join(a, "alice", "code", "opd-store"), os.path.join(b, "alice", "code", "opd-store"))
    assert hs.scan(mts, [os.path.join(a, "alice", "code")], {"HISTORY_STORE_MIN_GB": "99999999"})["store"] is None
    assert hs.scan(mts, [os.path.join(a, "alice", "code")], {"OPD_STORE": "/x/y"})["store"] == "/x/y"
    install(os.path.join(b, "alice", "code", "opd-store"), venv=False)   # an existing store is a root and is taken
    rep = hs.scan(mts, [os.path.join(a, "alice", "code")], {"HISTORY_STORE_MIN_GB": "99999999"})
    assert rep["store"] == os.path.realpath(os.path.join(b, "alice", "code", "opd-store"))
    hs.write(rep)
    envf = os.path.join(str(tmp_path / "local" / "opd" / "run"), "control", "history_secret.env")
    assert "export OPD_STORE=" in open(envf).read()
    monkeypatch.setenv("HISTORY_DR_MIN_GB", "0"); hs.write(rep)
    assert "export OPD_STORE_DR=1" in open(envf).read()            # room for resume states: the store is the DR root too
    monkeypatch.setenv("HISTORY_DR_MIN_GB", "99999999"); hs.write(rep)
    assert "OPD_STORE_DR" not in open(envf).read()                 # a nearly full disk: the store only


def test_base_models_are_found_outside_installs_and_in_hf_caches(tmp_path, monkeypatch):
    a, b, mts = machine(tmp_path)
    root = str(tmp_path / "local" / "opd" / "run")
    monkeypatch.setattr(hs, "ROOT", root)
    mk(a, {"alice/pretrained/Qwen3-0.6B/config.json": 5, "alice/pretrained/Qwen3-0.6B/model.safetensors": 1000001,
           "alice/code/proj/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/abc/config.json": 5,
           "alice/code/proj/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/abc/model.safetensors": 1000001,
           "alice/half/Qwen3-8B/config.json": 5})                      # incomplete: not offered
    rep = hs.scan(mts, [os.path.join(a, "alice", "code", "train.py")], {})
    md = rep["model_dirs"]
    assert set(md) == {"Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"} and md["Qwen3-0.6B"][0].endswith("pretrained/Qwen3-0.6B")
    assert md["Qwen3-4B"][-1].endswith("snapshots/abc") and any(p.endswith("opd-run/models/Qwen3-1.7B") for p in md["Qwen3-1.7B"])
    assert hs.summary(rep)["models"] == ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"]
    hs.write(rep)
    import subprocess
    env = dict(os.environ, OPD_ROOT=root)
    r = subprocess.run([sys.executable, hs.__file__, "--model", "Qwen3-0.6B"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip().endswith("pretrained/Qwen3-0.6B")
    assert subprocess.run([sys.executable, hs.__file__, "--model", "Qwen3-8B"], capture_output=True, text=True, env=env).returncode == 1
