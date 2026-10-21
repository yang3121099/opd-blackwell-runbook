"""Unit tests of fetch_model.py (runbook v2.5.4e): no network; the Hub API and hf_hub_download are replaced."""
import hashlib
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_model as F  # noqa: E402

W = b"weights" * 1000
C = b'{"a": 1}'
SHA_W = hashlib.sha256(W).hexdigest()


def tree(prefix):
    return [types.SimpleNamespace(path=f"{prefix}/config.json", size=len(C), lfs=None),
            types.SimpleNamespace(path=f"{prefix}/model.safetensors", size=len(W), lfs={"sha256": SHA_W}),
            types.SimpleNamespace(path=f"{prefix}", size=None, lfs=None)]   # a folder entry


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub
    state = {"list_fail": set(), "calls": [], "bad_weights": False}

    class Api:
        def __init__(self, endpoint=None, token=None):
            self.ep = endpoint
            state.setdefault("tokens", []).append((endpoint, token))

        def get_paths_info(self, repo, paths, repo_type=None):   # v2.5.7g: one file
            state["paths_info"] = paths; state.setdefault("repo_types", []).append(repo_type)
            return [types.SimpleNamespace(path=paths[0], size=len(W), lfs={"sha256": SHA_W})]

        def list_repo_tree(self, repo, path_in_repo, recursive=True, repo_type=None):
            state.setdefault("repo_types", []).append(repo_type)
            if self.ep in state["list_fail"]:
                raise OSError("HEAD failed")
            return tree(path_in_repo)

    def dl(repo, filename, local_dir, endpoint=None, token=None, repo_type=None):
        state["calls"].append((endpoint, filename)); state.setdefault("dl_repo_types", []).append(repo_type)
        state.setdefault("dl_tokens", []).append((endpoint, token))
        p = os.path.join(local_dir, filename); os.makedirs(os.path.dirname(p), exist_ok=True)
        data = C if filename.endswith(".json") else (b"x" * len(W) if state["bad_weights"] else W)
        open(p, "wb").write(data)
        return p

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    monkeypatch.setattr(F, "ENDPOINTS", ["https://huggingface.co", "https://hf-mirror.com"])
    return state


def test_first_endpoint_fails_second_completes(hub, tmp_path):
    hub["list_fail"].add("https://huggingface.co")
    r = F.fetch("o/r", "runs/X/global_step_40/actor/huggingface", str(tmp_path))
    assert r["ok"] and r["endpoint"] == "https://hf-mirror.com" and r["files"] == 2
    assert [e["error"] for e in r["errors"]] == ["OSError"] * F.TRIES
    assert os.path.isfile(tmp_path / "runs/X/global_step_40/actor/huggingface/.fetch_ok")
    assert all(t is False for ep, t in hub["tokens"] + hub["dl_tokens"] if "mirror" in ep)    # no token to the mirror
    assert all(t is None for ep, t in hub["tokens"] if ep == "https://huggingface.co")


def test_partial_directory_is_completed(hub, tmp_path):
    d = tmp_path / "runs/X/global_step_40/actor/huggingface"; d.mkdir(parents=True)
    (d / "config.json").write_bytes(C)
    (d / "model.safetensors").write_bytes(W[:100])     # a broken earlier download
    r = F.fetch("o/r", "runs/X/global_step_40/actor/huggingface", str(tmp_path))
    assert r["ok"] and [f for _, f in hub["calls"]] == ["runs/X/global_step_40/actor/huggingface/model.safetensors"]
    assert (d / "model.safetensors").read_bytes() == W


def test_sha_mismatch_is_not_ok(hub, tmp_path):
    hub["bad_weights"] = True
    r = F.fetch("o/r", "runs/X/s/actor/huggingface", str(tmp_path))
    assert not r["ok"] and {e["error"] for e in r["errors"]} == {"Incomplete"}
    assert not os.path.exists(tmp_path / "runs/X/s/actor/huggingface/.fetch_ok")


def test_cached_after_success(hub, tmp_path):
    assert F.fetch("o/r", "runs/X/s/actor/huggingface", str(tmp_path))["ok"]
    hub["calls"].clear(); hub["list_fail"].update(F.ENDPOINTS)    # network gone: the marker answers
    r = F.fetch("o/r", "runs/X/s/actor/huggingface", str(tmp_path))
    assert r["ok"] and r.get("cached") and hub["calls"] == []


def test_nothing_under_prefix(hub, monkeypatch, tmp_path):
    monkeypatch.setattr(F, "listing", lambda api, repo, prefix: [])
    r = F.fetch("o/r", "runs/none", str(tmp_path))
    assert not r["ok"] and {e["error"] for e in r["errors"]} == {"NoFiles"}


def test_hf_transfer_off_after_a_failure(hub, monkeypatch, tmp_path):
    from huggingface_hub import constants as K
    import huggingface_hub
    monkeypatch.setattr(K, "HF_HUB_ENABLE_HF_TRANSFER", True)
    real, n = huggingface_hub.hf_hub_download, {"i": 0}
    def flaky(*a, **k):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("hf_transfer: connection reset")
        return real(*a, **k)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", flaky)
    r = F.fetch("o/r", "runs/X/s/actor/huggingface", str(tmp_path))
    assert r["ok"] and r["hf_transfer"] is False and r["errors"][0].get("hf_transfer_off")


def test_single_file(hub, tmp_path):   # v2.5.7g: an archived generation file (PROBE-TERM)
    f = "runs/e11/e11_P0_math500_T0.6_math500_no_think_t0.6.jsonl"
    r = F.fetch("o/r", f, str(tmp_path))
    assert r["ok"] and r["files"] == 1 and hub["paths_info"] == [f] and (tmp_path / f).read_bytes() == W
    assert os.path.isfile(str(tmp_path / f) + ".fetch_ok") and F.fetch("o/r", f, str(tmp_path)).get("cached")


def test_dataset_repo_type(hub, monkeypatch, tmp_path):   # [v2.5.9e] --dataset: listing and downloads with repo_type "dataset"
    monkeypatch.setattr(F, "REPO_TYPE", "dataset")
    r = F.fetch("own/ds", "data", str(tmp_path))
    assert r["ok"] and set(hub["repo_types"]) == {"dataset"} and set(hub["dl_repo_types"]) == {"dataset"}
    assert json.load(open(tmp_path / "data" / ".fetch_ok"))["kind"] == "dataset"
    monkeypatch.setattr(F, "REPO_TYPE", None)
    r = F.fetch("own/m", "model", str(tmp_path / "m"))
    assert r["ok"] and hub["repo_types"][-1] is None and json.load(open(tmp_path / "m" / "model" / ".fetch_ok"))["kind"] == "model"


def test_dataset_flag_reaches_the_child(monkeypatch, tmp_path):   # [v2.5.9e] the download child gets --dataset too
    seen = {}

    class P:
        pid = 1

        def __init__(self, args, **kw):
            seen["args"] = args

        def communicate(self, timeout=None):
            return json.dumps({"ok": True, "files": 1}), None
    monkeypatch.setattr(F.subprocess, "Popen", P)
    monkeypatch.setattr(F, "REPO_TYPE", "dataset")
    F.locked_fetch("own/ds", "", str(tmp_path))
    assert seen["args"][-1] == "--dataset" and seen["args"][2] == "--child"


@pytest.fixture
def par(hub, monkeypatch):   # [v2.5.9q] FETCH_PARALLEL=1 with par_fetch.download replaced (it writes the right bytes)
    import par_fetch
    calls = []

    def fake(url, dst, size, conns, may_run=None, stop=None, **kw):
        calls.append({"url": url, "conns": conns, "size": size, "may_run": may_run(), "stop": stop()})
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(C if dst.endswith(".json") else W)
        return {"bytes": size, "segments": 1, "resumed": 0, "seconds": 0.0}
    monkeypatch.setattr(par_fetch, "download", fake)
    monkeypatch.setenv("FETCH_PARALLEL", "1")
    monkeypatch.delenv("FETCH_CONNS_DAY", raising=False); monkeypatch.delenv("FETCH_CONNS_NIGHT", raising=False)
    return calls


def test_parallel_by_day_defers_a_large_file(hub, par, monkeypatch, tmp_path):   # with a day cap set (the default has none)
    monkeypatch.setattr(F, "night", lambda t=None: False); monkeypatch.setenv("FETCH_DAY_MAX_BYTES", "100"); monkeypatch.setenv("FETCH_CONNS_DAY", "8")
    r = F.fetch("Qwen/Qwen3-1.7B", "m", str(tmp_path))
    assert r["ok"] is False and r["deferred"] is True and r["errors"][-1]["error"] == "Deferred"
    assert [c["size"] for c in par] == [len(C)] and par[0]["conns"] == 8   # the small file only, with the day's connections
    assert hub["calls"] == [] and not os.path.exists(tmp_path / "m" / ".fetch_ok")   # nothing through hf_hub_download


def test_parallel_at_night(hub, par, monkeypatch, tmp_path):
    monkeypatch.setattr(F, "night", lambda t=None: True); monkeypatch.setenv("FETCH_DAY_MAX_BYTES", "100")
    r = F.fetch("Qwen/Qwen3-1.7B", "m", str(tmp_path))
    assert r["ok"] is True and os.path.exists(tmp_path / "m" / ".fetch_ok") and hub["calls"] == []
    assert [c["conns"] for c in par] == [32, 32] and all(c["may_run"] and not c["stop"] for c in par)   # small file first
    assert par[0]["size"] == len(C) and "/Qwen/Qwen3-1.7B/resolve/main/m/config.json" in par[0]["url"]


def test_parallel_waits_for_uploads_and_counts_from_the_environment(hub, par, monkeypatch, tmp_path):
    monkeypatch.setattr(F, "night", lambda t=None: False); monkeypatch.setenv("FETCH_CONNS_DAY", "5")
    monkeypatch.setattr(F, "uploads_timing_out", lambda: True)
    r = F.fetch("Qwen/Qwen3-1.7B", "m", str(tmp_path))
    assert r["ok"] is True and [c["conns"] for c in par] == [5, 5] and not any(c["may_run"] for c in par)   # may_run says wait


def test_night_window_and_upload_check(tmp_path, monkeypatch):
    import calendar
    t = lambda h: calendar.timegm((2026, 10, 21, h, 30, 0))
    assert [F.night(t(h)) for h in (13, 14, 23, 0)] == [False, True, True, False]   # 21:30 / 22:30 / 07:30 / 08:30 Beijing
    monkeypatch.setenv("OPD_ROOT", str(tmp_path)); os.makedirs(tmp_path / "logs")
    assert F.uploads_timing_out() is False
    (tmp_path / "logs" / "heartbeat.err").write_text("x\nUPLOAD_TIMEOUT runs/A\n")
    assert F.uploads_timing_out() is True
    old = os.path.getmtime(tmp_path / "logs" / "heartbeat.err") - 7200
    os.utime(tmp_path / "logs" / "heartbeat.err", (old, old))
    assert F.uploads_timing_out() is False   # more than an hour ago


def test_by_day_as_at_night_by_default(hub, par, monkeypatch, tmp_path):   # [v2.5.9r] Runming 10-21 23:3x Beijing
    monkeypatch.setattr(F, "night", lambda t=None: False); monkeypatch.delenv("FETCH_DAY_MAX_BYTES", raising=False)
    monkeypatch.setenv("OPD_ROOT", str(tmp_path / "root"))
    r = F.fetch("Qwen/Qwen3-1.7B", "m", str(tmp_path))
    assert r["ok"] is True and [c["conns"] for c in par] == [32, 32] and not any(c["stop"] for c in par)   # no cap, no stop


def test_control_env_wins_over_the_environment(hub, par, monkeypatch, tmp_path):   # [v2.5.9r] setenv reaches the running loop
    os.makedirs(tmp_path / "root"); (tmp_path / "root" / "control.env").write_text("FETCH_CONNS_DAY=64\n")
    monkeypatch.setenv("OPD_ROOT", str(tmp_path / "root")); monkeypatch.setenv("FETCH_CONNS_DAY", "8")
    monkeypatch.setattr(F, "night", lambda t=None: False)
    assert F.conns_now() == 64 and F.day_max() == int(1e13)

