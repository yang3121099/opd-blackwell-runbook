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

        def list_repo_tree(self, repo, path_in_repo, recursive=True):
            if self.ep in state["list_fail"]:
                raise OSError("HEAD failed")
            return tree(path_in_repo)

    def dl(repo, filename, local_dir, endpoint=None, token=None):
        state["calls"].append((endpoint, filename))
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
