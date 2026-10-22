"""[v2.5.9y1] wheel_fetch.py with fake PyPI metadata and downloads (no network): the wheel picked, the proxy probe, par_fetch
only for a large wheel after a slow probe, sha256 checks, a file already there kept."""
import hashlib
import json
import os

import wheel_fetch as W


def test_pick_prefers_cp310_then_abi3_linux():
    urls = [{"packagetype": "sdist", "filename": "x-1.tar.gz"},
            {"packagetype": "bdist_wheel", "filename": "x-1-cp38-abi3-manylinux2014_aarch64.whl"},
            {"packagetype": "bdist_wheel", "filename": "x-1-cp38-abi3-manylinux1_x86_64.whl"},
            {"packagetype": "bdist_wheel", "filename": "x-1-cp311-cp311-manylinux_2_17_x86_64.whl"}]
    assert W.pick(urls)["filename"] == "x-1-cp38-abi3-manylinux1_x86_64.whl"
    urls.append({"packagetype": "bdist_wheel", "filename": "x-1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"})
    assert W.pick(urls)["filename"].startswith("x-1-cp310-cp310")
    assert W.pick([{"packagetype": "bdist_wheel", "filename": "y-2-py3-none-any.whl"}])["filename"] == "y-2-py3-none-any.whl"
    assert W.pick([{"packagetype": "bdist_wheel", "filename": "z-1-cp312-cp312-win_amd64.whl"}]) is None


def _env(tmp_path, sizes, slow=True):
    """Fake metadata / transports. sizes: {name: bytes}; content = name repeated; the probe (first) takes 30 s when slow (0.4 MB/s for 12 MB)."""
    data = {n: (n.encode() * (s // len(n) + 1))[:s] for n, s in sizes.items()}
    calls = []

    def meta(name, version):
        return {"name": name, "version": version, "file": f"{name}-{version}-py3-none-any.whl", "size": len(data[name]),
                "sha256": hashlib.sha256(data[name]).hexdigest(), "url": f"https://files.pythonhosted.org/p/{name}.whl"}

    def single(url, dst):
        calls.append(("single", url)); open(dst, "wb").write(data[url.split("/")[-1][:-4]])

    def par(url, dst, size, conns):
        calls.append(("par", url, conns)); open(dst, "wb").write(data[url.split("/")[-1][:-4]])
    t = {"now": 0.0}

    def clock():
        t["now"] += 30.0 if slow else 0.001
        return t["now"]
    return meta, single, par, clock, calls, data


def test_slow_probe_sends_the_large_wheel_through_par_fetch(tmp_path):
    meta, single, par, clock, calls, data = _env(tmp_path, {"transformers": 12_000_000, "vllm": 40_000_000, "mathruler": 15_000})
    r = W.fetch_all(str(tmp_path), ["transformers==4.57.1", "vllm==0.11.0", "mathruler==0.1.0"], single, par, meta, lambda: 64, clock)
    assert r["ok"] and r["probe_MBps"] < 1.0
    assert [c[0] for c in calls] == ["single", "single", "par"] and calls[2][2] == 64   # [v2.5.9z3] probe, then smallest first
    w = {x["name"]: x for x in r["wheels"]}
    assert w["vllm"]["conns"] == 64 and w["transformers"]["conns"] == 1 and w["vllm"]["host"] == "files.pythonhosted.org"
    man = {w["name"]: w for w in json.load(open(tmp_path / "manifest.json"))["wheels"]}
    assert man["vllm"]["sha256"] == hashlib.sha256(data["vllm"]).hexdigest()


def test_fast_probe_keeps_one_connection(tmp_path):
    meta, single, par, clock, calls, _ = _env(tmp_path, {"transformers": 12_000_000, "vllm": 40_000_000}, slow=False)
    r = W.fetch_all(str(tmp_path), ["transformers==4.57.1", "vllm==0.11.0"], single, par, meta, lambda: 64, clock)
    assert r["ok"] and r["probe_MBps"] > 1.0 and [c[0] for c in calls] == ["single", "single"]


def test_sha256_mismatch_fails_and_removes_only_that_file(tmp_path):
    meta, single, par, clock, calls, data = _env(tmp_path, {"transformers": 1000, "mathruler": 500})

    def bad_meta(name, version):
        m = meta(name, version)
        if name == "mathruler":
            m["sha256"] = "0" * 64
        return m
    r = W.fetch_all(str(tmp_path), ["transformers==4.57.1", "mathruler==0.1.0"], single, par, bad_meta, lambda: 8, clock)
    assert not r["ok"] and "sha256" in r["wheels"][1]["error"]
    assert not os.path.exists(tmp_path / "mathruler-0.1.0-py3-none-any.whl") and os.path.exists(tmp_path / "transformers-4.57.1-py3-none-any.whl")


def test_a_good_file_already_there_is_kept(tmp_path):
    meta, single, par, clock, calls, data = _env(tmp_path, {"transformers": 1000})
    open(tmp_path / "transformers-4.57.1-py3-none-any.whl", "wb").write(data["transformers"])
    r = W.fetch_all(str(tmp_path), ["transformers==4.57.1"], single, par, meta, lambda: 8, clock)
    assert r["ok"] and r["wheels"][0]["kept"] and calls == []


def test_a_download_error_is_reported(tmp_path):
    meta, single, par, clock, calls, data = _env(tmp_path, {"transformers": 1000})

    def boom(url, dst):
        raise OSError("proxy said no")
    r = W.fetch_all(str(tmp_path), ["transformers==4.57.1"], boom, par, meta, lambda: 8, clock)
    assert not r["ok"] and "proxy said no" in r["wheels"][0]["error"] and r["probe_MBps"] is None


def test_the_probe_is_a_small_wheel_even_when_listed_last(tmp_path):   # [v2.5.9z3] 10-21: vllm listed first was its own probe
    meta, single, par, clock, calls, data = _env(tmp_path, {"vllm": 40_000_000, "mathruler": 15_000, "transformers": 12_000_000})
    r = W.fetch_all(str(tmp_path), ["vllm==0.11.0", "mathruler==0.1.0", "transformers==4.57.1"], single, par, meta, lambda: 64, clock)
    assert [w["name"] for w in r["wheels"]] == ["transformers", "mathruler", "vllm"] and r["probe_MBps"] < 1.0
    assert [c[0] for c in calls] == ["single", "single", "par"]


def test_wheel_par_first_skips_the_probe(tmp_path, monkeypatch):   # [v2.5.9z34] a lone large pin straight through par_fetch
    meta, single, par, clock, calls, _ = _env(tmp_path, {"nvidia-cuda-nvrtc-cu12": 40_000_000})
    monkeypatch.setenv("WHEEL_PAR_FIRST", "1")
    r = W.fetch_all(str(tmp_path), ["nvidia-cuda-nvrtc-cu12==12.9.86"], single, par, meta, lambda: 16, clock)
    assert r["ok"] and [c[0] for c in calls] == ["par"] and calls[0][2] == 16 and r["probe_MBps"] is None and r["wheels"][0]["conns"] == 16
    monkeypatch.delenv("WHEEL_PAR_FIRST")
    meta, single, par, clock, calls, _ = _env(tmp_path / "b", {"nvidia-cuda-nvrtc-cu12": 40_000_000})
    r = W.fetch_all(str(tmp_path / "b"), ["nvidia-cuda-nvrtc-cu12==12.9.86"], single, par, meta, lambda: 16, clock)
    assert [c[0] for c in calls] == ["single"]   # without it: the lone wheel is its own one-connection probe, as before


def test_wheel_par_first_keeps_a_small_first_wheel_single(tmp_path, monkeypatch):
    meta, single, par, clock, calls, _ = _env(tmp_path, {"mathruler": 15_000})
    monkeypatch.setenv("WHEEL_PAR_FIRST", "1")
    r = W.fetch_all(str(tmp_path), ["mathruler==0.1.0"], single, par, meta, lambda: 16, clock)
    assert r["ok"] and [c[0] for c in calls] == ["single"]
