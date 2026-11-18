"""test_model_verify.py: a directory is the published model only with every weight file's sha256 (the list from a file here)."""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_verify as mv


def mkmodel(tmp_path, monkeypatch, tamper=None, drop=None):
    d = tmp_path / "Qwen3-x"; d.mkdir()
    files = {"model-00001-of-00002.safetensors": b"a" * 5000, "model-00002-of-00002.safetensors": b"b" * 3000, "config.json": b"{}"}
    sib = []
    for n, b in files.items():
        s = {"rfilename": n, "size": len(b)}
        if n.endswith(".safetensors"):
            s["lfs"] = {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
        sib.append(s)
        if n != drop:
            (d / n).write_bytes(b if n != tamper else b[:-1] + b"z")
    lst = tmp_path / "list.json"; lst.write_text(json.dumps({"siblings": sib}))
    monkeypatch.setenv("MODEL_VERIFY_LIST", str(lst))
    return str(d)


def test_published_copy_verifies(tmp_path, monkeypatch):
    assert mv.verify(mkmodel(tmp_path, monkeypatch), "Qwen/Qwen3-x")[0] == 0


def test_same_size_other_bytes_is_a_mismatch(tmp_path, monkeypatch):
    rc, msg = mv.verify(mkmodel(tmp_path, monkeypatch, tamper="model-00002-of-00002.safetensors"), "Qwen/Qwen3-x")
    assert rc == 1 and "sha256" in msg


def test_missing_file_is_a_mismatch(tmp_path, monkeypatch):
    assert mv.verify(mkmodel(tmp_path, monkeypatch, drop="model-00001-of-00002.safetensors"), "Qwen/Qwen3-x")[0] == 1


def test_no_list_is_unknown(tmp_path, monkeypatch):
    d = mkmodel(tmp_path, monkeypatch)
    monkeypatch.setenv("MODEL_VERIFY_LIST", str(tmp_path / "none.json"))
    assert mv.verify(d, "Qwen/Qwen3-x")[0] == 2
