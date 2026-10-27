"""[v2.5.9z96a] VERIFY-PRIV-LFS's check on synthetic files: ok, missing, a size mismatch, a sha mismatch; the shipped CSV's shape."""
import csv
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import verify_lfs_csv as V  # noqa: E402


def test_the_four_outcomes():
    root = tempfile.mkdtemp(); os.makedirs(os.path.join(root, "runs", "EVAL-a"))
    data = b"x" * 2_000_000
    open(os.path.join(root, "runs", "EVAL-a", "ok.jsonl"), "wb").write(data)
    open(os.path.join(root, "runs", "EVAL-a", "bad.jsonl"), "wb").write(data)
    open(os.path.join(root, "runs", "EVAL-a", "short.jsonl"), "wb").write(data[:1_000_000])
    sha = hashlib.sha256(data).hexdigest()
    src = os.path.join(root, "c.csv")
    with open(src, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tier", "task", "filename", "size_MB", "sha256", "git_oid", "pushedAt"])
        w.writerow(["A", "EVAL-a", "runs/EVAL-a/ok.jsonl", "2.0", sha, "g", "t"])
        w.writerow(["B", "EVAL-a", "runs/EVAL-a/bad.jsonl", "2.0", "0" * 64, "g", "t"])
        w.writerow(["B", "EVAL-a", "runs/EVAL-a/short.jsonl", "2.0", sha, "g", "t"])
        w.writerow(["C", "EVAL-a", "runs/EVAL-a/gone.jsonl", "2.0", sha, "g", "t"])
    out = os.path.join(root, "r.json"); sys.argv = ["x", src, root, out]; V.main()
    r = json.load(open(out))
    assert r["counts"] == {"ok": 1, "missing": 1, "size": 1, "sha": 1} and r["result"] == "MISMATCH" and r["bytes_read"] == 4_000_000
    assert sorted(x["why"] for x in r["not_ok"]) == ["missing", "sha", "size"] and r["by_tier"]["B"] == {"ok": 0, "missing": 0, "size": 1, "sha": 1}


def test_the_shipped_csv():
    rows = list(csv.DictReader(open(os.path.join(HERE, "..", "tasks", "private_repo_lfs_20261027.csv"), encoding="utf-8")))
    assert len(rows) == 759 and all(r["filename"].startswith("runs/") for r in rows) and all(len(r["sha256"]) == 64 for r in rows)
