"""VERIFY-PRIV-LFS (runbook [v2.5.9z96a]; Fable 10-27 02:5x UTC, private_repo_cleanup_20261027 6.4 step 1): read-only, the CPU lane.
Before any deletion of the private repository's 759 raw jsonl LFS objects is put to Runming, each one's node copy is checked:
for every row of tasks/private_repo_lfs_20261027.csv (tier, task, filename = the node path under $OPD_ROOT, size_MB, sha256 = the LFS
file oid, git_oid, pushedAt) the file must exist, have that size (to the CSV's 0.1 MB rounding) and that sha256 (the content hash the
LFS oid is). The two tier-E rows' repository path was nodes/<node>/runs/...; their node copy is runs/... (the CSV carries that).
Nothing is changed or deleted. report.json: counts (ok / missing / size / sha) overall and by tier, the bytes read, and every row that
is not ok (path, why, local size, local sha prefix). Usage: python verify_lfs_csv.py <csv> <root> <out.json>
"""
import csv
import hashlib
import json
import os
import sys
import time


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def check(row, root):
    p = os.path.join(root, row["filename"])
    if not os.path.isfile(p):
        return "missing", None, None
    size = os.path.getsize(p)
    if abs(size / 1e6 - float(row["size_MB"])) > 0.06:
        return "size", size, None
    sha = sha256_of(p)
    return ("ok" if sha == row["sha256"] else "sha"), size, sha


def main():
    src, root, out = sys.argv[1:4]
    t0 = time.time(); rows = list(csv.DictReader(open(src, encoding="utf-8")))
    res = {"rows": len(rows), "counts": {"ok": 0, "missing": 0, "size": 0, "sha": 0}, "by_tier": {}, "bytes_read": 0, "not_ok": []}
    for r in rows:
        why, size, sha = check(r, root)
        res["counts"][why] += 1
        t = res["by_tier"].setdefault(r["tier"], {"ok": 0, "missing": 0, "size": 0, "sha": 0}); t[why] += 1
        if size is not None and why in ("ok", "sha"):
            res["bytes_read"] += size
        if why != "ok":
            res["not_ok"].append({"tier": r["tier"], "path": r["filename"], "why": why, "local_bytes": size, "csv_MB": r["size_MB"],
                                  "local_sha": (sha or "")[:16], "csv_sha": r["sha256"][:16]})
    res["seconds"] = round(time.time() - t0, 1)
    res["result"] = "PASS" if res["counts"]["ok"] == len(rows) else "MISMATCH"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    c = res["counts"]
    print(f"VERIFY_LFS {c['ok']}/{len(rows)} ok, missing {c['missing']}, size {c['size']}, sha {c['sha']}; {res['bytes_read'] / 1e9:.1f} GB read in {res['seconds']} s")


if __name__ == "__main__":
    main()
