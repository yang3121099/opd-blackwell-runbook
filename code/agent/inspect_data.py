"""Describe a fetched dataset repository (runbook [v2.5.9c], git_fetch.sh): JSON on stdout, no absolute paths.

For each data file (.parquet / .json / .jsonl / .csv, at most 50, largest first): its path inside the repository, bytes,
sha256, number of records (parquet rows, a JSON list's length or a JSON object's keys, JSONL / CSV lines) and the first
record's fields with their types and a 300-character preview; the licence file's first non-empty lines with a guess of its
SPDX id. A parquet file needs pandas (the node's venv); without it the file is listed with "rows": null.
[v2.5.9e] --list-only (a code repository): the file count, bytes and the files per top-level folder, nothing parsed.
Usage: python inspect_data.py DIR [--repo owner/name] [--commit SHA] [--list-only]
"""
import argparse
import hashlib
import json
import os

EXT = (".parquet", ".json", ".jsonl", ".csv")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def preview(v):
    try:
        s = json.dumps(v, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(v)
    return s[:300]


def fields(rec):
    if isinstance(rec, dict):
        return {str(k): {"type": type(v).__name__, "preview": preview(v)} for k, v in rec.items()}
    return {"_value": {"type": type(rec).__name__, "preview": preview(rec)}}


def describe(path):
    d = {"bytes": os.path.getsize(path), "sha256": sha256(path), "rows": None, "first": None}
    try:
        if path.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(path)
            d["rows"] = len(df)
            if len(df):
                d["first"] = fields({c: (v.tolist() if hasattr(v, "tolist") else v) for c, v in df.iloc[0].items()})
        elif path.endswith(".jsonl"):
            with open(path) as f:
                lines = [x for x in f if x.strip()]
            d["rows"] = len(lines)
            d["first"] = fields(json.loads(lines[0])) if lines else None
        elif path.endswith(".json"):
            obj = json.load(open(path))
            if isinstance(obj, list):
                d["rows"] = len(obj); d["first"] = fields(obj[0]) if obj else None
            elif isinstance(obj, dict):
                d["rows"] = len(obj); k = next(iter(obj), None); d["first"] = {"_key": preview(k), **fields(obj[k])} if k is not None else None
        elif path.endswith(".csv"):
            with open(path, errors="ignore") as f:
                lines = f.read().splitlines()
            d["rows"] = max(len(lines) - 1, 0); d["first"] = {"header": preview(lines[0]) if lines else None}
    except Exception as e:   # described as far as it goes
        d["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return d


def licence(root):
    for name in sorted(os.listdir(root)):
        if name.upper().startswith(("LICENSE", "LICENCE", "COPYING")) and os.path.isfile(os.path.join(root, name)):
            lines = [x.strip() for x in open(os.path.join(root, name), errors="ignore") if x.strip()][:3]
            text = " ".join(lines)
            spdx = ("Apache-2.0" if "Apache License" in text else "MIT" if "MIT License" in text or "Permission is hereby granted" in text
                    else "CC-BY-4.0" if "Attribution 4.0" in text else None)
            return {"file": name, "first_lines": lines, "spdx_guess": spdx}
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--repo"); ap.add_argument("--commit")
    ap.add_argument("--list-only", action="store_true")
    a = ap.parse_args()
    if a.list_only:   # [v2.5.9e]
        top, n, b = {}, 0, 0
        for dp, dn, fn in os.walk(a.dir):
            dn[:] = [x for x in dn if x != ".git"]
            for f in fn:
                rel = os.path.relpath(os.path.join(dp, f), a.dir); k = rel.split(os.sep)[0] if os.sep in rel else "."
                top[k] = top.get(k, 0) + 1; n += 1; b += os.path.getsize(os.path.join(dp, f))
        print(json.dumps({"repo": a.repo, "commit": a.commit, "licence": licence(a.dir), "list_only": True, "files": n,
                          "total_bytes": b, "top": dict(sorted(top.items()))}, indent=1))
        return
    files = []
    for dp, dn, fn in os.walk(a.dir):
        dn[:] = [x for x in dn if not x.startswith(".")]
        for f in fn:
            if f.endswith(EXT) and not f.startswith("."):
                files.append(os.path.join(dp, f))
    files = sorted(files, key=lambda p: -os.path.getsize(p))[:50]
    out = {"repo": a.repo, "commit": a.commit, "licence": licence(a.dir),
           "data_files": {os.path.relpath(p, a.dir): describe(p) for p in sorted(files)},
           "total_bytes": sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(a.dir) for f in fn)}
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
