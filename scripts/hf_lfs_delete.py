#!/usr/bin/env python3
"""[v2.5.9z96b] Permanent LFS deletion from the private ops repository, for Runming to run himself (Fable private_repo_cleanup_20261027 §6;
Runming 10-27: "ops 私库 759 个原始 jsonl 全部 LFS 永久删除 ... 都在执行者节点核过文件在、sha256 对之后做，每批一条 events").
The executor wrote this script and does not run it with --execute: permanent deletion is the owner's own action. It is wired to
no task, verb or loop: the node never runs it on its own.

An object is deleted only when all of these hold:
  1. it is a row of the cleanup CSV (tier, task, filename = repository path, size_MB, sha256 = LFS oid, ...);
  2. the node's VERIFY-PRIV-LFS report (runbook z96a) lists no problem for that row, and the report covers every CSV row;
  3. HF's LFS listing has that sha256 with the CSV's git oid beside it, at that size (to the CSV's 0.1 MB rounding);
  4. no path in the current tree outside the CSV points at the same oid (a deleted oid breaks every pointer to it).
Anything else is kept and printed with the reason. Order A -> C -> B -> D -> E (Fable); one log line per tier batch: files, GB,
the HF answer (ok, or the status code and the server's message), the repository's usedStorage before and after and the GB freed
(HF may report the drop with a delay); one final usedStorage reading. The CSV may name tier E by its repository path
(nodes/<node>/runs/...) or by its node path (runs/..., the runbook copy); both are matched.

Without --execute nothing is deleted: the plan (per tier: files, GB, kept with reasons) is printed and written to --plan.
With --execute the plan is recomputed from a fresh listing and deleted tier by tier; the first error stops the run.
Needs huggingface_hub >= 0.27 (list_lfs_files / permanently_delete_lfs_files). Never reads or prints a token.
Usage: python hf_lfs_delete.py --repo <ops repo> --csv <cleanup csv> --report <report.json> [--tiers A,C,B,D,E] [--execute]
"""
import argparse, csv, json, os, re, sys, time

TIERS = "A,C,B,D,E"
CHUNK = 100   # objects per API call; one tier is one logged batch


def node_path(path):
    # tier E lives at nodes/<node>/runs/... in the repository and at runs/... on the node (z96a CSV)
    return re.sub(r"^nodes/[^/]+/", "", path)


def load_rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def load_report(path, rows):
    rep = json.load(open(path, encoding="utf-8"))
    if rep.get("rows") != len(rows):
        sys.exit(f"STOP: the node report covers {rep.get('rows')} rows, the CSV has {len(rows)}")
    return {(r["tier"], r["path"]): r["why"] for r in rep.get("not_ok", [])}


def plan(rows, bad_on_node, lfs, tree, tiers):
    """lfs: {sha256: (size, object)} from the LFS listing; tree: {path: sha256} of the current tree's LFS pointers."""
    csv_paths = {node_path(r["filename"]) for r in rows}
    out = {t: {"delete": [], "keep": []} for t in tiers}
    for r in rows:
        t = r["tier"].split("-")[0]
        if t not in out:
            continue
        oid, why = r["sha256"], None
        if (r["tier"], node_path(r["filename"])) in bad_on_node:
            why = "node: " + bad_on_node[(r["tier"], node_path(r["filename"]))]
        elif oid not in lfs:
            why = "not in the LFS listing"
        elif r.get("git_oid") and r["git_oid"] not in (lfs[oid][1].oid, lfs[oid][1].file_oid):
            why = "git oid on HF differs from the CSV"
        elif abs(lfs[oid][0] / 1e6 - float(r["size_MB"])) > 0.06:
            why = f"size on HF {lfs[oid][0]} B vs CSV {r['size_MB']} MB"
        else:
            other = sorted(p for p, o in tree.items() if o == oid and node_path(p) not in csv_paths)
            if other:
                why = f"oid also used by {other[0]}" + (f" (+{len(other) - 1})" if len(other) > 1 else "")
        out[t]["keep" if why else "delete"].append({"path": r["filename"], "oid": oid, "bytes": lfs.get(oid, (0,))[0],
                                                     **({"why": why} if why else {})})
    return out


def summary(p):
    lines = []
    for t, d in p.items():
        gb = sum(x["bytes"] for x in d["delete"]) / 1e9
        lines.append(f"tier {t}: delete {len(d['delete'])} files, {gb:.2f} GB; keep {len(d['keep'])}")
        lines += [f"  keep {x['path']}: {x['why']}" for x in d["keep"]]
    return lines


def used_storage(api, repo):
    try:
        info = api.model_info(repo, expand=["usedStorage"])
        return getattr(info, "used_storage", None) or getattr(info, "usedStorage", None)
    except Exception as e:   # a reading only; never blocks the batch
        return f"unread ({type(e).__name__})"


def hf_answer(e):
    """A failed request's status code and the first 200 characters of the server's message, token- and path-like strings scrubbed."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    msg = " ".join(str(getattr(e, "server_message", None) or "").split())
    msg = re.sub(r"hf_[A-Za-z0-9]{6,}|[Bb]earer\s+\S+|(?<![A-Za-z0-9_.-])/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+", "<scrubbed>", msg)[:200]
    return f"{code or type(e).__name__}" + (f" {msg}" if msg else "")


def freed(before, after):
    return round((before - after) / 1e9, 2) if isinstance(before, int) and isinstance(after, int) else None


def listings(api, repo):
    # an LFS listing entry carries two ids (oid, file_oid): the 64-hex one is the content sha256 the CSV and the tree use
    lfs = {}
    for f in api.list_lfs_files(repo, repo_type="model"):
        for x in (f.oid, f.file_oid):
            if isinstance(x, str) and len(x) == 64:
                lfs[x] = (f.size, f)
    tree = {}
    for f in api.list_repo_tree(repo, repo_type="model", recursive=True):
        lf = getattr(f, "lfs", None)
        sha = (lf.get("sha256") if isinstance(lf, dict) else getattr(lf, "sha256", None)) if lf else None
        if sha:
            tree[f.path] = sha
    return lfs, tree


def run(api, a):
    rows = load_rows(a.csv)
    bad = load_report(a.report, rows)
    tiers = a.tiers.split(",")
    lfs, tree = listings(api, a.repo)
    p = plan(rows, bad, lfs, tree, tiers)
    for line in summary(p):
        print(line)
    json.dump(p, open(a.plan, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if not a.execute:
        print(f"DRY RUN: nothing deleted; plan in {os.path.basename(a.plan)}")   # no node path on the screen
        return 0
    log = open(a.log, "a", encoding="utf-8")
    for t in tiers:
        objs = [lfs[x["oid"]][1] for x in p[t]["delete"]]
        if not objs:
            continue
        before, t0 = used_storage(api, a.repo), time.time()
        for i in range(0, len(objs), CHUNK):
            try:
                api.permanently_delete_lfs_files(a.repo, objs[i:i + CHUNK], rewrite_history=True, repo_type="model")
            except Exception as e:
                ev = {"event": "lfs_delete_failed", "tier": t, "deleted_before_error": i, "of": len(objs),
                      "hf": hf_answer(e), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                log.write(json.dumps(ev, ensure_ascii=False) + "\n"); log.flush()
                print("STOP:", json.dumps(ev, ensure_ascii=False))
                return 1
        after = used_storage(api, a.repo)
        ev = {"event": "lfs_deleted", "tier": t, "files": len(objs), "gb": round(sum(x["bytes"] for x in p[t]["delete"]) / 1e9, 2),
              "kept": len(p[t]["keep"]), "hf": "ok", "used_storage_before": before, "used_storage_after": after,
              "freed_gb": freed(before, after), "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        log.write(json.dumps(ev, ensure_ascii=False) + "\n"); log.flush()
        print(json.dumps(ev, ensure_ascii=False))
    ev = {"event": "lfs_delete_end", "used_storage": used_storage(api, a.repo), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log.write(json.dumps(ev, ensure_ascii=False) + "\n"); log.flush()
    print(json.dumps(ev, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--report", required=True, help="the node's VERIFY-PRIV-LFS report.json")
    ap.add_argument("--tiers", default=TIERS)
    ap.add_argument("--plan", default="hf_lfs_delete_plan.json")
    ap.add_argument("--log", default="hf_lfs_delete_events.jsonl")
    ap.add_argument("--execute", action="store_true", help="really delete (permanent); default is a dry run")
    a = ap.parse_args(argv)
    if not a.repo.endswith("-ops"):   # [ops-lead A1 10-27] never pointed at a weights / public repository by mistake
        sys.exit("STOP: --repo must be the private ops repository (a name ending in -ops)")
    import huggingface_hub
    if not hasattr(huggingface_hub.HfApi, "permanently_delete_lfs_files"):
        sys.exit(f"STOP: huggingface_hub {huggingface_hub.__version__} has no permanently_delete_lfs_files; needs >= 0.27")
    return run(huggingface_hub.HfApi(), a)


if __name__ == "__main__":
    sys.exit(main())
