"""Tests for hf_lfs_delete.py against a fake HfApi: only node-verified, listing-matched, unshared objects are deleted,
the dry run deletes nothing, and the first API error stops the run. Run: python -m pytest scripts/test_hf_lfs_delete.py"""
import csv, hashlib, json, os, sys, types

sys.path.insert(0, os.path.dirname(__file__))
import hf_lfs_delete as H

FIELDS = ["tier", "task", "filename", "size_MB", "sha256", "git_oid", "pushedAt"]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def gid(s):
    return hashlib.sha1(s.encode()).hexdigest()


class Api:
    def __init__(self, lfs, tree, fail_at=None):
        self.lfs, self.tree, self.fail_at, self.deleted, self.calls = lfs, tree, fail_at, [], 0

    def list_lfs_files(self, repo, repo_type=None):
        return iter(self.lfs)

    def list_repo_tree(self, repo, repo_type=None, recursive=False):
        return iter(types.SimpleNamespace(path=p, lfs={"sha256": o}) for p, o in self.tree.items())

    def model_info(self, repo, expand=None):
        return types.SimpleNamespace(usedStorage=10**9 - 1000 * len(self.deleted))

    def permanently_delete_lfs_files(self, repo, objs, rewrite_history=True, repo_type=None):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            e = RuntimeError("forbidden"); e.response = types.SimpleNamespace(status_code=403)
            e.server_message = "Forbidden: token hf_" + "x" * 12 + " at /srv/a/b"
            raise e
        self.deleted += [o.file_oid for o in objs]


def obj(name, size, swapped=False):
    # as on the Hub (Fable 10-27): fileOid is the content sha256 the batch endpoint takes, oid the 40-hex git oid;
    # swapped=True is the other reading of the docs, which the script must handle the same way
    a, b = (sha(name), gid(name)) if swapped else (gid(name), sha(name))
    return types.SimpleNamespace(oid=a, file_oid=b, filename=name, size=size)


def make_inputs(tmp_path, rows, not_ok=(), report_rows=None):
    c = tmp_path / "lfs.csv"
    with open(c, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader()
        for r in rows:
            w.writerow(r)
    rep = tmp_path / "report.json"
    rep.write_text(json.dumps({"rows": len(rows) if report_rows is None else report_rows, "not_ok": list(not_ok)}))
    return c, rep


def row(tier, name, mb):
    return {"tier": tier, "task": "T", "filename": name, "size_MB": str(mb), "sha256": sha(name), "git_oid": gid(name),
            "pushedAt": "2026-10-26T00:00:00.000Z"}


def args(tmp_path, c, rep, execute=False, tiers=H.TIERS):
    return H.main(["--repo", "o/r-ops", "--csv", str(c), "--report", str(rep), "--tiers", tiers,
                   "--plan", str(tmp_path / "plan.json"), "--log", str(tmp_path / "ev.jsonl")] + (["--execute"] if execute else []))


def fake_hub(monkeypatch, api):
    hub_api = type("HfApi", (), {"permanently_delete_lfs_files": None, "__new__": lambda cls: api})
    mod = types.SimpleNamespace(__version__="0.36.2", HfApi=hub_api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


ROWS = [row("A-VLM含图", "runs/a/x.jsonl", 2.0), row("C-方法链", "runs/c/y.jsonl", 1.0), row("C-方法链", "runs/c/z.jsonl", 1.0),
        row("B-已关闭", "runs/b/w.jsonl", 3.0), row("D-训练", "runs/d/v.jsonl", 0.5), row("E-nodes", "nodes/n1/runs/e/u.jsonl", 0.1)]


def full_api(swapped=False, **kw):
    lfs = [obj(r["filename"], int(float(r["size_MB"]) * 1e6), swapped) for r in ROWS]
    tree = {r["filename"]: r["sha256"] for r in ROWS}
    return Api(lfs, tree, **kw)


def test_dry_run_deletes_nothing(tmp_path, monkeypatch):
    c, rep = make_inputs(tmp_path, ROWS); api = full_api(); fake_hub(monkeypatch, api)
    assert args(tmp_path, c, rep) == 0
    assert api.calls == 0 and not (tmp_path / "ev.jsonl").exists()
    p = json.load(open(tmp_path / "plan.json"))
    assert [len(p[t]["delete"]) for t in "ACBDE"] == [1, 2, 1, 1, 1]


def test_execute_order_and_one_event_per_tier(tmp_path, monkeypatch):
    c, rep = make_inputs(tmp_path, ROWS); api = full_api(); fake_hub(monkeypatch, api)
    assert args(tmp_path, c, rep, execute=True) == 0
    assert api.deleted == [sha(r["filename"]) for r in ROWS]
    ev = [json.loads(l) for l in open(tmp_path / "ev.jsonl")]
    assert [e.get("tier") for e in ev] == list("ACBDE") + [None] and all(e["event"] == "lfs_deleted" for e in ev[:5])
    assert ev[1]["files"] == 2 and ev[0]["used_storage_before"] == 10**9 and ev[0]["hf"] == "ok"
    assert ev[0]["freed_gb"] == 0.0 and ev[-1]["event"] == "lfs_delete_end" and ev[-1]["used_storage"] == 10**9 - 6000


def test_node_problem_keeps_row_tier_e_path_mapped(tmp_path, monkeypatch):
    bad = [{"tier": "C-方法链", "path": "runs/c/y.jsonl", "why": "sha"}, {"tier": "E-nodes", "path": "runs/e/u.jsonl", "why": "missing"}]
    c, rep = make_inputs(tmp_path, ROWS, not_ok=bad); api = full_api(); fake_hub(monkeypatch, api)
    assert args(tmp_path, c, rep, execute=True) == 0
    assert sha("runs/c/y.jsonl") not in api.deleted and sha("nodes/n1/runs/e/u.jsonl") not in api.deleted
    assert len(api.deleted) == 4
    p = json.load(open(tmp_path / "plan.json"))
    assert p["E"]["keep"][0]["why"] == "node: missing"


def test_listing_mismatches_keep_rows(tmp_path, monkeypatch):
    c, rep = make_inputs(tmp_path, ROWS); api = full_api(); fake_hub(monkeypatch, api)
    api.lfs[0].size = 9_000_000                                  # A: size differs
    api.lfs[1].oid = gid("other")                                # C y: git oid differs
    api.lfs = [o for o in api.lfs if o.filename != "runs/b/w.jsonl"]   # B: not listed
    api.tree["status/kept.jsonl"] = sha("runs/d/v.jsonl")        # D: oid shared with a kept path
    assert args(tmp_path, c, rep, execute=True) == 0
    assert sorted(api.deleted) == sorted([sha("runs/c/z.jsonl"), sha("nodes/n1/runs/e/u.jsonl")])
    p = json.load(open(tmp_path / "plan.json"))
    assert p["A"]["keep"][0]["why"].startswith("size on HF") and p["B"]["keep"][0]["why"] == "not in the LFS listing"
    assert p["C"]["keep"][0]["why"] == "git oid on HF differs from the CSV"
    assert p["D"]["keep"][0]["why"] == "oid also used by status/kept.jsonl"


def test_first_error_stops(tmp_path, monkeypatch):
    c, rep = make_inputs(tmp_path, ROWS); api = full_api(fail_at=2); fake_hub(monkeypatch, api)
    assert args(tmp_path, c, rep, execute=True) == 1
    assert api.deleted == [sha("runs/a/x.jsonl")]
    ev = [json.loads(l) for l in open(tmp_path / "ev.jsonl")]
    assert [e["event"] for e in ev] == ["lfs_deleted", "lfs_delete_failed"] and ev[1]["tier"] == "C"
    assert ev[1]["hf"].startswith("403 Forbidden") and "hf_x" not in ev[1]["hf"] and "/srv" not in ev[1]["hf"]


def test_report_must_cover_every_row(tmp_path, monkeypatch):
    c, rep = make_inputs(tmp_path, ROWS, report_rows=5); api = full_api(); fake_hub(monkeypatch, api)
    try:
        args(tmp_path, c, rep, execute=True)
        assert False, "should stop"
    except SystemExit as e:
        assert "covers 5 rows" in str(e)
    assert api.calls == 0


def test_old_hub_refused(tmp_path, monkeypatch):
    c, rep = make_inputs(tmp_path, ROWS)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(__version__="0.25.0", HfApi=type("A", (), {})))
    try:
        args(tmp_path, c, rep, execute=True)
        assert False, "should stop"
    except SystemExit as e:
        assert "needs >= 0.27" in str(e)


def test_either_id_convention(tmp_path, monkeypatch):
    c, rep = make_inputs(tmp_path, ROWS); api = full_api(swapped=True); fake_hub(monkeypatch, api)
    assert args(tmp_path, c, rep, execute=True) == 0
    assert len(api.deleted) == len(ROWS)


def test_node_csv_tier_e_paths(tmp_path, monkeypatch):
    # the runbook copy of the CSV names tier E by its node path; the tree has the repository path: still deleted, not "also used"
    rows = [dict(r, filename=H.node_path(r["filename"])) for r in ROWS]
    c, rep = make_inputs(tmp_path, rows); api = full_api(); fake_hub(monkeypatch, api)
    assert args(tmp_path, c, rep, execute=True) == 0
    assert len(api.deleted) == len(ROWS)


def test_only_an_ops_repository(tmp_path, monkeypatch):
    c, rep = make_inputs(tmp_path, ROWS); api = full_api(); fake_hub(monkeypatch, api)
    try:
        H.main(["--repo", "o/opd-lab-b300-pub", "--csv", str(c), "--report", str(rep), "--execute"])
        assert False, "should stop"
    except SystemExit as e:
        assert "-ops" in str(e)
    assert api.calls == 0
