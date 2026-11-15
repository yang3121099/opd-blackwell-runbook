"""[z172] (11-15 incident, uk-8: two preflight runs of E12b-P0think-8k-c100 renamed its REAL local checkpoint dir aside) a node-profile dry
run copies the node's control.env, which may export CKPT_LOCAL; run_e12.sh's ckpt_lib.py prepare then ran on the real local disk (another
run id -> os.replace to ckpt.old_<id>). Now the sandbox points CKPT_LOCAL / ARCHIVE_ROOT / CKPT_CANDIDATES_EXTRA into itself, and
ckpt_lib.py refuses any local / shared root outside DRYRUN_SB. The end-to-end case reproduces the incident on a fake node root (~40 s)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deps  # noqa: E402

T = "E12b-P0think-8k-c100"


def tree(d):
    return sorted((os.path.relpath(os.path.join(a, f), d), open(os.path.join(a, f)).read()) for a, _, fs in os.walk(d) for f in fs)


def test_node_dry_run_leaves_the_real_local_dir_alone(tmp_path, monkeypatch):
    for k in ("OPD_ROOT", "CKPT_ROOT", "CKPT_LOCAL", "QUEUE_FILE"):
        monkeypatch.delenv(k, raising=False)
    root, ck, real_local = tmp_path / "opd", tmp_path / "ck", tmp_path / "reallocal"
    (root / "control").mkdir(parents=True); ck.mkdir()
    (root / "paths.env").write_text(f"CKPT_ROOT={ck}\n")
    (root / "control.env").write_text(f"export CKPT_LOCAL={real_local}\nexport ARCHIVE_ROOT={tmp_path / 'realarchive'}\nexport SAVE=1\n")
    (root / "control" / "queue_name").write_text("8gs\n")
    g = root / "runs" / "PROBE-E12B-P0THINK-S44"; g.mkdir(parents=True); (g / "report.json").write_text(json.dumps({"result": "PASS"})); (g / ".done").write_text("")
    d = ck / "adhoc_models/result/runs/E12b-P0think-8k/global_step_50/actor/huggingface"; d.mkdir(parents=True); (d / ".fetch_ok").write_text("x")
    lk = real_local / T / "ckpt"; (lk / "global_step_50").mkdir(parents=True)
    (lk / ".run_id").write_text("2026-11-15T03:18:08Z-47125670d10d"); (lk / "latest_checkpointed_iteration.txt").write_text("50")
    (lk / "global_step_50" / "data.pt").write_text("state")
    before = tree(real_local)
    rec, tail = deps.dry_record(T, 60, str(ck), "node", str(root))
    assert rec and rec["result"] == "ok", tail
    assert tree(real_local) == before and sorted(os.listdir(real_local / T)) == ["ckpt"], os.listdir(real_local / T)   # nothing renamed / added
    assert not (tmp_path / "realarchive").exists()


def test_ckpt_lib_refuses_paths_outside_the_sandbox(tmp_path):
    sb = tmp_path / "sb"; (sb / "opd" / "runs" / "T").mkdir(parents=True); out = tmp_path / "real"; (out / "T" / "ckpt").mkdir(parents=True)
    (out / "T" / "ckpt" / ".run_id").write_text("other"); cfg = sb / "opd" / "runs" / "T" / "config.yaml"; cfg.write_text("started: x\n")
    env = dict(os.environ, DRYRUN_SB=str(sb), CKPT_LOCAL=str(out), CKPT_ROOT=str(sb / "ck"), OPD_ROOT=str(sb / "opd"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "ckpt_lib.py"), "prepare", "T", str(cfg)], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "CKPT_SANDBOX_ESCAPE prepare" in r.stderr and sorted(os.listdir(out / "T")) == ["ckpt"]
    env["CKPT_LOCAL"] = str(sb / "local")   # inside: allowed
    r = subprocess.run([sys.executable, os.path.join(HERE, "ckpt_lib.py"), "prepare", "T", str(cfg)], env=env, capture_output=True, text=True)
    assert r.returncode == 0 and "CKPT_SANDBOX_ESCAPE" not in r.stderr, r.stderr
