"""[v2.5.9z123] mount_probe.py (read-only mount probe, Runming 11-02 via Fable): /proc/mounts parsing, our base left out, a B300-era
runs/ found below a persistent mount with its saved steps and raw generation files, nothing written, a summary without paths;
and the control rules accept `mounts` with no argument."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control_rules as R  # noqa: E402
import mount_probe as M  # noqa: E402

PROC = """overlay / overlay rw,relatime 0 0
proc /proc proc rw 0 0
tmpfs /dev tmpfs rw 0 0
sysfs /sys sysfs ro 0 0
/dev/nvme0n1 /code ext4 rw 0 0
server:/export {old} nfs4 rw 0 0
/dev/sdb /mnt/with\\040space xfs rw 0 0
cgroup2 /sys/fs/cgroup cgroup2 rw 0 0
"""


def test_real_mounts_skips_pseudo_and_unescapes():
    ms = M.real_mounts(PROC.format(old="/persist"))
    assert ms == [("/", "overlay"), ("/code", "ext4"), ("/mnt/with space", "xfs"), ("/persist", "nfs4")]


def _tree(tmp_path):
    old = tmp_path / "persist"; base = tmp_path / "code" / "opd" / "run"
    td = old / "someone" / "opd-run" / "runs" / "E10d-P5tail-s43"
    hf = td / "ckpt" / "global_step_25" / "actor" / "huggingface"; hf.mkdir(parents=True)
    (hf / "model.safetensors").write_bytes(b"x" * 1000); (hf / "config.json").write_text("{}")
    (td / "global_step_50" / "actor" / "huggingface_bf16").mkdir(parents=True)                    # no safetensors: not a saved step
    ev = old / "someone" / "opd-run" / "runs" / "EVAL-e10d-P5tail-s43"; ev.mkdir(parents=True)
    (ev / "e10d_s25_math500_T0.6_math500_no_think_t0.6.jsonl").write_text("{}\n" * 10)
    (ev / "x_math500_no_think_t1.0.jsonl.part").write_text("{}\n")
    (ev / "x.metrics.json").write_text("{}")
    (old / "someone" / "opd-run" / "runs" / "notes").mkdir()                                       # not a task name
    (old / "other" / "runs").mkdir(parents=True); (old / "other" / "runs" / "random").mkdir()       # a runs/ that is not ours
    base.mkdir(parents=True); (base / "runs" / "E12c-P5tail-8k").mkdir(parents=True)               # our current base: not scanned
    return old, base


def _snapshot(root):
    return sorted((dp, tuple(sorted(fs)), tuple(sorted(ds))) for dp, ds, fs in os.walk(root))


def test_probe_finds_the_old_runs_and_writes_nothing(tmp_path):
    old, base = _tree(tmp_path)
    proc = PROC.format(old=str(old)).replace("/dev/nvme0n1 /code ext4", f"/dev/nvme0n1 {tmp_path / 'code'} ext4")
    before = _snapshot(tmp_path)
    rep = M.probe(proc, {"OPD_ROOT": str(base), "MP_SECONDS": "30"})
    assert _snapshot(tmp_path) == before                                                          # read-only
    assert str(old) in rep["scanned"] and str(tmp_path / "code") not in rep["scanned"]
    runs = {k: v for k, v in rep["runs"].items()}
    assert list(runs) == [str(old / "someone" / "opd-run" / "runs")]                                # the other runs/ has no task child
    tasks = runs[str(old / "someone" / "opd-run" / "runs")]
    assert set(tasks) == {"E10d-P5tail-s43", "EVAL-e10d-P5tail-s43"}
    assert [(s["step"], s["place"]) for s in tasks["E10d-P5tail-s43"]["steps"]] == [(25, "ckpt/huggingface")]
    assert tasks["EVAL-e10d-P5tail-s43"]["raw_files"] == 2 and tasks["E10d-P5tail-s43"]["raw_files"] == 0
    ours = {m["mount"]: m["ours"] for m in rep["mounts"]}
    assert ours["/"] and ours[str(tmp_path / "code")] and not ours[str(old)]
    s = M.summary(rep)
    assert "2 task dirs; 1 saved steps" in s and "2 raw generation files" in s
    assert str(tmp_path) not in s and "someone" not in s                                          # counts only, no paths
    json.dumps(rep)                                                                                # the report is JSON


def test_old_base_candidates_and_budget(tmp_path):
    old, base = _tree(tmp_path)
    rep = M.probe("overlay / overlay rw 0 0\n", {"OPD_ROOT": str(base), "OLD_BASE_CANDS": f"{tmp_path / 'nope'}:{old / 'someone'}"})
    assert rep["scanned"] == [str(old / "someone")] and len(rep["runs"]) == 1                       # a missing candidate is skipped
    rep = M.probe(PROC.format(old=str(old)), {"OPD_ROOT": str(base), "MP_ENTRIES": "2"})
    assert rep["truncated"] is True                                                                 # the entry budget ends the walk


def test_rules_accept_mounts_without_arguments():
    R.parse({"cmd": "mounts", "reason": "t"})
    try:
        R.parse({"cmd": "mounts /somewhere", "reason": "t"}); assert False, "no argument allowed"
    except Exception:
        pass
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 111)
