"""[v2.5.9z72] (ops-lead 10-23 18:05 UTC, item 1) the inventory: a weight step is in the cloud only when one repository holds every
safetensors at the same size, queued when its destination is in weights_pending.txt, else local; other files by category (logs /
readings / other), the resume state apart, hard links counted once; a done reading with no summary in the cloud is missing;
pend queues local weight steps of tasks not running and the missing readings; the verb's rules."""
import json
import os
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402
import control_rules as R  # noqa: E402


def w(p, n):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(b"x" * n)


@pytest.fixture
def node(monkeypatch):
    root, ckpt = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "control")); os.makedirs(os.path.join(root, "logs"))
    monkeypatch.setattr(C, "ROOT", root); monkeypatch.setattr(C, "CDIR", os.path.join(root, "control"))
    monkeypatch.setenv("OPD_ROOT", root); monkeypatch.setenv("CKPT_ROOT", ckpt)
    # a training whose ckpt lives on the checkpoint disk (the symlink), steps 25 (uploaded), 50 (final, local), 40 (queued)
    ck = os.path.join(ckpt, "opd-ckpt", "E10b-A", "ckpt"); r = os.path.join(root, "runs", "E10b-A"); os.makedirs(r)
    os.symlink(ck, os.path.join(r, "ckpt"))
    for s in (25, 40, 50):
        w(f"{ck}/global_step_{s}/actor/huggingface_bf16/model-1.safetensors", 1000)
        w(f"{ck}/global_step_{s}/actor/huggingface_bf16/model-2.safetensors", 500)
        w(f"{ck}/global_step_{s}/actor/huggingface_bf16/config.json", 10)
    w(f"{ck}/global_step_50/actor/huggingface/model-1.safetensors", 2000)      # the fp32 copy beside it
    w(f"{ck}/global_step_50/actor/optim_rank0.pt", 7000)                        # resume state
    open(os.path.join(r, "final_step"), "w").write("50\n")
    w(os.path.join(r, "train.log"), 300); w(os.path.join(r, "rollouts", "1.jsonl"), 400)
    # a reading: done, summary in the cloud, a big jsonl local only; another done reading with nothing up
    e = os.path.join(root, "runs", "EVAL-x"); w(os.path.join(e, "x.metrics.json"), 20); w(os.path.join(e, "gen.jsonl"), 900)
    open(os.path.join(e, ".done"), "w").write("")
    f = os.path.join(root, "runs", "EVAL-y"); w(os.path.join(f, "y.metrics.json"), 30); open(os.path.join(f, ".done"), "w").write("")
    w(os.path.join(root, "logs", "E10b-A.log"), 100)
    # the queued step's hold (a hard link to the run's file: counted once)
    hold = os.path.join(ckpt, "weights_pending", "runs", "E10b-A", "global_step_40", "actor", "huggingface"); os.makedirs(hold)
    for n in ("model-1", "model-2"):
        os.link(f"{ck}/global_step_40/actor/huggingface_bf16/{n}.safetensors", os.path.join(hold, f"{n}.safetensors"))
    open(os.path.join(root, "control", "weights_pending.txt"), "w").write(f"2026-10-23T18:00:00Z\truns/E10b-A/global_step_40/actor/huggingface\t{hold}\t1500\n")
    remote = {"public": {"runs/E10b-A/global_step_25/actor/huggingface/model-1.safetensors": 1000,
                         "runs/E10b-A/global_step_25/actor/huggingface/model-2.safetensors": 500,
                         "runs/E10b-A/global_step_50/actor/huggingface/model-1.safetensors": 1000},   # s50: one file only
              "private": {"runs/EVAL-x/x.metrics.json": 20, "runs/E10b-A/rollouts/1.jsonl": 400},
              "old": {"runs/EVAL-y/y.metrics.json": 31}}                                              # another size: not counted
    return root, ckpt, remote


def test_scan(node):
    root, ckpt, remote = node
    rep = C.inventory_scan(remote, root=root, ckpt=ckpt, busy=lambda t: False)
    st = {(x["task"], x["step"]): x for x in rep["weights_local"] + rep["weights_queued"]}
    assert rep["weights_in_cloud"] == 1                                                     # s25: every file at its size
    assert st[("E10b-A", 50)]["state"] == "local" and st[("E10b-A", 50)]["final"] is True   # s50: a file missing
    assert st[("E10b-A", 40)]["state"] == "queued"
    g = rep["gb"]
    assert g["weights"]["local_only"] == round(1530 / 1e9, 1) and g["weights"]["second_copy"] == round(2000 / 1e9, 1)
    assert rep["readings_missing"] == ["EVAL-y"]                                             # EVAL-x's summary is up
    assert rep["local_only_by_task_gb"]["EVAL-x"] == round(900 / 1e9, 2)                     # its jsonl is not
    assert rep["repos"] == {"public": 3, "private": 2, "old": 1}


def test_bytes_by_category_exact(node):
    root, ckpt, remote = node
    g = C.inventory_scan(remote, root=root, ckpt=ckpt, busy=lambda t: False, unit=1)["gb"]
    assert g["weights"] == {"local_only": 1500 + 3 * 10, "queued": 1500, "in_cloud": 1500, "second_copy": 2000}   # + the configs; the hold once
    assert g["resume_state_local"] == 7000
    assert g["logs"] == {"local_only": 300 + 100, "queued": 0, "in_cloud": 0}                 # train.log + logs/
    assert g["other"] == {"local_only": 3, "queued": 0, "in_cloud": 400}                      # final_step; the rollouts are up
    assert g["readings"] == {"local_only": 900 + 30, "queued": 0, "in_cloud": 20}             # EVAL-x jsonl, EVAL-y summary


def test_pend(node, monkeypatch):
    root, ckpt, remote = node
    import remote as RM
    calls = []
    monkeypatch.setattr(RM, "defer_weight", lambda src, dst: calls.append(("w", os.path.relpath(src, root), dst)) or 1)
    monkeypatch.setattr(RM, "pend_result", lambda src, dst: calls.append(("r", os.path.relpath(src, root), dst)))
    rep = C.inventory_scan(remote, root=root, ckpt=ckpt, busy=lambda t: False)
    assert C.inventory_pend(rep, root=root) == (1, 1)
    assert calls == [("w", "runs/E10b-A/ckpt/global_step_50/actor/huggingface_bf16", "runs/E10b-A/global_step_50/actor/huggingface"),
                     ("r", "runs/EVAL-y", "runs/EVAL-y")]
    calls.clear(); open(os.path.join(root, "runs", "EVAL-y", ".upload_pending"), "w").write("")
    rep = C.inventory_scan(remote, root=root, ckpt=ckpt, busy=lambda t: True)                 # the training running now
    assert C.inventory_pend(rep, root=root) == (0, 0) and calls == []


def test_rules():
    R.parse({"cmd": "inventory", "reason": "t"}); R.parse({"cmd": "inventory pend", "reason": "t"})
    with pytest.raises(R.Reject):
        R.parse({"cmd": "inventory all", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 85)


def test_backoff_rejects(monkeypatch):
    monkeypatch.setattr(C, "_backoff_left", lambda: 120)
    assert C.v_inventory([], 1)[0] == "rejected"


def test_do_inventory_receipt_and_report(node, monkeypatch):   # the background half: report file, private upload, receipt
    root, ckpt, remote = node
    got, puts = [], []
    monkeypatch.setattr(C, "_repo_runs", lambda k: remote[k])
    monkeypatch.setattr(C, "running_pids", lambda t=None: [])
    monkeypatch.setattr(C, "remote_put", lambda src, dst, kind=None: puts.append(dst) or True)
    monkeypatch.setattr(C, "receipt", lambda seq, entry, status, detail, upload=True, **kw: got.append((seq, status, detail, kw)))
    monkeypatch.setattr(C, "node_id", lambda: "abcd1234")
    C.do_inventory(7)
    seq, status, detail, kw = got[0]
    assert seq == 7 and status == "ok" and detail.startswith("local only: weights ") and "1 done readings with no summary" in detail
    assert root not in detail and ckpt not in detail and "/" not in detail                 # no paths in a receipt
    assert puts == ["logs/abcd1234/inventory"]
    rep = json.load(open(os.path.join(root, "control", "inventory", "inventory.json")))
    assert rep["readings_missing"] == ["EVAL-y"] and "pended_weights" not in rep
    monkeypatch.setattr(C, "_repo_runs", lambda k: (_ for _ in ()).throw(OSError("x")))
    got.clear(); C.do_inventory(8)
    assert got[0][1] == "failed" and "listing failed" in got[0][2]
