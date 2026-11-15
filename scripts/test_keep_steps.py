"""[z168] (Fable board v22, the c100 s25 loss 11-15) keep_steps.py (the registered steps hard-linked out of the local rotation, published to
the shared disk's adhoc_models/result with fetch_model.py's .fetch_ok) and deps.py's RETAIN check. No GPU, no network. Stdlib + pytest."""
import json
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deps  # noqa: E402
import fetch_model as F  # noqa: E402
import keep_steps as K  # noqa: E402


def step(ck, s, bf16=False):
    d = ck / f"global_step_{s}" / "actor" / ("huggingface_bf16" if bf16 else "huggingface"); d.mkdir(parents=True)
    (d / "config.json").write_text("{}"); (d / "model.safetensors").write_bytes(os.urandom(4096) + str(s).encode())
    return d


def mk(tmp_path, uw="25,50", latest=26):
    local, ckr, opd = tmp_path / "local", tmp_path / "ckpt", tmp_path / "opd"
    ck = local / "T" / "ckpt"; ck.mkdir(parents=True); ckr.mkdir(); (opd / "runs" / "T").mkdir(parents=True)
    (ck / "latest_checkpointed_iteration.txt").write_text(str(latest))
    (opd / "runs" / "T" / "upload.log").write_text(f"UPLOAD_SETTINGS 2026-11-15T00:00:00Z T UPLOAD_WEIGHTS={uw} KEEP_LOCAL_HF=0 KEEP_BF16=1 DRY=0\n")
    return local, ckr, opd, ck


def test_a_registered_step_is_kept_and_published(tmp_path):
    local, ckr, opd, ck = mk(tmp_path)
    src = step(ck, 25); step(ck, 26); want = (src / "model.safetensors").read_bytes()
    K.one_pass(str(local), str(ckr), str(opd))
    pub = ckr / "adhoc_models" / "result" / "runs" / "T" / "global_step_25" / "actor" / "huggingface"
    assert (pub / "model.safetensors").read_bytes() == want and (pub / ".fetch_ok").is_file()
    c = F.cached("runs/T/global_step_25/actor/huggingface", str(ckr / "adhoc_models" / "result"))   # a result: reading finds it
    assert c and c["ok"] and c["files"] == 2
    assert not (tmp_path / "local.keep" / "T" / "global_step_25").exists()   # the local link goes once published
    shutil.rmtree(ck / "global_step_25"); K.one_pass(str(local), str(ckr), str(opd))   # rotated away later: the copy stays
    assert (pub / "model.safetensors").read_bytes() == want


def test_the_link_survives_the_rotation(tmp_path):
    local, ckr, opd, ck = mk(tmp_path)
    src = step(ck, 25); want = (src / "model.safetensors").read_bytes(); k = tmp_path / "k"
    K.link_tree(str(src), str(k)); shutil.rmtree(ck / "global_step_25")   # verl deletes the step after the link
    assert (k / "model.safetensors").read_bytes() == want


def test_not_kept_before_complete_nor_unregistered(tmp_path):
    local, ckr, opd, ck = mk(tmp_path, latest=24)
    step(ck, 25); step(ck, 24)
    assert K.one_pass(str(local), str(ckr), str(opd)) == []   # latest 24 < 25: s25 still being written
    (ck / "latest_checkpointed_iteration.txt").write_text("25")
    assert [x[1] for x in K.one_pass(str(local), str(ckr), str(opd))] == [25]   # s24 is not a registered step
    local2, ckr2, opd2, ck2 = mk(tmp_path / "b", uw="all", latest=30); step(ck2, 25)
    assert K.one_pass(str(local2), str(ckr2), str(opd2)) == []   # "all": nothing kept (every step goes up)


def test_bf16_copy_and_keep_steps_from_control_env(tmp_path):
    local, ckr, opd, ck = mk(tmp_path, uw="50", latest=40)
    step(ck, 30, bf16=True); (opd / "control.env").write_text("export KEEP_STEPS=30\n")
    assert [x[1] for x in K.one_pass(str(local), str(ckr), str(opd))] == [30]
    assert (ckr / "adhoc_models/result/runs/T/global_step_30/actor/huggingface/.fetch_ok").is_file()


def test_keep_steps_sh_once(tmp_path):
    local, ckr, opd, ck = mk(tmp_path); step(ck, 25)
    rb = tmp_path / "rb" / "scripts"; rb.mkdir(parents=True)
    for f in ("keep_steps.sh", "keep_steps.py"):
        shutil.copy(os.path.join(HERE, f), rb / f)
    (rb / "env.sh").write_text(f"export OPD_ROOT={opd} RUNBOOK={tmp_path / 'rb'} CKPT_ROOT={ckr} CKPT_LOCAL={local}\n")
    r = subprocess.run(["bash", str(rb / "keep_steps.sh")], env=dict(os.environ, KEEP_ONCE="1"), capture_output=True, text=True, timeout=60)
    assert "KEEP_OK T s25" in r.stdout and "PUB_OK T s25" in r.stdout, r.stdout + r.stderr
    assert (opd / "control" / "loop_keep_steps").is_file()
    rt = open(os.path.join(HERE, "run_tasks.sh")).read()
    assert rt.count('[ -n "${CKPT_LOCAL:-}" ] && [ "$(cenv KEEP_STEPS_OFF 0)" != 1 ] && ensure_loop keep_steps') == 2
    assert "ckpt_mirror keep_steps" in open(os.path.join(HERE, "loops.sh")).read()


def rec(uw):
    return {"upload": {"launch": [{"env": {"UPLOAD_WEIGHTS": uw}}]}}


def test_retain(tmp_path, monkeypatch, capsys):
    root = tmp_path / "r"; (root / "control").mkdir(parents=True)
    for k in ("CKPT_LOCAL", "HF_TOKEN", "MS_TOKEN", "UPLOAD_DRY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(deps.subprocess, "run", lambda *k, **kw: SimpleNamespace(stdout=""))   # no keep_steps.sh process on this machine
    assert deps.retain(rec("25,50"), str(root), "1", "50") == 0 and "no local rotation" in capsys.readouterr().out
    (root / "paths.env").write_text("CKPT_LOCAL=/var/tmp/opd/ckptlocal\n")
    assert deps.retain(rec("25,50"), str(root), "1", "50") == 1   # the c100 case: SAVE=1, keep 2, nothing else
    out = capsys.readouterr().out
    assert "RETAIN s25 MISSING" in out and "s50" not in out.replace("of 50", "")   # the final step is the rotation's newest
    assert deps.retain(rec("25,50"), str(root), "25", "50") == 0 and "the local rotation keeps it" in capsys.readouterr().out
    (root / "control" / "loop_keep_steps").write_text(str(int(time.time())))
    assert deps.retain(rec("25,50"), str(root), "1", "50") == 0 and "keep_steps.sh (the runbook loop, alive)" in capsys.readouterr().out
    (root / "control" / "loop_keep_steps").write_text(str(int(time.time()) - 900))   # a dead loop does not count
    monkeypatch.setattr(deps.subprocess, "run", lambda *k, **kw: SimpleNamespace(stdout=""))   # no keep_steps.sh process either
    assert deps.retain(rec("25,50"), str(root), "1", "50") == 1
    monkeypatch.setattr(deps.subprocess, "run", lambda cmd, **kw: SimpleNamespace(stdout="4242\n" if cmd[0] == "pgrep" else " 4242 bash /var/tmp/opd/bin/keep_steps.sh\n"))
    assert deps.retain(rec("25,50"), str(root), "1", "50") == 0   # [z171] executor B's node-kit copy counts
    assert "a keep_steps.sh process (4242 bash /var/tmp/opd/bin/keep_steps.sh; the steps it keeps are not checked)" in capsys.readouterr().out
    monkeypatch.setattr(deps.subprocess, "run", lambda *k, **kw: SimpleNamespace(stdout=""))
    monkeypatch.setenv("HF_TOKEN", "x")
    assert deps.retain(rec("25,50"), str(root), "1", "50") == 0 and "the upload" in capsys.readouterr().out
    assert deps.retain(rec("all"), str(root), "1", "50") == 0


def test_preflight_train_line_on_a_refusal_z170(tmp_path, monkeypatch, capsys):   # [z170] (executor B 11-15) a refusal is "unknown", not "not a training"
    from types import SimpleNamespace
    a = SimpleNamespace(task="X", sec=1, root=str(tmp_path))
    monkeypatch.setattr(deps, "dry_record", lambda *k, **kw: ({"result": "exit:2", "dispatch": {"first_line": "unknown task X"}}, ""))
    assert deps.preflight(a, str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "NODE_REFUSAL exit:2" in out and "TRAIN unknown (the node dry run ended before the launch" in out and "not a training" not in out
    monkeypatch.setattr(deps, "dry_record", lambda *k, **kw: ({"result": "ok", "train": []}, ""))
    assert deps.preflight(a, str(tmp_path)) == 0 and "TRAIN none (not a training" in capsys.readouterr().out


def test_resume_line_reads_the_local_dir_z171(tmp_path, monkeypatch, capsys):   # [z171] (executor B, c100 11-15) local steps 49 / 50 = a resume, not "fresh"
    import ckpt_lib as CL
    root, ck, local = tmp_path / "opd", tmp_path / "ckpt", tmp_path / "local"
    (root / "runs" / "T").mkdir(parents=True); (root / "paths.env").write_text(f"CKPT_LOCAL={local}\n")
    monkeypatch.delenv("CKPT_LOCAL", raising=False)
    cfg = root / "runs" / "T" / "config.yaml"; cfg.write_text("task: T\nstarted: 2026-11-15T10:00:00Z\n")
    lk = local / "T" / "ckpt"
    for st in (49, 50):
        (lk / f"global_step_{st}").mkdir(parents=True)
    (lk / "latest_checkpointed_iteration.txt").write_text("50"); (lk / ".run_id").write_text(CL.run_id(str(cfg))); CL.write_marker(str(lk), 50)
    deps.resume_line("T", str(root), str(ck), str(lk), "auto")
    out = capsys.readouterr().out
    assert f"RESUME mode=auto: from step 50 (local: {lk / 'global_step_50'})" in out and "local steps [49, 50] (latest 50)" in out, out
    (lk / ".run_id").write_text("another-run")   # another run's dir: not this run's steps
    deps.resume_line("T", str(root), str(ck), str(lk), "auto")
    assert "a fresh start - no step of this run" in capsys.readouterr().out
    cfg.unlink(); deps.resume_line("T", str(root), str(ck), str(lk), "auto")
    assert "no runs/T/config.yaml, so no run id" in capsys.readouterr().out
    (root / "paths.env").write_text(""); (lk / ".run_id").write_text("x")   # no CKPT_LOCAL: the trainer dir's latest file, as before z171
    deps.resume_line("T", str(root), str(ck), str(lk), "auto")
    assert f"RESUME mode=auto: from step 50 ({lk / 'latest_checkpointed_iteration.txt'})" in capsys.readouterr().out
