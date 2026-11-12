"""[z155] Unit tests of ckpt_lib.py on tiny synthetic checkpoints (no GPU, no network): manifests (full and sampled sha256), run ids, the mirror
state machine (a killed copy is finished, never abandoned; the shared marker only names verified steps; lag lines; keep-2 only after verify),
resume selection (run id, local > shared > pulled), prepare, the space guard, a pulled step. Run: python -m pytest -q scripts/test_ckpt_lib.py"""
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_lib as C

BIG, WIN = 100, 16   # files > 100 bytes get the sampled hash with 16-byte windows


def mkstep(task_ckpt, step, payload=b"x", big_payload=None):
    d = os.path.join(task_ckpt, f"global_step_{step}", "actor")
    os.makedirs(os.path.join(d, "huggingface"), exist_ok=True)
    open(os.path.join(d, "model_world_size_1_rank_0.pt"), "wb").write((big_payload or bytes(range(256)) * 4) + payload)
    open(os.path.join(d, "huggingface", "config.json"), "w").write(json.dumps({"step": step}))
    return os.path.dirname(d)


def cfg(tmp, started="2026-11-12T10:00:00Z", extra=""):
    p = os.path.join(tmp, "config.yaml")
    open(p, "w").write(f"task: E12b-X\nvariant: 8k\nsteps: 50\nstarted: {started}\n{extra}")
    return p


def test_manifest_full_and_sampled_and_verify(tmp_path):
    d = mkstep(str(tmp_path), 5)
    m = C.manifest(d, "rid", 5, BIG, WIN)
    open(os.path.join(d, "actor", "extra_state.pt"), "w").write("small")
    m = C.manifest(d, "rid", 5, BIG, WIN)
    big = m["files"]["actor/model_world_size_1_rank_0.pt"]; small = m["files"]["actor/extra_state.pt"]
    assert "sampled_sha256" in big and len(big["windows"]) == 6 and tuple(big["windows"][0]) == (0, WIN)
    assert "sha256" in small and C.verify(d, json.loads(json.dumps(m))) == ""
    assert not [f for f in m["files"] if f.startswith("actor/huggingface")]          # the HF export is not covered
    os.remove(os.path.join(d, "actor/huggingface/config.json"))                       # (upload_sync may thin it out on the shared copy)
    assert C.verify(d, m) == ""
    p = os.path.join(d, "actor/model_world_size_1_rank_0.pt"); b = bytearray(open(p, "rb").read())
    b[3] ^= 1; open(p, "wb").write(bytes(b))                                   # inside the first window: caught
    assert C.verify(d, m).startswith("hash")
    b[3] ^= 1; b.append(0); open(p, "wb").write(bytes(b))
    assert C.verify(d, m).startswith("size")
    os.remove(os.path.join(d, "actor/extra_state.pt"))
    assert C.verify(d, m).startswith("missing")


def test_run_id(tmp_path):
    a = C.run_id(cfg(str(tmp_path)))
    assert a.startswith("2026-11-12T10:00:00Z-") and C.run_id(cfg(str(tmp_path), extra="resumed: later\n")) == a   # a resume keeps it
    open(os.path.join(tmp_path, "config.yaml"), "w").write("task: E12b-X\nvariant: 16k\nsteps: 50\nstarted: 2026-11-12T10:00:00Z\n")
    assert C.run_id(os.path.join(tmp_path, "config.yaml")) != a                 # another config: another run
    open(os.path.join(tmp_path, "config.yaml"), "w").write("task: x\n")
    assert C.run_id(os.path.join(tmp_path, "config.yaml")) is None


def test_mirror_basic_and_lag(tmp_path, capsys):
    L, S = str(tmp_path / "local/T/ckpt"), str(tmp_path / "shared/T/ckpt")
    mkstep(L, 1); mkstep(L, 2); C.write_marker(L, 2)
    assert C.mirror_pass("T", L, S, "rid", BIG, WIN) == 2
    out = capsys.readouterr().out
    assert "MIRROR_LAG T local 2 shared 0 behind 2" in out and "MIRRORED T 2" in out and "MIRRORED T 1" not in out   # jumped to the newest
    assert C.steps(S) == [2] and C.read_man(os.path.join(S, "global_step_2"))["run_id"] == "rid" and not os.path.exists(os.path.join(S, ".mirror_inflight"))
    assert C.mirror_pass("T", L, S, "rid", BIG, WIN) == 2 and "MIRROR_LAG" not in capsys.readouterr().out   # caught up: quiet


def test_killed_copy_is_finished_first_then_jump(tmp_path, capsys, monkeypatch):
    L, S = str(tmp_path / "local/T/ckpt"), str(tmp_path / "shared/T/ckpt")
    for s in (1, 2, 3):
        mkstep(L, s)
    C.write_marker(L, 1)
    real = shutil.copy2; calls = {"n": 0}
    def dying(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("killed mid-copy")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(C.shutil, "copy2", dying)
    with pytest.raises(KeyboardInterrupt):
        C.mirror_pass("T", L, S, "rid", BIG, WIN)
    assert C.marker(S) == 0 and os.path.isdir(os.path.join(S, "global_step_1.part")) and not os.path.isdir(os.path.join(S, "global_step_1"))
    assert open(os.path.join(S, ".mirror_inflight")).read() == "1"           # the shared marker never named the partial step
    monkeypatch.setattr(C.shutil, "copy2", real)
    C.write_marker(L, 3)                                                       # training moved on meanwhile
    capsys.readouterr()
    assert C.mirror_pass("T", L, S, "rid", BIG, WIN) == 3
    out = capsys.readouterr().out
    assert out.index("MIRRORED T 1") < out.index("MIRRORED T 3") and "MIRRORED T 2" not in out   # step 1 finished, then a jump to 3
    assert C.steps(S) == [1, 3] and C.marker(S) == 3


def test_verify_failure_keeps_old_steps_and_marker(tmp_path, capsys, monkeypatch):
    L, S = str(tmp_path / "local/T/ckpt"), str(tmp_path / "shared/T/ckpt")
    for s in (1, 2):
        mkstep(L, s); C.write_marker(L, s); C.mirror_pass("T", L, S, "rid", BIG, WIN)
    mkstep(L, 3); C.write_marker(L, 3)
    real = C._copy_tree
    def corrupt(src, dst):
        real(src, dst); p = os.path.join(dst, "actor/model_world_size_1_rank_0.pt"); b = bytearray(open(p, "rb").read()); b[0] ^= 1; open(p, "wb").write(bytes(b))
    monkeypatch.setattr(C, "_copy_tree", corrupt)
    capsys.readouterr()
    assert C.mirror_pass("T", L, S, "rid", BIG, WIN) == 2
    assert "MIRROR_VERIFY_FAIL T 3" in capsys.readouterr().out and C.steps(S) == [1, 2]   # nothing pruned before a new step verified
    monkeypatch.setattr(C, "_copy_tree", real)
    assert C.mirror_pass("T", L, S, "rid", BIG, WIN) == 3 and C.steps(S) == [2, 3]          # now 1 goes


def test_abandon_only_when_the_local_step_is_gone(tmp_path, capsys):
    L, S = str(tmp_path / "local/T/ckpt"), str(tmp_path / "shared/T/ckpt")
    mkstep(L, 4); C.write_marker(L, 4); os.makedirs(S)
    open(os.path.join(S, ".mirror_inflight"), "w").write("2"); os.makedirs(os.path.join(S, "global_step_2.part"))
    assert C.mirror_pass("T", L, S, "rid", BIG, WIN) == 4
    out = capsys.readouterr().out
    assert "MIRROR_ABANDON T 2 local step missing" in out and not os.path.exists(os.path.join(S, "global_step_2.part"))


def test_resume_selection_and_prepare(tmp_path, capsys):
    L, S = str(tmp_path / "local/T/ckpt"), str(tmp_path / "shared/T/ckpt")
    os.makedirs(L); open(os.path.join(L, ".run_id"), "w").write("A")
    mkstep(L, 3); C.write_marker(L, 3)                                         # local, no manifest yet: counts (run A, marker covers it)
    best, why = C.select_resume(L, S, "A")
    assert best[:2] == (3, 0)
    for s, rid in ((4, "A"), (6, "B")):
        d = mkstep(S, s); C.write_json(os.path.join(d, "manifest.json"), C.manifest(d, rid, s, BIG, WIN))
    best, why = C.select_resume(L, S, "A")
    assert best[:2] == (4, 1) and any("shared step 6: run id B" in w for w in why)     # another run's step ignored, with a reason
    d = mkstep(S, 5); C.write_json(os.path.join(d, "manifest.json"), C.manifest(d, "A", 5, BIG, WIN, source="hf"))
    assert C.select_resume(L, S, "A")[0][:2] == (5, 2)                         # pulled step 5 is the highest
    d = mkstep(L, 5); C.write_json(os.path.join(d, "manifest.json"), C.manifest(d, "A", 5, BIG, WIN))
    assert C.select_resume(L, S, "A")[0][:2] == (5, 0)                         # equal step: local wins
    shutil.rmtree(L)                                                           # local wiped: shared / pulled
    assert C.prepare("T", L, S, "A") == 5 and C.marker(L) == 5 and C.verify(os.path.join(L, "global_step_5"), C.read_man(os.path.join(S, "global_step_5"))) == ""
    assert os.path.isfile(os.path.join(L, "global_step_5", "actor", "huggingface", "config.json"))   # the HF export comes along
    assert "CKPT_RESUME T 5 hf" in capsys.readouterr().out


def test_prepare_moves_another_runs_local_dir_aside(tmp_path, capsys):
    L, S = str(tmp_path / "local/T/ckpt"), str(tmp_path / "shared/T/ckpt")
    os.makedirs(L); open(os.path.join(L, ".run_id"), "w").write("old-run-000000000000"); mkstep(L, 7); C.write_marker(L, 7)
    assert C.prepare("T", L, S, "new-run-111111111111") == 0
    assert os.path.isdir(L + ".old_000000000000") and C.steps(L) == [] and open(os.path.join(L, ".run_id")).read() == "new-run-111111111111"
    assert "CKPT_LOCAL_ASIDE" in capsys.readouterr().out                       # renamed, nothing deleted


def test_space_guard(tmp_path, monkeypatch):
    L, S = str(tmp_path / "local/T/ckpt"), str(tmp_path / "shared/T/ckpt")
    mkstep(S, 2)
    need = C.expected_bytes(L, S, str(tmp_path / "nomodel"))
    assert need == C.du(os.path.join(S, "global_step_2")) > 0
    assert C.expected_bytes(L, str(tmp_path / "none"), str(tmp_path / "nomodel"), est_gb="2") == 2 << 30
    m = tmp_path / "model"; m.mkdir(); (m / "w.safetensors").write_bytes(b"0" * 1000)
    assert C.expected_bytes(str(tmp_path / "a"), str(tmp_path / "b"), str(m)) == 10000
    monkeypatch.setattr(C.shutil, "disk_usage", lambda p: type("U", (), {"free": 2999})())
    assert C.space_ok(str(tmp_path / "loc"), 1000) == (False, 2999)
    monkeypatch.setattr(C.shutil, "disk_usage", lambda p: type("U", (), {"free": 3000})())
    assert C.space_ok(str(tmp_path / "loc"), 1000) == (True, 3000)


def test_place_pulled(tmp_path):
    src = mkstep(str(tmp_path / "dl"), 9)
    C.write_json(os.path.join(src, "manifest_full.json"), C.manifest(src, "A", 9, big=float("inf"), source="hf"))
    S = str(tmp_path / "shared/T/ckpt")
    assert C.place_pulled("T", src, S) == 9 and C.marker(S) == 9 and C.read_man(os.path.join(S, "global_step_9"))["source"] == "hf"
    src2 = mkstep(str(tmp_path / "dl2"), 10)
    C.write_json(os.path.join(src2, "manifest_full.json"), C.manifest(src2, "A", 10, big=float("inf"), source="hf"))
    p = os.path.join(src2, "actor/model_world_size_1_rank_0.pt"); b = bytearray(open(p, "rb").read()); b[-1] ^= 1; open(p, "wb").write(bytes(b))
    assert C.place_pulled("T", src2, S) is None and C.marker(S) == 9


def test_push_uses_the_newest_verified_step(tmp_path):
    S = str(tmp_path / "shared/T/ckpt")
    for s in (2, 4):
        d = mkstep(S, s); C.write_json(os.path.join(d, "manifest.json"), C.manifest(d, "A", s, BIG, WIN))
    mkstep(S, 6)                                                               # unverified: not pushed
    calls = []
    class Api:
        def upload_folder(self, **k): calls.append(k)
    assert C.push("T", S, "me/ops", api=Api()) == 4
    assert calls[0]["path_in_repo"] == "resume/T/global_step_4" and os.path.exists(os.path.join(S, "global_step_4", "manifest_full.json"))


def test_mirror_loop_one_pass_in_a_sandbox(tmp_path):   # scripts/ckpt_mirror.sh itself (MIRROR_ONCE=1): a local step reaches the shared disk
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    rb = tmp_path / "rb"; (rb / "scripts").mkdir(parents=True)
    for f in ("ckpt_mirror.sh", "ckpt_lib.py"):
        shutil.copy(os.path.join(here, f), rb / "scripts" / f)
    root, loc, sh = tmp_path / "root", tmp_path / "local", tmp_path / "shared"
    (rb / "scripts" / "env.sh").write_text(f"export OPD_ROOT={root} RUNBOOK={rb} CKPT_ROOT={sh} CKPT_LOCAL={loc}\n")
    (root / "runs" / "T").mkdir(parents=True); cfg(str(root / "runs" / "T"))
    L = str(loc / "T" / "ckpt"); mkstep(L, 3); C.write_marker(L, 3)
    out = subprocess.run(["bash", str(rb / "scripts" / "ckpt_mirror.sh")], env={**os.environ, "MIRROR_ONCE": "1"}, capture_output=True, text=True).stdout
    assert "MIRRORED T 3" in out and C.marker(str(sh / "T" / "ckpt")) == 3 and (root / "control" / "loop_ckpt_mirror").exists()
    (rb / "scripts" / "env.sh").write_text(f"export OPD_ROOT={root} RUNBOOK={rb} CKPT_ROOT={sh}\n")
    out = subprocess.run(["bash", str(rb / "scripts" / "ckpt_mirror.sh")], env={k: v for k, v in os.environ.items() if k != "CKPT_LOCAL"},
                         capture_output=True, text=True).stdout
    assert "CKPT_MIRROR_OFF" in out                                            # no CKPT_LOCAL: the loop does nothing


def test_wiring_z155():   # run_e12 / run_tasks / loops / upload_sync: the local dir only with CKPT_LOCAL, today's behaviour without it
    here = os.path.dirname(os.path.abspath(__file__))
    e12 = open(os.path.join(here, "run_e12.sh")).read()
    assert "trainer.default_local_dir=${CKDIR:-$OUT/ckpt} ${CKKEEP:-}" in e12 and 'CKKEEP="++trainer.max_actor_ckpt_to_keep=2"' in e12
    assert 'echo "WAIT_INPUTS $T$NN: local checkpoint disk low ($SP)"; exit 1' in e12 and "ckpt_lib.py prepare $NAME $OUT/config.yaml" in e12
    assert e12.index("ckpt_lib.py space") < e12.index("ckpt_lib.py prepare") < e12.index("verl.trainer.main_ppo")
    rt = open(os.path.join(here, "run_tasks.sh")).read()
    assert rt.count('[ -n "${CKPT_LOCAL:-}" ] && ensure_loop ckpt_mirror') == 2
    assert "uploader gh_heartbeat weights_replay ckpt_mirror" in open(os.path.join(here, "loops.sh")).read()
    assert '[ -n "${CKPT_LOCAL:-}" ] || for d in $RUN/ckpt/global_step_*; do' in open(os.path.join(here, "upload_sync.sh")).read()


def test_interval_rule():   # the smallest n >= 2 with save / (n x step) <= 0.15; tonight's 8B numbers (save 104-115 s, step 237-262 s)
    assert C.raw_interval(104, 262) == 3 and C.raw_interval(110, 250) == 3
    assert C.raw_interval(115, 250) == 4          # 115 / (3 x 250) = 0.153 > 0.15
    assert C.raw_interval(115, 237) == 4
    assert C.raw_interval(10, 250) == 2           # never below 2
    assert C.raw_interval(75, 250) == 2           # exactly 0.15 at n = 2 counts
    assert C.raw_interval(300, 100) == 20
    # aligned to the registered intermediate upload steps (s25 must exist): 3 -> 5; nothing in the way -> unchanged; impossible -> None
    assert C.aligned_interval(3, 50, [25, 50]) == 5
    assert C.aligned_interval(3, 50, [50]) == 3
    assert C.aligned_interval(2, 100, [25, 50, 75, 100]) == 5
    assert C.aligned_interval(3, 50, [10, 25]) == 5
    assert C.aligned_interval(4, 50, [3, 10]) is None


def log_lines(steps):   # verl console lines; steps: [(step, step_s, save_s or None)]
    out = ["(TaskRunner pid=1) some ray line"]
    for s, t, sv in steps:
        extra = f" - timing_s/save_checkpoint:{sv}" if sv is not None else ""
        out.append(f"(TaskRunner pid=1) step:{s} - actor/pg_loss:0.1 - timing_s/gen:50.0{extra} - timing_s/step:{t} - perf/x:1")
    return out


def test_measure_from_the_train_log():
    assert C.measure(log_lines([(1, 300, None), (2, 250, None)])) is None            # no save yet
    assert C.measure(log_lines([(1, 400, 110)])) is None                            # a save, no plain step yet
    save, med, n = C.measure(log_lines([(1, 400, None), (2, 250, None), (3, 240, None), (4, 260, None), (5, 365, 115)]))
    assert (save, med, n) == (115, 255, 4)                                         # median of 400/250/240/260 = 255; the save step excluded
    save, med, n = C.measure(log_lines([(1, 262, None), (2, 237, None), (3, 250, None), (4, 350, 104)]))
    assert (save, med, n) == (104, 250, 3)


def test_cost_pass_once_per_run_and_interval_for(tmp_path, capsys):
    tl, st = str(tmp_path / "train.log"), str(tmp_path / "control" / "ckpt_interval.json")
    open(tl, "w").write("\n".join(log_lines([(1, 262, None), (2, 237, None)])) + "\n")
    assert C.cost_pass("E12b-X", tl, "r1", st) is None and not os.path.exists(st)     # not measurable yet
    open(tl, "a").write("\n".join(log_lines([(3, 250, None), (4, 350, 104)])[1:]) + "\n")
    r = C.cost_pass("E12b-X", tl, "r1", st)
    assert r["save_freq"] == 3 and r["run_id"] == "r1"
    assert "CKPT_COST E12b-X save 104 s step 250 s ratio 0.42 -> save_freq 3" in open(tl).read()
    assert "CKPT_COST E12b-X" in capsys.readouterr().out
    n_lines = len(open(tl).read().splitlines())
    assert C.cost_pass("E12b-X", tl, "r1", st)["save_freq"] == 3 and len(open(tl).read().splitlines()) == n_lines   # once per run id
    assert C.interval_for("E12b-X", 50, [25, 50], st) == 5 and C.interval_for("E12b-X", 50, [50], st) == 3
    assert C.interval_for("other", 50, [50], st) is None
    # a new run id is measured again
    open(tl, "a").write("\n".join(log_lines([(1, 100, None), (2, 300, 10)])[1:]) + "\n")
    assert C.cost_pass("E12b-X", tl, "r2", st)["run_id"] == "r2"


def test_interval_cli_and_wiring(tmp_path):
    import subprocess
    os.makedirs(tmp_path / "control")
    json.dump({"E12b-X": {"run_id": "r", "save_freq": 3}}, open(tmp_path / "control" / "ckpt_interval.json", "w"))
    here = os.path.dirname(os.path.abspath(__file__))
    run = lambda *a: subprocess.run([sys.executable, os.path.join(here, "ckpt_lib.py"), "interval", *a], capture_output=True, text=True,
                                    env={**os.environ, "OPD_ROOT": str(tmp_path)}).stdout.strip()
    assert run("E12b-X", "50", "25,50") == "5" and run("E12b-X", "50", "50") == "3" and run("nope", "50", "50") == ""
    e12 = open(os.path.join(here, "run_e12.sh")).read()
    assert "SAVE_SET=${SAVE:-}" in e12 and e12.index("SAVE_SET=${SAVE:-}") < e12.index("[ $T = E12e-P0think-8k ] && [ -z \"${SAVE:-}\" ] && SAVE=5")
    i = e12.index("ckpt_lib.py interval $NAME $STEPS \"$UW\"")
    assert e12.index('SAVE=${UW%%,*}') < i < e12.index("trainer.save_freq=$SAVE")
    assert 'if [ -z "$SAVE_SET" ] && [ $SM = 0 ] && [ -n "${CKPT_LOCAL:-}" ] && [[ "$CONTENTS" == *optimizer* ]]; then' in e12
