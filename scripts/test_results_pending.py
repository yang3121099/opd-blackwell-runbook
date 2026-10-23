"""[v2.5.9z65] (Fable / ops-lead 10-23, Runming 12:2x Beijing) a finished reading whose upload fails is done: run_queue_eval.sh hands
it to the results queue (remote.py pend) instead of failing the attempt; term_probe.sh the same; the heartbeat says
phase upload_pending and counts the queued results; the pulse carries both."""
import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402

RQE = open(os.path.join(HERE, "run_queue_eval.sh")).read()
TP = open(os.path.join(HERE, "term_probe.sh")).read()


def upload_block():
    a = RQE.index("if python $RUNBOOK/scripts/remote.py put $OUT runs/$T; then"); b = RQE.index('echo "TASK_FINISHED')
    return RQE[a:b] + 'echo "TASK_FINISHED $T ok=$ok"\n'


@pytest.mark.parametrize("put,pend,ok,done,out", [
    (0, 0, 1, True, None),                    # uploaded: done, no marker work
    (1, 0, 1, True, "UPLOAD_PENDING EVAL-a"),  # upload failed, queued: done all the same (no attempt burnt)
    (1, 1, 1, False, "upload failed"),        # the queue refused it: failed as before
    (1, 0, 0, False, "upload failed"),        # a reading missing: never queued, failed as before
])
def test_the_upload_block(tmp_path, put, pend, ok, done, out):
    rb = tmp_path / "rb" / "scripts"; rb.mkdir(parents=True); calls = tmp_path / "calls"
    (rb / "remote.py").write_text("import os, sys\nopen(os.environ['CALLS'], 'a').write(' '.join(sys.argv[1:]) + chr(10))\n"
                                  f"sys.exit({put} if sys.argv[1] == 'put' else {pend})\n")
    o = tmp_path / "EVAL-a"; o.mkdir(); (o / ".upload_pending").write_text("old\n")
    sh = f"RUNBOOK={tmp_path / 'rb'}; T=EVAL-a; OUT={o}; ok={ok}\n" + upload_block()
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env=dict(os.environ, CALLS=str(calls)))
    assert (o / ".done").exists() == done, r.stdout + r.stderr
    if out:
        assert out in r.stdout
    c = calls.read_text().splitlines()
    assert c[0] == f"put {o} runs/EVAL-a"
    assert (len(c) > 1) == (put == 1 and ok == 1)                                         # pend only after a failed put of complete readings
    if put == 0:
        assert not (o / ".upload_pending").exists()                                       # a successful upload clears an old marker


def test_term_probe_queues_its_report_only_when_the_measurement_succeeded():
    a = TP.index("if python $RUNBOOK/scripts/remote.py put $D/up runs/$T"); blk = TP[a:TP.index("\n", TP.index("else step", a))]
    assert "elif [ $rc = 0 ] && python $RUNBOOK/scripts/remote.py pend $D/up runs/$T" in blk and 'rc=1; fi' in blk
    assert TP.index("> $D/.gpu_released") < a                                             # after z65a's release


def test_heartbeat_phase_and_count():
    hb = open(os.path.join(HERE, "heartbeat.sh")).read()
    assert 'if t["state"] == "done" and os.path.exists(os.path.join(d, ".upload_pending")):' in hb and 't["phase"] = "upload_pending"' in hb
    assert '"results_pending": results_pending(),' in hb
    ns = {"os": os}
    src = hb[hb.index("def results_pending():"):hb.index("live = live_tasks()")]
    root = tempfile.mkdtemp(); os.makedirs(os.path.join(root, "control")); ns["root"] = root
    exec(src, ns)
    assert ns["results_pending"]() == 0
    open(os.path.join(root, "control", "weights_pending.txt"), "w").write(
        "t\truns/EVAL-a\t/x\t1\nt\truns/E1/global_step_50/actor/huggingface\t/y\t2\nt\truns/PROBE-TERM4\t/z\t3\nbad line\n")
    assert ns["results_pending"]() == 2                                                   # results only, not the weights


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "control"))
    monkeypatch.setenv("OPD_ROOT", d); monkeypatch.setattr(C, "ROOT", d); monkeypatch.setattr(C, "CDIR", os.path.join(d, "control"))
    return d


def test_pulse_carries_upload_pending(root):
    st = {"tasks": {"EVAL-a": {"state": "done", "phase": "upload_pending"}, "EVAL-b": {"state": "done"}}, "results_pending": 3}
    p = C.build_pulse(st, "0, 0, 0, 275040\n")
    assert p["tasks"]["EVAL-a"]["phase"] == "upload_pending" and "phase" not in p["tasks"]["EVAL-b"] and p["results_pending"] == 3
    assert C.build_pulse({"tasks": {}, "results_pending": "x"}, "")["results_pending"] is None
