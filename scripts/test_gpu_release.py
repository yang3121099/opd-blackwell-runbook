"""[v2.5.9z65a] (ops-lead 22:1x UTC 10-22: GPU0, GPU3 and GPU5 idle at once while their evaluations uploaded) a GPU whose task
finished its GPU work (runs/<task>/.gpu_released) is free for the next task while the task uploads: run_tasks.sh's gpu_held /
free_gpus with stub pgrep / ps / nvidia-smi; the release points of run_queue_eval.sh and term_probe.sh; launch() clears the
marker at every start and skips its reap once released (a failed upload still counts its attempt); the heartbeat and the pulse
say "uploading" with no GPU; setenv GPU_RELEASE=0 holds the GPUs as before."""
import os
import re
import subprocess
import sys
import tempfile
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import control as C  # noqa: E402
import control_rules as R  # noqa: E402

RT = open(os.path.join(HERE, "run_tasks.sh")).read()


def fn(name):
    lines = RT.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(name + "() {"))
    if re.search(r"\}\s*(#.*)?$", lines[i]) and lines[i].count("{") == lines[i].count("}"):
        return lines[i]
    j = next(k for k in range(i + 1, len(lines)) if lines[k] == "}" or lines[k].rstrip().endswith("done; }"))
    return "\n".join(lines[i:j + 1])


@pytest.fixture
def node(tmp_path):
    b = tmp_path / "bin"; b.mkdir()
    (b / "pgrep").write_text('#!/bin/bash\n[ "$1" = -f ] && shift\ngrep -qE -- "$1" "$FAKE/ps.txt"\n')
    (b / "ps").write_text('#!/bin/bash\ncat "$FAKE/ps.txt"\n')
    (b / "nvidia-smi").write_text('#!/bin/bash\ni=""; p=""; for a in "$@"; do [ "$p" = -i ] && i=$a; p=$a; done\n'
                                  'case "$*" in *query-compute-apps*) cat "$FAKE/apps_$i" 2> /dev/null;; *memory.used*) cat "$FAKE/mem_$i" 2> /dev/null || echo 0;; esac\n')
    for f in b.iterdir():
        f.chmod(0o755)
    root = tmp_path / "opd"; (root / "runs" / "EVAL-a").mkdir(parents=True); (root / "runs" / "EVAL-b").mkdir(parents=True)

    class N:
        pass
    n = N(); n.tmp, n.root, n.bin = tmp_path, root, b
    n.ps = lambda *lines: (tmp_path / "ps.txt").write_text("".join(l + "\n" for l in lines))
    n.mark = lambda task, gpu: (root / "runs" / task / ".gpu_released").write_text(f"{gpu} 2026-10-22T22:20:00Z\n")

    def free():
        sh = "\n".join([f"OPD_ROOT={root}; NGPU=2", fn("cenv"), fn("gpu_held"), fn("free_gpus"), "echo $(free_gpus)"])
        r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True, env=dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", FAKE=str(tmp_path)))
        assert r.returncode == 0, r.stderr
        return r.stdout.split()
    n.free = free
    return n


TASK_A = ["bash -c bash /rb/scripts/task.sh 0 EVAL-a > /x/logs/EVAL-a.log 2>&1", "bash /rb/scripts/task.sh 0 EVAL-a",
          "bash /rb/scripts/run_queue_eval.sh 0 EVAL-a"]


def test_a_task_holds_its_gpu_until_it_releases_it(node):
    node.ps(*TASK_A)
    assert node.free() == ["1"]                                   # generating: GPU0 held, GPU1 free
    node.mark("EVAL-a", 0)
    assert node.free() == ["0", "1"]                              # its GPU work over, uploading: GPU0 free for the next task


def test_the_next_task_on_a_released_gpu_holds_it_again(node):
    node.ps(*TASK_A, "bash /rb/scripts/task.sh 0 EVAL-b"); node.mark("EVAL-a", 0)
    assert node.free() == ["1"]                                   # EVAL-b runs there now: never given out a third time


def test_held_when_switched_off_or_the_marker_is_another_gpus(node):
    node.ps(*TASK_A); node.mark("EVAL-a", 1)
    assert node.free() == ["1"]                                   # the marker names GPU1: GPU0 still held
    node.mark("EVAL-a", 0); (node.root / "control.env").write_text("export GPU_RELEASE=0\n")
    assert node.free() == ["1"]                                   # setenv GPU_RELEASE=0: as before z65a
    (node.root / "control.env").write_text("export GPU_RELEASE=0\nexport GPU_RELEASE=1\n")
    assert node.free() == ["0", "1"]                              # the last setting wins, as cenv reads it


def test_still_held_while_nvidia_smi_shows_processes_or_memory(node):
    node.ps(*TASK_A); node.mark("EVAL-a", 0)
    (node.tmp / "apps_0").write_text("4242\n")
    assert node.free() == ["1"]                                   # a process left on it: not free whatever the marker says
    (node.tmp / "apps_0").unlink(); (node.tmp / "mem_0").write_text("90000\n")
    assert node.free() == ["1"]                                   # memory still in use: not free


def test_an_adhoc_evaluation_or_an_unread_task_holds_the_gpu(node):
    node.ps("bash /rb/scripts/run_adhoc_eval.sh 0 512"); node.mark("EVAL-a", 0)
    assert node.free() == ["1"]                                   # an ad-hoc evaluation writes no marker
    node.ps("bash -c bash /rb/scripts/task.sh 0 EVAL-a > /x/logs/EVAL-a.log 2>&1")
    assert node.free() == ["1"]                                   # only the wrapper seen, no task name read: held


def test_launch_clears_the_marker_and_skips_its_reap_once_released():
    la = RT[RT.index("launch() {"):RT.index("stage_lines() {")]
    assert la.index("rm -f $OPD_ROOT/runs/$2/.gpu_released") < la.index("task.sh $1 $2")
    assert "[ $1 = cpu ] || [ -f $OPD_ROOT/runs/$2/.gpu_released ] || for _g in" in la
    # a failed upload after the release: the same .failed / attempts path as before (no attempt is spared, none added)
    assert la.count("attempts: ") == 2 and "gpu_released" not in la[la.index("else a="):]


def test_run_queue_eval_releases_after_its_last_gpu_step_and_before_the_upload():
    s = open(os.path.join(HERE, "run_queue_eval.sh")).read()
    rel = s.index('> $OUT/.gpu_released')
    assert s.index("rm -f $OUT/.gpu_released") < s.index('echo "EVAL_START') < s.index("forced_stop.py") < rel
    assert s.rindex('python $RUNBOOK/scripts/control.py reap $G $T', 0, rel) > s.index("forced_stop.py")   # reaped first
    assert rel < s.index('$1 == "#pair"') < s.index('$1 == "#budget"') < s.index("remote.py put $OUT runs/$T")
    assert s.index("rm -f $OUT/.gpu_released") > s.index('if [ "$G" = prefetch ]')   # a prefetch never clears a running task's marker
    assert "CUDA_VISIBLE_DEVICES=$G" not in s[rel:]                                # nothing after the release uses the GPU


def test_term_probe_releases_after_its_measurement():
    s = open(os.path.join(HERE, "term_probe.sh")).read()
    rel = s.index("> $D/.gpu_released")
    assert s.index("rm -f $D/.gpu_released") < s.index("term_probe.py --teacher") < rel < s.index("remote.py put $D/up")
    assert "CUDA_VISIBLE_DEVICES=$G" not in s[rel:]


def test_heartbeat_marks_the_task_uploading_without_a_gpu():
    s = open(os.path.join(HERE, "heartbeat.sh")).read()
    assert 'if lv and gpu_release_on and os.path.exists(os.path.join(d, ".gpu_released")):' in s
    assert 't["phase"] = "uploading"; t["released_gpu"] = t["gpu"]; t["gpu"] = None; t["gpus"] = []' in s
    assert s.index('t["gpus"] = lv[2] if lv else []') < s.index('t["phase"] = "uploading"')


@pytest.fixture
def root(monkeypatch):
    d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "control"))
    monkeypatch.setenv("OPD_ROOT", d); monkeypatch.setattr(C, "ROOT", d); monkeypatch.setattr(C, "CDIR", os.path.join(d, "control"))
    return d


def test_pulse_says_uploading(root):
    st = {"tasks": {"EVAL-a": {"state": "running", "gpu": None, "gpus": [], "kind": "eval", "phase": "uploading"},
                    "EVAL-b": {"state": "running", "gpu": 0, "gpus": [0], "kind": "eval", "phase": "something/else"}}}
    p = C.build_pulse(st, "0, 0, 0, 275040\n")
    assert p["tasks"]["EVAL-a"]["phase"] == "uploading" and p["tasks"]["EVAL-a"]["gpu"] is None
    assert "phase" not in p["tasks"]["EVAL-b"] and p["tasks"]["EVAL-b"]["gpu"] == 0


@pytest.mark.parametrize("kv", ["GPU_RELEASE=0", "GPU_RELEASE=1"])
def test_setenv_gpu_release(kv):   # [2.5.80]
    R.parse({"cmd": f"setenv {kv}", "reason": "t"})
    assert tuple(map(int, R.RULES_VERSION.split("."))) >= (2, 5, 80)


def _release_block(src, start, end):
    return src[src.index(start):src.index(end)]


@pytest.mark.parametrize("script,start,end,out", [
    ("run_queue_eval.sh", "# [v2.5.9z65a] (ops-lead 22:1x UTC 10-22) the GPU work is over", "while read -r PTAG PREFS", "OUT"),
    ("term_probe.sh", "python $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true\necho", "cp $D/report.json", "D")])
def test_the_release_lines_write_the_marker_and_hide_the_gpu(tmp_path, script, start, end, out):   # [ops-lead's A1 condition]
    s = open(os.path.join(HERE, script)).read()
    blk = _release_block(s, start, end)
    assert "export CUDA_VISIBLE_DEVICES=" in blk and blk.index("export CUDA_VISIBLE_DEVICES=") > blk.index(".gpu_released")
    rb = tmp_path / "rb" / "scripts"; rb.mkdir(parents=True); (rb / "control.py").write_text("import sys; open(sys.argv[0] + '.calls', 'a').write(' '.join(sys.argv[1:]) + chr(10))\n")
    d = tmp_path / "run"; d.mkdir()
    sh = f'step() {{ echo "$*"; }}; export CUDA_VISIBLE_DEVICES=3; RUNBOOK={tmp_path / "rb"}; G=3; T=EVAL-a; {out}={d}\n{blk}\n' \
         'python3 -c "import os; print(repr(os.environ.get(\'CUDA_VISIBLE_DEVICES\')))"'
    r = subprocess.run(["bash", "-c", sh], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines()[-1] == "\'\'"                                       # the GPU hidden from every later step
    assert (d / ".gpu_released").read_text().split()[0] == "3" and "GPU_RELEASED" in r.stdout
    assert (rb / "control.py.calls").read_text().split()[:3] == ["reap", "3", "EVAL-a"]            # reaped before the marker
