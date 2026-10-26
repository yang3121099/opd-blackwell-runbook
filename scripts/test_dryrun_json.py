"""[v2.6 S0] tests of the golden-corpus recorder (v26_impl_plan §2.10). Mac-runnable: T1 (the footprint on main), T2 (the
shims), T3 (the fixpoint on a toy runbook), T8 (every refusal and runs/ read of the runners is covered). Node-only
(DRYRUN_NODE_TESTS=1, on the node with its $OPD_ROOT / $CKPT_ROOT): T4 (five real names, read-only), T5 (determinism),
T6 / T7 (agreement with today's DRYRUN checks and known answers).

    python3 -m pytest -q scripts/test_dryrun_json.py"""
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RB = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import dryrun_json as J  # noqa: E402
import dryrun_shim as SH  # noqa: E402
import dryrun_normalize as N  # noqa: E402

NODE = os.environ.get("DRYRUN_NODE_TESTS") == "1"
node_only = pytest.mark.skipif(not NODE, reason="node-only (DRYRUN_NODE_TESTS=1)")


# ------------------------------------------------------------------------------------------------- T1: the footprint
NEW_FILES = {"scripts/dryrun_json.sh", "scripts/dryrun_json.py", "scripts/dryrun_shim.py", "scripts/dryrun_normalize.py",
             "scripts/test_dryrun_json.py"} | {f"scripts/dryrun_shims/{t}" for t in J.SHIMS}
ONE_LINE = {"scripts/task.sh", "scripts/run_tasks.sh", "scripts/control.py", "scripts/control_rules.py"}
ADD_ONLY = {"control/CHANGES.md", "scripts/test_v251.py", "tasks/queue.txt"}


def _git(*a):
    r = subprocess.run(["git", "-C", RB] + list(a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.returncode, r.stdout.decode()


def _s0_range():
    """[v2.5.9z88b] the diff T1 checks: the commit that added dryrun_json.py against its parent once it is committed (S0 on main
    since 15d3b94: later diffs on other branches are not S0's footprint), else the working tree against its merge base."""
    _, added = _git("log", "--diff-filter=A", "--format=%H", "--", "scripts/dryrun_json.py")
    if added.split():
        c = added.split()[-1]
        return [f"{c}~1", c], False
    rc, base = _git("merge-base", "HEAD", "origin/main")
    return ([base.strip()] if rc == 0 and base.strip() else None), True


def test_t1_footprint_new_files_and_one_line_names():
    rng, worktree = _s0_range()
    if not rng:
        pytest.skip("no origin/main to compare with")
    _, st = _git("diff", "--name-status", *rng)
    _, untracked = _git("ls-files", "--others", "--exclude-standard") if worktree else (0, "")
    changes = [l.split("\t") for l in st.strip().split("\n") if l] + [["A", f] for f in untracked.split() if f]
    for kind, f in changes:
        if kind.startswith("A"):
            assert f in NEW_FILES, f"a new file outside the footprint: {f}"
        else:
            assert kind == "M" and (f in ONE_LINE or f in ADD_ONLY), f"an existing file changed: {kind} {f}"
            _, d = _git("diff", "-U0", *rng, "--", f)
            body = [l for l in d.split("\n") if l[:1] in ("+", "-") and not l.startswith(("+++", "---"))]
            if f == "tasks/queue.txt":   # rows added; the #pad line gains the stage b26c and nothing else
                rm = [l[1:] for l in body if l.startswith("-")]
                ad = [l[1:] for l in body if l.startswith("+")]
                assert all(l.startswith("#pad ") and l + " b26c" in ad for l in rm), f"{f}: {rm}"
            elif f in ADD_ONLY:
                assert all(l.startswith("+") for l in body), f"{f}: a line removed or changed"
            else:   # every added line names CORPUS-RECORD (or is the rules version); a removed line is an added one minus the name
                ad = [l[1:] for l in body if l.startswith("+")]
                rm = [l[1:] for l in body if l.startswith("-")]
                assert all("CORPUS-RECORD" in l or "RULES_VERSION" in l for l in ad), f"{f}: {ad}"
                frags = ("CORPUS-RECORD|", "|CORPUS-RECORD", " CORPUS-RECORD) return 0;;")
                for l in rm:
                    assert "RULES_VERSION" in l or any(a.replace(fr, "", 1) == l for a in ad for fr in frags if fr in a), f"{f}: changed {l[:80]}"


# ------------------------------------------------------------------------------------------------------ T2: the shims
@pytest.mark.parametrize("argv,cls", [
    (["-m", "verl.trainer.main_ppo", "trainer.total_training_steps=3"], "train_cmd"),
    (["-m", "verl.trainer.fsdp_sft_trainer"], "train_cmd"),
    (["-m", "pytest", "-q"], "unit_test"),
    (["/rb/code/patches/dense_patch.py"], "patch"),
    (["code/analysis/e12_rollout_stats.py", "x"], "analysis"),
    (["/rb/code/eval/gen_eval.py", "--bench", "math500"], "gen_eval"),
    (["/rb/scripts/fetch_model.py", "pub", "runs/x", "/sb/ckpt/a"], "fetch"),
    (["/rb/scripts/control.py", "event", "T", "x"], "control"),
    (["/rb/scripts/control.py", "place_run", "T"], "control_unlisted"),
    (["/rb/scripts/remote.py", "backoff"], "remote"),
    (["/rb/scripts/remote.py", "put", "a", "b"], "remote"),
    (["-c", "import flash_attn"], "probe_fa"),
    (["-c", "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1]))", "Qwen/Qwen3-4B"], "hf_cache"),
    (["-c", "import importlib.util as u; print(u.find_spec('verl.workers.actor.dp_actor').origin)"], "find_spec"),
    (["-c", "import json; print(1)"], "py_forward"),
    (["-", "a"], "py_forward"),
    (["/rb/scripts/term_probe.py"], "py_forward"),
    (["code/weights/wops.py", "scale", "--out", "/sb/m"], "weights_op"),
    (["-m", "pip", "install", "x"], "pip"),
    (["code/train/build_sft_data.py", "--x"], "data_job"),
    (["/rb/code/agent/prep_tools_parquet.py"], "data_job"),
    (["/somewhere/else.py"], "unknown"),
])
def test_t2_python_classes(argv, cls):
    assert SH.classify_python(argv)[0] == cls


def _shim(tmp, tool, *argv, env=None, stdin=None):
    e = {"DRYRUN_REC": os.path.join(tmp, "rec.jsonl"), "DRYRUN_SB": tmp, "DRYRUN_T0": "1767225600", "PATH": os.environ["PATH"]}
    e.update(env or {})
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "dryrun_shim.py"), tool] + list(argv), env=e,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, input=stdin)
    recs = [json.loads(l) for l in open(e["DRYRUN_REC"])] if os.path.exists(e["DRYRUN_REC"]) else []
    return r.returncode, r.stdout.decode(), recs


def test_t2_bg_env_clock_gpu_net_and_heredocs():
    tmp = tempfile.mkdtemp()
    try:
        rc, _, recs = _shim(tmp, "setsid", "nohup", "bash", "/rb/scripts/upload_sync.sh", "/sb/opd/runs/T",
                            env={"UPLOAD_WEIGHTS": "25,50", "KEEP_BF16": "0", "SECRET_TOKEN": "x"})
        assert rc == 0 and recs[-1]["cls"] == "bg" and recs[-1]["env"] == {"UPLOAD_WEIGHTS": "25,50", "KEEP_BF16": "0"}
        _, a, _ = _shim(tmp, "date", "-u", "+%FT%TZ")
        _, b, _ = _shim(tmp, "date", "-u", "+%s")
        assert a.strip() == "2026-01-01T00:00:00Z" and b.strip() == "1767225601"   # pinned, one second per call
        _, g, _ = _shim(tmp, "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "3")
        _, n, _ = _shim(tmp, "nvidia-smi", "--query-gpu=name", "--format=csv,noheader")
        assert g.strip() == "0" and n.strip().split("\n") == [SH.GPU_NAME] * 8
        _, df, _ = _shim(tmp, "df", "-Pk", tmp)
        assert df.split("\n")[1].split()[3] == str(4 * 1024 ** 3) and df.split("\n")[1].split()[5] == os.path.realpath(tmp)
        for t in ("git", "curl", "wget", "huggingface-cli", "aria2c"):
            rc, _, recs = _shim(tmp, t, "clone", "x")
            assert rc == 1 and recs[-1]["cls"] == "net"
        rc, out, recs = _shim(tmp, "python", "-", stdin=b"import importlib\nfor modname in ('test_opd_dense',):\n  pass\n")
        assert rc == 0 and recs[-1]["cls"] == "unit_test" and "passed" in out
        rc, out, recs = _shim(tmp, "python", "-", "7", stdin=b"import sys; print(int(sys.argv[1]) * 6)\n",
                              env={"DRYRUN_REAL_PY": sys.executable})
        assert rc == 0 and out.strip() == "42" and recs[-1]["cls"] == "py_forward"   # a check: run for real, on its args
        rc, out, recs = _shim(tmp, "python", "-m", "verl.trainer.main_ppo", "trainer.total_training_steps=3")
        assert rc == 0 and re.findall(r"^step:(\d+) - ", out, re.M) == ["1", "2", "3"]
        rc, _, _ = _shim(tmp, "python", "-c", "import flash_attn", env={"VIRTUAL_ENV": "/x/venv-fa", "DRYRUN_FA": "venv:0,venv-fa:1"})
        rc2, _, _ = _shim(tmp, "python", "-c", "import flash_attn", env={"VIRTUAL_ENV": "/x/venv", "DRYRUN_FA": "venv:0,venv-fa:1"})
        assert (rc, rc2) == (0, 1)
        out_dir = os.path.join(tmp, "opd", "runs", "EVAL-x")
        _shim(tmp, "python", "/rb/code/eval/gen_eval.py", "--bench", "math500", "--temperature", "0.6", "--tag", "t1", "--out", out_dir)
        assert os.path.exists(os.path.join(out_dir, "t1_math500_no_think_t0.6.jsonl"))
        _shim(tmp, "python", "/rb/code/eval/gen_eval.py", "--bench", "math500", "--tag", "t2", "--out", "/elsewhere")
        assert not os.path.exists("/elsewhere")   # never outside the sandbox
    finally:
        shutil.rmtree(tmp)


def test_t2_normalize_and_leaks():
    sb = "/base/clean/T"
    ctx = N.Ctx(sb, sb + "/opd", sb + "/ckpt", sb + "/hm", sb + "/hf", sb + "/dr", "/rb", "3,5", 1767225600)
    r = N.normalize({"a": f"{sb}/opd/runs/T started 2026-01-01T00:00:05Z at 1767225603 by /rb/scripts/x.sh",
                     "CUDA_VISIBLE_DEVICES": "3,5"}, ctx)
    assert r == {"a": "${OPD_ROOT}/runs/T started ${T0}+5 at ${T0}+3 by ${RUNBOOK}/scripts/x.sh", "CUDA_VISIBLE_DEVICES": "G1,G2"}
    assert {w for w, _ in N.leak_hits("/" + "home/alice/x hf_" + "a" * 30 + " " + ".".join(["10", "1", "2", "3"]) + " nodebox7 x", ["nodebox7"])} == \
        {"a home directory", "an HF token", "an IP address", "a literal"}
    assert N.leak_hits("${OPD_ROOT}/runs/T 2.5.95 12.9.86") == []


# ------------------------------------------------------------------------------------------------ T3: a toy fixpoint
TOY_TASK_SH = r'''source "$(dirname "$0")/env.sh"; G=$1; T=$2
case $T in
  TOY) bash $RUNBOOK/scripts/toy.sh $G $T;;
  BAD) echo "something else went wrong"; exit 1;;
  *) echo "NOT_WHITELISTED $T"; exit 2;;
esac
'''
TOY_SH = r'''source "$(dirname "$0")/env.sh"; G=$1; T=$2
for P in PROBE-A PROBE-B; do grep -q '"result": "PASS"' $OPD_ROOT/runs/$P/report.json 2> /dev/null || { echo "$P has not passed: $T not started"; exit 3; }; done
[ -f $OPD_ROOT/models/Toy-1B/.fetch_ok ] || { echo "WAIT_INPUTS $T: Toy-1B not in (no .fetch_ok)"; exit 1; }
S=$OPD_ROOT/runs/START/ckpt/global_step_7/actor/huggingface
ls $S/*.safetensors > /dev/null 2>&1 || { echo "missing start checkpoint $S"; exit 1; }
mkdir -p $OPD_ROOT/runs/$T
echo "task: $T" > $OPD_ROOT/runs/$T/config.yaml; echo "started: $(date -u +%FT%TZ)" >> $OPD_ROOT/runs/$T/config.yaml
UPLOAD_WEIGHTS=2 setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$T > /dev/null 2>&1 &
python $RUNBOOK/code/patches/toy_patch.py
python3 -m verl.trainer.main_ppo trainer.total_training_steps=2 data.seed=42 2>&1 | tee -a $OPD_ROOT/runs/$T/train.log
LAST=$(grep -oE "(^|[ )])step:[0-9]+ - " $OPD_ROOT/runs/$T/train.log | tail -1 | grep -oE "[0-9]+")
echo "completed_steps: $LAST" >> $OPD_ROOT/runs/$T/config.yaml
'''


def _toy_runbook(tmp):
    rb = os.path.join(tmp, "rb")
    os.makedirs(os.path.join(rb, "scripts"))
    os.makedirs(os.path.join(rb, "tasks"))
    open(os.path.join(rb, "scripts", "env.sh"), "w").write('export OPD_ROOT=${OPD_ROOT:-$HOME/opd-run}\n'
                                                           'export RUNBOOK=${RUNBOOK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}\n'
                                                           '[ -f $OPD_ROOT/control.env ] && source $OPD_ROOT/control.env\n')
    open(os.path.join(rb, "scripts", "task.sh"), "w").write(TOY_TASK_SH)
    open(os.path.join(rb, "scripts", "toy.sh"), "w").write(TOY_SH)
    open(os.path.join(rb, "scripts", "run_tasks.sh"), "w").write("cpu_task() { case $1 in NONE) return 0;; esac; return 1; }\n")
    open(os.path.join(rb, "scripts", "control_rules.py"), "w").write('RULES_VERSION = "9.9.9"\n')
    open(os.path.join(rb, "tasks", "queue.txt"), "w").write("toy  open   TOY BAD\n")
    for f in ("dryrun_shim.py", "dryrun_normalize.py", "dryrun_json.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(rb, "scripts", f))
    shutil.copytree(os.path.join(HERE, "dryrun_shims"), os.path.join(rb, "scripts", "dryrun_shims"))
    return rb


def test_t3_toy_fixpoint_gates_inputs_start_and_findings():
    tmp = tempfile.mkdtemp()
    try:
        rb = _toy_runbook(tmp)
        out = os.path.join(tmp, "a", "dryrun_json.out")
        for stale in ("corpus/clean/GONE.json", "up/index.json"):   # N1: left by an earlier recording, cleared at the start
            os.makedirs(os.path.dirname(os.path.join(out, stale)), exist_ok=True)
            open(os.path.join(out, stale), "w").write("{}")
        env = {k: v for k, v in os.environ.items() if k not in ("OPD_ROOT", "CKPT_ROOT")}
        r = subprocess.run([sys.executable, os.path.join(rb, "scripts", "dryrun_json.py"), "--runbook", rb, "--base",
                            os.path.join(tmp, "dryrun_json.tmp"), "--out", out, "--profile", "clean", "TOY", "BAD"], env=env)
        assert r.returncode == 0
        rec = json.load(open(os.path.join(out, "corpus", "clean", "TOY.json")))
        assert rec["result"] == "ok" and rec["iterations"] == 5 and rec["gates"] == ["PROBE-A", "PROBE-B"]
        assert [p["path"] for p in rec["plants"]][-4:] == ["opd/models/Toy-1B/.fetch_ok", "opd/models/Toy-1B/model.safetensors",
                                                           "opd/models/Toy-1B/config.json",
                                                           "opd/runs/START/ckpt/global_step_7/actor/huggingface/model.safetensors"]
        assert rec["dispatch"]["runner"] == "${RUNBOOK}/scripts/toy.sh" and rec["dispatch"]["args"] == ["G1", "TOY"]
        assert rec["train"][0]["argv"] == ["-m", "verl.trainer.main_ppo", "trainer.total_training_steps=2", "data.seed=42"]
        assert rec["upload"]["launch"][0]["env"] == {"UPLOAD_WEIGHTS": "2"} and rec["patches"] == ["${RUNBOOK}/code/patches/toy_patch.py"]
        assert rec["config_yaml"] == "task: TOY\nstarted: ${T0}+0\n" and rec["config_tail"] == ["completed_steps: 2"]
        bad = json.load(open(os.path.join(out, "corpus", "clean", "BAD.json")))
        assert bad["result"] == "exit:1" and bad["stdout_tail"][-1] == "something else went wrong"
        f = json.load(open(os.path.join(out, "findings.json")))
        assert {"kind": "task", "profile": "clean", "task": "BAD", "result": "exit:1"} in f
        assert not [x for x in f if x["kind"] in ("leak", "runbook_changed")]
        assert not os.path.exists(os.path.join(tmp, "dryrun_json.tmp"))   # the sandboxes removed
        assert sorted(os.listdir(os.path.join(out, "corpus", "clean"))) == ["BAD.json", "TOY.json"]
        assert not os.path.exists(os.path.join(out, "up"))
        # determinism (the Mac half of T5): a second recording is byte-identical
        out2 = os.path.join(tmp, "b", "dryrun_json.out")
        subprocess.run([sys.executable, os.path.join(rb, "scripts", "dryrun_json.py"), "--runbook", rb, "--base",
                        os.path.join(tmp, "dryrun_json.tmp"), "--out", out2, "--profile", "clean", "TOY", "BAD"], env=env)
        for t in ("TOY", "BAD"):
            assert open(os.path.join(out, "corpus", "clean", f"{t}.json")).read() == open(os.path.join(out2, "corpus", "clean", f"{t}.json")).read()
    finally:
        shutil.rmtree(tmp)


def test_t3_base_guard():
    with pytest.raises(SystemExit):
        J.main(["--base", "/tmp/not-a-sandbox", "TOY"])
    tmp = tempfile.mkdtemp()
    try:   # N1: the output directory is cleared, so it too must carry its name (and lie on the checkpoint disk when one is set)
        keep = os.path.join(tmp, "results", "corpus", "x.json")
        os.makedirs(os.path.dirname(keep))
        open(keep, "w").close()
        with pytest.raises(SystemExit):
            J.main(["--base", os.path.join(tmp, "dryrun_json.tmp"), "--out", os.path.join(tmp, "results"), "TOY"])
        assert os.path.exists(keep) and not os.path.exists(os.path.join(tmp, "dryrun_json.tmp"))
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------------------------- T8: coverage
SAMPLE = {"$TASK": "E10b-P5", "$T": "E10b-P5", "$P": "PROBE-E12B-8K", "$PG": "PROBE-P5SELF", "$G8": "PROBE-P5-8K",
          "$m": "Qwen3-VL-2B-Instruct", "$(basename $f)": "Qwen3-VL-2B-Instruct", "${x#*=}": "Qwen3-4B", "$s": "3", "$SS": "25",
          "$R": "runs/E12b-P5-8k/global_step_25/actor/huggingface", "$M": "/sb/opd/runs/E10b-P5/ckpt/global_step_150/actor/huggingface",
          "${f#$OPD_ROOT/runs/}": "PROBE-DIFF17/labels_17.jsonl",
          "$XARM": "X1", "$(basename $m)": "Qwen2.5-Math-1.5B-Instruct", "$STUDENT": "/sb/opd/runs/A/ckpt/global_step_1/actor/huggingface",
          "$START": "/sb/opd/runs/A/ckpt/global_step_1/actor/huggingface", "$DS": "google__IFEval", "$MODEL": "Qwen/Qwen3-4B"}


def _runners():
    """every script task.sh dispatches"""
    return sorted(set(re.findall(r"bash \$RUNBOOK/scripts/([a-z0-9_]+\.sh)", open(os.path.join(HERE, "task.sh")).read())))


def test_t8_every_refusal_of_the_runners_is_classified():
    msgs = set()
    runners = _runners()
    assert len(runners) > 40 and "run_arm.sh" in runners and "run_vlm.sh" in runners
    for f in runners:
        msgs |= set(re.findall(r'echo "([^"]*(?:not started|WAIT_INPUTS|missing|MISSING|not in |not built)[^"]*)"',
                               open(os.path.join(HERE, f)).read()))
    msgs = {m for m in msgs if not any(re.search(p, m) for p, _ in J.NOT_TERMINAL)}
    assert len(msgs) > 25
    for m in sorted(msgs):
        s = m
        for k in sorted(SAMPLE, key=len, reverse=True):
            s = s.replace(k, SAMPLE[k])
        r = J.classify([s], "E10b-P5")
        assert r is not None and r[0] != "wait_other", f"no fixpoint row for: {m}"
        if r[0] == "gate_check":
            assert r[1]["gate"] in J.GATE_REPORTS, f"a custom gate predicate without a plant: {m}"


def test_t8_every_runs_read_of_the_runners_is_known():
    reads = set()
    for f in ("run_arm.sh", "run_e12.sh", "run_eval.sh", "run_sft.sh", "run_zero.sh", "run_vlm.sh"):
        reads |= set(re.findall(r"\$OPD_ROOT/runs/([A-Za-z0-9${}._-]+/[A-Za-z0-9_./${}-]+)", open(os.path.join(HERE, f)).read()))
    reads = {r for r in reads if not r.endswith("/report.json")}
    for r in sorted(reads):
        assert any(re.match(p, r) for p, _ in J.KNOWN_READS), f"a runs/ read the fixpoint does not cover: {r}"


def test_t8_control_and_remote_subcommands_are_listed():
    subs = set()
    for f in os.listdir(HERE):
        if f.endswith(".sh") and (f.startswith(("run_arm", "run_e12", "run_queue_eval", "env", "dr", "task", "probe_"))):
            subs |= set(re.findall(r"(control|remote)\.py ([a-z_]+)", open(os.path.join(HERE, f)).read()))
    for mod, sub in sorted(subs):
        assert sub in (SH.CONTROL if mod == "control" else SH.REMOTE), f"{mod}.py {sub}: not in the shim's table"


# ------------------------------------------------------------------------------------------------- node-only T4-T7
def _record_node(names, prof="both"):
    ck = os.environ["CKPT_ROOT"]
    top = os.path.join(ck, "dryrun_json.test")   # its own base and output: a live CORPUS-RECORD's are never touched
    shutil.rmtree(top, ignore_errors=True)
    out = os.path.join(top, "dryrun_json.out")
    subprocess.run([sys.executable, os.path.join(HERE, "dryrun_json.py"), "--runbook", RB, "--base", os.path.join(top, "dryrun_json.tmp"),
                    "--out", out, "--profile", prof] + names, check=True)
    return out


T4_NAMES = ["E10b-P5", "E12b-P5tail-8k", "PROBE-TAIL", "EVAL-e10d-P5tail", "BUILD-VENV-IFEVAL"]


@node_only
def test_t4_real_names_read_only():
    out = _record_node(T4_NAMES)
    f = json.load(open(os.path.join(out, "findings.json")))
    assert not [x for x in f if x["kind"] in ("leak", "runbook_changed")]
    for t in T4_NAMES:
        r = json.load(open(os.path.join(out, "corpus", "clean", f"{t}.json")))
        assert [c for c in r["calls"] if c["cls"] == "net"] == []


@node_only
def test_t5_determinism():
    a = _record_node(T4_NAMES, "clean")
    sa = {t: open(os.path.join(a, "corpus", "clean", f"{t}.json")).read() for t in T4_NAMES}
    b = _record_node(T4_NAMES, "clean")
    assert sa == {t: open(os.path.join(b, "corpus", "clean", f"{t}.json")).read() for t in T4_NAMES}


@node_only
def test_t7_known_answers():
    out = _record_node(["E10d-P5-akl", "E12b-P5tail-8k", "E10c-P5tail-c50"], "clean")
    r = lambda t: json.load(open(os.path.join(out, "corpus", "clean", f"{t}.json")))
    a = r("E10d-P5-akl")
    assert a["gates"] == ["PROBE-DENSE", "PROBE-DENSE-4B", "PROBE-4B"] and a["upload"]["launch"][0]["env"]["UPLOAD_WEIGHTS"] == "50"
    b = r("E12b-P5tail-8k")
    assert b["gates"] == ["PROBE-E12B-8K"] and b["upload"]["launch"][0]["env"]["UPLOAD_WEIGHTS"] == "25,50"
    assert "continued:" in r("E10c-P5tail-c50")["config_yaml"]


def test_t9_reap_put_event_never_execute():   # [ops-lead O7] the side-effect subcommands are recorded and never run
    tmp = tempfile.mkdtemp()
    try:
        for argv in (["/rb/scripts/control.py", "reap", "3", "T"], ["/rb/scripts/remote.py", "put", "/x", "logs/y"],
                     ["/rb/scripts/control.py", "event", "T", "done"], ["/rb/scripts/remote.py", "pend", "/x"]):
            rc, out, recs = _shim(tmp, "python3", *argv, env={"DRYRUN_REAL_PY": "/nonexistent/python"})   # an exec would fail
            assert rc == 0 and recs[-1]["cls"] == ("control" if "control" in argv[0] else "remote") and out == ""
        rc, out, _ = _shim(tmp, "python3", "/rb/scripts/control.py", "node")
        assert out.strip() == "${NODE}"
    finally:
        shutil.rmtree(tmp)


@pytest.mark.parametrize("var", ["DRYRUN_SB", "DRYRUN_REAL_PY"])
def test_z88e_entry_points_refuse_inside_a_sandbox(var, tmp_path):   # [v2.5.9z88e] both entries stop at once, writing nothing
    env = {k: v for k, v in os.environ.items() if not k.startswith("DRYRUN_")}; env[var] = "/sb"
    env["OPD_ROOT"] = str(tmp_path / "opd"); env["CKPT_ROOT"] = str(tmp_path / "ck")
    r = subprocess.run(["bash", os.path.join(HERE, "dryrun_json.sh"), "--all"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "CORPUS_RECORD_REFUSED" in r.stdout and not (tmp_path / "opd").exists()
    r = subprocess.run([sys.executable, os.path.join(HERE, "dryrun_json.py"), "--all"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "CORPUS_RECORD_REFUSED" in r.stdout and not (tmp_path / "ck").exists()


def test_z88e_self_names_never_recorded():
    names, _ = J.all_tasks(RB)
    assert "CORPUS-RECORD" not in names and "BUILD-RB-IDX1" not in names and set(J.SELF) >= {"CORPUS-RECORD", "BUILD-RB-IDX1"}


# ------------------------------------------------------------------------------------------------ [v2.5.9z88c] b / c / d, canary
def test_z88c_never_recorded_names():
    assert {"CORPUS-RECORD", "BUILD-RB-IDX1", "HEARTBEAT-UNSTICK", "PROBE-NET2"} <= set(J.SELF)
    names, _ = J.all_tasks(RB)
    assert not set(names) & set(J.SELF)


def test_z88c_named_self_tasks_are_skipped(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("OPD_ROOT", "CKPT_ROOT") and not k.startswith("DRYRUN_")}
    out = tmp_path / "a" / "dryrun_json.out"
    r = subprocess.run([sys.executable, os.path.join(HERE, "dryrun_json.py"), "--runbook", RB, "--base", str(tmp_path / "dryrun_json.tmp"),
                        "--out", str(out), "--profile", "clean", "PROBE-NET2", "CORPUS-RECORD"], env=env, capture_output=True, text=True, timeout=120)
    assert "skips CORPUS-RECORD PROBE-NET2" in r.stdout and not list((out / "corpus" / "clean").glob("*.json"))


def test_z88c_empty_values_match_nothing(tmp_path):   # ops-lead S1: a finally before RUN_ID is set must not match every process
    proc = tmp_path / "proc"; (proc / "7").mkdir(parents=True); (proc / "7" / "environ").write_bytes(b"DRYRUN_RUN_ID=\0PATH=/bin\0")
    assert J.env_pids("DRYRUN_RUN_ID", "", str(proc)) == [] and J.reap_env("DRYRUN_RUN_ID", "", str(proc)) == 0


def test_z88c_child_env_carries_the_run_id():
    old = J.RUN_ID
    try:
        J.RUN_ID = "R-1"
        assert J.child_env("/sb", RB, sys.executable, "x")["DRYRUN_RUN_ID"] == "R-1"
        J.RUN_ID = ""
        assert "DRYRUN_RUN_ID" not in J.child_env("/sb", RB, sys.executable, "x")
    finally:
        J.RUN_ID = old


def test_z88d_recording_the_recorder_refuses_at_once(tmp_path):   # d: a sandboxed CORPUS-RECORD stops at its first line
    base = tmp_path / "dryrun_json.tmp"; base.mkdir()
    rec = J.record_task("CORPUS-RECORD", "clean", "cpu", str(base), RB, J.Real("", ""), "", sys.executable,
                        ",".join(f"{k}:{v}" for k, v in J.CLEAN_FA.items()), 60, "h", "r")
    assert rec["result"] == "ok" and rec["dispatch"]["first_line"].startswith("CORPUS_RECORD_REFUSED"), rec["dispatch"]
    assert not [c for c in rec.get("calls", []) if "dryrun_json" in " ".join(c.get("argv", []))]   # no nested recording


def test_z88c_canary_decisions(tmp_path, monkeypatch):
    proc = tmp_path / "proc"; (proc / "self").mkdir(parents=True)
    rec = lambda t: {"result": "ok", "task": t}
    monkeypatch.setattr(J, "heartbeat_rounds", lambda opd, n=2, limit=600: True)
    monkeypatch.setattr(J, "env_pids", lambda k, v, p="/proc": [])
    st, why = J.canary(RB, str(tmp_path), ["EVAL-gm-06base", "E10b-P5"], rec, str(proc))
    assert st == "ok" and "EVAL-gm-06base" in why                         # the refusal is the real dryrun_json.sh's
    monkeypatch.setattr(J, "env_pids", lambda k, v, p="/proc": [123])
    monkeypatch.setattr(J, "reap_env", lambda k, v, p="/proc": 1)
    assert J.canary(RB, str(tmp_path), ["EVAL-gm-06base"], rec, str(proc))[0] == "fail"   # a leftover
    monkeypatch.setattr(J, "env_pids", lambda k, v, p="/proc": [])
    monkeypatch.setattr(J, "heartbeat_rounds", lambda opd, n=2, limit=600: False)
    assert J.canary(RB, str(tmp_path), ["EVAL-gm-06base"], rec, str(proc))[0] == "fail"   # the heartbeat stalled
    assert J.canary(RB, str(tmp_path), ["EVAL-gm-06base"], rec, str(tmp_path / "noproc"))[0] == "skipped"


@node_only
def test_z88c_reap_env_kills_a_new_session_child():
    tmp = tempfile.mkdtemp()
    try:
        sh = os.path.join(tmp, "two.sh")
        open(sh, "w").write("setsid sleep 60 &\nsleep 60\n")   # a child that leaves the group, as a nested run does
        p = subprocess.Popen(["bash", sh], start_new_session=True, env=dict(os.environ, DRYRUN_SB=tmp))
        time.sleep(1)
        assert len(J.env_pids("DRYRUN_SB", tmp)) >= 2
        assert J.reap_env("DRYRUN_SB", tmp) >= 2 and J.env_pids("DRYRUN_SB", tmp) == []
        p.wait(timeout=10)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_z88g_canary_first_then_all(tmp_path, monkeypatch):   # [v2.5.9z88g] the first --all records the canary only, the second all
    rb = _toy_runbook(str(tmp_path))
    for k in ("OPD_ROOT", "CKPT_ROOT"):
        monkeypatch.delenv(k, raising=False)
    runs = []
    monkeypatch.setattr(J, "canary", lambda *a, **k: runs.append(1) or ("ok", "EVAL-x recorded alone"))
    td = tmp_path / "td"; args = ["--runbook", rb, "--base", str(tmp_path / "dryrun_json.tmp"), "--out", str(tmp_path / "o" / "dryrun_json.out"),
                                   "--profile", "clean", "--all", "--task-dir", str(td), "--no-upload"]
    assert J.main(args) == 0
    rep = json.load(open(td / "report.json"))
    assert rep["result"] == "PASS" and "canary only" in rep["reason"] and len(runs) == 1
    assert not (tmp_path / "o" / "dryrun_json.out" / "corpus" / "clean" / "TOY.json").exists()   # nothing recorded yet
    assert list(td.glob("canary_ok.*")) and J.code_hash(rb) in list(td.glob("canary_ok.*"))[0].name
    (td / ".done").unlink()
    J.main(args)
    assert len(runs) == 1 and (tmp_path / "o" / "dryrun_json.out" / "corpus" / "clean" / "TOY.json").exists()   # no second canary
    assert json.load(open(td / "report.json"))["meta"]["canary"][0] == "passed earlier"
    open(os.path.join(rb, "scripts", "toy.sh"), "a").write("# a code change\n")
    assert not (td / f"canary_ok.{J.code_hash(rb)}").exists()                     # changed code: the canary again
