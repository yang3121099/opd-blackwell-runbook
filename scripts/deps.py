#!/usr/bin/env python3
"""deps.py (runbook [z165], Fable board v10, 11-15: Q-095 B's arm sat 20 min because its runner also wants PROBE-4B, which no list named):
everything a task needs on THIS root before it can start, from the two places that decide it:
  1. the queue: the task's dependencies in its stage line (<task>:<dep>+<dep>), followed transitively (each must have .done);
  2. the runner itself: the gates run_arm.sh / run_e12.sh / probes check for a PASS report (e.g. a 4B arm's implicit PROBE-4B) and the
     inputs they refuse to start without (.fetch_ok copies, start checkpoints) - found by dryrun_json.py's fixpoint, which runs the real
     task.sh in a throw-away sandbox (shims: nothing trains, uploads, fetches or touches a GPU) and plants exactly what each refusal names.
With --root <OPD_ROOT> (a node's root; CKPT_ROOT from its paths.env or --ckpt) every item is marked as it stands there:
  QUEUE_DEP <task> done|MISSING;  GATE <gate> PASS|FAIL|MISSING;  INPUT <path> present|MISSING
and the exit code is 0 only when nothing is missing (so `python3 scripts/deps.py E10d-P5q093t --root $OPD_ROOT && task.sh ...`).
Without --root it prints the lists (exit 0). stdlib only; ~30-60 s (the dry run).
Usage: python3 scripts/deps.py <task> [--queue tasks/queue_8gs.txt] [--root <OPD_ROOT>] [--ckpt <CKPT_ROOT>] [--sec 120]"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)


def queue_deps(qfile, task):
    """The task's transitive dependencies from the queue's stage lines (open / auto / locked), in first-seen order."""
    spec = {}
    for l in open(qfile):
        w = l.split()
        if len(w) >= 3 and not l.startswith("#") and w[1] in ("open", "auto", "locked"):
            for x in w[2:]:
                n, _, d = x.partition(":")
                spec[n] = [y for y in d.split("+") if y]
    if task not in spec:
        return None
    out, todo = [], list(spec[task])
    while todo:
        d = todo.pop(0)
        if d not in out:
            out.append(d); todo += spec.get(d, [])
    return out


def runner_needs(task, sec):
    """(gates, input paths, result) from dryrun_json.py's clean-profile record of the task."""
    tmp = tempfile.mkdtemp(prefix="deps_")
    try:
        base, out = os.path.join(tmp, "dryrun_json.tmp"), os.path.join(tmp, "dryrun_json.out"); os.makedirs(base)
        subprocess.run([sys.executable, os.path.join(HERE, "dryrun_json.py"), task, "--profile", "clean", "--base", base, "--out", out,
                        "--no-upload", "--per-task-sec", str(sec), "--runbook", RB], capture_output=True, text=True, timeout=sec * 6 + 60)
        f = glob.glob(os.path.join(out, "corpus", "clean", f"{task}.json"))
        if not f:
            return None, None, "no dry-run record (dryrun_json.py refused the task or timed out)"
        r = json.load(open(f[0]))
        gates = list(r.get("gates") or [])
        inputs = [p["path"] for p in r.get("plants") or [] if not p["path"].startswith("opd/runs/")]   # gate reports are listed as gates
        inputs = [p for p in inputs if not p.endswith("/model.safetensors") or not any(q == p.rsplit("/", 1)[0] + "/.fetch_ok" for q in inputs)]
        return gates, inputs, r.get("result")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ckpt_root(root, given):
    if given:
        return given
    try:
        for l in open(os.path.join(root, "paths.env")):
            if l.startswith("CKPT_ROOT="):
                return l.split("=", 1)[1].strip()
    except OSError:
        pass
    return os.path.join(os.path.dirname(root), "ckpt")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("task"); ap.add_argument("--queue", default=os.path.join(RB, "tasks", "queue_8gs.txt"))
    ap.add_argument("--root"); ap.add_argument("--ckpt"); ap.add_argument("--sec", type=int, default=120)
    a = ap.parse_args()
    qd = queue_deps(a.queue, a.task)
    if qd is None:
        print(f"NOT_IN_QUEUE {a.task} ({os.path.basename(a.queue)})")
        qd = []
    gates, inputs, res = runner_needs(a.task, a.sec)
    if gates is None:
        print(f"DRYRUN_FAILED {a.task}: {res}"); return 2
    missing = 0
    ck = ckpt_root(a.root, a.ckpt) if a.root else None
    for d in qd:
        st = "" if not a.root else (" done" if os.path.exists(os.path.join(a.root, "runs", d, ".done")) else " MISSING")
        missing += st == " MISSING"; print(f"QUEUE_DEP {d}{st}")
    for g in gates:
        st = ""
        if a.root:
            try:
                r = json.load(open(os.path.join(a.root, "runs", g, "report.json"))).get("result")
                st = " PASS" if r == "PASS" else f" {r or 'FAIL'}"
            except (OSError, ValueError):
                st = " MISSING"
            missing += st != " PASS"
        print(f"GATE {g}{st}")
    for p in inputs:
        st = ""
        if a.root:
            q = os.path.join(ck, p[len("ckpt/"):]) if p.startswith("ckpt/") else os.path.join(a.root, p[len("opd/"):] if p.startswith("opd/") else p)
            st = " present" if os.path.exists(q) else " MISSING"; missing += st == " MISSING"
        print(f"INPUT {p}{st}")
    print(f"DEPS {a.task}: {len(qd)} queue deps, {len(gates)} gates, {len(inputs)} inputs (dry run: {res})" + (f"; missing {missing}" if a.root else ""))
    return 1 if a.root and missing else 0


if __name__ == "__main__":
    sys.exit(main())
