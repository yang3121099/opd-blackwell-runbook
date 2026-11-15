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
Without --root it prints the lists (exit 0). Exit 2 = no conclusion: the dry run failed (DRYRUN_FAILED, with its last lines) or ended
before the launch (DRYRUN_INCOMPLETE). [z167] The dry run's sandbox is made on the checkpoint disk ($CKPT_ROOT / --ckpt / the root's
paths.env), as dryrun_json.py requires there. Do not pipe it (`| tail` hides the exit code) when it gates a start. stdlib only; ~30-60 s (the dry run).
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


def dry_record(task, sec, ck=None, profile="clean", root=None, queue=None):
    """[z168] dryrun_json.py's record of the task under a profile (clean: synthetic plants, the fixpoint; node: this root's control.env
    and gate reports, the refusal the node would give) and queue profile; (record or None, the dry run's last lines)."""
    ck = ck or os.environ.get("CKPT_ROOT") or None
    tmp = tempfile.mkdtemp(prefix="deps_", dir=ck if ck and os.path.isdir(ck) else None)
    env = dict(os.environ, **({"CKPT_ROOT": ck} if ck else {}), **({"OPD_ROOT": root} if root else {}))
    try:
        base, out = os.path.join(tmp, "dryrun_json.tmp"), os.path.join(tmp, "dryrun_json.out"); os.makedirs(base)
        cmd = [sys.executable, os.path.join(HERE, "dryrun_json.py"), task, "--profile", profile, "--base", base, "--out", out, "--no-upload",
               "--per-task-sec", str(sec), "--runbook", RB] + (["--queue", queue] if queue is not None else [])
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=sec * 6 + 60, env=env)
            tail = " | ".join((p.stdout + p.stderr).strip().splitlines()[-3:])
        except subprocess.TimeoutExpired:
            tail = f"timed out after {sec * 6 + 60} s"
        f = glob.glob(os.path.join(out, "corpus", profile, f"{task}.json"))
        return (json.load(open(f[0])) if f else None), tail
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def runner_needs(task, sec, ck=None):
    """(gates, input paths, result) from dryrun_json.py's clean-profile record of the task. [z167] The sandbox goes on the checkpoint disk
    when there is one (dryrun_json.py refuses a sandbox / output outside $CKPT_ROOT: executor B's first run on uk-8 11-15)."""
    ck = ck or os.environ.get("CKPT_ROOT") or None
    tmp = tempfile.mkdtemp(prefix="deps_", dir=ck if ck and os.path.isdir(ck) else None)
    env = dict(os.environ, **({"CKPT_ROOT": ck} if ck else {}))
    try:
        base, out = os.path.join(tmp, "dryrun_json.tmp"), os.path.join(tmp, "dryrun_json.out"); os.makedirs(base)
        try:
            p = subprocess.run([sys.executable, os.path.join(HERE, "dryrun_json.py"), task, "--profile", "clean", "--base", base, "--out", out,
                                "--no-upload", "--per-task-sec", str(sec), "--runbook", RB], capture_output=True, text=True, timeout=sec * 6 + 60, env=env)
            tail = " | ".join((p.stdout + p.stderr).strip().splitlines()[-3:])
        except subprocess.TimeoutExpired:
            tail = f"timed out after {sec * 6 + 60} s"
        f = glob.glob(os.path.join(out, "corpus", "clean", f"{task}.json"))
        if not f:
            return None, None, f"no dry-run record: {tail or 'dryrun_json.py printed nothing'}"
        r = json.load(open(f[0]))
        gates = list(r.get("gates") or [])
        inputs = [p["path"] for p in r.get("plants") or [] if not p["path"].startswith("opd/runs/")]   # gate reports are listed as gates
        inputs = [p for p in inputs if not p.endswith("/model.safetensors") or not any(q == p.rsplit("/", 1)[0] + "/.fetch_ok" for q in inputs)]
        return gates, inputs, r.get("result")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def queue_file(root):
    """The node's queue file as env.sh picks it: tasks/queue_<control/queue_name>.txt, else tasks/queue.txt."""
    try:
        qn = "".join(c for c in open(os.path.join(root, "control", "queue_name")).read() if c.isalnum() or c == "_")
        if qn:
            return os.path.join(RB, "tasks", f"queue_{qn}.txt")
    except (OSError, TypeError):
        pass
    return os.path.join(RB, "tasks", "queue.txt")


def argv_of(rec, key):
    for tr in rec.get("train") or []:
        for x in tr.get("argv") or []:
            if x.startswith(key + "="):
                return x.split("=", 1)[1]
    return None


def real_path(p, root, ck):
    return p.replace("${OPD_ROOT}", root).replace("${CKPT_ROOT}", ck) if p else p


def du_bytes(d, pattern="*.safetensors"):
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, pattern)))


def preflight(a, ck):
    """[z168] Fable board v20 rule 9, the four checks of a start on THIS root (the gates / inputs are above): the node's own refusal (the
    node profile: this root's control.env and gate reports), and for a training where each save goes (trainer.default_local_dir through
    its ckpt link), how often (save_freq, the effective SAVE), how big (~14 bytes / parameter: fp32 weights + Adam + the bf16 copy, from
    the start model's safetensors), the free space there, and where a reclaimed run resumes from. Returns the number of problems."""
    bad = 0
    node, tail = dry_record(a.task, a.sec, ck, "node", a.root)
    if node is None:
        print(f"NODE_DRYRUN_FAILED {tail}"); bad += 1
    elif node.get("result") != "ok":
        print(f"NODE_REFUSAL {node.get('result')}: {str((node.get('dispatch') or {}).get('first_line') or '')[:200]}"); bad += 1
    else:
        print("NODE_DRYRUN ok (this root's control.env and gate reports)")
    rec = node if node and node.get("train") else None
    if not rec:
        print("TRAIN none (not a training: no save / resume checks)"); return bad
    sf, ld, steps, mode = argv_of(rec, "trainer.save_freq"), argv_of(rec, "trainer.default_local_dir"), argv_of(rec, "trainer.total_training_steps"), argv_of(rec, "trainer.resume_mode")
    d = real_path(ld, a.root, ck); dd = os.path.realpath(d) if d and os.path.lexists(d) else os.path.join(ck, a.task, "ckpt")
    probe = dd
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free if probe else 0
    m = real_path(argv_of(rec, "actor_rollout_ref.model.path"), a.root, ck)
    per = int(du_bytes(m) / 2 * 14) if m and os.path.isdir(m) else 0
    n = (int(steps) // int(sf)) if sf and steps and sf.isdigit() and steps.isdigit() and int(sf) > 0 else 0
    print(f"SAVE every {sf} steps of {steps} ({n} saves) to {dd}; ~{per / 1e9:.1f} GB a save (start model {m}); free there {free / 1e9:.0f} GB")
    if per and free < 2 * per:
        print("DISK_LOW: less than two saves' room"); bad += 1
    latest = os.path.join(dd, "latest_checkpointed_iteration.txt")
    print(f"RESUME mode={mode}: " + (f"from step {open(latest).read().strip()} ({latest})" if os.path.isfile(latest) else "a fresh start (no saved state)"))
    return bad + retain(rec, a.root, sf, steps)


def node_env(root, key):
    """the node's value: the environment (preflight.sh sources env.sh), else the root's control.env / paths.env / node.env"""
    v = os.environ.get(key)
    for f in ("paths.env", "node.env", "control.env"):
        try:
            for l in open(os.path.join(root, f)):
                if l.strip().startswith((key + "=", "export " + key + "=")):
                    v = l.split("=", 1)[1].split("#")[0].strip().strip("'\"")
        except OSError:
            pass
    return v or ""


def retain(rec, root, sf, steps, now=None):
    """[z168] (Fable board v22, the c100 s25 loss 11-15) every registered intermediate step (the upload's UPLOAD_WEIGHTS < the final step)
    must survive until it is read: kept by the local rotation itself (no CKPT_LOCAL: no rotation; with CKPT_LOCAL verl keeps the newest
    2, so at most one save after it), else by keep_steps.sh (its loop alive on this root, control/loop_keep_steps < 5 min old), else by
    the upload (an HF / ModelScope credential in the environment, not UPLOAD_DRY). None of the three -> RETAIN ... MISSING."""
    import time
    up = ((rec.get("upload") or {}).get("launch") or [{}])[0].get("env", {}).get("UPLOAD_WEIGHTS", "all")
    if not (sf and steps and sf.isdigit() and steps.isdigit()) or up == "all":
        print(f"RETAIN registered steps {up}: nothing to check"); return 0
    sf, steps = int(sf), int(steps); uw = sorted({int(x) for x in up.split(",") if x.isdigit() and int(x) < steps})
    local = node_env(root, "CKPT_LOCAL")
    try:
        alive = (now or time.time()) - int(open(os.path.join(root, "control", "loop_keep_steps")).read().strip()) < 300
    except (OSError, ValueError):
        alive = False
    upl = bool(os.environ.get("HF_TOKEN") or os.environ.get("MS_TOKEN")) and os.environ.get("UPLOAD_DRY", "0") != "1"
    bad = 0
    for u in uw:
        saves_after = len({k for k in range(u + 1, steps + 1) if k % sf == 0} | {steps})
        if not local:
            how = "no local rotation (checkpoints on the shared disk)"
        elif saves_after <= 1:
            how = f"the local rotation keeps it (keep 2, {saves_after} save after it)"
        elif alive:
            how = f"keep_steps.sh (loop alive) - {saves_after} saves after it would rotate it away"
        elif upl:
            how = f"the upload (a credential set) - {saves_after} saves after it would rotate it away locally"
        else:
            print(f"RETAIN s{u} MISSING: save every {sf} of {steps} with CKPT_LOCAL ({saves_after} saves after it, verl keeps 2), keep_steps.sh "
                  "not running on this root (run_tasks starts it; or bash scripts/keep_steps.sh), no upload credential"); bad += 1; continue
        print(f"RETAIN s{u} ok: {how}")
    return bad


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
    ap.add_argument("--gpus", help="[z168] the GPU list it will get: checked with gpus_lib.sh's gpus_check, task.sh's own check")
    ap.add_argument("--preflight", action="store_true", help="[z168] also the node's own refusal (node profile), where each save goes / how "
                    "big / the free space there, and where a reclaimed run resumes from (scripts/preflight.sh)")
    a = ap.parse_args()
    if a.root and a.queue == os.path.join(RB, "tasks", "queue_8gs.txt"):
        a.queue = queue_file(a.root)   # [z168] the root's own queue (control/queue_name), as env.sh picks it
    qd = queue_deps(a.queue, a.task)
    if qd is None:
        print(f"NOT_IN_QUEUE {a.task} ({os.path.basename(a.queue)})")
        qd = []
    ck = ckpt_root(a.root, a.ckpt) if a.root else (a.ckpt or None)
    gates, inputs, res = runner_needs(a.task, a.sec, ck)
    if gates is None:
        print(f"DRYRUN_FAILED {a.task}: {res}"); return 2
    missing = 0
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
    if a.gpus:   # [z168] the same function task.sh refuses with
        q = os.environ.get("QUEUE_FILE") or queue_file(a.root)
        r = subprocess.run(["bash", "-c", f'source "{os.path.join(HERE, "gpus_lib.sh")}"; gpus_check "$0" "$1" "$2" "$3"', a.task, a.gpus, q,
                            os.environ.get("GPUS_MAX", "")], capture_output=True, text=True)
        ok = r.returncode == 0; missing += not ok
        print(f"GPUS {a.gpus} " + (f"ok ({os.path.basename(q)})" if ok else f"REFUSED: {r.stdout.strip()}"))
    if a.preflight and a.root:
        missing += preflight(a, ck)
    if res != "ok":   # [z167] the dry run did not reach the launch: the lists above may be incomplete - no conclusion
        print(f"DRYRUN_INCOMPLETE {a.task}: the dry run ended '{res}', not at the launch; the lists may miss gates (exit 2)"); return 2
    return 1 if a.root and missing else 0


if __name__ == "__main__":
    sys.exit(main())
