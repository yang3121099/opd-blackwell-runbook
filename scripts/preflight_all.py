"""[z168] (Fable board v20 rule 9 / v22) the FULL preflight enumeration - every task the runbook knows through the preflight dry run on the
CPU (dryrun_json.py's clean profile: the real task.sh up to the launch). Too slow for the unit tests (~8 min); run it before a push that
touches task.sh / the runners / the queues, and on a node before a new queue goes live. The unit test (test_preflight_enum.py) runs the
multi-GPU tasks and the board arms only.
  A. every task on the default queue: all reach the launch except KNOWN_NOT_OK (each with its reason; the set must match exactly);
  B. every task of the 8gs queue with --queue 8gs: all reach the launch.
Usage: python3 scripts/preflight_all.py   (exit 0 = all as expected, 1 = a difference, listed)"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dryrun_json as D  # noqa: E402

# runbook z168 (11-15): the tasks whose clean dry run already ended before the launch (present before z168, unchanged by it)
KNOWN_NOT_OK = {
    "PROBE-TERM": "term_probe.sh exits 1 in the clean sandbox (its reading inputs are not plantable by the fixpoint)",
    "PROBE-TERM2": "as PROBE-TERM", "PROBE-TERM3": "as PROBE-TERM", "PROBE-TERM4": "as PROBE-TERM", "PROBE-TERM4-06B": "as PROBE-TERM",
    "PROBE-TERM4-4B": "as PROBE-TERM", "PROBE-TERM4-EOSW": "as PROBE-TERM", "PROBE-TERM4-FRET": "as PROBE-TERM",
    "PROBE-WDIFF": "probe_wdiff exits 1 in the clean sandbox (5 models' weights not plantable)",
    "DIAG-WALL-BIMODAL": "DIAG_FAILED in the clean sandbox (a diagnostic of past readings)",
    "DIAG-VLM-LOOP2": "max_iterations: waits on a VLM reading the fixpoint does not plant",
}


def record(tasks, queue, sec=60):
    tmp = tempfile.mkdtemp(prefix="pfenum_")
    try:
        base, out = os.path.join(tmp, "dryrun_json.tmp"), os.path.join(tmp, "dryrun_json.out"); os.makedirs(base)
        env = {k: v for k, v in os.environ.items() if k not in ("OPD_ROOT", "CKPT_ROOT", "QUEUE_FILE", "CKPT_LOCAL")}
        cmd = [sys.executable, os.path.join(HERE, "dryrun_json.py"), *tasks, "--profile", "clean", "--base", base, "--out", out, "--no-upload",
               "--per-task-sec", str(sec), "--runbook", RB] + (["--queue", queue] if queue else [])
        subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        return {json.load(open(f))["task"]: json.load(open(f)) for f in glob.glob(os.path.join(out, "corpus", "clean", "*.json"))}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def why(r):
    return (r["result"], str((r.get("dispatch") or {}).get("first_line") or "")[:120])


def main():
    problems = []
    recs = record(["--all"], None)
    bad = {t: why(r) for t, r in recs.items() if r["result"] != "ok"}
    problems += [f"A: {t} no longer reaches the launch: {bad[t]}" for t in sorted(set(bad) - set(KNOWN_NOT_OK))]
    problems += [f"A: {t} reaches the launch now: drop it from KNOWN_NOT_OK" for t in sorted(set(KNOWN_NOT_OK) - set(bad))]
    names = sorted(set(D.queue_tasks(os.path.join(RB, "tasks", "queue_8gs.txt"))))
    r8 = record(names, "8gs")
    problems += [f"B: {t} (8gs) has no record" for t in sorted(set(names) - set(r8))]
    problems += [f"B: {t} (8gs) does not reach the launch: {why(r)}" for t, r in sorted(r8.items()) if r["result"] != "ok"]
    print(f"PREFLIGHT_ALL A {len(recs)} tasks ({len(bad)} not ok, {len(KNOWN_NOT_OK)} known), B {len(r8)} of {len(names)} 8gs tasks")
    for p in problems:
        print("  " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
