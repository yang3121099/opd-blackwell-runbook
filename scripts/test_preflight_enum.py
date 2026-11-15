"""[z168] (Fable board v20 rule 9 / v22; Runming 11-15: no more "a new machine finds a missing whitelist entry / gate") the multi-GPU tasks
and the board's arms go through the preflight dry run on the CPU (dryrun_json.py's clean profile: the real task.sh up to the launch) on the
8gs queue: every #gpus > 1 task of queue_8gs.txt and every arm of tasks/board_arms.txt reaches the launch with as many GPUs as its row /
board line says (the count task.sh now checks); a board arm's line and its queue row agree; a "pending" arm is not in the runbook yet.
The full enumeration (every task, both queues) is scripts/preflight_all.py (~8 min, not a unit test). ~3 min. Stdlib only."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import dryrun_json as D  # noqa: E402
import preflight_all as P  # noqa: E402

Q8 = os.path.join(RB, "tasks", "queue_8gs.txt")


def board():
    out = {}
    for l in open(os.path.join(RB, "tasks", "board_arms.txt")):
        w = l.split()
        if w and not l.startswith("#"):
            assert len(w) >= 3 and w[1] == "#gpus" and w[2] in "12345678", l
            out[w[0]] = (int(w[2]), len(w) > 3 and w[3] == "pending:")
    return out


def test_board_lines_agree_with_the_queue_rows():
    rows = {t: n for t, n in D.queue_gpus(Q8).items()}
    with open(Q8) as f:
        rowed = {l.split()[1]: int(l.split()[2]) for l in f if l.startswith("#gpus ") and len(l.split()) == 3}
    for arm, (n, pending) in board().items():
        if arm in rowed:
            assert rowed[arm] == n, f"{arm}: board_arms.txt says #gpus {n}, queue_8gs.txt {rowed[arm]}"
        assert rows.get(arm, 1) == n, arm   # dryrun_json's count = gpus_lib's (the row, else the board line)


def test_multi_gpu_tasks_and_board_arms_reach_the_launch():
    gpus = D.queue_gpus(Q8); b = board()
    known = set(D.all_tasks(RB)[0]) | set(D.queue_tasks(Q8))
    live = [a for a, (_, pending) in b.items() if not pending]
    for arm, (_, pending) in b.items():
        if pending:
            assert arm not in known, f"{arm} is in the runbook now: drop its 'pending' mark in tasks/board_arms.txt"
    names = sorted({t for t in D.queue_tasks(Q8) if gpus.get(t, 1) > 1} | set(live))
    recs = P.record(names, "8gs")
    missing = sorted(set(names) - set(recs))
    assert not missing, f"no dry-run record (not a task the runbook knows?): {missing}"
    bad = {t: P.why(r) for t, r in recs.items() if r["result"] != "ok"}
    assert not bad, f"multi-GPU / board tasks whose preflight dry run does not reach the launch: {bad}"
    for t, r in recs.items():
        g = (r.get("dispatch") or {}).get("gpu_arg")
        assert g and len(g.split(",")) == gpus.get(t, 1), (t, g, gpus.get(t))
    assert len(names) > 20
