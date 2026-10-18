"""Problem-level paired comparison of two evaluations (the lab's code/analysis/paired_bootstrap.py, same method; runbook
v2.5.6r runs it on the node at the end of a queue evaluation, reviewer A1 2026-10-18 06:3x UTC).
Each input is a gen_eval jsonl (one line per generation: pid, correct) or a per-problem score file written by
`--reduce` ({"pid": [n_correct, n_samples], ...}). Each run is reduced to its per-problem accuracy (mean over its own
samples; the two runs may have different numbers of samples per problem, which is reported). Difference = mean over
common problems of acc_a - acc_b, with a percentile bootstrap over problems (10000 resamples, seed 0) and the win /
loss counts (problems where acc_a > acc_b, acc_a < acc_b).
Usage: python paired_bootstrap.py --reduce <jsonl> > <scores.json>
       python paired_bootstrap.py <a: jsonl or scores.json> <b: jsonl or scores.json> [--label A-B]
       python paired_bootstrap.py --pairs <out.json> <a.jsonl> <b1> [<b2> ...]   (one entry per b; a missing b is listed)
"""
import json
import os
import random
import sys


def reduce(path):
    s = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            c = s.setdefault(r["pid"], [0, 0])
            c[0] += int(bool(r["correct"])); c[1] += 1
    return s


def load(path):
    if path.endswith(".jsonl"):
        return reduce(path)
    return json.load(open(path))


def paired(A, B, label):
    pids = sorted(set(A) & set(B))
    da = [A[p][0] / A[p][1] for p in pids]; db = [B[p][0] / B[p][1] for p in pids]
    d = [x - y for x, y in zip(da, db)]
    n = len(d); mean = sum(d) / n
    rng = random.Random(0); boots = []
    for _ in range(10000):
        boots.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return {"label": label, "problems": n, "acc_a": round(100 * sum(da) / n, 2), "acc_b": round(100 * sum(db) / n, 2),
            "diff": round(100 * mean, 2), "ci95": [round(100 * boots[249], 2), round(100 * boots[9749], 2)],
            "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d), "ties": sum(x == 0 for x in d),
            "samples_per_problem": [sorted({A[p][1] for p in pids}), sorted({B[p][1] for p in pids})]}


def main():
    a = sys.argv[1:]
    if a[0] == "--reduce":
        print(json.dumps(reduce(a[1]))); return
    if a[0] == "--pairs":
        out, sc, refs = a[1], a[2], a[3:]
        A = load(sc); res = {"reading": os.path.basename(sc), "method": "paired bootstrap over problems, 10000 resamples, seed 0, 95% percentile", "pairs": []}
        for r in refs:
            if not os.path.isfile(r):
                res["pairs"].append({"ref": os.path.relpath(r, os.path.dirname(os.path.dirname(r))), "error": "missing"}); continue
            p = paired(A, load(r), f"{os.path.basename(sc)} - {os.path.basename(r)}")
            p["ref"] = os.path.join(os.path.basename(os.path.dirname(r)), os.path.basename(r)); res["pairs"].append(p)
        json.dump(res, open(out, "w"), indent=1); print(json.dumps(res)); return
    label = a[a.index("--label") + 1] if "--label" in a else "a-b"
    print(json.dumps(paired(load(a[0]), load(a[1]), label)))


if __name__ == "__main__":
    main()
