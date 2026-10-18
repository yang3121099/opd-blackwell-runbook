"""Budget-dependence curves from long generations (runbook v2.5.7g; reviewer 2026-10-18 18:4x UTC, readings A: where
does P5 - base turn from negative to positive as the generation budget grows).
One generation at a long budget (e.g. 32768) stands for every shorter budget B: a sample that ended within B tokens
counts as it was graded; one that did not is cut to its first B tokens and graded again (gen_eval.grade, the EVAL-v1
grader: last \\boxed{} only), and counts as truncated at B. This equals generating at B with the same sampler (the first
B tokens are drawn the same way). Budgets above a reading's own max_tokens are not reported for it.
Per reading and budget: avg accuracy over all samples (avg@n), truncation rate, mean tokens (capped at B), the number of
samples graded again. Per reading (except the first, the base) and budget: the problem-level paired bootstrap against
the base (paired_bootstrap.paired: 10000 resamples, seed 0, 95% percentile CI). Per reading: the crossing budget, the
smallest listed B where reading - base > 0 after a budget with reading - base <= 0 (and the log2-interpolated budget
between the two), and the smallest B where the CI excludes 0 from below.
Usage: python budget_curve.py --out <out.json> [--budgets 2048,4096,8192,16384,32768] --tokenizer <model dir>
           <base.jsonl> <reading.jsonl> [<reading.jsonl> ...]
A reading is complete when its <stem>.metrics.json exists next to it; a missing or incomplete one is listed, not used.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
from paired_bootstrap import paired  # noqa: E402


def metrics_of(path):
    m = path[:-len(".jsonl")] + ".metrics.json"
    return json.load(open(m)) if os.path.isfile(m) else None


def at_budget(recs, B, decode, grade):
    """[(pid, correct, truncated, tokens)] of the samples as generated at budget B."""
    out, regraded = [], 0
    for r in recs:
        n = r["n_tokens"]
        if n <= B:
            out.append((r["pid"], bool(r["correct"]), r["finish_reason"] == "length", n))
        else:
            ok, _ = grade(decode(r["token_ids"][:B]), r["gold"]); regraded += 1
            out.append((r["pid"], bool(ok), True, B))
    return out, regraded


def summarise(rows):
    n = len(rows)
    return {"acc": round(100 * sum(c for _, c, _, _ in rows) / n, 2), "trunc_rate": round(sum(t for _, _, t, _ in rows) / n, 4),
            "mean_tokens": round(sum(k for _, _, _, k in rows) / n, 1), "samples": n}


def per_problem(rows):
    s = {}
    for pid, c, _, _ in rows:
        x = s.setdefault(pid, [0, 0]); x[0] += int(c); x[1] += 1
    return s


def crossing(budgets, diffs, lows):
    """diffs[i] = reading - base at budgets[i] (None where not available)."""
    res = {"first_positive": None, "log2_interp": None, "first_ci_positive": None,
           "positive_at_all": all(d is not None and d > 0 for d in diffs) and any(d is not None for d in diffs)}
    for i in range(1, len(budgets)):
        a, b = diffs[i - 1], diffs[i]
        if a is not None and b is not None and a <= 0 < b:
            res["first_positive"] = budgets[i]
            t = (0 - a) / (b - a)
            res["log2_interp"] = int(round(2 ** (math.log2(budgets[i - 1]) + t * (math.log2(budgets[i]) - math.log2(budgets[i - 1])))))
            break
    for B, lo in zip(budgets, lows):
        if lo is not None and lo > 0:
            res["first_ci_positive"] = B; break
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="2048,4096,8192,16384,32768")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("readings", nargs="+")
    a = ap.parse_args()
    budgets = sorted(int(x) for x in a.budgets.split(","))
    from gen_eval import grade
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True)   # as vLLM's text (special tokens skipped)
    res = {"method": __doc__.split("\n")[1].strip() + " One long generation, cut at each budget and graded again (EVAL-v1 grader).",
           "budgets": budgets, "base": None, "readings": {}, "missing": [], "pairs_vs_base": {}, "crossing": {}}
    loaded = []
    for p in a.readings:
        name = os.path.basename(p)[:-len(".jsonl")] if p.endswith(".jsonl") else os.path.basename(p)
        m = metrics_of(p) if os.path.isfile(p) else None
        if m is None:
            res["missing"].append(name); loaded.append((name, None, None)); continue
        recs = [json.loads(x) for x in open(p)]
        mt = int(m.get("max_tokens") or max(r["n_tokens"] for r in recs))
        loaded.append((name, recs, mt))
    base = loaded[0]
    res["base"] = base[0]
    tables = {}
    for name, recs, mt in loaded:
        if recs is None:
            continue
        r = {"max_tokens": mt, "by_budget": {}}
        tables[name] = {}
        for B in budgets:
            if B > mt:
                continue
            rows, k = at_budget(recs, B, decode, grade)
            r["by_budget"][str(B)] = dict(summarise(rows), regraded=k); tables[name][B] = per_problem(rows)
        res["readings"][name] = r
    if base[1] is not None:
        for name, recs, _ in loaded[1:]:
            if recs is None:
                continue
            pr, diffs, lows = {}, [], []
            for B in budgets:
                if B in tables[name] and B in tables[base[0]]:
                    p = paired(tables[name][B], tables[base[0]][B], f"{name} - {base[0]} @ {B}")
                    pr[str(B)] = p; diffs.append(p["diff"]); lows.append(p["ci95"][0])
                else:
                    diffs.append(None); lows.append(None)
            res["pairs_vs_base"][name] = pr
            res["crossing"][name] = crossing(budgets, diffs, lows)
    json.dump(res, open(a.out, "w"), indent=1)
    # a compact table on stdout: accuracy / truncation rate per budget, then reading - base [CI]
    print("budget".ljust(40) + "".join(str(B).rjust(22) for B in budgets))
    for name, r in res["readings"].items():
        print(name[:39].ljust(40) + "".join((f"{r['by_budget'][str(B)]['acc']:.2f} / {100 * r['by_budget'][str(B)]['trunc_rate']:.1f}%" if str(B) in r["by_budget"] else "-").rjust(22) for B in budgets))
    for name, pr in res["pairs_vs_base"].items():
        print(("- base: " + name)[:39].ljust(40) + "".join((f"{pr[str(B)]['diff']:+.2f} [{pr[str(B)]['ci95'][0]:+.1f},{pr[str(B)]['ci95'][1]:+.1f}]" if str(B) in pr else "-").rjust(22) for B in budgets))
        print("  crossing".ljust(40) + json.dumps(res["crossing"][name]))
    if res["missing"]:
        print("missing or incomplete: " + ", ".join(res["missing"]))


if __name__ == "__main__":
    main()
