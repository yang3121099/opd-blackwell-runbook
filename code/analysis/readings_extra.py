"""READINGS-EXTRA (runbook [v2.5.9x1]; ops-lead's / the reviewer's CPU pad 10-22): extra views of readings that exist, nothing
generated again, no reading's own numbers changed.

1. split (Astra (7): MATH-500's problems 1-100 are the in-training validation set): for each `#split READINGS-EXTRA <label>
   <task>/<file>.jsonl` queue line, the MATH-500 16K reading's accuracy (avg@n as the reading, 100 x mean of `correct` over its
   samples), mean tokens and truncation rate over problems 1-100 and 101-500 (MATH-500's own order: pid math500-<i>, i < 100),
   all 500 (a check against its metrics.json), and, when PROBE-DIFF17's labels_17.jsonl is there, over MATH-500's easy (r >= 0.5)
   and hard problems. Written next to the reading as <task>/<stem>.split.json.
2. frontier (the reviewer's cost-frontier table): for each `#frontier READINGS-EXTRA <label> <task>/<file> [<reading>]` line, the
   16K mean length and truncation rate: from a budget_curve*.json, its <reading>'s 16384 budget (by_budget["16384"]); from a
   metrics.json, its mean_tokens / trunc_rate (with its max_tokens). Written to READINGS-EXTRA/frontier_fill.json.
3. [v2.5.9z5] pairs (the reviewer 10-22, Q-F): for each `#pairq READINGS-EXTRA <label> <task>/<a>.jsonl <task>/<b>.jsonl` line, the
   problem-level paired bootstrap of a against b (code/analysis/paired_bootstrap.py, the lab's method run for #pair lines: per-problem
   accuracy, 10000 resamples, seed 0, 95% percentile CI, wins / losses / ties). Written to READINGS-EXTRA/pairs.json.
4. [v2.5.9z7] coverage (the reviewer 10-22, Q-052 (b)): for each `#coverq READINGS-EXTRA <label> <task>/<a>.jsonl <task>/<b>.jsonl
   [<name>=<task>/<c>.jsonl ...]` line, the problems b solves at least once and a never (b_only), the reverse (a_only), and the
   counts solved by a, by b, by both and by neither, over the problems both readings have; each listed problem with every reading's
   n_correct / n_samples (an extra reading without the problem: null). Written to READINGS-EXTRA/coverage.json.
5. [v2.5.9z13] (the reviewer's Q-030 checks 10-22) lensel: for each `#lensel READINGS-EXTRA <label> <task> [<task> ...]` line, per training
   step of each length-selected arm (runs/<task>/lensel.jsonl), the EOS rate, truncation rate and mean length of the kept rollouts and of
   all of them, and the first step whose kept EOS rate is below 90%. Written to READINGS-EXTRA/lensel.json.
6. [v2.5.9z13] tails: for each `#tails READINGS-EXTRA <label> <task>/<file>.jsonl <n> <seed> <tail_tokens>` line, n samples drawn with
   random.Random(seed) from the reading, each sample's last ~tail_tokens tokens (the text cut in proportion to n_tokens), classed "loop"
   (the tail ends in >= 3 exact copies of a unit of 8-400 characters, or compresses to < LOOP_ZLIB 0.2 of its bytes with zlib) or "long"
   (long reasoning / restatement); the counts and one tail end (<= 300 characters) per class. Written to READINGS-EXTRA/tails.json.
7. [v2.5.9z14] (the reviewer's Q-056, Runming's question 10-22) behaviour fingerprints of MATH-500 16K readings: `#fpauto READINGS-EXTRA
   <label>` takes every reading under runs/ named <stem>_math500_T0.6_math500_<mode>_t0.6.jsonl with 4 samples a problem and max_tokens
   16384 (its metrics.json), named <stem>; `#fpmodel READINGS-EXTRA <label> <name> <task>/<file>.jsonl [<cap>]` adds one under a name,
   with <cap> read at that budget (a sample counts as correct only when n_tokens <= cap; its length is min(n_tokens, cap): a 32K reading
   as a 16K one); `#fppair READINGS-EXTRA <label> <name> <ref> [<ref> ...]` lines are the table's rows. Per model the per-problem accuracy
   (avg over its samples) and mean length vectors; per pair Pearson r of accuracy (r_acc) and of length (r_len), and over the problems
   binarised at accuracy >= 0.5 the phi coefficient and the agreement rate; for all problems, 1-100 and 101-500. fingerprint.json.
Inputs missing on the node are listed, not guessed. Usage: python readings_extra.py <runs dir> <queue file> <out dir>
"""
import json
import os
import sys


def pid_index(pid):
    return int(str(pid).rsplit("-", 1)[-1])


def stats(recs):
    if not recs:
        return None
    pids = {r["pid"] for r in recs}
    return {"acc": round(100 * sum(1 for r in recs if r["correct"]) / len(recs), 2),
            "mean_tokens": round(sum(r["n_tokens"] for r in recs) / len(recs), 1),
            "trunc_rate": round(sum(1 for r in recs if r.get("finish_reason") == "length") / len(recs), 4),
            "n_problems": len(pids), "n_samples": len(recs)}


def read_records(path):
    """The fields a split needs, one line at a time (the files carry texts and token ids)."""
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out.append({"pid": r["pid"], "correct": bool(r["correct"]), "n_tokens": int(r["n_tokens"]), "finish_reason": r.get("finish_reason")})
    return out


def split(recs, easy=None):
    """{"1-100", "101-500", "all"[, "easy", "hard"]}: stats of each part; easy = {index: r} of MATH-500 from PROBE-DIFF17."""
    res = {"1-100": stats([r for r in recs if pid_index(r["pid"]) < 100]),
           "101-500": stats([r for r in recs if pid_index(r["pid"]) >= 100]), "all": stats(recs)}
    if easy:
        res["easy"] = stats([r for r in recs if easy.get(pid_index(r["pid"]), -1) >= 0.5])
        res["hard"] = stats([r for r in recs if 0 <= easy.get(pid_index(r["pid"]), -1) < 0.5])
        res["labels"] = "PROBE-DIFF17 labels_17.jsonl (MATH-500, 1.7B base non-thinking k=8: easy = r >= 0.5)"
    return res


def frontier(path, reading=None):
    d = json.load(open(path))
    if "readings" in d:   # a budget curve
        if reading not in d["readings"]:
            raise KeyError(f"reading {reading} not in {os.path.basename(path)}")
        b = d["readings"][reading]["by_budget"]["16384"]
        return {"mean_tokens_16k": b["mean_tokens"], "trunc_16k": b["trunc_rate"], "acc_16k": b.get("acc"), "how": "budget curve at 16384"}
    return {"mean_tokens_16k": d["mean_tokens"], "trunc_16k": d["trunc_rate"], "acc_16k": d.get("avg_acc"),
            "how": f"metrics.json (max_tokens {d.get('max_tokens')})"}


def pid_order(pid):
    try:
        return (0, pid_index(pid), str(pid))
    except ValueError:
        return (1, 0, str(pid))


def coverage(A, B, extras):
    """[v2.5.9z7] (Q-052) A, B, each extra: {pid: [n_correct, n_samples]} (paired_bootstrap.reduce). Over the problems A and B
    both have: a_only = solved by A at least once and never by B, b_only the reverse, each with every reading's counts."""
    common = sorted(set(A) & set(B), key=pid_order)
    row = lambda p: dict({"pid": p, "a": A[p], "b": B[p]}, **{k: E.get(p) for k, E in extras.items()})
    sa, sb = {p for p in common if A[p][0] > 0}, {p for p in common if B[p][0] > 0}
    return {"problems": len(common), "a_solved": len(sa), "b_solved": len(sb), "both": len(sa & sb),
            "neither": len(common) - len(sa | sb), "a_only": [row(p) for p in common if p in sa - sb],
            "b_only": [row(p) for p in common if p in sb - sa], "not_in_both": sorted(set(A) ^ set(B), key=pid_order)}


def lensel_steps(path):
    """{step: {"kept": [eos %, trunc %, mean len], "all": [...], "groups": n}} of a lensel.jsonl (one line per prompt group: len,
    trunc, eos of every rollout and the kept indices)."""
    by = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                x = json.loads(line); by.setdefault(int(x["step"]), []).append(x)
    stat = lambda v: [round(100 * sum(e for _, _, e in v) / len(v), 1), round(100 * sum(t for _, t, _ in v) / len(v), 1),
                      round(sum(l for l, _, _ in v) / len(v))] if v else None
    out = {}
    for s, g in sorted(by.items()):
        allv = [(l, t, e) for x in g for l, t, e in zip(x["len"], x["trunc"], x["eos"])]
        kept = [(x["len"][i], x["trunc"][i], x["eos"][i]) for x in g for i in x["kept"]]
        out[s] = {"kept": stat(kept), "all": stat(allv), "groups": len(g)}
    return out


LOOP_ZLIB = 0.2


def loop_period(s, pmin=8, pmax=400, cycles=3):
    """The shortest period p (pmin..pmax characters) such that s ends with `cycles` exact copies of its last p characters, else None."""
    for p in range(pmin, min(pmax, len(s) // cycles) + 1):
        if s.endswith(s[-p:] * cycles):
            return p
    return None


def tail_of(rec, tail_tokens):
    text, n = rec.get("text") or "", int(rec.get("n_tokens") or 0)
    k = len(text) if n <= tail_tokens else max(1, int(len(text) * tail_tokens / n))
    return text[-k:]


def tails(path, n, seed, tail_tokens):
    """n samples of a reading (random.Random(seed).sample over its lines, two streaming passes), each tail classed loop / long."""
    import random
    import zlib
    with open(path) as f:
        total = sum(1 for line in f if line.strip())
    pick = set(random.Random(seed).sample(range(total), min(n, total)))
    rows, i = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            if i in pick:
                r = json.loads(line); t = tail_of(r, tail_tokens); b = t.encode("utf-8")
                per = loop_period(t); z = round(len(zlib.compress(b, 9)) / max(len(b), 1), 3)
                rows.append({"pid": r.get("pid"), "sample": r.get("sample"), "n_tokens": r.get("n_tokens"), "finish_reason": r.get("finish_reason"),
                             "correct": bool(r.get("correct")), "class": "loop" if per or z < LOOP_ZLIB else "long", "period": per, "zlib": z,
                             "end": t[-300:]})
            i += 1
    out = {"lines": total, "n": len(rows), "seed": seed, "tail_tokens": tail_tokens, "loop_zlib": LOOP_ZLIB,
           "counts": {c: sum(1 for x in rows if x["class"] == c) for c in ("loop", "long")},
           "truncated": sum(1 for x in rows if x["finish_reason"] == "length"),
           "example": {c: next((x["end"] for x in rows if x["class"] == c), None) for c in ("loop", "long")},
           "samples": [{k: v for k, v in x.items() if k != "end"} for x in rows]}
    return out


FP_NAME = r"^(?P<stem>.+)_math500_T0\.6_math500_(?:no_think|think)_t0\.6\.jsonl$"


def fp_vectors(path, cap=None):
    """{index: (accuracy, mean length)} of a MATH-500 reading, read at budget cap when given."""
    acc, ln = {}, {}
    for r in read_records(path):
        i = pid_index(r["pid"]); n = r["n_tokens"]
        ok = r["correct"] and (cap is None or n <= cap)
        acc.setdefault(i, []).append(1.0 if ok else 0.0); ln.setdefault(i, []).append(min(n, cap) if cap else n)
    return {i: (sum(acc[i]) / len(acc[i]), sum(ln[i]) / len(ln[i])) for i in acc}


def _pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = sum((a - mx) ** 2 for a in x); sy = sum((b - my) ** 2 for b in y)
    if sx == 0 or sy == 0:
        return None
    return round(sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) ** 0.5, 4)


def fp_compare(va, vb, part=None):
    """r_acc, r_len, phi, agree (and the problem count) of two models over their common problems in part (None: all; "1-100"; "101-500")."""
    keep = lambda i: part is None or (i < 100 if part == "1-100" else i >= 100)
    idx = sorted(i for i in va if i in vb and keep(i))
    aa, ab = [va[i][0] for i in idx], [vb[i][0] for i in idx]
    ba, bb = [1.0 if x >= 0.5 else 0.0 for x in aa], [1.0 if x >= 0.5 else 0.0 for x in ab]
    return {"n": len(idx), "r_acc": _pearson(aa, ab), "r_len": _pearson([va[i][1] for i in idx], [vb[i][1] for i in idx]),
            "phi": _pearson(ba, bb), "agree": round(sum(1 for x, y in zip(ba, bb) if x == y) / len(idx), 4) if idx else None}


def fp_auto(runs):
    """{stem: <task>/<file>} of the 16K avg@4 T0.6 MATH-500 readings under runs/ (metrics.json max_tokens 16384, 2000 lines)."""
    import re
    out = {}
    for t in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
        d = os.path.join(runs, t)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            m = re.match(FP_NAME, f)
            if not m or m.group("stem") in out:
                continue
            mp = os.path.join(d, f[:-len(".jsonl")] + ".metrics.json")
            try:
                mx = json.load(open(mp)).get("max_tokens") if os.path.exists(mp) else None
            except ValueError:
                mx = None
            if mx != 16384:
                continue
            with open(os.path.join(d, f)) as fh:
                if sum(1 for line in fh if line.strip()) != 2000:
                    continue
            out[m.group("stem")] = f"{t}/{f}"
    return out


def safe(rel):
    """[v2.5.9z3] a row's path: relative, no '..', under runs/ (anything else is refused and listed as missing)."""
    return bool(rel) and not os.path.isabs(rel) and ".." not in rel.split("/") and "\\" not in rel


def queue_rows(qfile, kind):
    rows = []
    for line in open(qfile):
        f = line.split()
        if len(f) >= 4 and f[0] == f"#{kind}" and f[1] == "READINGS-EXTRA":
            rows.append(f[2:])
    return rows


def main(runs, qfile, out):
    os.makedirs(out, exist_ok=True)
    easy = {}
    lp = os.path.join(runs, "PROBE-DIFF17", "labels_17.jsonl")
    if os.path.exists(lp):
        for line in open(lp):
            r = json.loads(line)
            if r.get("split") == "math500":
                easy[int(r["pos"])] = float(r["r"])
    summary = {"split": {}, "frontier": {}, "missing": []}
    for label, rel in (r[:2] for r in queue_rows(qfile, "split")):
        p = os.path.join(runs, rel)
        if not safe(rel):
            summary["missing"].append(f"{rel}: refused path"); continue
        if not os.path.exists(p):
            summary["missing"].append(rel); continue
        s = split(read_records(p), easy)
        mp = p[:-len(".jsonl")] + ".metrics.json"
        if os.path.exists(mp):
            s["metrics_avg_acc"] = json.load(open(mp)).get("avg_acc")
            if s["metrics_avg_acc"] is not None and abs(s["all"]["acc"] - float(s["metrics_avg_acc"])) > 0.01:   # [v2.5.9z3] pid parsing check
                summary.setdefault("mismatch", []).append(f"{label}: all {s['all']['acc']} vs metrics {s['metrics_avg_acc']}")
        dst = os.path.join(out, "up", "runs", os.path.dirname(rel)); os.makedirs(dst, exist_ok=True)
        stem = os.path.basename(rel)[:-len(".jsonl")]
        json.dump(dict(s, reading=rel), open(os.path.join(dst, stem + ".split.json"), "w"), indent=1)
        json.dump(dict(s, reading=rel), open(p[:-len(".jsonl")] + ".split.json", "w"), indent=1)
        summary["split"][label] = {**{k: s[k] for k in s if k in ("1-100", "101-500", "all", "easy", "hard", "metrics_avg_acc")}, "reading": rel}
    for row in queue_rows(qfile, "frontier"):
        label, rel, reading = row[0], row[1], (row[2] if len(row) > 2 else None)
        if not safe(rel):
            summary["missing"].append(f"{rel}: refused path"); continue
        p = os.path.join(runs, rel)
        try:
            summary["frontier"][label] = dict(frontier(p, reading), source=rel + (f" [{reading}]" if reading else ""))
        except (OSError, KeyError, ValueError) as e:
            summary["missing"].append(f"{rel}: {type(e).__name__}")
    summary["pairs"] = {}
    for row in queue_rows(qfile, "pairq"):   # [v2.5.9z5]
        label, a, b = (row + [None, None])[:3]
        if not (a and b and safe(a) and safe(b)):
            summary["missing"].append(f"{a} / {b}: refused path"); continue
        pa, pb = os.path.join(runs, a), os.path.join(runs, b)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            summary["missing"] += [x for x, q in ((a, pa), (b, pb)) if not os.path.exists(q)]; continue
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import paired_bootstrap as PB
        summary["pairs"][label] = dict(PB.paired(PB.load(pa), PB.load(pb), label), a=a, b=b)
    summary["coverage"] = {}
    for row in queue_rows(qfile, "coverq"):   # [v2.5.9z7]
        label, a, b, ex = row[0], row[1], (row[2] if len(row) > 2 else None), [x.split("=", 1) for x in row[3:]]
        if not (a and b and safe(a) and safe(b)) or any(len(x) != 2 or not safe(x[1]) for x in ex):
            summary["missing"].append(f"{label}: refused path"); continue
        pa, pb = os.path.join(runs, a), os.path.join(runs, b)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            summary["missing"] += [x for x, q in ((a, pa), (b, pb)) if not os.path.exists(q)]; continue
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import paired_bootstrap as PB
        extras = {}
        for name, rel in ex:
            if os.path.exists(os.path.join(runs, rel)):
                extras[name] = PB.reduce(os.path.join(runs, rel))
            else:
                summary["missing"].append(rel)
        summary["coverage"][label] = dict(coverage(PB.reduce(pa), PB.reduce(pb), extras), a=a, b=b,
                                          extras={name: rel for name, rel in ex}, extras_missing=[n for n, _ in ex if n not in extras])
    summary["lensel"] = {}
    for row in queue_rows(qfile, "lensel"):   # [v2.5.9z13]
        label, ts = row[0], row[1:]
        res = {}
        for t in ts:
            p = os.path.join(runs, t, "lensel.jsonl")
            if not safe(t) or "/" in t:
                summary["missing"].append(f"{t}: refused path"); continue
            if not os.path.exists(p):
                summary["missing"].append(f"{t}/lensel.jsonl"); continue
            st = lensel_steps(p)
            res[t] = {"steps": st, "kept_eos_below_90_from": next((s for s, v in st.items() if v["kept"] and v["kept"][0] < 90), None)}
        if res:
            summary["lensel"][label] = res
    summary["tails"] = {}
    for row in queue_rows(qfile, "tails"):   # [v2.5.9z13]
        label, rel = row[0], (row[1] if len(row) > 1 else None)
        try:
            n, seed, tt = int(row[2]), int(row[3]), int(row[4])
        except (IndexError, ValueError):
            summary["missing"].append(f"{label}: #tails needs <file> <n> <seed> <tail_tokens>"); continue
        if not (rel and safe(rel)):
            summary["missing"].append(f"{label}: refused path"); continue
        p = os.path.join(runs, rel)
        if not os.path.exists(p):
            summary["missing"].append(rel); continue
        summary["tails"][label] = dict(tails(p, n, seed, tt), reading=rel)
    summary["fingerprint"] = {}
    fp_rows = [l.split() for l in open(qfile)]
    fp_rows = [f for f in fp_rows if len(f) >= 3 and f[0] in ("#fpauto", "#fpmodel", "#fppair") and f[1] == "READINGS-EXTRA"]
    fp_full = {}
    for label in sorted({f[2] for f in fp_rows}):   # [v2.5.9z14]
        models = {}
        if any(f[0] == "#fpauto" and f[2] == label for f in fp_rows):
            models.update({k: (v, None) for k, v in fp_auto(runs).items()})
        for row in queue_rows(qfile, "fpmodel"):
            if row[0] == label and len(row) >= 3:
                name, rel, cap = row[1], row[2], (int(row[3]) if len(row) > 3 and row[3].isdigit() else None)
                if not safe(rel):
                    summary["missing"].append(f"{name}: refused path"); continue
                models[name] = (rel, cap)
        vec = {}
        for name, (rel, cap) in sorted(models.items()):
            p = os.path.join(runs, rel)
            if os.path.exists(p):
                vec[name] = fp_vectors(p, cap)
            else:
                summary["missing"].append(rel)
        names = sorted(vec)
        mat = {part or "all": {a: {b: fp_compare(vec[a], vec[b], part) for b in names if b != a} for a in names} for part in (None, "1-100", "101-500")}
        rows = []
        for row in queue_rows(qfile, "fppair"):
            if row[0] != label:
                continue
            a = row[1]
            for b in row[2:]:
                if a in vec and b in vec:
                    rows.append({"a": a, "b": b, **{part: mat[part][a][b] for part in ("all", "1-100", "101-500")}})
                else:
                    summary["missing"].append(f"{label} pair {a} / {b}: no reading")
        summary["fingerprint"][label] = {"models": {n: {"reading": models[n][0], "cap": models[n][1],
                                                        "acc": round(100 * sum(v[0] for v in vec[n].values()) / len(vec[n]), 2),
                                                        "mean_len": round(sum(v[1] for v in vec[n].values()) / len(vec[n]), 1),
                                                        "problems": len(vec[n])} for n in names},
                                         "pairs": rows}
        fp_full[label] = dict(summary["fingerprint"][label], matrix=mat)
    fd = os.path.join(out, "up", "runs", "READINGS-EXTRA"); os.makedirs(fd, exist_ok=True)
    if fp_full:
        json.dump(fp_full, open(os.path.join(fd, "fingerprint.json"), "w"))
    if summary["lensel"]:
        json.dump(summary["lensel"], open(os.path.join(fd, "lensel.json"), "w"), indent=1)
    if summary["tails"]:
        json.dump(summary["tails"], open(os.path.join(fd, "tails.json"), "w"), indent=1, ensure_ascii=False)
    json.dump(summary["pairs"], open(os.path.join(fd, "pairs.json"), "w"), indent=1)
    json.dump(summary["coverage"], open(os.path.join(fd, "coverage.json"), "w"), indent=1)
    json.dump(summary["frontier"], open(os.path.join(fd, "frontier_fill.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(fd, "summary.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    lines = ["label | 1-100 acc / tok | 101-500 acc / tok | all acc (metrics) | easy acc / tok | hard acc / tok"]
    for k, s in summary["split"].items():
        f = lambda x: "—" if not s.get(x) else f"{s[x]['acc']} / {s[x]['mean_tokens']}"
        lines.append(f"{k} | {f('1-100')} | {f('101-500')} | {s['all']['acc']} ({s.get('metrics_avg_acc')}) | {f('easy')} | {f('hard')}")
    lines.append("frontier: label | 16K mean tokens | 16K trunc | source")
    lines += [f"{k} | {v['mean_tokens_16k']} | {v['trunc_16k']} | {v['source']}" for k, v in summary["frontier"].items()]
    lines.append("pairs: label | acc a | acc b | diff [95% CI] | wins / losses / ties | problems")
    lines += [f"{k} | {v['acc_a']} | {v['acc_b']} | {v['diff']} [{v['ci95'][0]}, {v['ci95'][1]}] | {v['wins']} / {v['losses']} / {v['ties']} | {v['problems']}" for k, v in summary["pairs"].items()]
    lines.append("coverage: label | problems | a solved / b solved / both / neither | b only (a never) | a only (b never)")
    cnt = lambda c: "—" if c is None else f"{c[0]}/{c[1]}"
    for k, v in summary["coverage"].items():
        show = lambda rows: "; ".join(f"{r['pid']} a {cnt(r['a'])} b {cnt(r['b'])}" + "".join(f" {n} {cnt(r.get(n))}" for n in v["extras"]) for r in rows) or "none"
        lines.append(f"{k} | {v['problems']} | {v['a_solved']} / {v['b_solved']} / {v['both']} / {v['neither']} | {show(v['b_only'])} | {show(v['a_only'])}")
    for k, v in summary["fingerprint"].items():   # [v2.5.9z14]
        lines.append(f"fingerprint {k} ({len(v['models'])} models): a vs b | all r_acc / phi / agree / r_len | 1-100 | 101-500")
        f = lambda c: f"{c['r_acc']} / {c['phi']} / {c['agree']} / {c['r_len']}"
        lines += [f"{r['a']} vs {r['b']} | {f(r['all'])} | {f(r['1-100'])} | {f(r['101-500'])}" for r in v["pairs"]]
    for k, v in summary["lensel"].items():   # [v2.5.9z13]
        lines.append(f"lensel {k}: kept EOS below 90% from step " + ", ".join(f"{t} {x['kept_eos_below_90_from']}" for t, x in v.items()))
    for k, v in summary["tails"].items():
        lines.append(f"tails {k}: {v['n']} of {v['lines']} (seed {v['seed']}, last ~{v['tail_tokens']} tokens): loop {v['counts']['loop']}, long {v['counts']['long']}, truncated {v['truncated']}")
    lines.append("missing: " + (", ".join(summary["missing"]) or "none"))
    lines.append("all-500 vs metrics mismatch: " + ("; ".join(summary.get("mismatch", [])) or "none"))
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main(*sys.argv[1:4])
