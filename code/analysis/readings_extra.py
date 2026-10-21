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
    fd = os.path.join(out, "up", "runs", "READINGS-EXTRA"); os.makedirs(fd, exist_ok=True)
    json.dump(summary["frontier"], open(os.path.join(fd, "frontier_fill.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(fd, "summary.json"), "w"), indent=1)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    lines = ["label | 1-100 acc / tok | 101-500 acc / tok | all acc (metrics) | easy acc / tok | hard acc / tok"]
    for k, s in summary["split"].items():
        f = lambda x: "—" if not s.get(x) else f"{s[x]['acc']} / {s[x]['mean_tokens']}"
        lines.append(f"{k} | {f('1-100')} | {f('101-500')} | {s['all']['acc']} ({s.get('metrics_avg_acc')}) | {f('easy')} | {f('hard')}")
    lines.append("frontier: label | 16K mean tokens | 16K trunc | source")
    lines += [f"{k} | {v['mean_tokens_16k']} | {v['trunc_16k']} | {v['source']}" for k, v in summary["frontier"].items()]
    lines.append("missing: " + (", ".join(summary["missing"]) or "none"))
    lines.append("all-500 vs metrics mismatch: " + ("; ".join(summary.get("mismatch", [])) or "none"))
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


if __name__ == "__main__":
    main(*sys.argv[1:4])
