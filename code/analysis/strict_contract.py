"""C1 strict-contract rejudge of existing readings (runbook v2.5.7o; reviewer 2026-10-19 06:0x UTC, queue v3 V3-c4 without
training): how much of each MATH500 reading's accuracy survives the strict delivery contract.
Contract (V3-c4): a sample counts only when it is correct under the EVAL-v1 grader (the reading's own `correct`: the last
\\boxed{} graded by math_verify), it ended by itself (finish_reason "stop", i.e. EOS, not the length limit) and its answer
holds exactly one complete \\boxed{...} (balanced braces). The answer is the whole text without thinking; in a thinking
reading it is the text after the last </think> (no </think>: no answer). The softer variant `same` allows several boxes
when all hold the same content (whitespace ignored); everything else as the contract.
v2.5.7s (reviewer 07:3x UTC): `same` is the C1 contract of record; "exactly one box" is kept as the multi-box rate. Per
reading also where `same` loses the lenient accuracy: of the correct samples, the share each clause rejects (not ended at
EOS, an unclosed box, no box, boxes with different contents), and the same losses exclusively, in points of accuracy, in
the order not EOS > unclosed > no box > inconsistent (they sum to acc - acc_same).
Per reading: lenient accuracy (avg over samples, as gen_eval), contract accuracy, their difference, `same` accuracy, and
the shares of samples that ended at EOS / hold one box / several boxes / no box / an unclosed box, and of the correct
samples, the share each contract clause rejects. No generation, no GPU, no grading again: the text and `correct` of each
record are read as written. With --pair A B, the problem-level paired bootstrap of A - B (paired_bootstrap.paired) under
the lenient grader and under the contract.
Usage: python strict_contract.py --out <report.json> [--table <table.txt>] [--scan <runs dir>] [--pair A B ...] [<reading.jsonl> ...]
--scan takes every MATH500 reading under <runs dir>: <task>/<file>.jsonl and <task>/gen/<file>.jsonl with a <stem>.metrics.json
whose bench is math500 (forced-answer files excluded); readings are named <task>/<file> relative to the runs dir.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paired_bootstrap import paired  # noqa: E402


def boxes(text):
    """(contents of the complete \\boxed{...} in text, number of unclosed ones)."""
    out, unclosed, i = [], 0, text.find("\\boxed{")
    while i >= 0:
        j, depth = i + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
        if depth:
            unclosed += 1; break
        out.append(text[i + len("\\boxed{"): j - 1])
        i = text.find("\\boxed{", j)
    return out, unclosed


def answer_part(text, think):
    if not think:
        return text
    k = text.rfind("</think>")
    return text[k + len("</think>"):] if k >= 0 else None


def judge(r, think):
    """The contract clauses of one record: dict(correct, eos, nbox, unclosed, inconsistent, strict, same)."""
    ans = answer_part(r.get("text") or "", think)
    bx, unc = boxes(ans) if ans is not None else ([], 0)
    eos = r.get("finish_reason") == "stop"
    ok = bool(r.get("correct"))
    norm = {"".join(b.split()) for b in bx}
    return {"correct": ok, "eos": eos, "nbox": len(bx), "unclosed": unc, "answer": ans is not None, "inconsistent": len(norm) > 1,
            "strict": ok and eos and len(bx) == 1 and not unc,
            "same": ok and eos and len(bx) >= 1 and len(norm) == 1 and not unc}


def metrics_of(path):
    m = path[:-len(".jsonl")] + ".metrics.json"
    try:
        return json.load(open(m))
    except (OSError, ValueError):
        return None


def scan(runs):
    out = []
    for p in sorted(glob.glob(os.path.join(runs, "*", "*.jsonl")) + glob.glob(os.path.join(runs, "*", "gen", "*.jsonl"))):
        if p.endswith(".forced.jsonl"):
            continue
        m = metrics_of(p)
        if m and m.get("bench") == "math500":
            out.append(p)
    return out


def rejudge(path, name):
    m = metrics_of(path) or {}
    think = m.get("mode") == "think"
    n = 0; c = {k: 0 for k in ("correct", "strict", "same", "eos", "one", "multi", "none", "unclosed", "no_answer",
                                "c_not_eos", "c_multi", "c_none", "c_unclosed", "c_inconsistent",
                                "x_not_eos", "x_unclosed", "x_no_box", "x_inconsistent")}
    probs, gv = {}, set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line); j = judge(r, think); n += 1; gv.add(r.get("grader_version"))
        c["correct"] += j["correct"]; c["strict"] += j["strict"]; c["same"] += j["same"]; c["eos"] += j["eos"]
        c["one"] += j["nbox"] == 1; c["multi"] += j["nbox"] > 1; c["none"] += j["nbox"] == 0; c["unclosed"] += j["unclosed"] > 0
        c["no_answer"] += not j["answer"]
        if j["correct"]:
            c["c_not_eos"] += not j["eos"]; c["c_multi"] += j["nbox"] > 1; c["c_none"] += j["nbox"] == 0; c["c_unclosed"] += j["unclosed"] > 0
            c["c_inconsistent"] += j["inconsistent"]
            if not j["same"]:   # the first clause that rejects it
                c["x_not_eos" if not j["eos"] else "x_unclosed" if j["unclosed"] else "x_no_box" if j["nbox"] == 0 else "x_inconsistent"] += 1
        p = probs.setdefault(r["pid"], [0, 0, 0, 0]); p[0] += j["correct"]; p[1] += j["strict"]; p[2] += j["same"]; p[3] += 1
    if not n:
        return {"name": name, "error": "empty"}, None
    pct = lambda k, d=n: round(100 * c[k] / d, 2) if d else None
    res = {"name": name, "mode": m.get("mode"), "temperature": m.get("temperature"),
           "max_tokens": m.get("max_tokens"), "samples": n, "problems": len(probs), "grader_versions": sorted(map(str, gv)),
           "acc": pct("correct"), "acc_contract": pct("strict"), "diff": round(100 * (c["strict"] - c["correct"]) / n, 2),
           "acc_same": pct("same"), "metrics_avg_acc": m.get("avg_acc"),
           "share": {"eos": pct("eos"), "one_box": pct("one"), "multi_box": pct("multi"), "no_box": pct("none"),
                     "unclosed_box": pct("unclosed"), "no_answer_part": pct("no_answer")},
           "of_correct_rejected": {k[2:]: pct(k, c["correct"]) for k in ("c_not_eos", "c_multi", "c_none", "c_unclosed", "c_inconsistent")},
           "same_loss_points_exclusive": {k[2:]: pct(k) for k in ("x_not_eos", "x_unclosed", "x_no_box", "x_inconsistent")}}
    return res, probs


def table(rows):
    h = f"{'reading':58} {'mode':8} {'T':>4} {'max':>6} {'n':>5} {'acc':>7} {'contract':>9} {'diff':>7} {'same':>7} {'eos%':>6} {'1box%':>6} {'multi%':>7} {'none%':>6}"
    out = [h]
    for r in rows:
        if "error" in r:
            out.append(f"{r['name'][:58]:58} {r['error']}"); continue
        s = r["share"]
        out.append(f"{r['name'][:58]:58} {str(r['mode']):8} {str(r['temperature']):>4} {str(r['max_tokens']):>6} {r['samples']:>5} "
                   f"{r['acc']:>7.2f} {r['acc_contract']:>9.2f} {r['diff']:>+7.2f} {r['acc_same']:>7.2f} {s['eos']:>6.1f} "
                   f"{s['one_box']:>6.1f} {s['multi_box']:>7.1f} {s['no_box']:>6.1f}")
    out += ["", "same (the C1 contract): of the correct samples, % rejected by each clause; then the loss in points of accuracy,",
            "exclusive in the order not EOS > unclosed > no box > inconsistent (sums to acc - same)",
            f"{'reading':58} {'acc':>7} {'same':>7} {'notEOS%':>8} {'unclo%':>7} {'nobox%':>7} {'incon%':>7} {'L:notEOS':>9} {'L:unclo':>8} {'L:nobox':>8} {'L:incon':>8}"]
    for r in rows:
        if "error" in r:
            continue
        o, x = r["of_correct_rejected"], r["same_loss_points_exclusive"]
        f = lambda v, w: f"{v:>{w}.2f}" if v is not None else f"{'-':>{w}}"   # v2.5.7t: no correct sample -> no share
        out.append(f"{r['name'][:58]:58} {f(r['acc'], 7)} {f(r['acc_same'], 7)} {f(o['not_eos'], 8)} {f(o['unclosed'], 7)} {f(o['none'], 7)} "
                   f"{f(o['inconsistent'], 7)} {f(x['not_eos'], 9)} {f(x['unclosed'], 8)} {f(x['no_box'], 8)} {f(x['inconsistent'], 8)}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=None)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"))
    ap.add_argument("readings", nargs="*")
    a = ap.parse_args()
    paths = scan(a.scan) if a.scan else []
    paths += [p for p in a.readings if p not in paths]
    base = os.path.abspath(a.scan) if a.scan else None
    name = lambda p: os.path.relpath(os.path.abspath(p), base) if base else os.path.basename(p)
    res = {"method": __doc__.split("\n")[1].strip() + " " + " ".join(__doc__.split("\n")[2:6]).strip(),
           "readings": [], "pairs": [], "missing": []}
    per = {}
    for p in paths:
        if not os.path.isfile(p):
            res["missing"].append(name(p)); continue
        r, probs = rejudge(p, name(p)); res["readings"].append(r)
        if probs is not None:
            per[r["name"]] = probs
    for A, B in a.pair:
        if A not in per or B not in per:
            res["pairs"].append({"a": A, "b": B, "error": "reading missing"}); continue
        pr = {"a": A, "b": B}
        for k, i in (("lenient", 0), ("contract", 1), ("same", 2)):
            pr[k] = paired({p: [v[i], v[3]] for p, v in per[A].items()}, {p: [v[i], v[3]] for p, v in per[B].items()}, f"{A} - {B} ({k})")
        pr["diff_change"] = round(pr["contract"]["diff"] - pr["lenient"]["diff"], 2)
        res["pairs"].append(pr)
    json.dump(res, open(a.out, "w"), indent=1)
    t = table(res["readings"])
    if res["pairs"]:
        t += "\n\npair (A - B)".ljust(60) + "lenient [CI]".rjust(24) + "contract [CI]".rjust(24) + "change".rjust(9)
        for p in res["pairs"]:
            if "error" in p:
                t += f"\n{(p['a'] + ' - ' + p['b'])[:58]:58} {p['error']}"; continue
            f = lambda q: f"{q['diff']:+.2f} [{q['ci95'][0]:+.1f},{q['ci95'][1]:+.1f}]"
            t += f"\n{(p['a'] + ' - ' + p['b'])[:58]:58} {f(p['lenient']):>24}{f(p['contract']):>24}{p['diff_change']:>+9.2f}"
    if res["missing"]:
        t += "\nmissing: " + ", ".join(res["missing"])
    if a.table:
        open(a.table, "w").write(t + "\n")
    print(t)


if __name__ == "__main__":
    main()
