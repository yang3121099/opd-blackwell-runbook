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
v2.5.7v (reviewer 08:2x UTC, "when does the student have the answer, when does it hand it in"): per reading also
answer_ready: of the correct samples, the token position where the first box holding the final (graded) answer ends, and
how many tokens the sample still writes after it (median, p90, share > 500), plus the first box of any content over all
samples. Positions in the whole text (thinking included); tokens estimated as n_tokens x char position / text length.
v2.5.7w (reviewer 08:5x UTC): answer_ready also holds (a) histograms in 256-token bins up to 32768 (the last bin takes
the rest) of the ready position and of the tokens after it (correct samples), for cumulative plots; (b) first_box: the
share of all samples whose FIRST box is correct under the EVAL-v1 grader (gen_eval.grade on the text up to that box;
graded again only when its content differs from the last box's), next to acc (the last box), the first box's end
position (p50, samples with a box) and the mean tokens written after it (what stopping at the first box would save).
[v2.5.8d] (reviewer 13:3x UTC, Runming: was a changed answer changed to right or to wrong?): answer_ready.box_change =
the samples whose first and last boxes (whole text) differ in content, split by the EVAL-v1 grade of the first box
(first_box) and of the last (the reading's correct): first right / last wrong, first wrong / last right, both, neither;
counts and points of all samples (first_right_last_wrong - first_wrong_last_right = first_box_acc - acc).
v2.5.7x (reviewer 09:3x UTC, "does P5 copy the thinking mode's two-part structure?"): answer_ready also gives, over the
correct samples, the share with two or more boxes in the whole text all holding the same content (and with any
content), and, for samples that close </think>, the share whose first final-answer box ends before the last </think>
and the tokens after the last </think> (p50 / p90: the summary part).
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
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paired_bootstrap import paired  # noqa: E402


def boxes_at(text):
    """([(start, end, content)] of the complete \\boxed{...} in text, number of unclosed ones)."""
    out, unclosed, i = [], 0, text.find("\\boxed{")
    while i >= 0:
        j, depth = i + len("\\boxed{"), 1
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
        if depth:
            unclosed += 1; break
        out.append((i, j, text[i + len("\\boxed{"): j - 1]))
        i = text.find("\\boxed{", j)
    return out, unclosed


def boxes(text):
    """(contents of the complete \\boxed{...} in text, number of unclosed ones)."""
    bl, unclosed = boxes_at(text)
    return [c for _, _, c in bl], unclosed


BIN, NBIN = 256, 128   # v2.5.7w: histogram bins (256 tokens, up to 32768; the last bin takes the rest)
_GRADE = None


def grade(text, gold):
    """EVAL-v1's grader (gen_eval.grade: the last \\boxed{} of text against gold), imported on first use."""
    global _GRADE
    if _GRADE is None:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
        from gen_eval import grade as g
        _GRADE = g
    return bool(_GRADE(text, gold)[0])


def hist(xs):
    h = [0] * NBIN
    for x in xs:
        h[min(int(x) // BIN, NBIN - 1)] += 1
    return h


def first_box(r):
    """(first box correct under EVAL-v1, char end of the first box) or (False, None) without a box."""
    text = r.get("text") or ""
    allb, _ = boxes_at(text)
    if not allb:
        return False, None
    s, e, c = allb[0]
    if "".join(c.split()) == "".join(allb[-1][2].split()):
        return bool(r.get("correct")), e      # the same content as the graded last box
    return grade(text[:e], str(r.get("gold", ""))), e


def ready(text, ans_boxes, correct):
    """(char end of the first box holding the final answer or None, char start of the first box or None, len(text))."""
    allb, _ = boxes_at(text)
    first = allb[0][0] if allb else None
    if not correct or not (ans_boxes or allb):
        return None, first, len(text)
    target = "".join((ans_boxes or [c for _, _, c in allb])[-1].split())
    end = next((e for _, e, c in allb if "".join(c.split()) == target), None)
    return end, first, len(text)


def tail_pattern(text, allb, nbytes=30):
    """v2.5.7ze (reviewer 16:2x UTC, Q-018: where does a correct sample stop?): the last nbytes bytes of text, with the
    last box's content replaced by <A> and every digit by 9, so the same ending of different problems counts once."""
    if allb:
        s, e, _ = allb[-1]
        text = text[:s] + "\\boxed{<A>}" + text[e:]
    return re.sub(r"[0-9]", "9", text.encode("utf-8")[-nbytes:].decode("utf-8", "ignore"))


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
    fc, fb, L = ready(r.get("text") or "", bx, ok)
    return {"fc_char": fc, "fb_char": fb, "chars": L,"correct": ok, "eos": eos, "nbox": len(bx), "unclosed": unc, "answer": ans is not None, "inconsistent": len(norm) > 1,
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
    probs, gv = {}, set(); ft, at, fbt, ntk = [], [], [], []; fb_ok, fb_end, fb_save = 0, [], []
    nc, m_same, m_any, th_n, th_before, post = 0, 0, 0, 0, 0, []   # v2.5.7x
    stop_after, tails = [], Counter()   # v2.5.7ze
    bc = Counter()   # [v2.5.8d] box_change
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
        nt = int(r.get("n_tokens") or 0); ntk.append(nt); tok = lambda ch: round(nt * ch / max(j["chars"], 1))
        if j["fb_char"] is not None:
            fbt.append(tok(j["fb_char"]))
        if j["correct"] and j["fc_char"] is not None:
            ft.append(tok(j["fc_char"])); at.append(max(nt - tok(j["fc_char"]), 0))
        if j["correct"]:   # v2.5.7x: the whole text's boxes, and the part after the last </think>
            text = r.get("text") or ""; allc = ["".join(c.split()) for _, _, c in boxes_at(text)[0]]; nc += 1
            m_any += len(allc) > 1; m_same += len(allc) > 1 and len(set(allc)) == 1
            k = text.rfind("</think>")
            if k >= 0:
                th_n += 1; post.append(max(nt - tok(k + len("</think>")), 0))
                th_before += j["fc_char"] is not None and j["fc_char"] <= k
        if j["correct"] and j["eos"]:   # v2.5.7ze: correct and ended by itself: tokens from the last box's end to EOS, and the ending
            text = r.get("text") or ""; allb = boxes_at(text)[0]
            if allb:
                stop_after.append(max(nt - tok(allb[-1][1]), 0))
            tails[tail_pattern(text, allb)] += 1
        ok1, e1 = first_box(r)   # v2.5.7w
        fb_ok += ok1
        allt = boxes_at(r.get("text") or "")[0]   # [v2.5.8d] the first and the last box differ: which one is right?
        if len(allt) >= 2 and "".join(allt[0][2].split()) != "".join(allt[-1][2].split()):
            last = bool(r.get("correct")); bc["n"] += 1
            bc["first_right_last_wrong" if ok1 and not last else "first_wrong_last_right" if last and not ok1 else "both_right" if ok1 else "both_wrong"] += 1
        if e1 is not None:
            fb_end.append(tok(e1)); fb_save.append(max(nt - tok(e1), 0))
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
           "same_loss_points_exclusive": {k[2:]: pct(k) for k in ("x_not_eos", "x_unclosed", "x_no_box", "x_inconsistent")},
           "answer_ready": {"correct_with_answer_box": len(ft), "first_correct_box_tok_p50": q(ft, .5), "first_correct_box_tok_p90": q(ft, .9),
                            "after_tok_p50": q(at, .5), "after_tok_p90": q(at, .9),
                            "after_gt_500_share": round(100 * sum(a > 500 for a in at) / len(at), 2) if at else None,
                            "first_box_tok_p50": q(fbt, .5), "n_tokens_p50": q(ntk, .5),
                            "bin_tokens": BIN, "ready_tok_hist": hist(ft), "after_tok_hist": hist(at),
                            "first_box_acc": round(100 * fb_ok / n, 2), "first_box_end_tok_p50": q(fb_end, .5),
                            "first_box_save_tok_mean": round(sum(fb_save) / len(fb_save), 1) if fb_save else None,
                            "correct_multi_box_same_share": round(100 * m_same / nc, 2) if nc else None,
                            "correct_multi_box_any_share": round(100 * m_any / nc, 2) if nc else None,
                            "correct_closing_think": th_n,
                            "answer_box_before_think_close_share": round(100 * th_before / th_n, 2) if th_n else None,
                            "post_think_tok_p50": q(post, .5), "post_think_tok_p90": q(post, .9),
                            "stop_n": sum(tails.values()), "stop_after_last_box_tok_p50": q(stop_after, .5),
                            "stop_after_last_box_tok_p90": q(stop_after, .9),
                            "stop_after_last_box_tok_mean": round(sum(stop_after) / len(stop_after), 1) if stop_after else None,
                            "stop_tail_top5": [[t, k, round(100 * k / sum(tails.values()), 2)] for t, k in tails.most_common(5)],
                            "box_change": {"n": bc["n"], "share": round(100 * bc["n"] / n, 2),   # [v2.5.8d]
                                           **{k: bc[k] for k in ("first_right_last_wrong", "first_wrong_last_right", "both_right", "both_wrong")},
                                           **{k + "_points": round(100 * bc[k] / n, 2) for k in ("first_right_last_wrong", "first_wrong_last_right")}}}}
    return res, probs


def q(xs, p):
    """the p-quantile (nearest rank) of xs, None when empty."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


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
    out += ["", "answer ready (v2.5.7v): correct samples, token position where the first box holding the final answer ends and the",
            "tokens written after it (p50 / p90; % after > 500), the first box of any content (p50, all samples), length p50",
            f"{'reading':58} {'n':>5} {'ready50':>8} {'ready90':>8} {'after50':>8} {'after90':>8} {'>500%':>6} {'1stbox50':>9} {'len50':>7} {'acc':>7} {'acc1st':>7} {'save1st':>8}"]
    for r in rows:
        a = r.get("answer_ready")
        if "error" in r or not a:
            continue
        g = lambda v, w: f"{v:>{w}}" if v is not None else f"{'-':>{w}}"
        out.append(f"{r['name'][:58]:58} {a['correct_with_answer_box']:>5} {g(a['first_correct_box_tok_p50'], 8)} {g(a['first_correct_box_tok_p90'], 8)} "
                   f"{g(a['after_tok_p50'], 8)} {g(a['after_tok_p90'], 8)} {g(a['after_gt_500_share'], 6)} {g(a['first_box_tok_p50'], 9)} {g(a['n_tokens_p50'], 7)} "
                   f"{g(r['acc'], 7)} {g(a.get('first_box_acc'), 7)} {g(a.get('first_box_save_tok_mean'), 8)}")
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
        if not set(per[A]) & set(per[B]):   # v2.5.7zm: e.g. readings of two benches; it crashed the whole run (ZeroDivisionError)
            res["pairs"].append({"a": A, "b": B, "error": "no common problems"}); continue
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
