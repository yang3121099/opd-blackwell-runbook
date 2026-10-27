"""Zero-GPU diagnostics of 10-27 (runbook [v2.5.9z99]; Fable's research_program 3i / 3j / 3l, the CPU lane, nothing generated):

  t32tails     Q-082: the truncated samples of E10b-P5-t32's MATH-500 16K T0.6 reading - every one classed as loop / unfinished /
               box_more (vlm_trunc_classify), and 20 drawn (seed 0) with their class, box, loop start / period and last 300 characters,
               to judge loop vs continuing writing.
  probegate    Q-078: the probe-gated choice between the 1.7B base's non-thinking and thinking answers, per MATH-500 problem, read at a
               16K budget off the 32K readings (budget_curve.at_budget: the lab's method), against always-think, never-think and the
               oracle (the true label). H1: gate >= always-think - 1 and its mean length <= 0.7 x; H2: |gate - oracle| < 1.
  wallbimodal  Q-085-A: per arm with rollout dumps, per step f_long (share of responses longer than 3,686 tokens), the short and the
               long component's mean length, f_long among the probe's easy / hard prompts; per prompt its first long step.
               H-A1 (a mixture, not a shift): while f_long rises from < 0.2 to > 0.8 each component's mean moves < 20 %;
               H-A2 (hard first): at the wall step f_long(hard) - f_long(easy) >= 0.2, or hard prompts' median first long step
               >= 4 steps before the easy ones'.
  verifyonset  DIAG-VERIFY-ONSET: 100 samples each of two readings (seed 0); for every candidate verification marker its presence,
               its first position relative to the length and to the first complete box; the registered set's gate: >= 60 % of
               the samples have their first marker before the first box and the median position is 40-85 % of the length.
  vlmloop2     DIAG-VLM-LOOP2: the loops among the VLM readings' truncated samples - where they start (relative), the repeated
               unit's length and kind (calc / verify / restate / other), whether a box came before; and per problem whether it
               loops in both readings of a pair (problem-driven) or in one only (decoding-driven).

Usage: python zero_gpu_diag.py <command> --out <out.json> ...   (see each command's arguments in main)
"""
import argparse
import json
import os
import random
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_trunc_classify import box_end, classify  # noqa: E402

WINDOW, REPEATS = 20, 3


def rows(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def loop_info(ids, window=WINDOW, repeats=REPEATS):
    """None, or the first window seen `repeats` times: {start, period, unit} (token positions; period = the distance between
    its first two occurrences)."""
    if len(ids) < window * repeats:
        return None
    seen = {}
    for i in range(len(ids) - window + 1):
        k = tuple(ids[i:i + window])
        seen.setdefault(k, []).append(i)
        if len(seen[k]) >= repeats:
            s, p = seen[k][0], seen[k][1] - seen[k][0]
            return {"start": s, "period": p, "unit": list(ids[s:s + p])}
    return None


def box_start(text):
    """The index of the first complete \\boxed{...}, or -1."""
    e = box_end(text)
    return text.rfind("\\boxed{", 0, e) if e >= 0 else -1


# ---- t32tails --------------------------------------------------------------------------------------------------------------
def t32tails(path, n=20, seed=0):
    trunc, counts = [], {"loop": 0, "unfinished": 0, "box_more": 0}
    total = 0
    for r in rows(path):
        total += 1
        if r.get("finish_reason") != "length":
            continue
        c, _ = classify(r); counts[c] += 1; trunc.append((r, c))
    pick = random.Random(seed).sample(trunc, min(n, len(trunc)))
    out = []
    for r, c in pick:
        ids, text = r.get("token_ids") or [], r.get("text") or ""
        li = loop_info(ids)
        out.append({"pid": r.get("pid"), "sample": r.get("sample"), "n_tokens": r.get("n_tokens"), "class": c,
                    "kind": "loop" if c == "loop" else "continuation", "box": box_end(text) >= 0,
                    "first_box_frac": round(box_start(text) / max(len(text), 1), 3) if box_end(text) >= 0 else None,
                    "loop_start_frac": round(li["start"] / max(len(ids), 1), 3) if li else None,
                    "loop_period": li["period"] if li else None, "tail": text[-300:]})
    t = max(len(trunc), 1)
    return {"samples": total, "truncated": len(trunc), "counts": counts, "shares": {k: round(v / t, 4) for k, v in counts.items()},
            "drawn": out, "drawn_loop": sum(1 for x in out if x["kind"] == "loop"), "drawn_continuation": sum(1 for x in out if x["kind"] != "loop")}


# ---- probegate ---------------------------------------------------------------------------------------------------------------
def per_problem(samples):
    """{pid: (accuracy in [0, 1], mean tokens)} from [(pid, correct, truncated, tokens, sample)]."""
    acc, ln = {}, {}
    for pid, ok, _, n, _ in samples:
        acc.setdefault(pid, []).append(1.0 if ok else 0.0); ln.setdefault(pid, []).append(n)
    return {p: (sum(acc[p]) / len(acc[p]), sum(ln[p]) / len(ln[p])) for p in acc}


def boot_ci(diffs, n=10000, seed=0):
    rnd, m = random.Random(seed), len(diffs)
    if not m:
        return None
    means = sorted(sum(diffs[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return [round(100 * means[int(0.025 * n)], 2), round(100 * means[int(0.975 * n) - 1], 2)]


def probegate(nothink, think, probe, truth, boot=10000):
    """nothink / think: {pid: (acc, len)}; probe / truth: {pid: p_easy / r}. Easy (>= 0.5) -> the non-thinking answer."""
    pids = sorted(set(nothink) & set(think) & set(probe), key=lambda p: (len(p), p))
    pol = {"always_think": lambda p: True, "never_think": lambda p: False,
           "probe_gate": lambda p: probe[p] < 0.5, "oracle": lambda p: truth.get(p, 0.0) < 0.5}
    per = {k: [think[p] if f(p) else nothink[p] for p in pids] for k, f in pol.items()}
    res = {k: {"acc": round(100 * sum(a for a, _ in v) / len(v), 2), "mean_len": round(sum(l for _, l in v) / len(v), 1),
               "think_share": round(sum(1 for p in pids if pol[k](p)) / len(pids), 4)} for k, v in per.items()}
    g, t, o = per["probe_gate"], per["always_think"], per["oracle"]
    res["gate_minus_think_ci"] = boot_ci([a - b for (a, _), (b, _) in zip(g, t)], boot)
    res["gate_minus_oracle_ci"] = boot_ci([a - b for (a, _), (b, _) in zip(g, o)], boot)
    G, T, O = res["probe_gate"], res["always_think"], res["oracle"]
    res["H1"] = G["acc"] >= T["acc"] - 1 and G["mean_len"] <= 0.7 * T["mean_len"]
    res["H2"] = abs(G["acc"] - O["acc"]) < 1
    res["problems"] = len(pids)
    return res


def math500_labels(path, field="r"):
    return {f"math500-{int(r['pos'])}": float(r[field]) for r in rows(path) if r.get("split") == "math500" and field in r}


# ---- wallbimodal -------------------------------------------------------------------------------------------------------------
USER = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)


def prompt_key(rendered):
    """diff_labels.key: sha1 (16 hex) of the last user message's content."""
    import hashlib
    users = USER.findall(rendered or "")
    text = users[-1] if users else (rendered or "")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def step_stats(lens, easy, thresh):
    """lens: [(key, tokens)]; easy: {key: bool}. f_long, the components' means, f_long among easy / hard prompts."""
    long_ = [n for _, n in lens if n > thresh]; short = [n for _, n in lens if n <= thresh]
    def f(sel):
        v = [n for k, n in lens if k in easy and easy[k] == sel]
        return round(sum(1 for n in v if n > thresh) / len(v), 4) if v else None
    return {"n": len(lens), "f_long": round(len(long_) / max(len(lens), 1), 4),
            "short_mean": round(sum(short) / len(short), 1) if short else None, "long_mean": round(sum(long_) / len(long_), 1) if long_ else None,
            "f_long_easy": f(True), "f_long_hard": f(False)}


def wall_hypotheses(steps, first_long, easy):
    """steps: [(step, stats)] in order. H-A1 over the rise from the last f_long < 0.2 to the first f_long > 0.8; H-A2 at the wall
    step (the first f_long >= 0.5) and by the prompts' first long steps."""
    out = {"wall_step": next((s for s, x in steps if x["f_long"] >= 0.5), None)}
    hi = next((i for i, (_, x) in enumerate(steps) if x["f_long"] > 0.8), None)
    lo = max((i for i, (_, x) in enumerate(steps[:hi] if hi is not None else []) if x["f_long"] < 0.2), default=None)
    if hi is not None and lo is not None:
        win = steps[lo:hi + 1]
        move = {}
        for comp in ("short_mean", "long_mean"):
            v = [x[comp] for _, x in win if x[comp]]
            move[comp] = round(max(v) / min(v) - 1, 4) if len(v) >= 2 else None
        out["rise"] = [steps[lo][0], steps[hi][0]]; out["component_move"] = move
        out["H_A1"] = all(m is not None and m < 0.2 for m in move.values())
    else:
        out["H_A1"] = None
    w = next((x for s, x in steps if s == out["wall_step"]), None)
    gap = (w["f_long_hard"] - w["f_long_easy"]) if w and w["f_long_hard"] is not None and w["f_long_easy"] is not None else None
    fe = [s for k, s in first_long.items() if s is not None and easy.get(k) is True]
    fh = [s for k, s in first_long.items() if s is not None and easy.get(k) is False]
    lead = statistics.median(fe) - statistics.median(fh) if fe and fh else None
    out.update(wall_gap_hard_minus_easy=round(gap, 4) if gap is not None else None, first_long_median_easy=statistics.median(fe) if fe else None,
               first_long_median_hard=statistics.median(fh) if fh else None, hard_lead_steps=lead)
    out["H_A2"] = (gap is not None and gap >= 0.2) or (lead is not None and lead >= 4)
    return out


def wallbimodal(run_dirs, count_tokens, easy, thresh=3686.0):
    arms = {}
    for d in run_dirs:
        rd = os.path.join(d, "rollouts")
        files = sorted((int(m.group(1)), f) for f in os.listdir(rd) if (m := re.fullmatch(r"(\d+)\.jsonl", f)))
        steps, first_long = [], {}
        for st, f in files:
            recs = list(rows(os.path.join(rd, f)))
            ns = count_tokens([r.get("output") or "" for r in recs])
            lens = [(prompt_key(r.get("input")), n) for r, n in zip(recs, ns)]
            steps.append((st, step_stats(lens, easy, thresh)))
            by = {}
            for k, n in lens:
                by.setdefault(k, []).append(n)
            for k, v in by.items():
                if k not in first_long or first_long[k] is None:
                    first_long[k] = st if sum(v) / len(v) > thresh else first_long.get(k)
        arms[os.path.basename(d.rstrip("/"))] = {"steps": [dict(step=s, **x) for s, x in steps], **wall_hypotheses(steps, first_long, easy)}
    return {"thresh": thresh, "arms": arms}


# ---- verifyonset -------------------------------------------------------------------------------------------------------------
MARKERS = {   # case-insensitive; registered by the executor 10-27 10:39Z from a non-gate reading (tail-s43 s50, 100 samples), not changed after
    "verify_kw": r"\b(?:let me|let'?s|to|i(?:'ll| will)|now) verify\b", "verification": r"\bverification\b", "double_check": r"\bdouble[- ]check",
    "check_again": r"\bcheck (?:again|once more)\b|\b(?:let me|let'?s) (?:re-?check|recheck)\b",
    "check_answer": r"\b(?:let me|let'?s|to) check (?:my|the|this|our) (?:answer|result|solution|work|calculations?)\b",
    "check_if_correct": r"\bcheck (?:if|whether) (?:this|that|it|the answer|my answer) (?:is|makes sense|works|holds|satisf)",
    "plug_back": r"\b(?:plug|substitut)\w* (?:\w+ )?(?:\w+ )?back\b", "sanity": r"\bsanity[- ]check", "confirm_kw": r"\b(?:let me|let'?s|to) confirm\b",
    "check_bare": r"\b(?:let me|let'?s) check\b", "wait": r"\bwait\b",   # reported, not registered: 100 % present, median 13 % / 7 % (mid-course)
}
REGISTERED = ("verify_kw", "verification", "double_check", "check_again", "check_answer", "check_if_correct", "plug_back", "sanity", "confirm_kw")


def onset(text, names):
    pos = [m.start() for n in names for m in [re.search(MARKERS[n], text, re.I)] if m]
    return min(pos) if pos else -1


def verifyonset(readings, n=100, seed=0, registered=REGISTERED):
    out = {"registered": list(registered), "readings": {}}
    for label, path in readings:
        recs = list(rows(path)); pick = random.Random(seed).sample(recs, min(n, len(recs)))
        cand = {}
        for name in MARKERS:
            hits = [(onset(r.get("text") or "", [name]), r.get("text") or "") for r in pick]
            present = [(p, t) for p, t in hits if p >= 0]
            before = [1 for p, t in present if box_start(t) < 0 or p < box_start(t)]
            cand[name] = {"present": round(len(present) / len(pick), 3), "before_first_box": round(len(before) / max(len(present), 1), 3),
                          "median_rel_len": round(statistics.median(p / max(len(t), 1) for p, t in present), 3) if present else None}
        firsts = [(onset(r.get("text") or "", registered), r.get("text") or "") for r in pick]
        ok = [(p, t) for p, t in firsts if p >= 0 and (box_start(t) < 0 or p < box_start(t))]
        rel = [p / max(len(t), 1) for p, t in ok]
        rel_box = [p / box_start(t) for p, t in ok if box_start(t) > 0]
        share, med = len(ok) / len(pick), (statistics.median(rel) if rel else None)
        out["readings"][label] = {"samples": len(pick), "candidates": cand, "first_before_box_share": round(share, 3),
                                  "median_rel_len": round(med, 3) if med is not None else None,
                                  "median_rel_first_box": round(statistics.median(rel_box), 3) if rel_box else None,
                                  "gate": share >= 0.6 and med is not None and 0.40 <= med <= 0.85}
    out["gate"] = all(v["gate"] for v in out["readings"].values())
    return out


# ---- vlmloop2 ----------------------------------------------------------------------------------------------------------------
VERIFY_W = re.compile(r"\b(?:verify|check|wait|confirm|double)\b", re.I)
RESTATE_W = re.compile(r"\b(?:the problem|the question|we need to|given that|find the|asks)\b", re.I)


def unit_kind(text):
    if VERIFY_W.search(text):
        return "verify"
    if RESTATE_W.search(text):
        return "restate"
    sym = sum(ch.isdigit() or ch in "+-*/=^()." for ch in text)
    return "calc" if sym >= 0.3 * max(len(text.strip()), 1) else "other"


def loops_of(path, decode):
    """{pid: [loop record]} of a reading's truncated samples that loop; all its pids in 'seen'."""
    out, seen = {}, set()
    for r in rows(path):
        seen.add(r.get("pid"))
        if r.get("finish_reason") != "length":
            continue
        ids = r.get("token_ids") or []; li = loop_info(ids)
        if not li:
            continue
        text = r.get("text") or ""; pre = len(decode(ids[:li["start"]]))
        b = box_end(text)
        out.setdefault(r.get("pid"), []).append({"start_rel": round(li["start"] / max(len(ids), 1), 3), "period": li["period"],
                                                 "kind": unit_kind(decode(li["unit"])), "box_before": 0 <= b <= pre})
    return out, seen


def vlmloop2(pairs, singles, decode):
    def agg(loops):
        recs = [x for v in loops.values() for x in v]
        if not recs:
            return {"loops": 0}
        kinds = {}
        for x in recs:
            kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
        return {"loops": len(recs), "start_rel_median": statistics.median(x["start_rel"] for x in recs),
                "start_rel_quartiles": statistics.quantiles([x["start_rel"] for x in recs], n=4) if len(recs) > 1 else None,
                "period_median": statistics.median(x["period"] for x in recs), "kinds": kinds,
                "box_before_share": round(sum(1 for x in recs if x["box_before"]) / len(recs), 4)}
    out = {"readings": {}, "pairs": {}}
    cache = {}
    for label, path in singles + [x for _, a, b in pairs for x in (a, b)]:
        if label not in cache:
            cache[label] = loops_of(path, decode); out["readings"][label] = agg(cache[label][0])
    for name, (la, _), (lb, _) in pairs:
        (A, sa), (B, sb) = cache[la], cache[lb]
        both = sa & sb
        out["pairs"][name] = {"problems": len(both), "both": sum(1 for p in both if p in A and p in B),
                              "only_" + la: sum(1 for p in both if p in A and p not in B), "only_" + lb: sum(1 for p in both if p in B and p not in A),
                              "neither": sum(1 for p in both if p not in A and p not in B)}
    return out


# ---- main --------------------------------------------------------------------------------------------------------------------
def labelled(items):
    return [tuple(x.split("=", 1)) for x in items]


def tokenizer_fns(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    return (lambda ids: tok.decode(ids, skip_special_tokens=True),
            lambda texts: [len(x) for x in tok(texts, add_special_tokens=False)["input_ids"]] if texts else [])


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--out", required=True)
    ap.add_argument("--input", action="append", default=[]); ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--tokenizer"); ap.add_argument("--probe"); ap.add_argument("--labels"); ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--budget", type=int, default=16384); ap.add_argument("--thresh", type=float, default=3686.0)
    a = ap.parse_args(argv)
    if a.cmd == "t32tails":
        res = t32tails(a.input[0], a.n or 20)
    elif a.cmd == "probegate":
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval"))
        from budget_curve import at_budget
        from gen_eval import grade
        decode, _ = tokenizer_fns(a.tokenizer)
        ins = dict(labelled(a.input))
        nt = per_problem(at_budget(list(rows(ins["nothink"])), a.budget, decode, grade)[0])
        th = per_problem(at_budget(list(rows(ins["think"])), a.budget, decode, grade)[0])
        truth = math500_labels(a.labels) if a.labels else math500_labels(a.probe, "r_true")
        res = dict(budget=a.budget, **probegate(nt, th, math500_labels(a.probe), truth))
    elif a.cmd == "wallbimodal":
        _, count = tokenizer_fns(a.tokenizer)
        easy = {r["key"]: float(r["r"]) >= 0.5 for r in rows(a.probe) if r.get("split") == "train"} if a.probe else {}
        res = wallbimodal(a.runs, count, easy, a.thresh)
    elif a.cmd == "verifyonset":
        res = verifyonset(labelled(a.input), a.n or 100)
    elif a.cmd == "vlmloop2":
        decode, _ = tokenizer_fns(a.tokenizer)
        pairs = []
        for p in a.pair:   # name=labelA:pathA,labelB:pathB
            name, rest = p.split("=", 1); (la, pa), (lb, pb) = [x.split(":", 1) for x in rest.split(",")]
            pairs.append((name, (la, pa), (lb, pb)))
        res = vlmloop2(pairs, labelled(a.input), decode)
    else:
        sys.exit(f"unknown command {a.cmd}")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("drawn", "steps", "arms", "readings", "pairs")})[:600])


if __name__ == "__main__":
    main()
