"""Q-064 p* (runbook [v2.5.9z75]; Runming 10-22 19:1x Beijing, pre-registered research_program 10-22): the critical thinking share of
a mixed teacher from the two teachers' sequence log-probs.

Reverse KL to a mixture of teachers has its fixed point at the (sequence-level) geometric mean prod q_i^{w_i}; a sequence's log-prob
grows with its length, so the non-thinking teacher's penalty on a long answer, Delta_long = E[log q_think(long) - log q_plain(long)],
dwarfs the thinking teacher's penalty on a short one, Delta_short = E[log q_plain(short) - log q_think(short)], and the long mode
wins only when the thinking share p > p* = Delta_long / (Delta_long + Delta_short).
Measured here: N problems of MATH-500 (seeded, the same for both sets) with one short answer each from P2 s150's reading and one long
answer from P5 s150's (gen_eval jsonl: pid, sample, token_ids), each scored by the 8B teacher twice: on the thinking template
(P5's reading: the thinking generation prompt and '<think>\\n', as the trainer renders it) and on the non-thinking template (P2's).
Reported: Delta_long, Delta_short and p* from sequence sums, the same from per-token means (H2's contrast), bootstrap 95% CIs over the
problems (10000, seed 0), the sets' lengths; and the same with TERM's thinking prompt (no '<think>\\n') as a sensitivity row.
Lines: H1 p* (sums) in (0.5, 0.75); the measured interval (0.5, 0.6) (mix 0.5 did not cross, mix60 did); H2 the per-token p*'s CI
outside (0.5, 0.6). Numerics in numpy (test_pstar.py); the model only in main().
Usage: python pstar.py --teacher <8B> --short <P2 reading.jsonl> --long <P5 reading.jsonl> --out <dir> [--n 256] [--seed 0]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED = (0.5, 0.6)


def pick(short, long_, n, seed):
    """The problems both readings answer (sample 0 each), n of them by a seeded draw, sorted: [(pid, short rec, long rec)]."""
    s = {r["pid"]: r for r in short if int(r.get("sample", 0)) == 0}
    l_ = {r["pid"]: r for r in long_ if int(r.get("sample", 0)) == 0}
    common = sorted(set(s) & set(l_), key=lambda p: int(str(p).rsplit("-", 1)[-1]))
    idx = sorted(np.random.RandomState(seed).choice(len(common), min(n, len(common)), replace=False))
    return [(common[i], s[common[i]], l_[common[i]]) for i in idx]


def pstar(lt_long, lp_long, lt_short, lp_short, len_long=None, len_short=None):
    """p* from per-problem log-probs (lt = thinking reading, lp = plain); with lengths, from per-token means instead of sums."""
    lt_long, lp_long, lt_short, lp_short = map(lambda x: np.asarray(x, dtype=float), (lt_long, lp_long, lt_short, lp_short))
    dl, ds = lt_long - lp_long, lp_short - lt_short
    if len_long is not None:
        dl, ds = dl / np.asarray(len_long, dtype=float), ds / np.asarray(len_short, dtype=float)
    a, b = float(dl.mean()), float(ds.mean())
    return {"delta_long": round(a, 4), "delta_short": round(b, 4), "p_star": round(a / (a + b), 4) if a + b > 0 else None}


def boot(f, n, reps=10000, seed=0):
    """95% CI of f(index array) over reps resamples of n problems."""
    rng = np.random.default_rng(seed); v = []
    for _ in range(reps):
        x = f(rng.integers(0, n, n))
        if x is not None:
            v.append(x)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if v else None


def summarize(rows):
    """rows: [{lt_long, lp_long, lt_short, lp_short, n_long, n_short}] -> sums, per-token, CIs, the lines."""
    A = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    n = len(rows)
    sums = pstar(A["lt_long"], A["lp_long"], A["lt_short"], A["lp_short"])
    tok = pstar(A["lt_long"], A["lp_long"], A["lt_short"], A["lp_short"], A["n_long"], A["n_short"])
    sums["p_star_ci95"] = boot(lambda i: pstar(A["lt_long"][i], A["lp_long"][i], A["lt_short"][i], A["lp_short"][i])["p_star"], n)
    tok["p_star_ci95"] = boot(lambda i: pstar(A["lt_long"][i], A["lp_long"][i], A["lt_short"][i], A["lp_short"][i], A["n_long"][i], A["n_short"][i])["p_star"], n)
    p, ci_t = sums["p_star"], tok["p_star_ci95"]
    return {"problems": n, "mean_len_long": round(float(A["n_long"].mean()), 1), "mean_len_short": round(float(A["n_short"].mean()), 1),
            "sums": sums, "per_token": tok,
            "H1": bool(p is not None and 0.5 < p < 0.75), "in_measured_interval": bool(p is not None and MEASURED[0] < p < MEASURED[1]),
            "H2": bool(ci_t is not None and (ci_t[1] <= MEASURED[0] or ci_t[0] >= MEASURED[1]))}


# ---------------------------------------------------------------------------------------------------------------- the model
def seq_logprob(model, prompt_ids, resp_ids, torch, chunk=2048):
    """sum of log q(resp_ids[k] | prompt, resp[:k]) under model, in chunks of positions (a 16K answer's logits stay small)."""
    ids = torch.tensor([prompt_ids + resp_ids], device="cuda")
    with torch.no_grad():
        h = model.model(input_ids=ids).last_hidden_state[0]
        tot, start = 0.0, len(prompt_ids) - 1
        tgt = torch.tensor(resp_ids, device="cuda")
        for i in range(0, len(resp_ids), chunk):
            lg = model.lm_head(h[start + i: start + i + min(chunk, len(resp_ids) - i)]).float()
            tot += float(torch.log_softmax(lg, -1).gather(1, tgt[i:i + chunk, None]).sum())
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True); ap.add_argument("--short", required=True); ap.add_argument("--long", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.join(HERE, "..", "eval")); import gen_eval
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    rd = lambda p: [json.loads(x) for x in open(p) if x.strip()]
    sel = pick(rd(a.short), rd(a.long), a.n, a.seed)
    bench = {r["pid"]: r for r in gen_eval.load_bench("math500")}
    tok = AutoTokenizer.from_pretrained(a.teacher)
    model = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16).to("cuda").eval()
    def prompt(pid, think, opener):
        msgs = [{"role": "user", "content": bench[pid]["problem"] + gen_eval.PROMPT_SUFFIX}]
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think) + ("<think>\n" if opener else "")
        return tok(s, add_special_tokens=False)["input_ids"]
    rows, rows_term = [], []
    for pid, s, l_ in sel:
        pt, pt0, pp = prompt(pid, True, True), prompt(pid, True, False), prompt(pid, False, False)
        ls, ll = list(s["token_ids"]), list(l_["token_ids"])
        r = {"lt_long": seq_logprob(model, pt, ll, torch), "lp_long": seq_logprob(model, pp, ll, torch),
             "lt_short": seq_logprob(model, pt, ls, torch), "lp_short": seq_logprob(model, pp, ls, torch), "n_long": len(ll), "n_short": len(ls)}
        rows.append(r)
        rows_term.append(dict(r, lt_long=seq_logprob(model, pt0, ll, torch), lt_short=seq_logprob(model, pt0, ls, torch)))
    t1 = time.time()
    rep = {"teacher": os.path.basename(a.teacher.rstrip("/")), "short": os.path.basename(a.short), "long": os.path.basename(a.long),
           "n": a.n, "seed": a.seed, **summarize(rows), "term_prompt": summarize(rows_term),
           "lines": {"H1": "p* (sums) in (0.5, 0.75)", "measured": "(0.5, 0.6): mix 0.5 did not cross, mix60 did",
                     "H2": "the per-token p*'s 95% CI outside (0.5, 0.6)"},
           "seconds": {"scoring": round(t1 - t0, 1)}}
    json.dump(rep, open(os.path.join(a.out, "pstar.json"), "w"), indent=1)
    with open(os.path.join(a.out, "rows.jsonl"), "w") as f:
        for (pid, _, _), r in zip(sel, rows):
            f.write(json.dumps({"pid": pid, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}}) + "\n")
    print("PSTAR", json.dumps({k: rep[k] for k in ("problems", "sums", "per_token", "H1", "in_measured_interval", "H2")}))


if __name__ == "__main__":
    main()
