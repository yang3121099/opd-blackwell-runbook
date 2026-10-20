"""Think win rate (TWR) and thinking activation shift (alpha), following 2610.14270 Eq. 1-4.

LLR(y) = (1/|y|) sum_t [ log p0(y_t | p_T, y_<t) - log p0(y_t | p_N, y_<t) ]   (2610.14270 Eq. 1: per-token MEAN,
  "the sum runs over all generated tokens of y", normalised by |y|), judge p0 = original model,
  p_T = think-mode chat prompt + '<think>\\n',  p_N = no-think chat prompt (ends with empty think block).
TWR(S) = P(LLR(y) > LLR(y')) + 0.5 P(=),  y ~ S, y' ~ R0 (judge's own no-think responses).
alpha  = <delta, v> / ||v||^2 at a middle layer, on fixed text R0:
  v     = h(p0, p_T, y) - h(p0, p_N, y);   delta = h(pk, p_N, y) - h(p0, p_N, y); cross-fitted odd/even problems.
Deviation from the paper (logged in output): responses truncated to --max_tok tokens (default 4096).

Usage:
  python twr_alpha.py --judge Qwen/Qwen3-1.7B --bench aime24 \
     --ref runs/g1/e01_q17_aime24_nothink_aime24_no_think_t0.7.jsonl \
     --score runs/g1/e01_opsa17_aime24_nothink_aime24_no_think_t0.7.jsonl \
     --model_k Tuwhy/Qwen3-1.7B-OPSA --out runs/leak/opsa17_aime24.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from gen_eval import PROMPT_SUFFIX, load_bench  # noqa: E402


def response_ids(tok, r, max_tok):
    """Prefer the generation-time token ids (gen_eval v2 stores them); otherwise re-tokenize.
    Scored text must be a NO-THINK response: p_T already ends with '<think>\\n', so a response that
    itself starts with <think> would get a double tag (R-001 suggestion 8)."""
    text = r["text"]
    assert not text.lstrip().startswith("<think>"), "score no-think responses only (text starts with <think>)"
    ids = r.get("token_ids") or tok(text, add_special_tokens=False)["input_ids"]
    return list(ids)[:max_tok]


def read(path, per_problem):
    rows, cnt = [], defaultdict(int)
    for line in open(path):
        r = json.loads(line)
        if cnt[r["pid"]] < per_problem:
            rows.append(r); cnt[r["pid"]] += 1
    return rows


def prompts(tok, problems):
    P_T, P_N = {}, {}
    for pid, prob in problems.items():
        m = [{"role": "user", "content": prob + PROMPT_SUFFIX}]
        P_T[pid] = tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True) + "<think>\n", add_special_tokens=False)["input_ids"]
        P_N[pid] = tok(tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]
    return P_T, P_N


@torch.no_grad()
def seq_lp(model, prefix, y):
    ids = torch.tensor([prefix + y], device=model.device)
    lg = model(input_ids=ids, use_cache=False).logits[0, len(prefix) - 1:len(prefix) + len(y) - 1].float().log_softmax(-1)
    return lg.gather(-1, torch.tensor(y, device=model.device)[:, None]).squeeze(-1)


@torch.no_grad()
def mean_hidden(model, prefix, y, layer):
    ids = torch.tensor([prefix + y], device=model.device)
    hs = model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states[layer]
    return hs[0, len(prefix):].float().mean(0)


def mann_whitney(a, b):
    gt = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    return gt / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ref", required=True, help="judge's own no-think generations (R0)")
    ap.add_argument("--score", nargs="+", required=True, help="generation jsonl files to score")
    ap.add_argument("--model_k", default=None, help="trained checkpoint for alpha (optional)")
    ap.add_argument("--layer", type=int, default=None, help="default: middle layer")
    ap.add_argument("--per_problem", type=int, default=4)
    ap.add_argument("--max_tok", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.judge)
    problems = {r["pid"]: r["problem"] for r in load_bench(args.bench)}
    P_T, P_N = prompts(tok, problems)
    judge = AutoModelForCausalLM.from_pretrained(args.judge, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    layer = args.layer if args.layer is not None else judge.config.num_hidden_layers // 2

    def llrs(rows):
        out = []
        for r in rows:
            y = response_ids(tok, r, args.max_tok)
            if not y:
                continue
            out.append(float((seq_lp(judge, P_T[r["pid"]], y) - seq_lp(judge, P_N[r["pid"]], y)).mean()))
        return out

    ref_rows = read(args.ref, args.per_problem)
    ref_llr = llrs(ref_rows)
    res = {"judge": args.judge, "bench": args.bench, "layer": layer, "max_tok": args.max_tok,
           "per_problem": args.per_problem, "ref": args.ref, "ref_mean_llr": sum(ref_llr) / len(ref_llr), "scores": {}}
    for path in args.score:
        l = llrs(read(path, args.per_problem))
        res["scores"][os.path.basename(path)] = {"twr": round(100 * mann_whitney(l, ref_llr), 2),
                                                "mean_llr": round(sum(l) / len(l), 5), "n": len(l)}
        print(os.path.basename(path), res["scores"][os.path.basename(path)], flush=True)

    mk = None
    if args.model_k:
        mk = AutoModelForCausalLM.from_pretrained(args.model_k, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        if mk.config.hidden_size != judge.config.hidden_size:   # [v2.5.8] a judge of another size: a clear line, the TWR kept
            res["alpha_error"] = (f"hidden size: judge {judge.config.hidden_size} != model_k {mk.config.hidden_size} "
                                  "(judge with the base of the student's own size)")
            print("ALPHA_REFUSED", res["alpha_error"], flush=True)
            mk = None
    if mk is not None:
        halves = {0: [], 1: []}
        for r in ref_rows:
            halves[int(r["pid"].split("-")[-1]) % 2].append(r)
        vs, ds = {}, {}
        for h, rows in halves.items():
            v = d = 0
            for r in rows:
                y = response_ids(tok, r, args.max_tok)
                h0n = mean_hidden(judge, P_N[r["pid"]], y, layer)
                v = v + (mean_hidden(judge, P_T[r["pid"]], y, layer) - h0n)
                d = d + (mean_hidden(mk, P_N[r["pid"]], y, layer) - h0n)
            vs[h], ds[h] = v / len(rows), d / len(rows)
        alpha_cf = 0.5 * (float(ds[1] @ vs[0]) / float(vs[0] @ vs[0]) + float(ds[0] @ vs[1]) / float(vs[1] @ vs[1]))
        v_all, d_all = (vs[0] + vs[1]) / 2, (ds[0] + ds[1]) / 2
        res["alpha_crossfit"] = round(alpha_cf, 4)
        res["alpha_same"] = round(float(d_all @ v_all) / float(v_all @ v_all), 4)
        res["cos_delta_v"] = round(float(d_all @ v_all) / (float(d_all.norm()) * float(v_all.norm()) + 1e-12), 4)
        print("alpha", res["alpha_crossfit"], "cos", res["cos_delta_v"], flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
