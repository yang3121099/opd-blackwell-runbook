"""DEER as a training-free baseline (runbook [z166], lab/registry/Q-102.md B / F7; arXiv 2504.15895 "Dynamic Early Exit in Reasoning
Models", public code github.com/iie-ycx/DEER vllm-deer-qwen3.py, MIT): while the model thinks, at each transition point ("Wait") the
reasoning so far is cut, "\\n**Final Answer**\\n\\boxed" is appended and a short trial answer is generated (<= 20 tokens, stop at
</think>, with log-probs); its confidence (geometric mean of the sampled tokens' probabilities from the 2nd token on, 0 unless the trial
closed </think>) above the threshold ends the thinking ("\\n</think>\\n\\n" appended, the answer generated freely); otherwise "Wait" is
appended and the thinking goes on. At most --max_judge_steps trials; the thinking budget is --think_ratio of --max_tokens (a probe
within 50 tokens of it also ends the thinking); a natural </think> ends it as well.

Reading conventions of this runbook (each difference from the public implementation is listed in DIFFERENCES below and in the
registry's run record): the gen_eval prompt (the problem + PROMPT_SUFFIX in the user message, the chat template with enable_thinking),
the reading's sampling for the thinking and the answer (T0.6, top_p 0.95, top_k 0, n samples a problem, each its own state and seeds),
gen_eval's grader and record format (<stem>.jsonl / .metrics.json, so outcomes / TERM4 / forced passes read it as any reading).
Tokens (Fable 11-15): the MAIN count is every generated token, the trial answers included (n_tokens, mean_tokens: the real cost, the
frontier's x axis); the kept response's tokens are a side column (n_tokens_kept, mean_tokens_kept).
lambda (Fable 11-15): the paper's 0.95 by default; for Q-102 B three runs on Q-098's 100-problem calibration subset (--split calib) at
0.90 / 0.95 / 0.99, `deer_eval.py select` picks one by the prompt baselines' rule (accuracy not more than 1 point below the base's own
native reading on the same subset, then the shortest mean_tokens), and that lambda alone is read on the 400-problem test set (--split test).

Usage: python deer_eval.py --model M --bench math500 [--split all|calib|test] --n 4 --threshold 0.95 --tag TAG --out DIR [--chunk 100]
       python deer_eval.py select --base <base calib .metrics.json> <run .metrics.json> ...
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CONTINUE = {"wait": "Wait", "alternatively": "Alternatively"}   # DEER --points 1 / 0
ANSWER_PROMPT = "\n**Final Answer**\n\\boxed"
THINK_END = "</think>"
SPLIT_SEED, SPLIT_K, SPLIT_N = 20261115, 100, 500   # Q-098: random.Random(20261115).sample(range(500), 100) = calibration
DIFFERENCES = [
    ("prompt", "system message 'Please reason step by step, and put your final answer within \\boxed{}.' + the problem",
     "gen_eval's: the problem + PROMPT_SUFFIX in the user message (the same instruction), enable_thinking; as every reading of the frontier"),
    ("thinking / answer sampling", "--temperature 0.0 (greedy), top_p 1.0, one sample", "the reading's T0.6, top_p 0.95, top_k 0, n samples, a seed per (problem, sample, round)"),
    ("trial-answer sampling", "SamplingParams() defaults: temperature 1.0, top_p 1.0 (no temperature given)", "the same (temperature 1.0, top_p 1.0, top_k 0), seeded"),
    ("budget", "--max_generated_tokens 16000, think_ratio 0.8 (Qwen3 script)", "--max_tokens 16384 (the readings'), think_ratio 0.8"),
    ("token count", "the kept thinking + answer (trial answers not counted)", "MAIN: every generated token incl. trial answers (n_tokens); kept = n_tokens_kept"),
    ("confidence with a 1-token trial", "division by zero (the state errors out)", "0.0 (no exit)"),
    ("grader", "its own utils/grader.py", "gen_eval.grade (the frontier's grader)"),
    ("budget reached in a think step", "moves to the answer at the next round", "moves to the answer in the same round (same outcome, one round fewer)"),
    ("engine", "one vLLM engine, no prefix caching", "one vLLM engine with enable_prefix_caching (speed only)"),
    ("resume", "none", "--chunk K blocks of problems, a rerun keeps the finished blocks (spot reclaims)"),
]


def calib_indices(n=SPLIT_N, seed=SPLIT_SEED, k=SPLIT_K):
    return set(random.Random(seed).sample(range(n), k))


def confidence(probs, closed, policy="avg2"):
    """DEER's calculate_average_max_prob_from_logprobs: the sampled tokens' probabilities from the 2nd to the last; 0 unless the trial
    answer closed </think>. avg2 = geometric mean (the default), avg1 = arithmetic, min."""
    ps = list(probs[1:])
    if not closed or not ps:
        return 0.0
    if policy == "min":
        return min(ps)
    if policy == "avg1":
        return sum(ps) / len(ps)
    return math.exp(sum(math.log(max(p, 1e-10)) for p in ps) / len(ps))


def seed_of(base, pid, s, rnd):
    return int(hashlib.sha256(f"{base}:{pid}:{s}:{rnd}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def run_deer(states, engine, tok_len, a):
    """Drive every state to its end. engine.generate(prompts, params) -> [SimpleNamespace(text, n, finish_reason, stop_reason, probs)];
    tok_len(text) = the tokenizer's count (DEER re-encodes the thinking history). Each state: prompt, pid, s; filled: think, answer, ..."""
    cont = CONTINUE[a.points]; limit = int(a.max_tokens * a.think_ratio); rnd = 0
    samp = dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k)
    active = [st for st in states if st["phase"] != "done"]
    while active:
        rnd += 1; reqs = []
        for st in active:
            kind = st["phase"]
            if kind == "think":
                room = limit - tok_len(st["think"])
                if room <= 0:
                    st["phase"], st["exit"] = "answer", "too_long"; kind = "answer"
                else:
                    stop = [cont, THINK_END] if st["steps"] < a.max_judge_steps else [THINK_END]
                    reqs.append((st, kind, st["prompt"] + st["think"], dict(samp, max_tokens=room, stop=stop, seed=seed_of(a.seed, st["pid"], st["s"], rnd))))
                    continue
            if kind == "probe":
                reqs.append((st, kind, st["prompt"] + st["think"] + ANSWER_PROMPT,
                             dict(temperature=1.0, top_p=1.0, top_k=0, max_tokens=a.prob_check_max_tokens, stop=[THINK_END], logprobs=1,
                                  seed=seed_of(a.seed + 1, st["pid"], st["s"], rnd))))
            elif kind == "answer":
                room = max(1, a.max_tokens - tok_len(st["think"]))
                reqs.append((st, kind, st["prompt"] + st["think"] + "\n" + THINK_END + "\n\n", dict(samp, max_tokens=room, seed=seed_of(a.seed + 2, st["pid"], st["s"], rnd))))
        outs = engine.generate([r[2] for r in reqs], [r[3] for r in reqs])
        for (st, kind, _, _), o in zip(reqs, outs):
            st["gen_tokens"] += o.n
            if kind == "think":
                st["think"] += o.text
                if o.stop_reason == THINK_END:
                    st["phase"], st["exit"] = "answer", "regular_end"
                else:   # a transition point, or the budget: the trial answer decides
                    st["phase"] = "probe"; st["steps"] += 1
            elif kind == "probe":
                st["probe_tokens"] += o.n
                c = confidence(o.probs, o.stop_reason == THINK_END, a.policy); st["trials"].append(round(c, 4))
                near = tok_len(st["think"]) >= limit - 50
                if c > a.threshold or near:
                    st["phase"], st["exit"] = "answer", ("too_long" if near else "high_prob")
                else:
                    st["phase"] = "think"
                    if not st["think"].strip().endswith(cont) and st["steps"] != 0:
                        st["think"] += cont
            else:
                st["answer"], st["finish_reason"], st["phase"] = o.text, o.finish_reason, "done"
        active = [st for st in states if st["phase"] != "done"]
    return states


def new_state(prompt, pid, s):
    return {"prompt": prompt, "pid": pid, "s": s, "phase": "think", "think": "", "answer": "", "steps": 0, "gen_tokens": 0,
            "probe_tokens": 0, "trials": [], "exit": None, "finish_reason": None}


def final_text(st):
    return st["think"] + "\n" + THINK_END + "\n\n" + st["answer"]


def record_of(G, r, st, encode):
    """gen_eval's record of the kept response (grading, positions, token ids), then the main count = every generated token, the kept
    count as n_tokens_kept, and DEER's own fields."""
    text = final_text(st)
    rec = G.make_record(r, st["s"], SimpleNamespace(text=text, token_ids=encode(text), finish_reason=st["finish_reason"]), "think")
    rec["n_tokens_kept"] = rec["n_tokens"]; rec["n_tokens"] = st["gen_tokens"]
    rec["deer"] = {"exit": st["exit"], "trials": st["trials"], "n_trials": len(st["trials"]), "probe_tokens": st["probe_tokens"]}
    return rec


class VllmEngine:
    def __init__(self, a):
        import gen_eval as G
        from vllm import LLM
        self.llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=G.model_len(a.model, a.max_tokens), seed=a.seed,
                       trust_remote_code=True, dtype=a.dtype, enable_prefix_caching=True)

    def generate(self, prompts, params):
        from vllm import SamplingParams
        outs = self.llm.generate(prompts, [SamplingParams(n=1, **p) for p in params], use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]; probs = []
            if c.logprobs:
                probs = [math.exp(lp[t].logprob) if t in lp else math.exp(next(iter(lp.values())).logprob) for t, lp in zip(c.token_ids, c.logprobs)]
            res.append(SimpleNamespace(text=c.text, n=len(c.token_ids), finish_reason=c.finish_reason, stop_reason=c.stop_reason, probs=probs))
        return res


def select(base_metrics, runs):
    """Q-098's rule for the prompt baselines, applied to the lambda runs on the calibration subset: keep the runs whose accuracy is not
    more than 1 point below the base's own native reading there, then the shortest mean_tokens (the main count); none -> None."""
    b = base_metrics["avg_acc"]
    ok = [r for r in runs if r["avg_acc"] >= b - 1.0]
    return min(ok, key=lambda r: (r["mean_tokens"], -r["avg_acc"])) if ok else None


def main():
    if sys.argv[1:2] == ["select"]:
        ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("runs", nargs="+")
        a = ap.parse_args(sys.argv[2:])
        runs = [dict(json.load(open(f)), file=f) for f in a.runs]
        best = select(json.load(open(a.base)), runs)
        print(json.dumps({"base_acc": json.load(open(a.base))["avg_acc"], "candidates": [{k: r.get(k) for k in ("file", "deer_threshold", "avg_acc", "mean_tokens")} for r in runs],
                          "selected": best and {k: best.get(k) for k in ("file", "deer_threshold", "avg_acc", "mean_tokens")}}, indent=1))
        return 0
    import gen_eval as G
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--bench", default="math500"); ap.add_argument("--split", default="all", choices=["all", "calib", "test"])
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0); ap.add_argument("--max_tokens", type=int, default=16384); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.95); ap.add_argument("--think_ratio", type=float, default=0.8); ap.add_argument("--max_judge_steps", type=int, default=10)
    ap.add_argument("--prob_check_max_tokens", type=int, default=20); ap.add_argument("--policy", default="avg2", choices=["avg2", "avg1", "min"])
    ap.add_argument("--points", default="wait", choices=sorted(CONTINUE)); ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--gpu_mem", type=float, default=0.9); ap.add_argument("--dtype", default="auto"); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); a.mode = "think"; a.tp = 1; a.prefill = ""; a.seed_mode = "shared"; a.user_suffix = ""; a.chunk_size = a.chunk
    rows = G.load_bench(a.bench, a.limit)
    if a.split != "all":
        assert a.bench == "math500" and len(rows) == SPLIT_N, "the calibration split is defined on the 500 MATH-500 problems"
        cal = calib_indices(); rows = [r for i, r in enumerate(rows) if (i in cal) == (a.split == "calib")]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    tok_len = lambda t: len(tok.encode(t, add_special_tokens=False))
    prompt_of = lambda r: G.one_bos(tok, tok.apply_chat_template([{"role": "user", "content": G.user_content(a.bench, r["problem"])}], tokenize=False,
                                                                 add_generation_prompt=True, enable_thinking=True))
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"{a.tag}_{a.bench}_think_t{a.temperature}")
    key = {k: getattr(a, k) for k in ("model", "bench", "split", "n", "temperature", "top_p", "top_k", "max_tokens", "seed", "threshold", "think_ratio",
                                      "max_judge_steps", "prob_check_max_tokens", "policy", "points", "chunk", "limit")}
    part, meta = stem + ".jsonl.part", stem + ".jsonl.part.meta"
    recs, done_blocks, gen_s = [], 0, 0.0
    if os.path.exists(part) and os.path.exists(meta):
        m = json.load(open(meta))
        if m.get("args") != key:
            raise SystemExit(f"RESUME_REFUSED {part}: written with other arguments; move it aside first")
        recs = [json.loads(l) for l in open(part) if l.strip()]; done_blocks, gen_s = m["blocks"], m["gen_seconds"]
        recs = [r for r in recs if r["pid"] in {x["pid"] for x in rows[:done_blocks * a.chunk]}]
    engine = VllmEngine(a)
    for b in range(done_blocks * a.chunk, len(rows), a.chunk):
        t0 = time.time(); block = rows[b:b + a.chunk]
        states = run_deer([new_state(prompt_of(r), r["pid"], s) for r in block for s in range(a.n)], engine, tok_len, a)
        by = {(st["pid"], st["s"]): st for st in states}
        with open(part, "a") as f:
            for r in block:
                for s in range(a.n):
                    rec = record_of(G, r, by[(r["pid"], s)], lambda t: tok.encode(t, add_special_tokens=False))   # MAIN n_tokens: every generated token
                    f.write(json.dumps(rec) + "\n"); recs.append(rec)
        gen_s += time.time() - t0
        json.dump({"args": key, "blocks": b // a.chunk + 1, "gen_seconds": gen_s}, open(meta, "w"))
    m = G.compute_metrics(a, rows, recs, gen_s)
    m.update({"deer_threshold": a.threshold, "deer_think_ratio": a.think_ratio, "deer_max_judge_steps": a.max_judge_steps, "deer_policy": a.policy,
              "deer_points": a.points, "split": a.split, "tokens_main": "every generated token incl. trial answers (mean_tokens)",
              "mean_tokens_kept": round(sum(r["n_tokens_kept"] for r in recs) / len(recs), 1),
              "mean_trials": round(sum(r["deer"]["n_trials"] for r in recs) / len(recs), 2),
              "exit": {k: sum(r["deer"]["exit"] == k for r in recs) for k in ("high_prob", "regular_end", "too_long")},
              "differences_from_public_code": [list(d) for d in DIFFERENCES]})
    os.replace(part, stem + ".jsonl"); os.remove(meta)
    json.dump(m, open(stem + ".metrics.json", "w"), indent=2)
    print(json.dumps({k: m[k] for k in ("model", "bench", "avg_acc", "mean_tokens", "mean_tokens_kept", "mean_trials", "exit")}), flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush(); sys.stderr.flush()
    try:   # [z169] (executor C, uk-share 11-15: three calibrations kept their vLLM engines, 0 % and ~242 GB each, 35 min after metrics.json)
        from gen_eval import _reap_children   # as gen_eval / forced_stop: no engine core outlives the reading (v2.5.6e)
        _reap_children()
    except Exception as e:   # noqa: BLE001 - the exit below must happen whatever the reaper does
        print(f"REAP_FAILED {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    os._exit(rc or 0)
