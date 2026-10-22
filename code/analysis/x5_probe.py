"""PROBE-X5's measurements (runbook [v2.5.9z46]; Q-057 E16-X5, the reviewer's gate 10-22 14:4x Beijing, ruling A 10:0x UTC). Two phases,
each its own process on the probe's GPU (vLLM):

  python x5_probe.py student --model <student> --out <dir> [--n 64 --max_tokens 3072]
      the student (Qwen2.5-Math-1.5B-Instruct, its own template) answers the first n MATH-500 problems (T 0.6, top-p 0.95, seed 0,
      max_model_len 4096: its positions); per sample the response ids, the finish reason, the terminal id, correct. <dir>/student.jsonl
  python x5_probe.py teacher --model <teacher> --out <dir> [--n 64 --max_tokens 16384]
      (a) the teacher's own answers to the same problems on its thinking template: the share that finish (an end token within
          max_tokens) and the share correct;
      (b) at every student stop (a response ending in 151645 or 151643) the teacher's p(151643) after the student's response, the
          prompt in the teacher's template ending in one '<think>\\n' (the arm's prefix) and in an empty thought (E16-X5e's), read
          as vLLM's prompt log-prob of 151643 appended (the vocabularies agree below 151643, so the student's ids go in as they are)
      <dir>/teacher.json: completion, accuracy, p_stop_think / p_stop_empty (mean, median, n), and x5e = (think < 0.3 and empty
      >= 0.5): the reviewer's condition for E16-X5e.
"""
import argparse
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval")); sys.path.insert(0, os.path.join(HERE, "..", "patches"))
STOP_IDS = (151645, 151643)
TEACHER_EOS = 151643


def messages(problem):
    import gen_eval as G
    return [{"role": "user", "content": problem + G.PROMPT_SUFFIX}]


def student_summary(recs):
    """Counts over the student's samples: terminal ids, truncation, accuracy."""
    n = len(recs)
    t45 = sum(1 for r in recs if r["terminal_id"] == 151645); t43 = sum(1 for r in recs if r["terminal_id"] == 151643)
    return {"n": n, "term_151645": t45, "term_151643": t43, "truncated": n - t45 - t43,
            "share_151645": round(t45 / n, 4) if n else None, "share_151643": round(t43 / n, 4) if n else None,
            "acc": round(sum(r["correct"] for r in recs) / n, 4) if n else None}


def stop_inputs(prefix_ids, resp_ids):
    """The teacher's ids for one student stop: its prompt, the student's response without the stop id, then the teacher's EOS."""
    assert resp_ids and resp_ids[-1] in STOP_IDS
    return list(prefix_ids) + list(resp_ids[:-1]) + [TEACHER_EOS]


def stats(ps):
    return {"mean": round(statistics.mean(ps), 4), "median": round(statistics.median(ps), 4), "n": len(ps)} if ps else {"mean": None, "median": None, "n": 0}


def x5e_condition(think, empty):
    return think is not None and empty is not None and think < 0.3 and empty >= 0.5


def run_student(a):
    import gen_eval as G
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = G.load_bench("math500", a.n)
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = [tok.apply_chat_template(messages(r["problem"]), tokenize=False, add_generation_prompt=True) for r in rows]
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=G.model_len(a.model, a.max_tokens), seed=0, dtype="auto")
    outs = llm.generate(prompts, SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=0))
    recs = []
    for r, o in zip(rows, outs):
        c = o.outputs[0]; ids = list(c.token_ids)
        recs.append({"pid": r.get("unique_id", r.get("pid")), "problem": r["problem"], "ids": ids, "finish": c.finish_reason,
                     "terminal_id": ids[-1] if ids else None, "correct": bool(G.grade(c.text, r["answer"])[0])})
    with open(os.path.join(a.out, "student.jsonl"), "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    s = student_summary(recs); json.dump(s, open(os.path.join(a.out, "student_summary.json"), "w"), indent=1)
    print("X5_STUDENT", json.dumps(s))


def run_teacher(a):
    import gen_eval as G
    import opd_x5 as X
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    rows = G.load_bench("math500", a.n)
    stu = [json.loads(x) for x in open(os.path.join(a.out, "student.jsonl"))]
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=G.model_len(a.model, a.max_tokens), seed=0, dtype="auto")
    # (a) its own answers on its thinking template (the template ends in '<think>\n' already)
    own = [X.teacher_prompt(tok.apply_chat_template(messages(r["problem"]), tokenize=False, add_generation_prompt=True), empty=False) for r in rows]
    own = [{"prompt_token_ids": tok(p, add_special_tokens=False)["input_ids"]} for p in own]   # the template writes the BOS: no second one
    outs = llm.generate(own, SamplingParams(n=1, temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=0))
    fin = [o.outputs[0].finish_reason == "stop" for o in outs]
    acc = [bool(G.grade(o.outputs[0].text, r["answer"])[0]) for r, o in zip(rows, outs)]
    # (b) p(151643) at the student's stops under the two prefixes
    res = {}
    for name, empty in (("think", False), ("empty", True)):
        ins = []
        for x in stu:
            if x["ids"] and x["ids"][-1] in STOP_IDS:
                p = X.teacher_prompt(tok.apply_chat_template(messages(x["problem"]), tokenize=False, add_generation_prompt=True), empty=empty)
                ins.append(stop_inputs(tok(p, add_special_tokens=False)["input_ids"], x["ids"]))
        ps = []
        if ins:
            o2 = llm.generate([{"prompt_token_ids": i} for i in ins], SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0))
            for o in o2:
                lp = o.prompt_logprobs[-1][TEACHER_EOS]
                ps.append(math.exp(lp.logprob if hasattr(lp, "logprob") else lp))
        res[name] = stats(ps)
    out = {"n": len(rows), "completion": round(sum(fin) / len(fin), 4), "accuracy": round(sum(acc) / len(acc), 4), "max_tokens": a.max_tokens,
           "p_stop_think": res["think"], "p_stop_empty": res["empty"],
           "x5e": x5e_condition(res["think"]["mean"], res["empty"]["mean"])}
    json.dump(out, open(os.path.join(a.out, "teacher.json"), "w"), indent=1)
    print("X5_TEACHER", json.dumps(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["student", "teacher"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=None)
    ap.add_argument("--gpu_mem", type=float, default=0.5)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.max_tokens is None:
        a.max_tokens = 3072 if a.phase == "student" else 16384
    (run_student if a.phase == "student" else run_teacher)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
