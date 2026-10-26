"""PROBE-VLM's checks (runbook [v2.5.9z], Q-051; research_program §3c), one step per call so each vLLM engine has its own process:

  same      (a2) the student's and the teacher's preprocessor configs (patch / merge / temporal patch size, min / max pixels, the image
            mean / std) and tokenizers (vocab, special tokens) equal field by field, and the teacher's chat-template prompt for the same
            image question = the student's + '<think>\\n' (what opd_vlmref.py assumes); -> same.json
  student   the student answers the first N (64) Geometry3K-train problems once (T=1.0, 4096 tokens, its own template); -> student.jsonl
  teacher   the teacher answers the same problems itself (thinking template, T=0.6, TEACHER_MAX tokens): finished share and accuracy
            (Do As I Do, 2610.20989: a teacher that does not finish is not a reliable target); -> teacher.jsonl
  pstop     the teacher scores each student answer that stopped, once per suffix ([v2.5.9z1] the reviewer 10-22): prompt = the
            student's prompt + the suffix + the student's answer + the end token, vLLM prompt_logprobs; p(stop) = the teacher's
            probability of that end token where the student stopped. Suffixes: think = '<think>\n' (P5's reading), empty =
            '<think>\n\n</think>\n\n' (an empty think block: the thinking-only teacher's non-thinking reading); -> pstop.jsonl
  verdict   PASS when the teacher's finished share >= 0.7 and either suffix's mean p(stop) >= 0.5 (thresholds from §3c); the suffix
            for the arm: think if it passes, else empty (E15-VLM-P5e); -> check_b.json
  [v2.5.9z89] (Q-080a's gate PROBE-VLM-COMPLETE, Fable's ruling 10-26 23:0x UTC)
  same      PVL_EXPECT_SUFFIX: the teacher prompt = the student's + this ('<think>\n' by default; '' for an Instruct teacher)
  student   PVL_STUDENT_MAX tokens (4096 by default; 1536 for Q-080a)
  complete  --teacher scores each finished student answer once with the suffix PVL_SUFFIX ('' by default): prompt = the
            student's prompt + the suffix + the answer + the end token, vLLM prompt_logprobs=1: p(stop) and whether the end
            token is the argmax where the student stopped -> PVL_OUT (complete.jsonl)
  complete_verdict  -> complete.json: the completion rate = the share of finished student answers where the teacher's
            (complete_teacher.jsonl) argmax is the end token, PASS when >= PVL_COMPLETE_MIN (0.9); also the teacher's mean
            p_stop_last, the student's own finished share (of n) and its own p_stop_last (complete_self.jsonl), and the
            thinking teacher's non-thinking reading (complete_think.jsonl, when there; no gate). A student finished share
            below 0.5 is noted, not a FAIL (Q-080a trains on every rollout).
Usage: python vlm_teacher_check.py <step> <out dir> --student <path> --teacher <path> --data <geo3k_train.parquet> [--n 64]
"""
import argparse
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "eval"))

SUFFIXES = {"think": "<think>\n", "empty": "<think>\n\n</think>\n\n"}
FIELDS = ("patch_size", "merge_size", "temporal_patch_size", "min_pixels", "max_pixels", "image_mean", "image_std", "size")


def rows(path, n):
    import pandas as pd
    df = pd.read_parquet(path).head(n)
    return df.to_dict("records")


def image(r):
    from PIL import Image
    x = r["images"][0] if not isinstance(r["images"], dict) else r["images"]
    return Image.open(io.BytesIO(x["bytes"] if isinstance(x, dict) else x)).convert("RGB")


def text_of(r):
    return str(r["prompt"][0]["content"]).replace("<image>", "").strip()


def chat(proc, r):
    return proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_of(r)}]}], tokenize=False, add_generation_prompt=True)


def cfg_fields(path):
    p = json.load(open(os.path.join(path, "preprocessor_config.json")))
    return {k: p.get(k) for k in FIELDS if k in p}


def step_same(a):
    from transformers import AutoProcessor
    ps, pt = AutoProcessor.from_pretrained(a.student), AutoProcessor.from_pretrained(a.teacher)
    cs, ct = cfg_fields(a.student), cfg_fields(a.teacher)
    ts, tt = ps.tokenizer, pt.tokenizer
    q = {"prompt": [{"role": "user", "content": "<image>Find x."}]}
    s_prompt, t_prompt = chat(ps, q), chat(pt, q)
    res = {"preprocessor_equal": cs == ct, "student_cfg": cs, "teacher_cfg": ct,
           "vocab_equal": ts.get_vocab() == tt.get_vocab(), "special_equal": ts.all_special_tokens == tt.all_special_tokens,
           "expected_suffix": os.environ.get("PVL_EXPECT_SUFFIX", "<think>\n"),   # [v2.5.9z89] '' for an Instruct teacher
           "template_teacher_is_student_plus_think": t_prompt == s_prompt + os.environ.get("PVL_EXPECT_SUFFIX", "<think>\n"),
           "student_prompt_tail": s_prompt[-40:], "teacher_prompt_tail": t_prompt[-40:]}
    res["pass"] = bool(res["preprocessor_equal"] and res["vocab_equal"] and res["special_equal"] and res["template_teacher_is_student_plus_think"])
    return res


def gen(model, data, n, out, temperature, max_tokens, seed=0):
    import vllm
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model)
    rs = rows(data, n)
    llm = vllm.LLM(model=model, seed=seed, max_model_len=max_tokens + 4096, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=float(os.environ.get("PVL_UTIL", "0.85")))
    outs = llm.generate([{"prompt": chat(proc, r), "multi_modal_data": {"image": image(r)}} for r in rs],
                        vllm.SamplingParams(n=1, temperature=temperature, top_p=1.0 if temperature >= 1 else 0.95, max_tokens=max_tokens, seed=seed))
    import gen_eval as G
    with open(out, "w") as f:
        for i, (r, o) in enumerate(zip(rs, outs)):
            c = o.outputs[0]
            ok, _ = G.grade(c.text, r["reward_model"]["ground_truth"])
            f.write(json.dumps({"i": i, "text": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids), "correct": bool(ok)}) + "\n")


def step_pstop(a):
    import vllm
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(a.teacher)
    end = os.environ.get("PVL_END_TOKEN", "<|im_end|>")
    end_id = proc.tokenizer.convert_tokens_to_ids(end)
    rs = rows(a.data, a.n)
    st = [json.loads(x) for x in open(os.path.join(a.out, "student.jsonl"))]
    items = [(r, s) for r, s in zip(rs, st) if s["finish_reason"] == "stop"]
    llm = vllm.LLM(model=a.teacher, seed=0, max_model_len=4096 + 4096 + 64, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=float(os.environ.get("PVL_UTIL", "0.85")))
    got = {}
    for name, suf in SUFFIXES.items():
        prompts = [{"prompt": base_prompt(proc, r) + suf + s["text"] + end, "multi_modal_data": {"image": image(r)}} for r, s in items]
        outs = llm.generate(prompts, vllm.SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0)) if prompts else []
        got[name] = [p_last(o, end_id) for o in outs]
    with open(os.path.join(a.out, "pstop.jsonl"), "w") as f:
        for j, (r, s) in enumerate(items):
            f.write(json.dumps({"i": s["i"], "student_tokens": s["n_tokens"], **{f"p_stop_{k}": got[k][j] for k in SUFFIXES}}) + "\n")


def base_prompt(proc, r):
    """The student's prompt as the teacher's template renders it: the teacher's generation prompt without its '<think>\\n' opening
    (a2 checks teacher = student + '<think>\\n'), so each suffix is added once."""
    t = chat(proc, r)
    return t[: -len("<think>\n")] if t.endswith("<think>\n") else t


def p_last(o, end_id):
    last_id = o.prompt_token_ids[-1]; lp = o.prompt_logprobs[-1] or {}
    v = lp.get(last_id)
    return math.exp(v.logprob) if v is not None and last_id == end_id else None


def p_last_top(o, end_id):
    """[v2.5.9z89] (p(stop), argmax is the end token) at the prompt's last position (prompt_logprobs=1: the actual and the top-1)."""
    last_id = o.prompt_token_ids[-1]; lp = o.prompt_logprobs[-1] or {}
    if last_id != end_id or last_id not in lp:
        return None, None
    top = max(lp.items(), key=lambda kv: kv[1].logprob)[0]
    return math.exp(lp[last_id].logprob), bool(top == end_id)


def step_complete(a):
    """[v2.5.9z89] --teacher (any scoring model) on each finished student answer with the suffix PVL_SUFFIX -> PVL_OUT."""
    import vllm
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(a.teacher)
    end = os.environ.get("PVL_END_TOKEN", "<|im_end|>"); end_id = proc.tokenizer.convert_tokens_to_ids(end)
    suf = os.environ.get("PVL_SUFFIX", "")
    rs = rows(a.data, a.n)
    st = [json.loads(x) for x in open(os.path.join(a.out, "student.jsonl"))]
    items = [(r, s) for r, s in zip(rs, st) if s["finish_reason"] == "stop"]
    llm = vllm.LLM(model=a.teacher, seed=0, max_model_len=int(os.environ.get("PVL_MAXLEN", "8192")), limit_mm_per_prompt={"image": 1},
                   gpu_memory_utilization=float(os.environ.get("PVL_UTIL", "0.85")))
    prompts = [{"prompt": base_prompt(proc, r) + suf + s["text"] + end, "multi_modal_data": {"image": image(r)}} for r, s in items]
    outs = llm.generate(prompts, vllm.SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0)) if prompts else []
    with open(os.path.join(a.out, os.environ.get("PVL_OUT", "complete.jsonl")), "w") as f:
        for (r, s), o in zip(items, outs):
            p, top = p_last_top(o, end_id)
            f.write(json.dumps({"i": s["i"], "student_tokens": s["n_tokens"], "p_stop": p, "argmax_end": top}) + "\n")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def step_complete_verdict(a):
    """[v2.5.9z89] PROBE-VLM-COMPLETE's numbers and its PASS (see the docstring)."""
    rd = lambda n: [json.loads(x) for x in open(os.path.join(a.out, n))] if os.path.exists(os.path.join(a.out, n)) else None
    st, te, me, th = rd("student.jsonl") or [], rd("complete_teacher.jsonl") or [], rd("complete_self.jsonl"), rd("complete_think.jsonl")
    fin = sum(x["finish_reason"] == "stop" for x in st)
    tops = [x["argmax_end"] for x in te if x.get("argmax_end") is not None]
    res = {"n": len(st), "student_finished": round(fin / max(1, len(st)), 4), "student_finished_n": fin,
           "completion_rate": round(sum(tops) / len(tops), 4) if tops else None, "scored": len(tops),
           "teacher_p_stop_last_mean": _mean([x.get("p_stop") for x in te]),
           "student_p_stop_last_mean": _mean([x.get("p_stop") for x in me]) if me is not None else None,
           "student_argmax_end": _mean([float(x["argmax_end"]) for x in me if x.get("argmax_end") is not None]) if me is not None else None,
           "thinking_teacher_nonthinking": None if th is None else {"p_stop_last_mean": _mean([x.get("p_stop") for x in th]),
                                                                    "argmax_end": _mean([float(x["argmax_end"]) for x in th if x.get("argmax_end") is not None])},
           "threshold": float(os.environ.get("PVL_COMPLETE_MIN", "0.9"))}
    res["note"] = "the student finished under half of its answers (noted, not a gate)" if res["student_finished"] < 0.5 else ""
    res["pass"] = bool(res["completion_rate"] is not None and res["completion_rate"] >= res["threshold"])
    return res


def step_verdict(a):
    st = [json.loads(x) for x in open(os.path.join(a.out, "student.jsonl"))]
    te = [json.loads(x) for x in open(os.path.join(a.out, "teacher.jsonl"))]
    ps = [json.loads(x) for x in open(os.path.join(a.out, "pstop.jsonl"))]
    res = {"n": len(st), "student_finished": round(sum(x["finish_reason"] == "stop" for x in st) / max(1, len(st)), 4),
           "student_acc": round(sum(x["correct"] for x in st) / max(1, len(st)), 4),
           "teacher_finished": round(sum(x["finish_reason"] == "stop" for x in te) / max(1, len(te)), 4),
           "teacher_acc": round(sum(x["correct"] for x in te) / max(1, len(te)), 4), "pstop": {},
           "thresholds": {"teacher_finished": float(os.environ.get("PVL_FINISH_MIN", "0.7")), "pstop_mean": float(os.environ.get("PVL_PSTOP_MIN", "0.5"))}}
    for k in SUFFIXES:
        vals = [x.get(f"p_stop_{k}") for x in ps if x.get(f"p_stop_{k}") is not None]
        res["pstop"][k] = {"n": len(vals), "mean": round(sum(vals) / len(vals), 4) if vals else None,
                           "median": round(sorted(vals)[len(vals) // 2], 4) if vals else None}
    ok = {k: (v["mean"] or 0) >= res["thresholds"]["pstop_mean"] for k, v in res["pstop"].items()}
    res["suffix"] = "think" if ok.get("think") else ("empty" if ok.get("empty") else None)
    res["suffix_text"] = SUFFIXES.get(res["suffix"])
    res["pass"] = bool(res["teacher_finished"] >= res["thresholds"]["teacher_finished"] and res["suffix"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["same", "student", "teacher", "pstop", "verdict", "complete", "complete_verdict"]); ap.add_argument("out")
    ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True); ap.add_argument("--data", default="")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.step == "same":
        json.dump(step_same(a), open(os.path.join(a.out, "same.json"), "w"), indent=1)
    elif a.step == "student":
        gen(a.student, a.data, a.n, os.path.join(a.out, "student.jsonl"), 1.0, int(os.environ.get("PVL_STUDENT_MAX", "4096")))   # [v2.5.9z89]
    elif a.step == "teacher":
        gen(a.teacher, a.data, a.n, os.path.join(a.out, "teacher.jsonl"), 0.6, int(os.environ.get("PVL_TEACHER_MAX", "16384")))
    elif a.step == "pstop":
        step_pstop(a)
    elif a.step == "complete":   # [v2.5.9z89]
        step_complete(a)
    elif a.step == "complete_verdict":
        json.dump(step_complete_verdict(a), open(os.path.join(a.out, "complete.json"), "w"), indent=1)
    else:
        json.dump(step_verdict(a), open(os.path.join(a.out, "check_b.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
