"""PROBE-VLM's checks (runbook [v2.5.9z], Q-051; research_program §3c), one step per call so each vLLM engine has its own process:

  same      (a2) the student's and the teacher's preprocessor configs (patch / merge / temporal patch size, min / max pixels, the image
            mean / std) and tokenizers (vocab, special tokens) equal field by field, and the teacher's chat-template prompt for the same
            image question = the student's + '<think>\\n' (what opd_vlmref.py assumes); -> same.json
  student   the student answers the first N (64) Geometry3K-train problems once (T=1.0, 4096 tokens, its own template); -> student.jsonl
  teacher   the teacher answers the same problems itself (thinking template, T=0.6, TEACHER_MAX tokens): finished share and accuracy
            (Do As I Do, 2610.20989: a teacher that does not finish is not a reliable target); -> teacher.jsonl
  pstop     the teacher scores each student answer that stopped: prompt = the teacher's own prompt + the student's answer + the end
            token, vLLM prompt_logprobs; p(stop) = the teacher's probability of that end token where the student stopped; -> pstop.jsonl
  verdict   PASS when the teacher's finished share >= 0.7 and the mean p(stop) >= 0.5 (both thresholds from §3c); -> check_b.json
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
           "template_teacher_is_student_plus_think": t_prompt == s_prompt + "<think>\n",
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
    prompts = [{"prompt": chat(proc, r) + s["text"] + end, "multi_modal_data": {"image": image(r)}} for r, s in items]
    outs = llm.generate(prompts, vllm.SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0))
    with open(os.path.join(a.out, "pstop.jsonl"), "w") as f:
        for (r, s), o in zip(items, outs):
            last_id = o.prompt_token_ids[-1]; lp = o.prompt_logprobs[-1] or {}
            v = lp.get(last_id)
            p = math.exp(v.logprob) if v is not None and last_id == end_id else None
            f.write(json.dumps({"i": s["i"], "end_is_last": last_id == end_id, "p_stop": p, "student_tokens": s["n_tokens"]}) + "\n")


def step_verdict(a):
    st = [json.loads(x) for x in open(os.path.join(a.out, "student.jsonl"))]
    te = [json.loads(x) for x in open(os.path.join(a.out, "teacher.jsonl"))]
    ps = [json.loads(x) for x in open(os.path.join(a.out, "pstop.jsonl"))]
    vals = [x["p_stop"] for x in ps if x["p_stop"] is not None]
    res = {"n": len(st), "student_finished": round(sum(x["finish_reason"] == "stop" for x in st) / max(1, len(st)), 4),
           "student_acc": round(sum(x["correct"] for x in st) / max(1, len(st)), 4),
           "teacher_finished": round(sum(x["finish_reason"] == "stop" for x in te) / max(1, len(te)), 4),
           "teacher_acc": round(sum(x["correct"] for x in te) / max(1, len(te)), 4),
           "pstop_n": len(vals), "pstop_mean": round(sum(vals) / len(vals), 4) if vals else None,
           "pstop_median": round(sorted(vals)[len(vals) // 2], 4) if vals else None,
           "thresholds": {"teacher_finished": float(os.environ.get("PVL_FINISH_MIN", "0.7")), "pstop_mean": float(os.environ.get("PVL_PSTOP_MIN", "0.5"))}}
    res["pass"] = bool(res["teacher_finished"] >= res["thresholds"]["teacher_finished"] and (res["pstop_mean"] or 0) >= res["thresholds"]["pstop_mean"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["same", "student", "teacher", "pstop", "verdict"]); ap.add_argument("out")
    ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True); ap.add_argument("--data", default="")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.step == "same":
        json.dump(step_same(a), open(os.path.join(a.out, "same.json"), "w"), indent=1)
    elif a.step == "student":
        gen(a.student, a.data, a.n, os.path.join(a.out, "student.jsonl"), 1.0, 4096)
    elif a.step == "teacher":
        gen(a.teacher, a.data, a.n, os.path.join(a.out, "teacher.jsonl"), 0.6, int(os.environ.get("PVL_TEACHER_MAX", "16384")))
    elif a.step == "pstop":
        step_pstop(a)
    else:
        json.dump(step_verdict(a), open(os.path.join(a.out, "check_b.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
