"""Image-question evaluation with vLLM (runbook [v2.5.9z], Q-051; E15-VLM's readings). The counterpart of gen_eval.py for the Qwen3-VL
models, run with venv-vl's python; the same arguments, file names (<tag>_<bench>_<mode>_t<T>.jsonl / .metrics.json) and record /
metrics fields, so #pair / #strict / #budget lines read its outputs as they read gen_eval's.

Benches (from the node's copies, nothing fetched here):
  geo3k     Geometry3K test (601), data/hf/hiyouga__geometry3k: the problem without its '<image>' placeholder + gen_eval's
            PROMPT_SUFFIX; graded as gen_eval grades MATH-500 (the last \\boxed{} under Math-Verify).
  mathvista MathVista testmini (1000), data/hf/AI4Math__MathVista (CC BY-SA 4.0, evaluation only): the dataset's own `query` +
            PROMPT_SUFFIX; graded by rule from the last \\boxed{} (no LLM extraction): multi_choice = the boxed text equal to the
            answer (normalised) or a letter A.. naming the answer's choice; integer / float = the number equal at the item's
            precision (0 decimals for integer); text / list = normalised equality, else Math-Verify.
The image goes in as one {"type": "image"} item before the text, rendered by the model's own processor chat template, and to vLLM as
multi_modal_data. Usage: python gen_eval_vl.py --model <path> --bench geo3k|mathvista --mode no_think|think --n 4 --temperature 0.6
--top_p 0.95 --top_k 0 --max_tokens 16384 --tag <tag> --out <dir> [--limit N --seed 0 --chunk K --dtype auto]
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_eval as G  # noqa: E402  (PROMPT_SUFFIX, grade, last_boxed, pass_at_k)

DATA = {"geo3k": ("data/hf/hiyouga__geometry3k", "test-*.parquet"), "mathvista": ("data/hf/AI4Math__MathVista", "testmini-*.parquet")}


def load(bench, root, limit=None):
    d, pat = DATA[bench]
    fs = sorted(glob.glob(os.path.join(root, d, "**", pat), recursive=True))
    if not fs:
        raise FileNotFoundError(f"no {pat} under {d}")
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    rows = []
    for i, r in enumerate(df.to_dict("records")):
        if bench == "geo3k":
            img = r["images"][0] if not isinstance(r["images"], dict) else r["images"]
            rows.append({"pid": f"geo3k-{i}", "text": str(r["problem"]).replace("<image>", "").strip() + G.PROMPT_SUFFIX,
                         "image": img, "answer": str(r["answer"]).strip(), "level": None, "source": "hiyouga/geometry3k"})
        else:
            rows.append({"pid": f"mathvista-{r.get('pid', i)}", "text": str(r["query"]).strip() + G.PROMPT_SUFFIX,
                         "image": r["decoded_image"], "answer": str(r["answer"]).strip(), "level": r.get("question_type"),
                         "choices": list(r["choices"]) if r.get("choices") is not None else None, "question_type": r.get("question_type"),
                         "answer_type": r.get("answer_type"), "precision": r.get("precision"), "source": "AI4Math/MathVista"})
    return rows[:limit] if limit else rows


def norm(s):
    s = str(s).strip().strip("$").replace("\\text{", "").replace("}", "").replace("{", "")
    return re.sub(r"\s+", " ", s).strip().rstrip(".").lower()


def boxed(text):
    """The content of the last \\boxed{...} (nested braces matched), or None."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None


def grade_mathvista(text, row):
    """(correct, pred) by rule from the last boxed answer."""
    pred = boxed(text)
    if pred is None:
        return False, None
    if row.get("question_type") == "multi_choice" and row.get("choices"):
        ch = [str(c) for c in row["choices"]]
        p = norm(pred)
        if p == norm(row["answer"]):
            return True, pred
        m = re.fullmatch(r"\(?([a-z])\)?", p)
        if m and ord(m.group(1)) - 97 < len(ch):
            return norm(ch[ord(m.group(1)) - 97]) == norm(row["answer"]), pred
        return False, pred
    if row.get("answer_type") in ("integer", "float"):
        try:
            x = float(re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))[-1])
            prec = int(row.get("precision") or 0) if row.get("answer_type") == "float" else 0
            return round(x, prec) == round(float(row["answer"]), prec), pred
        except (IndexError, ValueError):
            return False, pred
    if norm(pred) == norm(row["answer"]):
        return True, pred
    ok, _ = G.grade(text, row["answer"])
    return bool(ok), pred


def grade(bench, text, row):
    if bench == "geo3k":
        ok, pred = G.grade(text, row["answer"])
        return bool(ok), pred
    return grade_mathvista(text, row)


def image_of(x):
    from PIL import Image
    b = x.get("bytes") if isinstance(x, dict) else x
    return Image.open(io.BytesIO(b)).convert("RGB")


PROTOCOL = {"geo3k": "the last \\boxed{} under Math-Verify, as MATH-500",   # [v2.5.9z1] the reviewer: say what the reading is
            "mathvista": "rule-graded, not the official LLM-extraction protocol"}
PROC_FILES = ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json", "chat_template.jinja")


def ensure_processor(model, base):
    """A trained checkpoint saved without the processor's files gets them from the base it was trained from (copied, missing ones
    only), so vLLM and the chat template find them; nothing is overwritten. Returns the files copied."""
    done = []
    if os.path.isdir(model) and not os.path.exists(os.path.join(model, "preprocessor_config.json")) and base and os.path.isdir(base):
        import shutil
        for f in PROC_FILES:
            if os.path.exists(os.path.join(base, f)) and not os.path.exists(os.path.join(model, f)):
                shutil.copy2(os.path.join(base, f), os.path.join(model, f)); done.append(f)
    return done


def metrics(args, rows, recs, gen_s):
    per = defaultdict(list); n_tok = [r["n_tokens"] for r in recs]
    for r in recs:
        per[r["pid"]].append(r["correct"])
    acc = sum(sum(v) for v in per.values()) / max(1, sum(len(v) for v in per.values()))
    return {"model": args.model, "bench": args.bench, "mode": args.mode, "n": args.n, "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "max_tokens": args.max_tokens, "num_problems": len(rows), "avg_acc": round(100 * acc, 2),
            "pass_at_k": {k: round(100 * sum(G.pass_at_k(len(v), sum(v), k) for v in per.values()) / max(1, len(per)), 2) for k in [1, 2, 4, 8] if k <= args.n},
            "mean_tokens": round(sum(n_tok) / max(1, len(n_tok)), 1), "trunc_rate": round(sum(1 for r in recs if r["finish_reason"] == "length") / max(1, len(recs)), 4),
            "by_level": {}, "gen_seconds": round(gen_s, 1), "source": rows[0]["source"] if rows else None, "grader": "gen_eval_vl rule grader v1",
            "protocol": PROTOCOL[args.bench]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--bench", required=True, choices=sorted(DATA))
    ap.add_argument("--mode", default="no_think"); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=0); ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--tag", default=None); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--chunk", type=int, default=0); ap.add_argument("--dtype", default="auto")
    a = ap.parse_args()
    root = os.environ.get("OPD_ROOT", ".")
    rows = load(a.bench, root, a.limit)
    import vllm
    from transformers import AutoProcessor
    ensure_processor(a.model, os.environ.get("VL_PROCESSOR", os.path.join(root, "models", "Qwen3-VL-2B-Instruct")))
    proc = AutoProcessor.from_pretrained(a.model)
    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["text"]}]}]
        prompts.append({"prompt": proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), "multi_modal_data": {"image": image_of(r["image"])}})
    llm = vllm.LLM(model=a.model, seed=a.seed, dtype=a.dtype, max_model_len=a.max_tokens + 8192, limit_mm_per_prompt={"image": 1},
                   gpu_memory_utilization=float(os.environ.get("EVAL_VL_UTIL", "0.85")))
    sp = vllm.SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p, top_k=a.top_k if a.top_k > 0 else -1, max_tokens=a.max_tokens, seed=a.seed)
    t0 = time.time(); outs = llm.generate(prompts, sp); gen_s = time.time() - t0
    recs = []
    for r, o in zip(rows, outs):
        for si, c in enumerate(o.outputs):
            ok, pred = grade(a.bench, c.text, r)
            recs.append({"pid": r["pid"], "sample": si, "level": r["level"], "gold": r["answer"], "pred": pred, "correct": ok,
                         "n_tokens": len(c.token_ids), "finish_reason": c.finish_reason, "terminated": c.finish_reason != "length",
                         "think_closed": ("</think>" in c.text) if a.mode == "think" else None, "token_ids": list(c.token_ids), "text": c.text})
    os.makedirs(a.out, exist_ok=True)
    stem = f"{a.tag or os.path.basename(a.model.rstrip('/'))}_{a.bench}_{a.mode}_t{a.temperature}"
    with open(os.path.join(a.out, stem + ".jsonl"), "w") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")
    m = metrics(a, rows, recs, gen_s)
    json.dump(m, open(os.path.join(a.out, stem + ".metrics.json"), "w"), indent=1)
    print("GEN_EVAL_VL", json.dumps({k: m[k] for k in ("avg_acc", "mean_tokens", "trunc_rate", "num_problems")}))


if __name__ == "__main__":
    main()
