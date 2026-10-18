"""Helpers of the SPEED-PROBE task (runbook v2.5.6o; speed_probe.sh runs the steps).
  python speed_probe.py lp <model> <rollouts.jsonl> <n>   -> JSON: per-token log-probs of n rollout sequences (prompt +
        response, response tokens compared) through verl's plain path (logits / temperature 1.0, verl's
        logprobs_from_logits) and through verl's fused kernels (apply_monkey_patch, torch backend, the path
        use_fused_kernels=True takes in the actor and the teacher), bf16 like the actor; max / mean |difference|,
        pass = max < 1e-2 and mean < 1e-3 (the v2.5.4 protocol, B300 speed list 2026-10-17); run from G-OPD/verl
  python speed_probe.py report <dir>                      -> <dir>/report.json: the two log-prob checks, and per smoke
        variant (base / fused / gc0 / fused_gc0) exit code, peak GPU memory, steps 2-3 mean of each timing segment and
        its ratio to base, step 1-3 pg_loss and response length
"""
import json
import os
import re
import sys

TIMING = ("timing_s/gen", "timing_s/old_log_prob", "timing_s/ref", "timing_s/update_actor", "timing_s/step")
METRICS = ("actor/pg_loss", "response_length/mean")
VARIANTS = ("base", "fused", "gc0", "fused_gc0")


def lp(model_path, rollouts, n):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.torch_functional import logprobs_from_logits
    tok = AutoTokenizer.from_pretrained(model_path)
    seqs = []
    for line in open(rollouts):
        r = json.loads(line)
        p, o = r.get("input", r.get("prompt")), r.get("output", r.get("response"))
        if not p or not o:
            continue
        seqs.append((tok(p, add_special_tokens=False)["input_ids"], tok(o, add_special_tokens=False)["input_ids"][:4096]))
        if len(seqs) >= n:
            break
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    cls, orig = model.__class__, model.__class__.forward

    def run(fused):
        out = []
        with torch.no_grad():
            for p, o in seqs:
                ids = torch.tensor([p + o], device="cuda"); L = len(o)
                if fused:
                    out.append(model(input_ids=ids, use_cache=False, temperature=1.0, return_dict=True).log_probs[:, -L - 1:-1].float().cpu())
                else:
                    logits = model(input_ids=ids, use_cache=False).logits[:, -L - 1:-1, :]
                    out.append(logprobs_from_logits(logits.div_(1.0), ids[:, -L:]).float().cpu())
        return out
    try:
        plain = run(False)
        apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=True, fused_kernels_backend="torch")
        fused = run(True)
    finally:
        cls.forward = orig            # the patch is class-wide: undo it for the next model in this process
    d = torch.cat([(a - b).abs().flatten() for a, b in zip(plain, fused)])
    mx, mean = d.max().item(), d.mean().item()
    return {"model": os.path.basename(model_path.rstrip("/")), "sequences": len(seqs), "tokens": d.numel(),
            "max_abs_diff": mx, "mean_abs_diff": mean, "pass": bool(mx < 1e-2 and mean < 1e-3)}


def steps_of(text):
    rows = []
    for st, body in re.findall(r"(?:^|[ )])step:(\d+) - (.*?)(?=(?:^|[ )])step:\d+ - |\Z)", text, re.S):
        row = {"i": int(st)}   # the step index ("step" is the timing segment)
        for k in TIMING + METRICS:
            m = re.search(re.escape(k) + r":([-0-9.e]+)", body)
            row[k.split("/")[-1] if k.startswith("timing") else k] = float(m.group(1)) if m else None
        rows.append(row)
    return rows


def report(d):
    out = {"lp": {}, "variants": {}}
    for k in ("student", "teacher"):
        try:
            out["lp"][k] = json.load(open(os.path.join(d, f"lp_{k}.json")))
        except (OSError, ValueError):
            out["lp"][k] = None
    for v in VARIANTS:
        r = {}
        try:
            r["rc"] = int(open(os.path.join(d, f"{v}.rc")).read().strip())
        except (OSError, ValueError):
            r["rc"] = None
        try:
            vals = [float(x.split(",")[1]) for x in open(os.path.join(d, f"{v}.mem.csv")) if "," in x]
            r["mem_peak_mib"] = max(vals) if vals else None
        except (OSError, ValueError, IndexError):
            r["mem_peak_mib"] = None
        try:
            text = open(os.path.join(d, v, "train.log"), errors="ignore").read()
        except OSError:
            text = ""
        r["steps"] = steps_of(text)
        r["oom"] = "CUDA out of memory" in text or "OutOfMemoryError" in text
        late = [s for s in r["steps"] if s["i"] >= 2]   # step 1 carries the warm-up (graph capture, first allocation)
        r["mean_s23"] = {}
        for k in TIMING:
            n = k.split("/")[1]; vals = [s[n] for s in late if s.get(n) is not None]
            r["mean_s23"][n] = round(sum(vals) / len(vals), 2) if vals else None
        out["variants"][v] = r
    base = out["variants"]["base"]["mean_s23"]
    for v in VARIANTS:
        m = out["variants"][v]["mean_s23"]
        out["variants"][v]["ratio_to_base"] = {k: (round(base[k] / m[k], 3) if base.get(k) and m.get(k) else None) for k in m}
    lp_ok = all(out["lp"].get(k) and out["lp"][k].get("pass") for k in ("student", "teacher"))
    out["fused_pass"] = bool(lp_ok and out["variants"]["fused"]["rc"] == 0)
    out["gc0_pass"] = bool(out["variants"]["gc0"]["rc"] == 0 and not out["variants"]["gc0"]["oom"])
    out["fused_gc0_pass"] = bool(out["fused_pass"] and out["variants"]["fused_gc0"]["rc"] == 0 and not out["variants"]["fused_gc0"]["oom"])
    json.dump(out, open(os.path.join(d, "report.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "lp":
        r = lp(sys.argv[2], sys.argv[3], int(sys.argv[4])); print(json.dumps(r)); sys.exit(0 if r["pass"] else 1)
    elif c == "report":
        print(json.dumps(report(sys.argv[2])))
    else:
        sys.exit(__doc__)
