"""Helpers of the FA-SPEED probes (runbook v2.5.7c; fa_speed.sh runs them; Runming 2026-10-18 08:5x UTC: the
flash-attn speed-up first, more tests, on line soon).
  python fa_speed.py lp <model> <rollouts.jsonl> <n>  -> JSON: per-token log-probs of n rollout sequences (prompt +
        response, response tokens compared, batch 1, no padding) with attn_implementation sdpa and flash_attention_2,
        bf16 like the actor; max / mean / 99.9th-percentile |difference|; pass = mean < 2e-3 and p99.9 < 5e-2 (class B
        of the speed list: another kernel, other accumulation order; the reading tables note it)
  python fa_speed.py report <dir> <variant> [<variant> ...] -> <dir>/report.json: the log-prob checks, and per smoke
        variant exit code, out of memory, peak GPU memory, steps 2+ mean per timing segment and its ratio to the first
        variant (the baseline), response lengths
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_probe import TIMING, steps_of  # noqa: E402


def lp(model_path, rollouts, n):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    seqs = []
    for line in open(rollouts):
        r = json.loads(line)
        p, o = r.get("input", r.get("prompt")), r.get("output", r.get("response"))
        if p and o:
            seqs.append((tok(p, add_special_tokens=False)["input_ids"], tok(o, add_special_tokens=False)["input_ids"][:4096]))
        if len(seqs) >= n:
            break
    out = {}
    for impl in ("sdpa", "flash_attention_2"):
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation=impl).cuda().eval()
        res = []
        with torch.no_grad():
            for p, o in seqs:
                ids = torch.tensor([p + o], device="cuda"); L = len(o)
                logits = model(input_ids=ids, use_cache=False).logits[:, -L - 1:-1, :].float()
                res.append(torch.log_softmax(logits, -1).gather(-1, ids[:, -L:, None])[..., 0].cpu())
        out[impl] = res
        del model; torch.cuda.empty_cache()
    d = torch.cat([(a - b).abs().flatten() for a, b in zip(out["sdpa"], out["flash_attention_2"])])
    s = torch.sort(d).values
    p999 = s[min(len(s) - 1, int(0.999 * len(s)))].item()
    mean = d.mean().item()
    return {"model": os.path.basename(model_path.rstrip("/")), "sequences": len(seqs), "tokens": d.numel(),
            "max_abs_diff": d.max().item(), "mean_abs_diff": mean, "p999_abs_diff": p999, "pass": bool(mean < 2e-3 and p999 < 5e-2)}


def last_json(path):
    try:
        for line in reversed(open(path).read().splitlines()):
            try:
                return json.loads(line)
            except ValueError:
                continue
    except OSError:
        pass
    return None


def report(d, variants):
    out = {"lp": {k: last_json(os.path.join(d, f"lp_{k}.json")) for k in ("student", "teacher")}, "variants": {}}
    for v in variants:
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
        st = steps_of(text); r["steps"] = len(st)
        r["oom"] = "CUDA out of memory" in text or "OutOfMemoryError" in text
        late = [s for s in st if s["i"] >= 2]
        r["mean_s2plus"] = {}
        for k in TIMING:
            nme = k.split("/")[1]; vals = [s[nme] for s in late if s.get(nme) is not None]
            r["mean_s2plus"][nme] = round(sum(vals) / len(vals), 2) if vals else None
        r["response_length"] = [s.get("response_length/mean") for s in st]
        out["variants"][v] = r
    base = out["variants"][variants[0]]["mean_s2plus"] if variants else {}
    for v in variants:
        m = out["variants"][v]["mean_s2plus"]
        out["variants"][v]["speedup_vs_" + variants[0]] = {k: (round(base[k] / m[k], 3) if base.get(k) and m.get(k) else None) for k in m}
    lps = [x for x in out["lp"].values() if x is not None]
    out["lp_pass"] = bool(lps) and all(x.get("pass") for x in lps)
    out["smokes_ok"] = all(out["variants"][v]["rc"] == 0 and not out["variants"][v]["oom"] for v in variants)
    json.dump(out, open(os.path.join(d, "report.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "lp":
        r = lp(sys.argv[2], sys.argv[3], int(sys.argv[4])); print(json.dumps(r)); sys.exit(0 if r["pass"] else 1)
    elif c == "report":
        print(json.dumps(report(sys.argv[2], sys.argv[3:])))
    else:
        sys.exit(__doc__)
