"""[v3.8] sft_warm.py: plain supervised fine-tuning of a base model on one GPU (Fable 11-18; Runming: "a real training, e.g. SFT of the
base Qwen3-8B, that starts fast"). torch + transformers only (peft when it is there): it needs neither vLLM nor verl, so it can
run while the rest of the stack is still being installed. This is a real optimisation run - forward, loss, backward, optimizer
steps, a saved adapter - but NOT a registered experiment: nothing reads its result.
  data   every *.parquet under --data (text columns, chat-style lists of {"content"}), else generated arithmetic exercises
  model  --mode full (default; Runming 11-18: "full use of the GPU, not LoRA"): every parameter is trained, master weights in
         fp32, forward and backward under bf16 autocast. The optimizer is AdamW when weights + gradients + its two states fit the
         card with room for the batch, else SGD with momentum (one state): an 8B model in fp32 is 33 GB, so AdamW would need
         131 GB before any activation. --mode lora: rank-64 adapters on the bf16 base (light: for a card that is shared).
  loop   cosine schedule, batch --bsz of at most --maxlen tokens (one sample less after an out-of-memory), for --hours; a line
         every 10 steps; lora saves its adapter every 500 steps and at the end, full saves nothing (33 GB a copy)
Usage: CUDA_VISIBLE_DEVICES=<g> python sft_warm.py --model <dir> --out <dir> [--data <dir>...] [--hours 6] [--seed 0]"""
import argparse
import glob
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def texts_from_parquet(dirs, limit=20000):
    out = []
    try:
        import pandas as pd
    except ImportError:
        return out
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.parquet"))):
            try:
                df = pd.read_parquet(f)
            except Exception:
                continue
            for _, row in df.head(limit).iterrows():
                parts = []
                for v in row.values:
                    if isinstance(v, str) and len(v) > 20:
                        parts.append(v)
                    elif hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)):
                        parts += [x["content"] for x in v if isinstance(x, dict) and isinstance(x.get("content"), str)]
                    elif isinstance(v, dict):
                        parts += [x for x in v.values() if isinstance(x, str) and len(x) > 20]
                if parts:
                    out.append("\n\n".join(parts))
                if len(out) >= limit:
                    return out
    return out


def generated(n, rng):
    out = []
    for _ in range(n):
        a, b, c = rng.randint(10, 9999), rng.randint(10, 9999), rng.randint(2, 99)
        out.append(f"Problem: Compute ({a} + {b}) * {c} - {b}.\nSolution: First {a} + {b} = {a + b}. Then {a + b} * {c} = {(a + b) * c}. "
                   f"Finally {(a + b) * c} - {b} = {(a + b) * c - b}.\nAnswer: {(a + b) * c - b}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--hours", type=float, default=6.0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bsz", type=int, default=0); ap.add_argument("--maxlen", type=int, default=768); ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--mode", choices=["full", "lora"], default=os.environ.get("SFT_WARM_MODE", "full"))
    a = ap.parse_args()
    rng = random.Random(a.seed); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    data = texts_from_parquet(a.data)
    src = "parquet" if len(data) >= 200 else "generated"
    if src == "generated":
        data = generated(20000, rng)
    rng.shuffle(data)
    print(json.dumps({"event": "data", "source": src, "samples": len(data)}), flush=True)
    mode = a.mode
    if mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            mode = "full"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16 if mode == "lora" else torch.float32).cuda()
    model.config.use_cache = False
    if mode == "lora":
        model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    n = sum(p.numel() for p in params)
    free = torch.cuda.mem_get_info()[0] / 1e9
    if mode == "lora":
        opt_name, opt, lr, bsz = "adamw", torch.optim.AdamW(params, lr=a.lr or 1e-4, weight_decay=0.0), a.lr or 1e-4, a.bsz or 8
    elif n * 4 * 3 / 1e9 + 25 < free:      # gradients + AdamW's two states still to come, 25 GB for the batch
        opt_name, lr = "adamw", a.lr or 1e-5
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0); bsz = a.bsz or 8
    else:                                   # gradients + one momentum buffer
        opt_name, lr = "sgd_momentum", a.lr or 1e-3
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True); bsz = a.bsz or max(1, min(8, int((free - n * 4 * 2 / 1e9 - 6) / 10)))
    a.bsz = bsz
    total = 200000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    print(json.dumps({"event": "model", "mode": mode, "optimizer": opt_name, "lr": lr, "bsz": a.bsz, "trainable": n, "free_gb_after_load": round(free, 1),
                      "gpu": torch.cuda.get_device_name(0)}), flush=True)
    model.train(); t0 = time.time(); step = 0; i = 0; ema = None
    while time.time() - t0 < a.hours * 3600:
        batch = data[i:i + a.bsz]; i += a.bsz
        if len(batch) < a.bsz:
            rng.shuffle(data); i = 0; continue
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=a.maxlen).to("cuda")
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**enc, labels=labels).loss
            loss.backward()
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); del enc, labels; torch.cuda.empty_cache()
            a.bsz = max(1, a.bsz - 1); print(json.dumps({"event": "oom", "bsz_now": a.bsz}), flush=True); continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if step % 10 == 0:
            print(json.dumps({"step": step, "loss": round(loss.item(), 4), "loss_ema": round(ema, 4), "lr": sched.get_last_lr()[0],
                              "tok_per_s": round(step * a.bsz * enc["input_ids"].shape[1] / (time.time() - t0)), "mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}), flush=True)
        if step % 500 == 0 and mode == "lora":
            model.save_pretrained(os.path.join(a.out, "adapter"))
    if mode == "lora":
        model.save_pretrained(os.path.join(a.out, "adapter"))
    print(json.dumps({"event": "end", "steps": step, "hours": round((time.time() - t0) / 3600, 2), "loss_ema": ema}), flush=True)


if __name__ == "__main__":
    main()
