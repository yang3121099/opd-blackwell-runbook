"""Weight arithmetic for E02/E03 (CPU, bf16 safetensors).

  interp  : W = lam*A + (1-lam)*B                 (E03 model interpolation, A=Thinking, B=Instruct)
  scale   : W = Base + c*(FT - Base)               (E02/E03 delta scaling)
  graft   : W = Target + (FT - Base)               (E02 cross-variant grafting, Shadow-FT style)
  randdir : W = Target + per-tensor random dir with ||.|| = ||FT - Base||   (E02 control)
  sigma   : Shadow-FT relative gap between two checkpoints (no output model)

Tokenizer/config files are copied from the first model argument (or --cfg_from).
Examples:
  python wops.py interp  --a Qwen/Qwen3-4B-Thinking-2507 --b Qwen/Qwen3-4B-Instruct-2507 --lam 0.7 --out $OPD_ROOT/tmpm/mi_0.7
  python wops.py scale   --base Qwen/Qwen3-1.7B --ft Tuwhy/Qwen3-1.7B-OPSA --c 1.5 --out $OPD_ROOT/tmpm/opsa17_c1.5
  python wops.py graft   --target Qwen/Qwen3-4B-Thinking-2507 --base Qwen/Qwen3-4B --ft Tuwhy/Qwen3-4B-OPSA --out ...
  python wops.py sigma   --a Qwen/Qwen3-4B --b Qwen/Qwen3-4B-Thinking-2507
"""
import argparse
import glob
import json
import os
import shutil

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file


def local(repo):
    # Only the files needed for weights/tokenizer: hf-mirror redirects README/LICENSE/.gitattributes to
    # huggingface.co (308), which fails on machines without direct HF access (reported by the 2xH100 bring-up).
    return repo if os.path.isdir(repo) else snapshot_download(
        repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.jinja", "tokenizer*"])


def load(repo):
    d = local(repo)
    sd = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        sd.update(load_file(f))
    return sd, d


def save(sd, out, cfg_dir):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(cfg_dir):
        if not f.endswith(".safetensors") and not f.endswith(".index.json") and os.path.isfile(os.path.join(cfg_dir, f)):
            shutil.copy(os.path.join(cfg_dir, f), out)
    # shard at ~4GB
    shard, size, idx, n = {}, 0, {}, 0
    keys = sorted(sd)
    shards = []
    for k in keys:
        t = sd[k].contiguous()
        if size + t.numel() * t.element_size() > 4e9 and shard:
            shards.append(shard); shard, size = {}, 0
        shard[k] = t; size += t.numel() * t.element_size()
    shards.append(shard)
    for i, sh in enumerate(shards):
        name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        save_file(sh, os.path.join(out, name), metadata={"format": "pt"})
        for k in sh:
            idx[k] = name
    json.dump({"metadata": {}, "weight_map": idx}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
    print("saved", out, len(shards), "shards")


def common(a, b):
    ka, kb = set(a), set(b)
    if ka != kb:
        print("WARNING key mismatch:", len(ka - kb), len(kb - ka))
    return sorted(ka & kb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("op", choices=["interp", "scale", "graft", "randdir", "sigma"])
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--lam", type=float)
    ap.add_argument("--base"); ap.add_argument("--ft"); ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--target"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.op == "interp":
        A, da = load(args.a); B, _ = load(args.b)
        sd = {k: (args.lam * A[k].float() + (1 - args.lam) * B[k].float()).to(A[k].dtype) for k in common(A, B)}
        save(sd, args.out, da)
    elif args.op == "scale":
        Bs, db = load(args.base); F, _ = load(args.ft)
        sd = {k: (Bs[k].float() + args.c * (F[k].float() - Bs[k].float())).to(Bs[k].dtype) for k in common(Bs, F)}
        save(sd, args.out, db)
    elif args.op == "graft":
        T, dt = load(args.target); Bs, _ = load(args.base); F, _ = load(args.ft)
        sd = {k: (T[k].float() + (F[k].float() - Bs[k].float())).to(T[k].dtype) for k in common(T, F)}
        save(sd, args.out, dt)
    elif args.op == "randdir":
        T, dt = load(args.target); Bs, _ = load(args.base); F, _ = load(args.ft)
        g = torch.Generator().manual_seed(args.seed)
        sd = {}
        for k in common(T, F):
            d = F[k].float() - Bs[k].float()
            r = torch.randn(d.shape, generator=g)
            r = r * (d.norm() / (r.norm() + 1e-12))
            sd[k] = (T[k].float() + r).to(T[k].dtype)
        save(sd, args.out, dt)
    elif args.op == "sigma":
        A, _ = load(args.a); B, _ = load(args.b)
        num = den = 0.0
        for k in common(A, B):
            a, b = A[k].float(), B[k].float()
            num += (a - b).abs().sum().item(); den += a.abs().sum().item() + b.abs().sum().item()
        print(json.dumps({"a": args.a, "b": args.b, "sigma": round(num / den, 5)}))


if __name__ == "__main__":
    main()
