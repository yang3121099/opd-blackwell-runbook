"""Copy an HF checkpoint folder with every floating-point tensor cast to bfloat16 (exact bf16 rounding of the fp32
weights; config torch_dtype set to bfloat16). Usage: python to_bf16.py <src dir> <dst dir>"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1:3]; os.makedirs(dst)
for f in os.listdir(src):
    p = os.path.join(src, f)
    if f.endswith(".safetensors"):
        t = load_file(p)
        save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in t.items()}, os.path.join(dst, f), metadata={"format": "pt"})
    elif os.path.isfile(p):
        shutil.copy2(p, os.path.join(dst, f))
c = json.load(open(os.path.join(dst, "config.json"))); c["torch_dtype"] = "bfloat16"
json.dump(c, open(os.path.join(dst, "config.json"), "w"), indent=2)
