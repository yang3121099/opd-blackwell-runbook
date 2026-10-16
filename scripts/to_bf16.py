"""Copy an HF checkpoint folder with every floating-point tensor cast to bfloat16 (exact bf16 rounding of the fp32
weights; config torch_dtype set to bfloat16). Sharded folders (model-0000k-of-0000n.safetensors, e.g. every 1.7B
checkpoint) keep their shards; the index's metadata.total_size is set to the bf16 size.
Usage: python to_bf16.py <src dir> <dst dir>"""
import json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1:3]; os.makedirs(dst)
total = 0
for f in sorted(os.listdir(src)):
    p = os.path.join(src, f)
    if f.endswith(".safetensors"):
        t = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in load_file(p).items()}
        total += sum(v.numel() * v.element_size() for v in t.values())
        save_file(t, os.path.join(dst, f), metadata={"format": "pt"})
    elif os.path.isfile(p):
        shutil.copy2(p, os.path.join(dst, f))
c = json.load(open(os.path.join(dst, "config.json"))); c["torch_dtype"] = "bfloat16"
json.dump(c, open(os.path.join(dst, "config.json"), "w"), indent=2)
ix = os.path.join(dst, "model.safetensors.index.json")
if os.path.exists(ix):
    j = json.load(open(ix)); j.setdefault("metadata", {})["total_size"] = total
    json.dump(j, open(ix, "w"), indent=2)
