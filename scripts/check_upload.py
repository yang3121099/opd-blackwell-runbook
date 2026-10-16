"""Check a bf16 upload folder against a reference manifest (verify.sh step 4; reference/upload_manifest_*.json):
every required file present, weight files within the size tolerance, every floating-point tensor bf16, the tensor
count, the index's total_size equal to the shards' tensor bytes, and config torch_dtype bfloat16. Extra files (for
example chat_template.jinja from a newer transformers) are allowed.
Usage: python check_upload.py <folder> <manifest.json>      exit 0 = pass"""
import json, math, os, sys

from safetensors import safe_open

BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
d, man = sys.argv[1], json.load(open(sys.argv[2])); bad = []
if not os.path.isdir(d):
    sys.exit(f"UPLOAD_CHECK: no folder {d}")
for f, size in man["required"].items():
    p = os.path.join(d, f)
    if not os.path.exists(p):
        bad.append(f"missing {f}")
    elif size and abs(os.path.getsize(p) - size) > man["size_tolerance"] * size:
        bad.append(f"{f}: {os.path.getsize(p)} B, expected about {size} B")
n, tot, dtypes = 0, 0, set()
for f in sorted(x for x in os.listdir(d) if x.endswith(".safetensors")):
    with safe_open(os.path.join(d, f), "pt") as g:
        for k in g.keys():
            s = g.get_slice(k); dt = s.get_dtype(); n += 1
            tot += math.prod(s.get_shape()) * BYTES[dt]
            if dt in ("BF16", "F16", "F32", "F64"):
                dtypes.add(dt)
if dtypes != {"BF16"}:
    bad.append(f"floating dtypes {sorted(dtypes)}, expected only BF16")
if n != man["n_tensors"]:
    bad.append(f"{n} tensors, expected {man['n_tensors']}")
if tot != man["tensor_bytes"]:
    bad.append(f"{tot} tensor bytes, expected {man['tensor_bytes']}")
ix = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(ix):
    ts = json.load(open(ix)).get("metadata", {}).get("total_size")
    if ts != tot:
        bad.append(f"index total_size {ts} != tensor bytes {tot}")
if json.load(open(os.path.join(d, "config.json"))).get("torch_dtype") != "bfloat16":
    bad.append("config torch_dtype is not bfloat16")
print(f"UPLOAD_CHECK_DETAIL {n} tensors, {tot} tensor bytes, weight files: "
      + ", ".join(f"{f} {os.path.getsize(os.path.join(d, f))} B" for f in sorted(os.listdir(d)) if f.endswith(".safetensors")))
for b in bad:
    print("UPLOAD_CHECK_BAD", b)
sys.exit(1 if bad else 0)
