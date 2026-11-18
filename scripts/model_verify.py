"""[v3.11] model_verify.py <directory> <repo id>: is this directory the published model, byte for byte? (Fable 11-18; Runming: "find
the base models on the shared disks, do not download them".) A directory named Qwen3-1.7B on someone's disk may be a fine-tuned
or truncated copy; a registered experiment must start from the published weights. The Hub's file list (siblings with blobs:
size, and sha256 for LFS files) is read from the mirror or the Hub - a few kB - and every weight file (*.safetensors, *.bin) is
hashed locally; the other listed files are compared by size.
  exit 0  every listed weight file is there with the published sha256 (one line: VERIFIED ...)
  exit 1  a file is missing or differs (MISMATCH ...)
  exit 2  the list could not be read (UNKNOWN ...): the caller decides (a copy from an install of ours was downloaded by this
          runbook and is taken as before; a copy from anywhere else is not used without the check)
stdlib only. MODEL_VERIFY_LIST=<json file> gives the list for tests."""
import hashlib
import json
import os
import sys
import urllib.request

ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")


def listing(repo):
    if os.environ.get("MODEL_VERIFY_LIST"):
        return json.load(open(os.environ["MODEL_VERIFY_LIST"]))
    err = None
    for ep in ENDPOINTS:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{ep}/api/models/{repo}?blobs=true", headers={"User-Agent": "opd-runbook"}), timeout=40) as r:
                d = json.load(r)
            if d.get("siblings"):
                return d
        except Exception as e:   # the next endpoint
            err = e
    raise OSError(f"no file list for {repo}: {type(err).__name__}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify(d, repo):
    try:
        sib = listing(repo)["siblings"]
    except (OSError, ValueError, KeyError) as e:
        return 2, f"UNKNOWN {repo}: {e}"
    weights = [s for s in sib if s["rfilename"].endswith((".safetensors", ".bin")) and (s.get("lfs") or {}).get("sha256")]
    if not weights:
        return 2, f"UNKNOWN {repo}: the list names no weight file with a sha256"
    for s in weights:
        p = os.path.join(d, s["rfilename"])
        if not os.path.isfile(p):
            return 1, f"MISMATCH {repo}: {s['rfilename']} is missing"
        if os.path.getsize(p) != (s["lfs"].get("size") or s.get("size")):
            return 1, f"MISMATCH {repo}: {s['rfilename']} has {os.path.getsize(p)} bytes, published {s['lfs'].get('size') or s.get('size')}"
        if sha256(p) != s["lfs"]["sha256"]:
            return 1, f"MISMATCH {repo}: {s['rfilename']} differs from the published file (sha256)"
    for s in sib:
        n = s["rfilename"]
        if n in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json") and s.get("size") is not None:
            p = os.path.join(d, n)
            if not os.path.isfile(p) or os.path.getsize(p) != s["size"]:
                return 1, f"MISMATCH {repo}: {n} is missing or has another size"
    return 0, f"VERIFIED {repo}: {len(weights)} weight file(s) match the published sha256"


if __name__ == "__main__":
    rc, msg = verify(sys.argv[1], sys.argv[2])
    print(msg)
    sys.exit(rc)
