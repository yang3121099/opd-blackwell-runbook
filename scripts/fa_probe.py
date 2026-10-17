"""Helpers of the FA-PROBE task (runbook v2.5.3; fa_probe.sh runs the steps).
  python fa_probe.py wheel <TRUE|FALSE>        -> "<url> <sha256>" of the pinned flash-attn wheel for this torch ABI
  python fa_probe.py versions                  -> {"torch": ..., "vllm": ..., "python": ...} as JSON
  python fa_probe.py varlen                    -> exit 0 if flash_attn_varlen_func runs forward and backward on the GPU
                                                  and matches PyTorch SDPA per sequence (prints the max difference)
  python fa_probe.py report <dir>              -> <dir>/report.json from probe.json, mem.csv and smoke/train.log
The wheel is pinned to flash-attn v2.8.3.post2 (Dao-AILab release, 2026-10-13), torch 2.8, CUDA 12, Python 3.10, with the
sha256 digests of the release page; any other stack is refused (the probe then reports which one it found)."""
import json
import os
import re
import sys

REL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/"
WHEELS = {  # cxx11 ABI of torch -> (file name, sha256 from the release page)
    "TRUE": ("flash_attn-2.8.3.post2+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
             "203608ec486ec866a64b5debe1ad554b73d70632ee0f939c0b2e017057ecf23f"),
    "FALSE": ("flash_attn-2.8.3.post2+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl",
              "407e2c761e7546a8fe131d43d50cbc27f88bbeb603691ba6a8240f0b6ac5337b"),
}
TIMING = ("timing_s/gen", "timing_s/old_log_prob", "timing_s/ref", "timing_s/update_actor", "timing_s/step")


def wheel(abi):
    name, sha = WHEELS[abi.upper()]
    return REL + name.replace("+", "%2B"), sha, name


def versions():
    import platform
    out = {"python": platform.python_version()}
    try:
        import torch
        out.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cxx11abi=str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).upper())
    except Exception as e:
        out["torch"] = f"error {type(e).__name__}"
    try:
        import vllm
        out["vllm"] = vllm.__version__
    except Exception as e:
        out["vllm"] = f"error {type(e).__name__}"
    return out


def varlen():
    import torch
    import torch.nn.functional as F
    from flash_attn import flash_attn_varlen_func
    torch.manual_seed(0)
    lens = [37, 512, 1300]; H, D = 16, 128
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    tot = sum(lens)
    q, k, v = (torch.randn(tot, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    o = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens), causal=True)
    o.float().sum().backward()
    diff = 0.0
    for i, n in enumerate(lens):
        a, b = cu[i].item(), cu[i + 1].item()
        ref = F.scaled_dot_product_attention(q[a:b].transpose(0, 1)[None], k[a:b].transpose(0, 1)[None],
                                             v[a:b].transpose(0, 1)[None], is_causal=True)[0].transpose(0, 1)
        diff = max(diff, (ref.float() - o[a:b].float()).abs().max().item())
    ok = diff < 2e-2 and q.grad is not None and torch.isfinite(q.grad).all().item()
    print(json.dumps({"varlen_max_abs_diff": round(diff, 5), "grad_finite": bool(q.grad is not None and torch.isfinite(q.grad).all().item()),
                      "device": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())}))
    return ok


def report(d):
    pr = json.load(open(os.path.join(d, "probe.json"))) if os.path.exists(os.path.join(d, "probe.json")) else {}
    peak = None
    try:
        vals = [float(x.split(",")[1]) for x in open(os.path.join(d, "mem.csv")) if "," in x]
        peak = max(vals) if vals else None
    except OSError:
        pass
    steps = []
    try:
        text = open(os.path.join(d, "smoke", "train.log"), errors="ignore").read()
        for st, body in re.findall(r"(?:^|[ )])step:(\d+) - (.*?)(?=(?:^|[ )])step:\d+ - |\Z)", text, re.S):
            row = {"step": int(st)}
            for k in TIMING:
                m = re.search(re.escape(k) + r":([-0-9.e]+)", body)
                row[k.split("/")[1]] = float(m.group(1)) if m else None
            steps.append(row)
    except OSError:
        pass
    pr.update(mem_peak_mib=peak, steps=steps, smoke_steps=len(steps))
    json.dump(pr, open(os.path.join(d, "report.json"), "w"), indent=1)
    return pr


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "wheel":
        url, sha, name = wheel(sys.argv[2]); print(url, sha, name)
    elif c == "versions":
        print(json.dumps(versions()))
    elif c == "varlen":
        sys.exit(0 if varlen() else 1)
    elif c == "report":
        print(json.dumps(report(sys.argv[2])))
    else:
        print(__doc__); sys.exit(2)
