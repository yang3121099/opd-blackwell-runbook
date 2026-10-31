"""[v2.5.9z106] env_probe.py <out.json>: what this machine has, for the control command `env-probe` (connect-first, Fable 11-01;
ops-lead's A1 11-01 16:11 UTC). Standard library only (runs on the control venv); nvidia-smi, df, curl, git, aria2c, nvcc when present.
No path, host name, address or proxy value is written: disks by role (opd_root / ckpt_root / largest local), the old-base
candidates (node.env OLD_BASE_CANDS, ':'-separated) as counts, the proxy as set / not set, reachability as HTTP codes.
Prints one summary line."""
import json
import os
import shutil
import subprocess
import sys
import time


def sh(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def free_gb(path):
    try:
        st = os.statvfs(path)
        return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        return None


def http(url, proxy):
    env = dict(os.environ)
    if not proxy:
        for k in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
            env.pop(k, None)
    try:
        r = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "20", url],
                           capture_output=True, text=True, timeout=30, env=env)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def probe():
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    q = sh(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in q.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 5:
            gpus.append({"index": int(f[0]) if f[0].isdigit() else f[0], "name": f[1], "memory_total_mib": int(float(f[2])) if f[2].replace(".", "", 1).isdigit() else None,
                         "compute_cap": f[3]})
            out["driver"] = f[4]
    out["gpus"], out["gpu_count"] = gpus, len(gpus)
    head = sh(["nvidia-smi"])
    out["cuda_driver"] = head.split("CUDA Version:")[1].split()[0] if "CUDA Version:" in head else None
    busy = sh(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip()
    out["gpu_processes"] = len(busy.splitlines()) if busy else 0
    root = os.environ.get("OPD_ROOT", "")
    disks = {"opd_root": free_gb(root) if root else None, "ckpt_root": free_gb(os.environ.get("CKPT_ROOT", "")) if os.environ.get("CKPT_ROOT") else None}
    best = None
    for line in sh(["df", "-P", "-T"]).splitlines()[1:]:
        f = line.split()
        if len(f) >= 7 and f[1] in ("ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "overlay"):
            gb = round(int(f[4]) * 1024 / 1e9, 1)
            best = gb if best is None or gb > best else best
    disks["largest_local"] = best
    out["disk_free_gb"] = disks
    cands = [c for c in os.environ.get("OLD_BASE_CANDS", "").split(":") if c]
    out["old_base"] = {"candidates": len(cands), "found": sum(1 for c in cands if os.path.isdir(c)),
                       "with_venv": sum(1 for c in cands if os.path.isfile(os.path.join(c, "venv", "bin", "activate")))}
    proxy = any(os.environ.get(k) for k in ("https_proxy", "HTTPS_PROXY"))
    out["proxy_set"] = proxy
    out["no_proxy_set"] = bool(os.environ.get("no_proxy") and os.environ.get("NO_PROXY"))
    out["reach"] = {"github_direct": http("https://github.com/", False), "hf_direct": http("https://huggingface.co/api/models/Qwen/Qwen3-0.6B", False)}
    if proxy:
        out["reach"].update(github_proxy=http("https://github.com/", True), hf_proxy=http("https://huggingface.co/api/models/Qwen/Qwen3-0.6B", True))
    out["python"] = sys.version.split()[0]
    out["tools"] = {t: bool(shutil.which(t)) for t in ("git", "curl", "aria2c", "nvcc", "tmux", "flock", "setsid")}
    nv = sh(["nvcc", "--version"]) if shutil.which("nvcc") else ""
    out["nvcc"] = nv.split("release ")[1].split(",")[0] if "release " in nv else None
    out["cpus"] = os.cpu_count()
    try:
        out["mem_total_gb"] = round(int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) * 1024 / 1e9, 1)
    except (OSError, StopIteration, ValueError):
        out["mem_total_gb"] = None
    out["venvs"] = {v: os.path.isfile(os.path.join(root, v, "bin", "activate")) for v in ("venv-ctl", "venv", "venv-fa", "venv-vl", "venv-ifeval")} if root else {}
    return out


if __name__ == "__main__":
    o = probe()
    with open(sys.argv[1] + ".tmp", "w") as f:
        json.dump(o, f)
    os.replace(sys.argv[1] + ".tmp", sys.argv[1])
    names = sorted({g["name"] for g in o["gpus"]})
    print(f"{o['gpu_count']} GPU(s) {', '.join(names) or 'none'}; driver {o.get('driver')}, CUDA {o['cuda_driver']}; "
          f"{o['gpu_processes']} GPU process(es); free {o['disk_free_gb']}; proxy {'set' if o['proxy_set'] else 'not set'}; "
          f"reach {o['reach']}")
