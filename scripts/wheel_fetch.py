"""Pinned wheels from PyPI into a local directory for an offline install (runbook [v2.5.9y1], BUILD-VENV-VL; ops-lead's A1).

    python wheel_fetch.py <out dir> <name==version> ...      e.g. vllm==0.11.0 transformers==4.57.1

For each pin the wheel is picked from PyPI's JSON (https://pypi.org/pypi/<name>/<version>/json): a CPython 3.10 / abi3 /
py3-none-any build for Linux x86_64 (or any platform), with its size and the sha256 PyPI publishes. [v2.5.9z3] The probe of the node's
proxy is the smallest wheel of at least PROBE_MIN_MB (4; else the first), fetched first over ONE connection and timed (probe_MBps):
10-21 vllm, listed first, was its own 438 MB probe at 0.24 MB/s. A later wheel larger than PAR_MIN_MB (32) goes
through par_fetch.download over FETCH_CONNS connections when that probe is under PAR_BELOW_MBPS (1.0), else over one connection
too. Every file is checked against PyPI's sha256 (a mismatch removes nothing but the file just written and fails); a file already
there with the right sha256 is kept. <out dir>/manifest.json: per wheel name, version, file, size, sha256, url host, seconds, MB/s,
connections; and the probe. [v2.5.9z34] WHEEL_PAR_FIRST=1: the first wheel, when larger than PAR_MIN_MB, goes straight through
par_fetch (no probe): a lone large pin was its own one-connection probe (VENV-VL-NVRTC 10-22: 89.6 MB at 0.069 MB/s, a read timeout). Exit 0 when every pin is in, 1 otherwise. No tunnel, no proxy change: urllib with the environment's
proxy settings, as fetch_model.py. [v2.5.9z16] WHEEL_SDIST=<name>[,<name>]: those pins, when PyPI has no such wheel for them, are
fetched as their .tar.gz sdist (BUILD-VENV-IFEVAL: langdetect 1.0.9 publishes only an sdist).
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAR_MIN = float(os.environ.get("PAR_MIN_MB", "32")) * 1e6
PAR_BELOW = float(os.environ.get("PAR_BELOW_MBPS", "1.0"))
TAG_OK = re.compile(r"-(cp310-cp310|cp3\d-abi3|cp38-abi3|py3-none|py2\.py3-none)-")
PLAT_OK = re.compile(r"(manylinux[0-9_]*_x86_64|linux_x86_64|any)\.whl$")


def pick(urls):
    """The wheel to use from PyPI's 'urls' list, or None."""
    whs = [u for u in urls if u.get("packagetype") == "bdist_wheel" and TAG_OK.search(u["filename"]) and PLAT_OK.search(u["filename"])]
    whs.sort(key=lambda u: (0 if "cp310-cp310" in u["filename"] else 1, u["filename"]))
    return whs[0] if whs else None


def pick_sdist(urls):
    """[v2.5.9z16] the .tar.gz sdist from PyPI's 'urls' list, or None."""
    return next((u for u in urls if u.get("packagetype") == "sdist" and u["filename"].endswith(".tar.gz")), None)


def meta(name, version, opener=urllib.request.urlopen):
    d = json.load(opener(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60))
    w = pick(d.get("urls", []))
    # [v2.5.9z16] a pin named in WHEEL_SDIST (comma-separated: pure-Python packages that publish no wheel, langdetect) may be its
    # sdist, sha256-checked as a wheel is; the caller builds it offline (--no-build-isolation)
    if not w and name in os.environ.get("WHEEL_SDIST", "").split(","):
        w = pick_sdist(d.get("urls", []))
    if not w:
        raise ValueError(f"no Linux x86_64 / py3 wheel for {name}=={version}")
    return {"name": name, "version": version, "file": w["filename"], "size": int(w["size"]), "sha256": w["digests"]["sha256"],
            "url": w["url"]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def single(url, dst, timeout=120, opener=urllib.request.urlopen):
    tmp = dst + ".tmp"
    with opener(url, timeout=timeout) as r, open(tmp, "wb") as f:
        for b in iter(lambda: r.read(1 << 20), b""):
            f.write(b)
    os.replace(tmp, dst)


def conns():
    try:
        import fetch_model
        return int(fetch_model.conns_now())
    except Exception:
        return int(os.environ.get("FETCH_CONNS", "32"))


def order(metas):
    """[v2.5.9z3] the probe (smallest wheel >= PROBE_MIN_MB, else the first) first, then the others smallest first."""
    pmin = float(os.environ.get("PROBE_MIN_MB", "4")) * 1e6
    big = [m for m in metas if m["size"] >= pmin]
    probe = min(big, key=lambda m: m["size"]) if big else metas[0]
    return [probe] + sorted((m for m in metas if m is not probe), key=lambda m: m["size"])


def fetch_all(out, pins, single_fn=single, par_fn=None, meta_fn=meta, conns_fn=conns, clock=time.time):
    os.makedirs(out, exist_ok=True)
    res = {"wheels": [], "probe_MBps": None, "ok": True}
    metas = order([meta_fn(*pin.split("==", 1)) for pin in pins])
    for i, m in enumerate(metas):
        dst = os.path.join(out, m["file"])
        row = dict(m, host=re.sub(r"^https?://([^/]+)/.*$", r"\1", m["url"]), conns=1, seconds=0.0, MBps=None, kept=False)
        del row["url"]
        if os.path.exists(dst) and sha256(dst) == m["sha256"]:
            row["kept"] = True
        else:
            use_par = i > 0 and m["size"] > PAR_MIN and res["probe_MBps"] is not None and res["probe_MBps"] < PAR_BELOW
            if i == 0 and os.environ.get("WHEEL_PAR_FIRST") == "1" and m["size"] > PAR_MIN:   # [v2.5.9z34] no probe
                use_par = True
            t0 = clock()
            try:
                if use_par:
                    if par_fn is None:
                        import par_fetch
                        par_fn = par_fetch.download
                    row["conns"] = conns_fn()
                    par_fn(m["url"], dst, m["size"], row["conns"])
                else:
                    single_fn(m["url"], dst)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:300]
            dt = max(clock() - t0, 1e-6)
            row["seconds"], row["MBps"] = round(dt, 2), round(m["size"] / 1e6 / dt, 3)
            if i == 0 and "error" not in row and not use_par:   # [v2.5.9z34] a parallel first wheel is no one-connection probe
                res["probe_MBps"] = row["MBps"]
            if "error" not in row:
                got = sha256(dst)
                if got != m["sha256"]:
                    row["error"] = f"sha256 {got[:16]} != PyPI {m['sha256'][:16]}"
                    os.remove(dst)
        if "error" in row:
            res["ok"] = False
        res["wheels"].append(row)
    json.dump(res, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    r = fetch_all(sys.argv[1], sys.argv[2:])
    for w in r["wheels"]:
        print("WHEEL", w["file"], w["size"], w["sha256"][:16], "kept" if w["kept"] else f"{w['MBps']} MB/s x{w['conns']}", w.get("error", "ok"))
    print("PROBE_MBPS", r["probe_MBps"])
    sys.exit(0 if r["ok"] else 1)
