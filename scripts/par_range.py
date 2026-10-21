#!/usr/bin/env python3
"""[v2.5.9h] PROBE-NET2's measuring tool: N parallel HTTP Range connections to one file for SECS seconds, standard library
only (urllib + threads; the node's venv has no pip), data discarded. Through the node's proxy (urllib reads https_proxy).

The URL is resolved once (a bytes=0-0 request: the file size from Content-Range, the final URL after redirects, so the
connections do not each pay the redirect); thread i then reads segments i, i+N, i+2N, ... of SEG bytes (4 MB), each its own
request, until the deadline. Bytes count as they arrive, up to the deadline only.

Prints one JSON object: conns, secs, bytes, MBps (bytes / secs / 1e6), codes (HTTP status or error class -> count), failed
(requests that raised), conns_with_data, range_ok (some answer was 206), size; or error when the resolve request failed.
Usage: python3 par_range.py <url> <conns> <secs> [--seg BYTES] [--timeout S] [--source NAME] (added to the JSON with tool par_range)
"""
import argparse
import json
import re
import threading
import time
import urllib.error
import urllib.request

UA = {"User-Agent": "opd-probe-net2/1"}


def resolve(url, timeout):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
        size = int(m.group(1)) if m else int(r.headers.get("Content-Length") or 0)
        r.read(1)
        return r.geturl(), size, r.status


def measure(url, conns, secs, seg=4 << 20, timeout=30):
    out = {"conns": conns, "secs": secs, "bytes": 0, "MBps": 0.0, "codes": {}, "failed": 0, "conns_with_data": 0,
           "range_ok": False, "size": None, "error": None}
    try:
        final, size, code = resolve(url, timeout)
    except urllib.error.HTTPError as e:
        out["error"] = f"resolve http {e.code}"; out["codes"][str(e.code)] = 1; return out
    except Exception as e:
        out["error"] = f"resolve {type(e).__name__}: {str(e)[:120]}"; return out
    out["size"] = size; out["codes"][str(code)] = 1; out["range_ok"] = code == 206
    lock = threading.Lock(); got = [0] * conns
    t0 = time.monotonic(); deadline = t0 + secs

    def count(key):
        with lock:
            out["codes"][key] = out["codes"].get(key, 0) + 1

    def worker(i):
        off = (i * seg) % max(size, 1)
        while time.monotonic() < deadline:
            end = min(off + seg, size) - 1 if size else off + seg - 1
            req = urllib.request.Request(final, headers={**UA, "Range": f"bytes={off}-{end}"})
            try:
                left = max(1.0, deadline - time.monotonic())
                with urllib.request.urlopen(req, timeout=min(timeout, left + 5)) as r:
                    count(str(r.status))
                    if r.status == 206:
                        out["range_ok"] = True
                    while time.monotonic() < deadline:
                        b = r.read(65536)
                        if not b:
                            break
                        if time.monotonic() <= deadline:
                            got[i] += len(b)
            except urllib.error.HTTPError as e:
                count(str(e.code))
                with lock:
                    out["failed"] += 1
                time.sleep(1)
            except Exception as e:
                count(type(e).__name__)
                with lock:
                    out["failed"] += 1
                time.sleep(1)
            off += conns * seg
            if size and off >= size:
                off = (i * seg) % size

    ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(conns)]
    for t in ts:
        t.start()
    time.sleep(max(0.0, deadline - time.monotonic()))
    total = sum(got)   # the threads still reading past the deadline add nothing more (they check it)
    out.update(bytes=total, MBps=round(total / secs / 1e6, 4), conns_with_data=sum(1 for g in got if g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url"); ap.add_argument("conns", type=int); ap.add_argument("secs", type=float)
    ap.add_argument("--seg", type=int, default=4 << 20); ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--source", default=None)
    a = ap.parse_args()
    r = measure(a.url, a.conns, a.secs, a.seg, a.timeout)
    print(json.dumps({"tool": "par_range", "source": a.source, **r} if a.source else r), flush=True)


if __name__ == "__main__":
    main()
