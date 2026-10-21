#!/usr/bin/env python3
"""[v2.5.9q] Parallel, resumable download of one file in HTTP Range segments (standard library; fetch_model.py's background
pulls, FETCH_PARALLEL=1). PROBE-NET2 (10-21): the node's proxy limits each connection (~0.06-0.08 MB/s) and not the host,
so N connections give about N times the rate (hf.co 32: 2.3 MB/s, 30x one connection).

download(url, dst, size, conns, ...) writes <dst>.part with positional writes; every finished segment is recorded in
<dst>.part.segs (JSON, rewritten atomically), so a later call (after FETCH_TIMEOUT, a kill or a reboot) fetches only the
missing segments. The URL is resolved once per call (redirects followed, 308 included, as par_range.py), and again when a
signed URL stops answering (401 / 403 / 404). Before each segment a worker asks may_run(); while it says no (result uploads
timing out) the worker waits without holding a connection; once stop() says so (a large file when the night window ends)
the workers finish their segment and the call raises Deferred, keeping what is in. A segment that fails is
retried with a back-off (SEG_TRIES); a file whose segments keep failing raises. When every segment is in, the .part's size
must be the listed size; it is renamed to dst (the caller checks the sha256, fetch_model.missing()). No token is ever sent:
the pulls are public repositories.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from par_range import OPENER, UA  # noqa: E402   (redirects, 308 included)

SEG = int(os.environ.get("FETCH_SEG_BYTES", str(16 << 20)))
SEG_TRIES = int(os.environ.get("FETCH_SEG_TRIES", "6"))


class Deferred(Exception):
    """stop() said so (a large file at the end of the night window): the segments in are kept for the next call."""


class Stale(Exception):
    """The resolved URL stopped answering (a signed URL that expired): resolve again."""


def resolve(url, timeout=60):
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with OPENER.open(req, timeout=timeout) as r:
        r.read(1)
        return r.geturl()


def _segments(size, seg):
    return [(i, i * seg, min(size, (i + 1) * seg) - 1) for i in range((size + seg - 1) // seg)]


def _load_done(path):
    try:
        return set(json.load(open(path)))
    except (OSError, ValueError):
        return set()


def download(url, dst, size, conns, may_run=lambda: True, stop=lambda: False, seg=SEG, timeout=60, wait_s=30, log=lambda *a: None):
    """{"bytes": fetched now, "segments": total, "resumed": segments already there, "seconds": ...}; raises on failure."""
    part, rec = dst + ".part", dst + ".part.segs"
    segs = _segments(size, seg)
    done = _load_done(rec) if os.path.exists(part) else set()
    resumed = len(done)
    fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
    lock = threading.Lock(); state = {"url": resolve(url, timeout), "got": 0, "error": None, "deferred": False}
    todo = [s for s in segs if s[0] not in done]; t0 = time.time()

    def save():
        tmp = rec + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sorted(done), f)
        os.replace(tmp, rec)

    def fetch_one(i, a, b):
        req = urllib.request.Request(state["url"], headers={**UA, "Range": f"bytes={a}-{b}"})
        try:
            r = OPENER.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                raise Stale(str(e.code))
            raise
        with r:
            if r.status != 206:
                raise IOError(f"segment {i}: HTTP {r.status}, not 206")
            off, n = a, 0
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                os.pwrite(fd, buf, off); off += len(buf); n += len(buf)
        if n != b - a + 1:
            raise IOError(f"segment {i}: {n} bytes, expected {b - a + 1}")
        return n

    def worker():
        while True:
            with lock:
                if state["error"] or state["deferred"] or not todo:
                    return
                if stop():
                    state["deferred"] = True
                    return
                i, a, b = todo.pop(0)
            for attempt in range(1, SEG_TRIES + 1):
                while not may_run():   # uploads first / the night window: no connection held while waiting
                    time.sleep(wait_s)
                try:
                    n = fetch_one(i, a, b)
                    with lock:
                        done.add(i); state["got"] += n; save()
                    break
                except Stale:
                    with lock:
                        try:
                            state["url"] = resolve(url, timeout)
                        except Exception as e:   # noqa: BLE001
                            log(f"re-resolve failed: {type(e).__name__}")
                except Exception as e:   # noqa: BLE001
                    log(f"segment {i} attempt {attempt}: {type(e).__name__}: {str(e)[:120]}")
                time.sleep(min(60, 5 * attempt))
            else:
                with lock:
                    state["error"] = f"segment {i} failed {SEG_TRIES} times"
                return

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, min(conns, len(todo) or 1)))]
    try:
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        os.close(fd)
    if state["error"]:
        raise IOError(state["error"])
    if state["deferred"] and len(done) != len(segs):
        raise Deferred(f"{os.path.basename(dst)}: {len(done)} of {len(segs)} segments in, the rest deferred")
    if len(done) != len(segs) or os.path.getsize(part) != size:
        raise IOError(f"{os.path.basename(dst)}: {len(done)} of {len(segs)} segments, {os.path.getsize(part)} of {size} bytes")
    os.replace(part, dst)
    try:
        os.remove(rec)
    except OSError:
        pass
    return {"bytes": state["got"], "segments": len(segs), "resumed": resumed, "seconds": round(time.time() - t0, 1)}
