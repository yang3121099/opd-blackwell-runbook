"""[v2.5.9q] par_fetch.py against a local Range server (127.0.0.1 only, no proxy): the bytes, resuming from the segment
record, waiting while may_run() says no, stopping (Deferred) and going on, retries, a stale signed URL, a 308 entry."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import par_fetch as P

SIZE, SEG = 1 << 20, 64 << 10   # 16 segments
DATA = bytes(k % 251 for k in range(SIZE))


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(P.time, "sleep", lambda s: None)


def server(fail=None, stale_after=None):
    """fail: {offset: n} answers 500 n times for the segment at that offset; stale_after: /sN/ paths answer 403 after that
    many requests, /entry redirects (308) to a fresh /sN/."""
    st = {"starts": [], "n": 0, "gen": 0, "lock": threading.Lock(), "fail": dict(fail or {})}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/entry":
                with st["lock"]:
                    st["gen"] += 1; st["n"] = 0
                self.send_response(308); self.send_header("Location", f"/s{st['gen']}/f.bin"); self.end_headers(); return
            if stale_after is not None and self.path.startswith("/s"):
                with st["lock"]:
                    st["n"] += 1; n = st["n"]
                if self.path != f"/s{st['gen']}/f.bin" or n > stale_after:
                    self.send_response(403); self.end_headers(); return
            a, b = (int(x) for x in self.headers["Range"].split("=")[1].split("-")); b = min(b, SIZE - 1)
            with st["lock"]:
                st["starts"].append(a)
                if st["fail"].get(a, 0) > 0:
                    st["fail"][a] -= 1; self.send_response(500); self.end_headers(); return
            self.send_response(206); self.send_header("Content-Range", f"bytes {a}-{b}/{SIZE}")
            self.send_header("Content-Length", str(b - a + 1)); self.end_headers(); self.wfile.write(DATA[a:b + 1])

    s = ThreadingHTTPServer(("127.0.0.1", 0), H); s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}", st


def test_whole_file(tmp_path):
    s, base, st = server()
    try:
        r = P.download(base + "/f.bin", str(tmp_path / "f"), SIZE, 4, seg=SEG)
    finally:
        s.shutdown()
    assert open(tmp_path / "f", "rb").read() == DATA and r["segments"] == 16 and r["bytes"] == SIZE and r["resumed"] == 0
    assert not os.path.exists(tmp_path / "f.part") and not os.path.exists(tmp_path / "f.part.segs")


def test_resume_reads_only_the_missing_segments(tmp_path):
    with open(tmp_path / "f.part", "wb") as f:
        f.write(DATA[:2 * SEG])
    json.dump([0, 1], open(tmp_path / "f.part.segs", "w"))
    s, base, st = server()
    try:
        r = P.download(base + "/f.bin", str(tmp_path / "f"), SIZE, 3, seg=SEG)
    finally:
        s.shutdown()
    seg_starts = sorted(x for x in st["starts"] if x != 0 or len(st["starts"]) == 0)
    assert open(tmp_path / "f", "rb").read() == DATA and r["resumed"] == 2 and r["bytes"] == SIZE - 2 * SEG
    assert 0 not in seg_starts[1:] and SEG not in st["starts"]   # segment 1 never asked again (segment 0's start is the resolve)


def test_waits_while_may_run_says_no(tmp_path):
    n = {"k": 0}

    def may_run():
        n["k"] += 1; return n["k"] > 5
    s, base, st = server()
    try:
        P.download(base + "/f.bin", str(tmp_path / "f"), SIZE, 2, may_run=may_run, seg=SEG, wait_s=0)
    finally:
        s.shutdown()
    assert open(tmp_path / "f", "rb").read() == DATA and n["k"] > 5


def test_stop_defers_and_the_next_call_goes_on(tmp_path):
    n = {"k": 0}

    def stop():
        n["k"] += 1; return n["k"] > 4   # a few segments, then the window closes
    s, base, st = server()
    try:
        with pytest.raises(P.Deferred):
            P.download(base + "/f.bin", str(tmp_path / "f"), SIZE, 1, stop=stop, seg=SEG)
        done = json.load(open(tmp_path / "f.part.segs"))
        assert 0 < len(done) < 16 and not os.path.exists(tmp_path / "f")
        r = P.download(base + "/f.bin", str(tmp_path / "f"), SIZE, 4, seg=SEG)
    finally:
        s.shutdown()
    assert open(tmp_path / "f", "rb").read() == DATA and r["resumed"] == len(done)


def test_a_failing_segment_is_retried(tmp_path):
    s, base, st = server(fail={5 * SEG: 2})
    try:
        P.download(base + "/f.bin", str(tmp_path / "f"), SIZE, 4, seg=SEG)
    finally:
        s.shutdown()
    assert open(tmp_path / "f", "rb").read() == DATA and st["starts"].count(5 * SEG) == 3


def test_a_segment_failing_every_time_raises(tmp_path):
    s, base, st = server(fail={3 * SEG: 99})
    try:
        with pytest.raises(IOError, match="segment 3 failed"):
            P.download(base + "/f.bin", str(tmp_path / "f"), SIZE, 2, seg=SEG)
    finally:
        s.shutdown()
    assert not os.path.exists(tmp_path / "f") and os.path.exists(tmp_path / "f.part.segs")   # what was in is kept


def test_a_stale_signed_url_is_resolved_again_through_a_308(tmp_path):
    s, base, st = server(stale_after=6)   # the signed path stops answering after 6 requests
    try:
        P.download(base + "/entry", str(tmp_path / "f"), SIZE, 2, seg=SEG)
    finally:
        s.shutdown()
    assert open(tmp_path / "f", "rb").read() == DATA and st["gen"] >= 2   # resolved more than once
