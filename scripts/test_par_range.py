"""[v2.5.9h] par_range.py against a local Range server (127.0.0.1 only, no proxy): a per-connection rate limit scales with
the number of connections, a per-server limit does not, Range answers are 206, a missing file is an error."""
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import par_range  # noqa: E402

SIZE = 64 << 20


def server(per_conn_bps, total_bps=None):
    budget = {"t": time.monotonic(), "sent": 0, "lock": threading.Lock()}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path != "/f.bin":
                self.send_response(404); self.end_headers(); return
            a, b = (int(x) for x in self.headers["Range"].split("=")[1].split("-"))
            b = min(b, SIZE - 1)
            self.send_response(206); self.send_header("Content-Range", f"bytes {a}-{b}/{SIZE}")
            self.send_header("Content-Length", str(b - a + 1)); self.end_headers()
            left, chunk = b - a + 1, 16384
            try:
                while left > 0:
                    n = min(chunk, left)
                    if total_bps:   # the whole server shares one rate
                        while True:
                            with budget["lock"]:
                                if budget["sent"] + n <= (time.monotonic() - budget["t"]) * total_bps:
                                    budget["sent"] += n; break
                            time.sleep(0.005)
                    self.wfile.write(b"\0" * n); left -= n
                    time.sleep(n / per_conn_bps)
            except (BrokenPipeError, ConnectionResetError):
                pass

    s = ThreadingHTTPServer(("127.0.0.1", 0), H); s.daemon_threads = True
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_address[1]}"


def _noproxy():
    for k in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    os.environ["no_proxy"] = "127.0.0.1,localhost"


def test_per_connection_limit_scales():
    _noproxy(); s, base = server(per_conn_bps=2e6)
    try:
        one = par_range.measure(base + "/f.bin", 1, 2.0, seg=1 << 20)
        four = par_range.measure(base + "/f.bin", 4, 2.0, seg=1 << 20)
    finally:
        s.shutdown()
    assert one["range_ok"] and one["size"] == SIZE and one["failed"] == 0 and one["conns_with_data"] == 1
    assert four["conns_with_data"] == 4 and four["codes"].get("206", 0) >= 5
    assert 2.5 < four["MBps"] / one["MBps"] < 5.5, (one["MBps"], four["MBps"])


def test_total_limit_does_not_scale():
    _noproxy(); s, base = server(per_conn_bps=50e6, total_bps=2e6)
    try:
        one = par_range.measure(base + "/f.bin", 1, 2.0, seg=1 << 20)
        four = par_range.measure(base + "/f.bin", 4, 2.0, seg=1 << 20)
    finally:
        s.shutdown()
    assert 0.6 < four["MBps"] / one["MBps"] < 1.6, (one["MBps"], four["MBps"])


def test_missing_file_is_an_error():
    _noproxy(); s, base = server(per_conn_bps=1e6)
    try:
        r = par_range.measure(base + "/missing.bin", 4, 1.0)
    finally:
        s.shutdown()
    assert r["error"] == "resolve http 404" and r["bytes"] == 0 and r["MBps"] == 0.0


def test_308_is_followed_with_the_range():   # [v2.5.9p] hf-mirror answers the resolve with 308 (Python 3.9's urllib raised)
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    _noproxy(); s, base = server(per_conn_bps=5e6)

    class R(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(308); self.send_header("Location", base + "/f.bin"); self.end_headers()
    r = ThreadingHTTPServer(("127.0.0.1", 0), R); r.daemon_threads = True
    threading.Thread(target=r.serve_forever, daemon=True).start()
    try:
        out = par_range.measure(f"http://127.0.0.1:{r.server_address[1]}/x", 2, 1.0, seg=1 << 20)
    finally:
        r.shutdown(); s.shutdown()
    assert out["error"] is None and out["range_ok"] and out["size"] == SIZE and out["bytes"] > 0, out
