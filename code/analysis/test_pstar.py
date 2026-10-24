"""[v2.5.9z75] (Q-064) p*'s numerics without a GPU: the problem draw, p* from sums and from per-token means, the bootstrap CI, the lines."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pstar as P  # noqa: E402


def test_pick_common_sample0_seeded():
    short = [{"pid": f"math500-{i}", "sample": s, "token_ids": [i]} for i in range(10) for s in (0, 1)]
    long_ = [{"pid": f"math500-{i}", "sample": 0, "token_ids": [i, i]} for i in range(3, 12)]
    a, b = P.pick(short, long_, 4, 0), P.pick(short, long_, 4, 0)
    assert [x[0] for x in a] == [x[0] for x in b] and len(a) == 4                          # seeded: the same draw
    assert all(3 <= int(p.split("-")[1]) <= 9 and s["sample"] == 0 for p, s, _ in a)         # common problems, sample 0
    assert [int(x[0].split("-")[1]) for x in a] == sorted(int(x[0].split("-")[1]) for x in a)
    assert len(P.pick(short, long_, 100, 0)) == 7                                           # all of them when n is larger


def test_pstar_sums_and_per_token():
    # long answers: the thinking teacher +60 nats over the plain one; short: the plain one +20 over the thinking one
    r = P.pstar([-100, -120], [-160, -180], [-30, -32], [-10, -12])
    assert (r["delta_long"], r["delta_short"], r["p_star"]) == (60.0, 20.0, 0.75)
    t = P.pstar([-100, -120], [-160, -180], [-30, -32], [-10, -12], [600, 600], [100, 100])   # per token: 0.1 against 0.2
    assert (t["delta_long"], t["delta_short"], t["p_star"]) == (0.1, 0.2, round(0.1 / 0.3, 4))


def test_summary_lines_and_ci():
    rs = np.random.RandomState(0); rows = []
    for _ in range(80):
        nl, ns = 3000, 600
        rows.append({"lt_long": -0.30 * nl, "lp_long": -0.30 * nl - 55 + rs.randn(), "lt_short": -0.2 * ns - 45 + rs.randn(),
                     "lp_short": -0.2 * ns, "n_long": nl, "n_short": ns})
    s = P.summarize(rows)
    assert s["problems"] == 80 and abs(s["sums"]["p_star"] - 0.55) < 0.01 and s["H1"] and s["in_measured_interval"]
    lo, hi = s["sums"]["p_star_ci95"]; assert lo < 0.55 < hi and hi - lo < 0.05
    # per token: 55 / 3000 against 45 / 600 -> p* about 0.196, well below the measured interval -> H2
    assert abs(s["per_token"]["p_star"] - (55 / 3000) / (55 / 3000 + 45 / 600)) < 0.01 and s["H2"]
