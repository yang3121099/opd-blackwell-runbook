"""[v2.5.9z78] metrics_patch.py against verl's verl/utils/metric/utils.py (GOPD_METRIC_UTILS; the node: the installed file): the anchor
once, the patched text compiles and is idempotent; the patched reduce_metrics takes PROBE-8B's ragged per-rank lists and gives the old
numbers on rectangular ones and on plain lists."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_patch as P  # noqa: E402


def _patched():
    p = os.environ.get("GOPD_METRIC_UTILS")
    if not p:
        return None, None
    src = open(p).read()
    new, changed = P.patch_text(src)
    assert changed and new.count(P.MARK) == 1 and P.patch_text(new) == (new, False)
    ns_old, ns_new = {}, {}
    exec(compile(src, p, "exec"), ns_old); exec(compile(new, p, "exec"), ns_new)
    return ns_old["reduce_metrics"], ns_new["reduce_metrics"]


def test_ragged_rectangular_plain():
    old, new = _patched()
    if old is None:
        return
    ragged = {"opd/x": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0], [7.0]], "x/max_v": [[1.0], [9.0, 2.0]], "x/min_v": [[3.0, -1.0], [0.5]]}
    r = new(dict(ragged))
    assert r["opd/x"] == np.mean([1, 2, 3, 4, 5, 6, 7]) and r["x/max_v"] == 9.0 and r["x/min_v"] == -1.0
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old(dict(ragged)); raised = False
    except ValueError:
        raised = True
    if tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 24):          # older numpy only warns
        assert raised                                                               # the bug PROBE-8B hit
    rect = {"a": [[1.0, 2.0], [3.0, 4.0]], "b/max_q": [[1.0, 5.0], [2.0, 0.0]]}
    plain = {"c": [1.0, 2.0, 4.0], "d/min_z": [3.0, 1.0]}
    for m in (rect, plain):
        assert new(dict(m)) == old(dict(m))
