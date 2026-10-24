"""[v2.5.9z78] metrics_patch.py against verl's verl/utils/metric/utils.py (GOPD_METRIC_UTILS; the node: the installed file): the anchor
once, the patched text compiles and is idempotent; the patched reduce_metrics takes PROBE-8B's ragged per-rank lists and gives the old
numbers on rectangular ones and on plain lists. [v2.5.9z81] the concat hook against verl/protocol.py (GOPD_PROTOCOL): E10e-P5self's ranks
with different metric keys merge, agreeing keys give the old dict; the installed files are read as <file>.orig once an arm patched them."""
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
    if P.MARK in src and os.path.exists(p + ".orig"):   # [v2.5.9z81] (z78b) an arm patched the installed file already
        src = open(p + ".orig").read()
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


def _concat():   # [v2.5.9z81] (old DataProto, new DataProto) from verl/protocol.py
    p = os.environ.get("GOPD_PROTOCOL")
    if not p:
        return None, None
    src = open(p).read()
    if P.CMARK in src and os.path.exists(p + ".orig"):
        src = open(p + ".orig").read()
    new, changed = P.patch_concat_text(src)
    assert changed and new.count(P.CMARK) == 1 and P.patch_concat_text(new) == (new, False)
    try:
        import torch  # noqa: F401  (protocol.py imports torch / tensordict: the node)
    except ImportError:
        compile(new, p, "exec"); return None, None
    import importlib.util as U
    mods = []
    for i, text in enumerate((src, new)):
        spec = U.spec_from_loader(f"_proto_{i}", loader=None); m = U.module_from_spec(spec)
        exec(compile(text, p, "exec"), m.__dict__); mods.append(m.DataProto)
    return mods


def _proto(D, metrics):
    import torch
    from tensordict import TensorDict
    return D(batch=TensorDict({"x": torch.zeros(2, 1)}, batch_size=[2]), meta_info={"metrics": metrics})


def test_concat_ranks_with_different_keys():   # E10e-P5self: rank 1 had no EOS-terminated response, so no opd/eos_adv_mean
    old, new = _concat()
    if old is None:
        return
    ranks = [{"actor/pg_loss": [0.1, 0.2], "opd/eos_adv_mean": [-0.5]}, {"actor/pg_loss": [0.3]}, {"actor/pg_loss": [0.4], "opd/trunc_tail_adv_mean": [0.2]}]
    m = new.concat([_proto(new, r) for r in ranks]).meta_info["metrics"]
    assert m == {"actor/pg_loss": [[0.1, 0.2], [0.3], [0.4]], "opd/eos_adv_mean": [[-0.5]], "opd/trunc_tail_adv_mean": [[0.2]]}
    try:
        old.concat([_proto(old, r) for r in ranks]); raised = False
    except AssertionError:
        raised = True
    assert raised                                                                    # the failure E10e-P5self hit
    same = [{"a": [1.0], "b/max_v": [2.0]}, {"a": [3.0], "b/max_v": [4.0]}]
    assert new.concat([_proto(new, r) for r in same]).meta_info["metrics"] == old.concat([_proto(old, r) for r in same]).meta_info["metrics"]


def test_concat_hook_text_alone():   # the Mac: the anchor once, compiles, idempotent (no torch needed)
    _concat()
