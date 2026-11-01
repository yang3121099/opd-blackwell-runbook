"""[v2.5.9z118] opd_variants' reward shaping (research_program 3q): TOPD's bounded reward r~ = log(a e^r + 1 - a) and the suffix credit
A_k = r_k + (1 / (T - k)) sum_{j > k} r_j. The torch tests run where torch is (the node: probe_fret.sh runs them on the arm's python);
the numpy tests check the same reverse-cumulative algorithm against the definition everywhere. No exact float compares in torch."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
except ImportError:
    torch = None


def fret_loop(r, m):
    """The definition, position by position: valid positions 1..T in order; A_k = r_k + mean(r_j, j > k), 0 added at the last."""
    out = np.zeros_like(r, dtype=np.float64)
    for b in range(r.shape[0]):
        idx = [i for i in range(r.shape[1]) if m[b, i] > 0]
        T = len(idx)
        for n, i in enumerate(idx, start=1):
            later = [r[b, j] for j in idx[n:]]
            out[b, i] = r[b, i] + (sum(later) / (T - n) if T - n > 0 else 0.0)
    return out


def fret_np(r, m):
    """opd_variants.fret's algorithm in numpy (flip, cumsum, subtract self, divide by the later count)."""
    m = m.astype(np.float64); rf = r * m
    later_sum = np.flip(np.cumsum(np.flip(rf, 1), 1), 1) - rf
    later_n = np.flip(np.cumsum(np.flip(m, 1), 1), 1) - m
    return (rf + later_sum / np.maximum(later_n, 1)) * m


def _data(seed=0):
    g = np.random.default_rng(seed)
    r = g.normal(0, 2, size=(5, 12)); m = np.zeros((5, 12))
    for b, L in enumerate((12, 7, 1, 0, 4)):
        m[b, :L] = 1
    r[m == 0] = 9.0          # padding holds junk: it must never leak in
    return r, m


def test_fret_algorithm_matches_the_definition():
    r, m = _data()
    assert np.allclose(fret_np(r, m), fret_loop(r, m), rtol=1e-9, atol=1e-12)
    a = fret_np(r, m)
    assert np.all(a[m == 0] == 0)                                         # padding 0
    assert np.isclose(a[2, 0], r[2, 0])                                    # a one-token response: its own reward
    assert np.isclose(a[0, 11], r[0, 11]) and np.isclose(a[0, 10], r[0, 10] + r[0, 11])   # last: own; second to last: + the last


def test_topd_formula_and_range_numpy():
    a = 0.2
    r = np.array([-50.0, -3.5, 0.0, 2.9, 30.0])
    t = np.logaddexp(r + math.log(a), math.log(1 - a))
    assert np.allclose(t, np.log(a * np.exp(r) + 1 - a), rtol=1e-12)
    assert np.isclose(np.log(a * math.exp(-3.5) + 0.8), -0.2155, atol=1e-3) and np.isclose(np.log(a * 19 + 0.8), 1.526, atol=1e-3)   # the note's two numbers
    assert t.min() >= math.log(0.8) - 1e-12 and t.max() <= math.log(a * math.exp(r.max()) + 0.8) + 1e-9 and abs(t[2]) < 1e-12


def _torch():
    if torch is None:
        return None
    import opd_variants as V
    return V


def _env(**kw):
    old = {k: os.environ.get(k) for k in ("OPD_FRET", "OPD_TOPD_ALPHA")}
    for k in old:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in kw.items()})
    return old


def _restore(old):
    for k, v in old.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


def test_torch_fret_and_topd_match_numpy():
    V = _torch()
    if V is None:
        return
    r, m = _data(1)
    rt, mt = torch.tensor(r, dtype=torch.float32), torch.tensor(m, dtype=torch.float32)
    assert torch.allclose(V.fret(rt, mt), torch.tensor(fret_loop(r, m), dtype=torch.float32), rtol=1e-5, atol=1e-5)
    t = V.topd(rt, mt, 0.2)
    ref = np.logaddexp(r + math.log(0.2), math.log(0.8)) * m
    assert torch.allclose(t, torch.tensor(ref, dtype=torch.float32), rtol=1e-5, atol=1e-6)
    assert torch.all(t[mt == 0] == 0)
    big = V.topd(torch.tensor([[200.0, -200.0]]), torch.ones(1, 2), 0.2)   # no overflow at either end
    assert torch.isfinite(big).all() and abs(big[0, 0].item() - (200 + math.log(0.2))) < 1e-3 and abs(big[0, 1].item() - math.log(0.8)) < 1e-5
    assert V.fret(rt.to(torch.bfloat16), mt).dtype == torch.bfloat16      # the input's dtype kept


def test_shape_switches():
    V = _torch()
    if V is None:
        return
    r, m = _data(2)
    rt, mt = torch.tensor(r, dtype=torch.float32), torch.tensor(m, dtype=torch.float32)
    old = _env()
    try:
        out, met = V.shape(rt, mt)
        assert out is rt and met == {} and not V.shaping_active()                  # both off: the advantage untouched
        os.environ["OPD_FRET"] = "1"
        out, met = V.shape(rt, mt)
        assert torch.allclose(out, V.fret(rt, mt)) and "shape/fret_abs_mean" in met and "shape/topd_min" not in met
        os.environ.pop("OPD_FRET"); os.environ["OPD_TOPD_ALPHA"] = "0.2"
        out, met = V.shape(rt, mt)
        assert torch.allclose(out, V.topd(rt, mt, 0.2)) and "shape/fret_abs_mean" not in met
        assert met["shape/topd_min"] >= math.log(0.8) - 1e-5 and met["shape/topd_max"] <= math.log(0.2 * met["shape/rho_max"] + 0.8) + 1e-4
        os.environ["OPD_FRET"] = "1"
        out, met = V.shape(rt, mt)
        assert torch.allclose(out, V.fret(V.topd(rt, mt, 0.2), mt), rtol=1e-5, atol=1e-6)   # topd first, then the credit
        os.environ["OPD_TOPD_ALPHA"] = "1.5"
        try:
            V.shape(rt, mt); assert False, "alpha outside (0, 1) must fail"
        except ValueError:
            pass
    finally:
        _restore(old)


def test_torch_importable_on_the_node():
    """probe_fret.sh counts this module's torch tests as run only when torch imports (its own check); here a no-op without torch."""
    assert torch is None or hasattr(torch, "flip")
