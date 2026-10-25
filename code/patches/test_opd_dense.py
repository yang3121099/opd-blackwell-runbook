"""[v2.5.9z76] (Q-065) dense OPD's divergences against brute force over the whole vocabulary (numpy; torch when installed):
a teacher whose mass outside its top-K is spread evenly makes the top-K forms exact; AKL's head / tail split and weights."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opd_dense as D  # noqa: E402


def _case(seed=0, N=5, V=40, K=6):
    rs = np.random.RandomState(seed)
    z = rs.randn(N, V) * 2
    tl = rs.randn(N, V) * 3                                   # the teacher's full logits
    tq = np.exp(tl - tl.max(-1, keepdims=True)); tq /= tq.sum(-1, keepdims=True)
    ids = np.argsort(-tq, -1)[:, :K]
    top = np.take_along_axis(tq, ids, -1)
    q = np.zeros_like(tq) + ((1 - top.sum(-1)) / (V - K))[:, None]   # the even remainder the top-K form assumes
    np.put_along_axis(q, ids, top, -1)
    return z, ids, np.log(top), q


def brute(z, q, mu_):
    lp = z - (np.log(np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)) + z.max(-1, keepdims=True)); p = np.exp(lp)
    rkl = (p * (lp - np.log(q))).sum(-1); fkl = (q * (np.log(q) - lp)).sum(-1)
    out = []
    for i in range(len(q)):
        order = np.argsort(-q[i], kind="stable"); cum = np.cumsum(q[i][order])
        head = order[: max(1, int(np.searchsorted(cum, mu_, side="right")))]   # [v2.5.9z76b] the official code: cum <= mu, and the first
        m = np.zeros(len(q[i]), bool); m[head] = True
        gh, gt = np.abs(q[i] - p[i])[m].sum(), np.abs(q[i] - p[i])[~m].sum()
        out.append(gh / (gh + gt))
    w = np.array(out)
    return rkl, fkl, w, w * fkl + (1 - w) * rkl


def test_against_brute_force():
    for seed in range(4):
        z, ids, tlp, q = _case(seed)
        d = D.dense_terms(z, ids, tlp, 0.5)
        rkl, fkl, w, akl = brute(z, q, 0.5)
        assert np.allclose(d["rkl"], rkl, atol=1e-9) and np.allclose(d["fkl"], fkl, atol=1e-9)
        assert np.allclose(d["mass_topk"], np.exp(tlp).sum(-1))
        # the head must fall inside the top-K for the weights to match (mu below the top-K mass)
        ok = np.exp(tlp).sum(-1) >= 0.5
        assert np.allclose(d["w_head"][ok], w[ok], atol=1e-9) and np.allclose(d["akl"][ok], akl[ok], atol=1e-9)


def test_zero_when_equal_and_nonnegative():
    V, K = 30, 30                                                   # K = V: the teacher fully known
    rs = np.random.RandomState(5); z = rs.randn(4, V)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    ids = np.argsort(-lp, -1)[:, :K]; tlp = np.take_along_axis(lp, ids, -1)
    d = D.dense_terms(z, ids, tlp, 0.5)
    assert np.allclose(d["rkl"], 0, atol=1e-6) and np.allclose(d["fkl"], 0, atol=1e-6) and np.allclose(d["akl"], 0, atol=1e-6)   # eps floor on r
    z2, ids2, tlp2, _ = _case(7)
    d2 = D.dense_terms(z2, ids2, tlp2, 0.5)
    assert (d2["rkl"] > -1e-12).all() and (d2["fkl"] > -1e-12).all()


def test_env_and_pearson():
    old = {k: os.environ.get(k) for k in ("OPD_DENSE", "OPD_DENSE_K", "OPD_DENSE_MU")}
    try:
        os.environ["OPD_DENSE"] = "akl"; os.environ["OPD_DENSE_K"] = "32"; os.environ["OPD_DENSE_MU"] = "0.6"
        assert D.mode() == "akl" and D.active() and D.topk_k() == 32 and D.mu() == 0.6
        os.environ["OPD_DENSE"] = "fkl"; assert D.mode() == "" and not D.active()
        os.environ.pop("OPD_DENSE"); assert not D.active()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    a = np.arange(10.0); assert abs(D.pearson(a, 2 * a + 1) - 1) < 1e-12 and D.pearson(a[:2], a[:2]) is None
    assert D.pearson(a, np.ones(10)) is None


def test_loss_term_env():   # [v2.5.9z85] (Q-079) OPD_DENSE_LOSS picks the loss while the hooks stay on rkl / akl
    old = {k: os.environ.get(k) for k in ("OPD_DENSE", "OPD_DENSE_LOSS")}
    try:
        os.environ.pop("OPD_DENSE_LOSS", None)
        os.environ["OPD_DENSE"] = "akl"; assert D.loss_term() == "akl"
        os.environ["OPD_DENSE"] = "rkl"; os.environ["OPD_DENSE_LOSS"] = "fkl"; assert D.mode() == "rkl" and D.active() and D.loss_term() == "fkl"
        os.environ["OPD_DENSE_LOSS"] = "bogus"; assert D.loss_term() == "rkl"
        os.environ.pop("OPD_DENSE"); os.environ["OPD_DENSE_LOSS"] = "fkl"; assert D.loss_term() == "" and not D.active()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_fkl_gradient_is_p_minus_q():   # [v2.5.9z85] (Q-079) the forward KL's gradient in the logits is p - q (q with its even remainder)
    try:
        import torch
    except ImportError:   # the Mac has no torch; PROBE-DENSE / PROBE-DENSE-4B run this on the node
        return
    z, ids, tlp, q = _case(5)
    zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    D.dense_terms(zt, torch.tensor(ids), torch.tensor(tlp), 0.5)["fkl"].sum().backward()
    p = np.exp(z - z.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    assert np.allclose(zt.grad.numpy(), p - q, atol=1e-8)
    dn = D.dense_terms(z, ids, tlp, 0.5)
    loss, _ = D.dense_loss_rows(torch.tensor(z, dtype=torch.float32, requires_grad=True), torch.tensor(ids), torch.tensor(tlp, dtype=torch.float32), "fkl", chunk=2, mu_=0.5)
    assert np.allclose(loss.detach().numpy(), dn["fkl"], atol=1e-4)


def test_torch_matches_numpy_and_grad():
    try:
        import torch
    except ImportError:   # the Mac has no torch; PROBE-DENSE runs this on the node
        return
    z, ids, tlp, _ = _case(3)
    dn = D.dense_terms(z, ids, tlp, 0.5)
    zt = torch.tensor(z, dtype=torch.float64, requires_grad=True)
    dt = D.dense_terms(zt, torch.tensor(ids), torch.tensor(tlp), 0.5)
    for k in ("rkl", "fkl", "akl", "w_head", "mass_topk"):
        assert np.allclose(dt[k].detach().numpy(), dn[k], atol=1e-9), k
    dt["rkl"].sum().backward(); assert torch.isfinite(zt.grad).all() and zt.grad.abs().sum() > 0
    i, l = D.topk_logprobs(torch.tensor(z), 6)
    assert torch.allclose(l.exp().sum(-1), torch.softmax(torch.tensor(z), -1).topk(6).values.sum(-1).float(), atol=1e-6)
    loss, st = D.dense_loss_rows(torch.tensor(z, dtype=torch.float32, requires_grad=True), torch.tensor(ids), torch.tensor(tlp, dtype=torch.float32), "akl", chunk=2, mu_=0.5)
    assert loss.shape == (5,) and np.allclose(loss.detach().numpy(), dn["akl"], atol=1e-4) and set(st) == {"rkl", "fkl", "w_head", "mass_topk"}


def test_akl_head_official_boundary():   # [v2.5.9z76b] (the literature reader's example) [0.3, 0.3, 0.2, 0.2]: official {1}, paper {1, 2}
    B = D.backend(np.zeros(1))
    h = D.akl_head(np.cumsum(np.array([[0.3, 0.3, 0.2, 0.2]]), -1), 0.5, B)
    assert h.tolist() == [[True, False, False, False]]
    assert D.akl_head(np.cumsum(np.array([[0.7, 0.2, 0.1]]), -1), 0.5, B).tolist() == [[True, False, False]]       # top-1 > mu: the same in both
    assert D.akl_head(np.cumsum(np.array([[0.2, 0.2, 0.1, 0.5]]), -1), 0.5, B).tolist() == [[True, True, True, False]]  # exactly mu stays in the head
    try:
        import torch
    except ImportError:   # the Mac has no torch; PROBE-DENSE runs this on the node
        return
    t = D.akl_head(torch.cumsum(torch.tensor([[0.3, 0.3, 0.2, 0.2]], dtype=torch.float64), -1), 0.5, D.backend(torch.zeros(1)))
    assert t.tolist() == [[True, False, False, False]]


def test_dense_terms_boundary_case():   # [v2.5.9z76b] top-1 < mu: the weight follows the official head ({1}), not the paper's ({1, 2})
    V, K = 10, 4
    z = np.random.RandomState(7).randn(1, V)
    top = np.array([[0.3, 0.3, 0.2, 0.15]]); ids = np.array([[0, 1, 2, 3]])
    q = np.full((1, V), (1 - top.sum()) / (V - K)); q[0, :K] = top[0]
    d = D.dense_terms(z, ids, np.log(top), 0.5)
    rkl, fkl, w, akl = brute(z, q, 0.5)
    assert np.allclose(d["w_head"], w, atol=1e-9) and np.allclose(d["akl"], akl, atol=1e-9)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True)); p = np.exp(lp)
    gap = np.abs(q - p)[0]; paper = gap[:2].sum() / gap.sum()
    assert abs(float(d["w_head"][0]) - paper) > 1e-3                              # the paper's head would weigh differently
