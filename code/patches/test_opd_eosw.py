"""[v2.5.9z102] (Q-087) opd_eosw.py: the last complete box, the weight window (numpy and torch), and - torch, on the node - the loss with
every weight 1 equal to the unweighted loss (allclose, rtol 1e-6 / atol 1e-7; ops-lead's A1 condition), the advantage factor equal to a
per-token loss factor under PPO's clip and dual clip, and the window's tokens getting w times the gradient."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opd_eosw as E  # noqa: E402


def test_last_box_end():
    s = "a \\boxed{1} b \\boxed{\\frac{1}{2}} c"
    assert E.last_box_end(list(s)) == s.rindex("}")                               # one piece per character
    assert E.last_box_end(["x ", "\\boxed{", "4", "}", ".", " \\boxed{", "5"]) == 3   # the last box never closes: the one before
    assert E.last_box_end(["\\boxed{", "4", "}", " and ", "\\boxed{", "{7}", "}", "\n"]) == 6
    assert E.last_box_end(list("no box here")) == -1 and E.last_box_end([]) == -1


def test_weights_window_numpy():
    valid = np.array([[1] * 10, [1] * 10, [1] * 6 + [0] * 4, [1] * 10])
    be = np.array([2, -1, 3, 9])
    w, win = E.weights(be, valid, 64.0, 4, np)
    assert w.dtype == np.float32 and w.shape == (4, 10)
    assert list(np.where(win[0])[0]) == [3, 4, 5, 6] and set(w[0][win[0]]) == {64.0} and set(w[0][~win[0]]) == {1.0}
    assert not win[1].any() and (w[1] == 1).all()                                    # no complete box: 1 everywhere
    assert list(np.where(win[2])[0]) == [4, 5]                                       # padding never weighted
    assert not win[3].any()                                                          # the box closes on the last token
    st = E.stats(be, win, valid)
    assert (st["rows"], st["rows_box"], st["weighted"]) == (4, 3, 6) and st["weighted_frac"] == round(6 / 36, 6)


def test_params_from_the_environment(monkeypatch=None):
    old = {k: os.environ.pop(k, None) for k in ("OPD_EOSW_W", "OPD_EOSW_N", "OPD_EOSW")}
    try:
        assert E.params() == (64.0, 16) and not E.active()
        os.environ.update({"OPD_EOSW_W": "8", "OPD_EOSW_N": "4", "OPD_EOSW": "1"})
        assert E.params() == (8.0, 4) and E.active()
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


# ---- torch (the node; on a machine without torch these return early, probe_eosw.sh fails when torch is missing) ---------------------
def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _ppo_loss(torch, log_prob, old_log_prob, adv, mask, cl=0.2, ch=0.2, c=3.0):
    """verl 0.6.1's compute_policy_loss_vanilla (clip + dual clip) with agg_loss token-mean."""
    ratio = torch.exp(log_prob - old_log_prob)
    l1 = -adv * ratio
    l2 = -adv * torch.clamp(ratio, 1 - cl, 1 + ch)
    c1 = torch.maximum(l1, l2)
    c2 = torch.min(-adv * c, c1)
    pg = torch.where(adv < 0, c2, c1)
    return (pg * mask).sum() / (mask.sum() + 1e-8), pg


def _batch(torch, B=4, L=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    old = -torch.rand(B, L, generator=g, dtype=torch.float64) * 3
    lp = old + (torch.rand(B, L, generator=g, dtype=torch.float64) - 0.5) * 0.8      # ratios on both sides of the clip
    adv = torch.randn(B, L, generator=g, dtype=torch.float64) * 2
    mask = torch.ones(B, L, dtype=torch.float64); mask[1, 30:] = 0; mask[3, 12:] = 0
    return lp, old, adv, mask


def test_weights_torch_match_numpy():
    torch = _torch()
    if torch is None:
        return
    valid = np.array([[1] * 12, [1] * 7 + [0] * 5, [1] * 12])
    be = np.array([1, 2, -1])
    wn, _ = E.weights(be, valid, 64.0, 16, np)
    wt, _ = E.weights(torch.tensor(be), torch.tensor(valid), 64.0, 16, torch)
    assert wt.dtype == torch.float32 and torch.allclose(wt, torch.tensor(wn), rtol=1e-6, atol=1e-7)


def test_unit_weights_leave_the_loss_unchanged():
    torch = _torch()
    if torch is None:
        return
    lp, old, adv, mask = _batch(torch)
    base, _ = _ppo_loss(torch, lp, old, adv, mask)
    ones, _ = E.weights(torch.full((4,), -1, dtype=torch.long), mask, 64.0, 16, torch)          # no box: every weight 1
    w1, _ = E.weights(torch.tensor([5, 2, 9, 0]), mask, 1.0, 16, torch)                         # a window, but w = 1
    for w in (ones, w1):
        got, _ = _ppo_loss(torch, lp, old, adv * w.to(adv.dtype), mask)
        assert torch.allclose(got, base, rtol=1e-6, atol=1e-7), (got, base)


def test_advantage_factor_is_a_per_token_loss_factor():
    torch = _torch()
    if torch is None:
        return
    lp, old, adv, mask = _batch(torch, seed=1)
    w, win = E.weights(torch.tensor([3, 10, 20, 1]), mask, 64.0, 16, torch)
    w = w.to(adv.dtype)
    weighted, _ = _ppo_loss(torch, lp, old, adv * w, mask)
    _, pg = _ppo_loss(torch, lp, old, adv, mask)
    direct = (pg * w * mask).sum() / (mask.sum() + 1e-8)                               # the token-mean's denominator unweighted
    assert torch.allclose(weighted, direct, rtol=1e-6, atol=1e-7)
    assert win.any() and (adv[win] < 0).any() and (adv[win] > 0).any()                # both clip branches inside the window


def test_window_tokens_get_w_times_the_gradient():
    torch = _torch()
    if torch is None:
        return
    lp, old, adv, mask = _batch(torch, seed=2)
    w, win = E.weights(torch.tensor([4, 0, 15, 2]), mask, 64.0, 16, torch)
    g = []
    for f in (torch.ones_like(adv), w.to(adv.dtype)):
        x = lp.clone().requires_grad_(True)
        loss, _ = _ppo_loss(torch, x, old, adv * f, mask)
        loss.backward(); g.append(x.grad)
    assert torch.allclose(g[1][win], 64 * g[0][win], rtol=1e-6, atol=1e-7)
    assert torch.allclose(g[1][~win], g[0][~win], rtol=1e-6, atol=1e-7)
