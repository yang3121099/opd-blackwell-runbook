"""Gate G2: unit tests for advantage variants. Run: python -m pytest -q test_opd_variants.py"""
import torch
from opd_variants import opd_k1, bin_permuted_teacher, bin_mean_teacher, fixed_negative, compute, adv_stats


def _toy(B=3, T=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    old = -torch.rand(B, T, generator=g) * 5          # student log-probs in [-5, 0]
    ref = old + 0.3 * torch.randn(B, T, generator=g)  # teacher correlated with student
    mask = torch.ones(B, T)
    mask[1, 150:] = 0                                  # a shorter response
    return old, ref, mask, g


def test_k1_sign_and_mask():
    old, ref, mask, _ = _toy()
    a = opd_k1(old, ref, mask)
    assert torch.all(a[1, 150:] == 0)
    i = (ref - old > 0) & (mask > 0)
    assert torch.all(a[i] > 0)  # teacher likes the token more than student -> positive advantage


def test_bin_permutation_preserves_bin_stats_and_breaks_alignment():
    old, ref, mask, g = _toy()
    a = bin_permuted_teacher(old, ref, mask, n_bins=10, generator=g)
    perm_ref = a + old
    assert torch.all(a[1, 150:] == 0)
    for b in range(old.shape[0]):
        idx = torch.nonzero(mask[b] > 0).squeeze(-1)
        order = torch.argsort(old[b, idx])
        for chunk in torch.chunk(order, 10):
            pos = idx[chunk]
            # same multiset of teacher values inside each student-confidence bin
            assert torch.allclose(torch.sort(perm_ref[b, pos]).values, torch.sort(ref[b, pos]).values)
    # per-token alignment is reduced: residual correlation (beyond student log-prob) drops
    r_true = torch.corrcoef(torch.stack([(ref - old)[mask > 0], ref[mask > 0]]))[0, 1]
    r_perm = torch.corrcoef(torch.stack([(perm_ref - old)[mask > 0], ref[mask > 0]]))[0, 1]
    assert r_perm < r_true


def test_fixed_negative_selects_lowest_fraction():
    old, _, mask, _ = _toy()
    adv, sel = fixed_negative(old, mask, frac=0.2, value=-0.5)
    for b in range(old.shape[0]):
        n = int(mask[b].sum())
        assert int(sel[b].sum()) == round(0.2 * n)
        chosen = old[b][sel[b] > 0]
        rest = old[b][(sel[b] == 0) & (mask[b] > 0)]
        assert chosen.max() <= rest.min()
        assert torch.all(adv[b][sel[b] > 0] == -0.5)
    assert torch.all(sel[1, 150:] == 0)


def test_compute_dispatch():
    old, ref, mask, g = _toy()
    for v in ["P2", "P3", "P4", "P5"]:
        adv, lm = compute(v, old, ref, mask, generator=g)
        assert adv.shape == old.shape and lm.shape == old.shape


def test_adv_stats_and_p3_diagnostics():
    from opd_variants import adv_stats
    old, ref, mask, g = _toy()
    a0 = opd_k1(old, ref, mask)
    a3, lm3 = compute("P3", old, ref, mask, generator=g)
    s3 = adv_stats("P3", a0, a3, mask, lm3)
    assert 0 <= s3["opd/p3_r2"] <= 1 and "opd/p3_var_ratio" in s3
    a4, lm4 = compute("P4", old, ref, mask)
    s4 = adv_stats("P4", a0, a4, mask, lm4)
    assert abs(s4["opd/loss_token_frac"] - 0.2) < 0.02  # loss only on the selected 20%
    assert abs(s4["opd/adv_abs_mean"] - 0.5) < 1e-6
    assert lm4.sum() < mask.sum()  # response_mask itself is untouched by the caller


def test_p3_seeded_generator_reproducible():
    import os, importlib, opd_variants
    os.environ["OPD_SEED"] = "7"
    opd_variants._GENS.clear()
    old, ref, mask, _ = _toy()
    a1, _ = opd_variants.compute("P3", old, ref, mask)
    opd_variants._GENS.clear()
    a2, _ = opd_variants.compute("P3", old, ref, mask)
    assert torch.equal(a1, a2)


def test_p6_sparse_topk_partitions_p2():
    old, ref, mask, _ = _toy()
    a2, _ = compute("P2", old, ref, mask)
    a6, lm6 = compute("P6", old, ref, mask)
    a6r, lm6r = compute("P6r", old, ref, mask)
    assert torch.equal(lm6, mask) and torch.equal(lm6r, mask)       # loss mask = full response
    assert torch.allclose(a6 + a6r, a2)                              # top + rest = P2
    for b in range(old.shape[0]):
        n = int(mask[b].sum())
        assert int((a6[b] != 0).sum()) <= round(0.2 * n)
        if (a6[b] != 0).any() and (a6r[b] != 0).any():
            assert a6[b][a6[b] != 0].abs().min() >= a6r[b][a6r[b] != 0].abs().max()


def test_shadow_and_overlap_diagnostics():
    from opd_variants import adv_stats
    old, ref, mask, _ = _toy()
    a2, lm = compute("P2", old, ref, mask)
    ent = torch.rand_like(old)
    s = adv_stats("P2", a2, a2, mask, lm, entropy=ent, old_log_prob=old, ref_log_prob=ref)
    assert 0 <= s["opd/p3shadow_r2"] <= 1
    assert 0 <= s["opd/topA_vs_topH_overlap"] <= 1
    s_same = adv_stats("P2", a2, a2, mask, lm, entropy=a2.abs(), old_log_prob=old, ref_log_prob=ref)
    assert s_same["opd/topA_vs_topH_overlap"] > 0.99   # identical scores -> full overlap


def test_termination_diagnostics():
    from opd_variants import adv_stats
    old, ref, mask, _ = _toy(B=3, T=200)
    a2, lm = compute("P2", old, ref, mask)
    resp = torch.full(old.shape, 11, dtype=torch.long)
    resp[0, 199] = 151645          # row 0 terminated at the last position
    resp[1, 149] = 151645          # row 1 terminated (mask ends at 150)
    # row 2: no EOS -> truncated
    s = adv_stats("P2", a2, a2, mask, lm, responses=resp)
    assert abs(s["opd/terminated_frac"] - 2 / 3) < 1e-6
    assert abs(s["opd/eos_adv_mean"] - (a2[0, 199] + a2[1, 149]).item() / 2) < 1e-5
    assert abs(s["opd/trunc_tail_adv_mean"] - a2[2, -32:].mean().item()) < 1e-5


def test_p3b_bin_mean_keeps_bin_means_and_removes_within_bin_variation():
    old, ref, mask, _ = _toy()
    a = bin_mean_teacher(old, ref, mask, n_bins=10)
    m_ref = a + old
    assert torch.all(a[1, 150:] == 0)
    for b in range(old.shape[0]):
        idx = torch.nonzero(mask[b] > 0).squeeze(-1)
        for chunk in torch.chunk(torch.argsort(old[b, idx]), 10):
            pos = idx[chunk]
            assert torch.allclose(m_ref[b, pos], ref[b, pos].mean().expand(pos.numel()), atol=1e-6)
    # deterministic (no generator), dispatched via compute, and P3 diagnostics are logged for P3b
    a2, lm = compute("P3b", old, ref, mask)
    assert torch.equal(a, a2) and torch.equal(lm, mask)
    st = adv_stats("P3b", opd_k1(old, ref, mask), a2, mask, lm)
    assert 0 <= st["opd/p3_r2"] <= 1


def test_eos_in_low20_diagnostic():
    old, ref, mask, _ = _toy(B=3, T=200)
    a2, lm = compute("P2", old, ref, mask)
    resp = torch.full(old.shape, 11, dtype=torch.long)
    resp[0, 199] = 151645
    resp[1, 149] = 151645
    old = old.clone()
    old[0, 199] = -50.0            # row 0: EOS is the least confident token -> inside the lowest 20%
    old[1, 149] = 0.0              # row 1: EOS fully confident -> outside
    s = adv_stats("P2", a2, a2, mask, lm, old_log_prob=old, ref_log_prob=ref, responses=resp)
    assert abs(s["opd/eos_in_low20_frac"] - 0.5) < 1e-6
    assert abs(s["opd/eos_student_logp_mean"] - (-25.0)) < 1e-5


def _toy_resp(old, mask):
    resp = torch.full(old.shape, 11, dtype=torch.long)
    resp[0, 199] = 151645          # row 0 terminated at its last valid token
    resp[1, 149] = 151643          # row 1 terminated (mask ends at 150) with <|endoftext|>
    resp[2, 100] = 151668          # row 2: a spontaneous </think> at position 100, then truncated (no EOS at the end)
    return resp


def test_p2noeos_zeroes_only_final_eos():
    old, ref, mask, _ = _toy()
    resp = _toy_resp(old, mask)
    a2, _ = compute("P2", old, ref, mask)
    a, lm = compute("P2noEOS", old, ref, mask, responses=resp)
    assert a[0, 199] == 0 and a[1, 149] == 0 and torch.equal(lm, mask)
    diff = (a != a2) & (mask > 0)
    assert diff.sum() <= 2 and torch.equal(a[2], a2[2])   # truncated row untouched


def test_pstop_keeps_eos_window_only():
    old, ref, mask, _ = _toy()
    resp = _toy_resp(old, mask)
    a2, _ = compute("P2", old, ref, mask)
    a, lm = compute("Pstop", old, ref, mask, responses=resp)
    assert torch.equal(a[0, 191:200], a2[0, 191:200]) and torch.all(a[0, :191] == 0)
    assert torch.equal(a[1, 141:150], a2[1, 141:150]) and torch.all(a[1, :141] == 0)
    assert torch.all(a[2] == 0) and torch.equal(lm, mask)


def test_p6rand_and_top5_fractions():
    old, ref, mask, g = _toy()
    a2, _ = compute("P2", old, ref, mask)
    ar, _ = compute("P6rand", old, ref, mask, generator=g)
    at, _ = compute("P6top5", old, ref, mask)
    for b, n in [(0, 200), (1, 150), (2, 200)]:
        nz = (ar[b] != 0).sum().item(); assert nz == round(0.2 * n)
        assert torch.equal(ar[b][ar[b] != 0], a2[b][ar[b] != 0])
        kt = (at[b] != 0).sum().item(); assert kt == round(0.05 * n)
        thr = a2[b, :n].abs().sort(descending=True).values[kt - 1]
        assert torch.all(at[b][at[b] != 0].abs() >= thr - 1e-6)


def test_p4eospen_forces_first_marker():
    old, ref, mask, _ = _toy()
    resp = _toy_resp(old, mask)
    old = old.clone(); old[0, 199] = 0.0; old[2, 100] = 0.0     # markers fully confident: P4 alone would not pick them
    a, sel = compute("P4EOSpen", old, ref, mask, responses=resp)
    a4, sel4 = compute("P4", old, ref, mask)
    assert a[0, 199] == -0.5 and sel[0, 199] == 1 and a4[0, 199] == 0
    assert a[2, 100] == -0.5 and sel[2, 100] == 1                # </think> counts as the first marker
    assert (sel - sel4).sum() <= 3


def test_termination_marker_diagnostics():
    old, ref, mask, _ = _toy()
    resp = _toy_resp(old, mask)
    old = old.clone(); old[2, 100] = -50.0                        # </think> is the least confident token of row 2
    a2, lm = compute("P2", old, ref, mask)
    s = adv_stats("P2", a2, a2, mask, lm, old_log_prob=old, ref_log_prob=ref, responses=resp)
    assert abs(s["opd/think_close_frac"] - 1 / 3) < 1e-6
    assert 0 <= s["opd/term_in_low20_frac"] <= 1


# v2.5.7h (Q-014): token-selective OPD with the P5 teacher prompt
def test_p5top10_keeps_the_largest_tenth():
    old, ref, mask, _ = _toy()
    a2, _ = compute("P2", old, ref, mask)
    at, lm = compute("P5top10", old, ref, mask)
    assert torch.equal(lm, mask)
    for b, n in [(0, 200), (1, 150), (2, 200)]:
        k = (at[b] != 0).sum().item(); assert k == round(0.1 * n)
        assert torch.equal(at[b][at[b] != 0], a2[b][at[b] != 0])
        thr = a2[b, :n].abs().sort(descending=True).values[k - 1]
        assert torch.all(at[b][at[b] != 0].abs() >= thr - 1e-6)


def test_p5rand10m_matches_the_top10_mass():
    from opd_variants import rand_matched
    old, ref, mask, _ = _toy()
    a2, _ = compute("P2", old, ref, mask)
    ar, _ = compute("P5rand10m", old, ref, mask, generator=torch.Generator().manual_seed(7))
    ar2, _ = compute("P5rand10m", old, ref, mask, generator=torch.Generator().manual_seed(7))
    at, _ = compute("P5top10", old, ref, mask)
    assert torch.equal(ar, ar2)                                         # fixed seed, same selection
    for b, n in [(0, 200), (1, 150), (2, 200)]:
        sel = ar[b] != 0
        assert sel.sum().item() == round(0.1 * n) and not torch.equal(sel, at[b] != 0)
        assert torch.allclose(ar[b].abs().sum(), at[b].abs().sum(), rtol=1e-4)   # same sum of |A| as the top 10%
        sc = ar[b][sel] / a2[b][sel]
        assert torch.allclose(sc, sc[0].expand_as(sc), rtol=1e-4) and sc[0] >= 1   # one scale per response
    z = torch.zeros(1, 10); m = torch.ones(1, 10)
    out, scales, chosen = rand_matched(z, m, 0.1, torch.Generator().manual_seed(0))
    assert torch.all(out == 0) and scales == [1.0] and chosen.sum().item() == 1   # nothing to scale: no division by zero


def test_override_only_through_its_hook():
    import os
    old, ref, mask, _ = _toy()
    p6, _ = compute("P6", old, ref, mask); p6r, _ = compute("P6r", old, ref, mask); top, _ = compute("P5top10", old, ref, mask)
    os.environ["OPD_VARIANT_OVERRIDE"] = "P5top10"
    try:
        a, _ = compute("P6", old, ref, mask); b, _ = compute("P6r", old, ref, mask)
        assert torch.equal(a, top) and torch.equal(b, p6r)              # P6 is the hook; P6r is not touched
    finally:
        del os.environ["OPD_VARIANT_OVERRIDE"]
    assert torch.equal(compute("P6", old, ref, mask)[0], p6)          # no override: P6 as before (20%)


def test_selection_stats_and_token_classes():
    import os
    from opd_variants import token_class
    assert [token_class(151645, ""), token_class(5, "Wait"), token_class(5, " however"), token_class(5, " 3"),
            token_class(5, "\\frac"), token_class(5, "^{"), token_class(5, " the"), token_class(5, " x")] == [0, 1, 1, 2, 2, 2, 3, 3]
    old, ref, mask, _ = _toy()
    resp = _toy_resp(old, mask)
    a2, _ = compute("P2", old, ref, mask)
    os.environ["OPD_VARIANT_OVERRIDE"] = "P5top10"
    try:
        at, lm = compute("P6", old, ref, mask, responses=resp)
        s = adv_stats("P6", a2, at, mask, lm, old_log_prob=old, ref_log_prob=ref, responses=resp)
        assert abs(s["opd/sel_kept_frac"] - 55 / 550) < 1e-6 and 0 <= s["opd/sel_eos_kept_frac"] <= 1
        assert "opd/rand_scale_mean" not in s
        ar, lm = compute("P5rand10m", old, ref, mask, generator=torch.Generator().manual_seed(1), responses=resp)   # not P6's override
    finally:
        del os.environ["OPD_VARIANT_OVERRIDE"]
    s = adv_stats("P5rand10m", a2, ar, mask, lm, old_log_prob=old, ref_log_prob=ref, responses=resp)
    assert s["opd/rand_scale_mean"] >= 1 and abs(s["opd/sel_kept_frac"] - 55 / 550) < 1e-6
    assert s["opd/rand_scale_p10"] <= s["opd/rand_scale_p50"] <= s["opd/rand_scale_p90"] <= s["opd/rand_scale_max"] and s["opd/rand_scale_cap_frac"] == 0
    s = adv_stats("P6", a2, at, mask, lm, old_log_prob=old, ref_log_prob=ref, responses=resp)
    assert "opd/sel_kept_frac" not in s                                 # nothing pending: a real P6 arm logs none of it



def test_kept_share_counts_picked_tokens_with_zero_advantage():   # v2.5.7l: PROBE-TOP10 2026-10-19 00:1x UTC
    import os
    old, ref, mask, _ = _toy()
    ref = ref.clone(); ref[:, ::3] = old[:, ::3]                        # a third of the tokens: teacher = student, A = 0
    resp = _toy_resp(old, mask)
    a2, _ = compute("P2", old, ref, mask)
    for v in ("P5top10", "P5rand10m"):
        a, lm = compute(v, old, ref, mask, generator=torch.Generator().manual_seed(3), responses=resp)
        s = adv_stats(v, a2, a, mask, lm, old_log_prob=old, ref_log_prob=ref, responses=resp)
        assert abs(s["opd/sel_kept_frac"] - 55 / 550) < 1e-6, (v, s["opd/sel_kept_frac"])
