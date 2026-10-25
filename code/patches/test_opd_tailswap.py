"""[v2.5.9n] (T4) unit tests of tail-swap OPD: the swap (numpy; torch when installed), the trainer hook's text against the real
ray_trainer.py (GOPD_RAY_TRAINER; with the length-selection hook too), and second_pass with stand-ins for verl (torch only).
Run: python -m pytest -q test_opd_tailswap.py, or any plain runner of the test_ functions (PROBE-TAIL runs them on the node)."""
import os
import sys
import types

import numpy as np

import opd_tailswap as S


def _case(xp):
    th = xp.asarray([[-1.0] * 5, [-2.0] * 5, [-3.0] * 5])
    pl = xp.asarray([[-0.5] * 5, [-0.7] * 5, [-0.9] * 5])
    box = xp.asarray([1, -1, 2])
    valid = xp.asarray([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    return th, pl, box, valid


def test_swap_numpy():
    th, pl, box, valid = _case(np)
    out, st = S.swap(th, pl, box, valid, np)
    assert out[0].tolist() == [-1.0, -1.0, -0.5, -0.5, -0.5]   # the box closes at 1: 2..4 from the non-thinking reading
    assert out[1].tolist() == [-2.0] * 5                        # no box: P5's reading everywhere
    assert out[2].tolist() == [-3.0, -3.0, -3.0, -0.9, -3.0]   # the padding (last) is never swapped
    assert (st["rows"], st["rows_box"], st["swapped"], st["valid"], st["violations"]) == (3, 2, 4, 13, 0)
    assert abs(st["delta_mean"] - (0.5 * 3 + 2.1) / 4) < 1e-9 and st["lp_think_finite"] == 1 and st["lp_plain_finite"] == 1


def test_swap_flags_a_non_finite_reading():
    th, pl, box, valid = _case(np)
    pl = pl.copy(); pl[0, 3] = -np.inf
    _, st = S.swap(th, pl, box, valid, np)
    assert st["lp_plain_finite"] == 0 and st["lp_think_finite"] == 1


def test_swap_torch():
    try:
        import torch
    except ImportError:   # the Mac has no torch; PROBE-TAIL runs this on the node
        return
    th, pl, box, valid = _case(torch)
    out, st = S.swap(th, pl, box, valid, torch)
    assert out[0].tolist() == [-1.0, -1.0, -0.5, -0.5, -0.5] and st["swapped"] == 4 and st["violations"] == 0


def test_hook_text_on_the_real_trainer():   # GOPD_RAY_TRAINER=<G-OPD's ray_trainer.py> (the node: the installed one)
    import lensel_patch as LP
    import tailswap_patch as P
    rt = os.environ.get("GOPD_RAY_TRAINER")
    if not rt:
        return
    src = open(rt).read()
    if (P.MARK in src or LP.MARK in src) and os.path.exists(rt + ".orig"):   # [v2.5.9z80] an arm patched the installed file already
        src = open(rt + ".orig").read()                                       # (PROBE-TAIL-BUD 10-24): the unpatched copy patch_file kept
    for base in (src, LP.patch_text(src)[0] if LP.MARK not in src else src):   # alone and with the length-selection hook
        new, changed = P.patch_text(base)
        assert changed and new.count(P.MARK) == 1
        compile(new, rt, "exec")
        i = new.index(P.MARK); assert "self.ref_policy_wg.compute_ref_log_prob(batch)" in new[i - 600:i]
        assert P.patch_text(new) == (new, False)   # idempotent
        if LP.MARK in base:
            assert new.count(LP.MARK) == 1


def test_second_pass_with_stand_ins():
    try:
        import torch
    except ImportError:
        return
    seen = {}

    def prepare(batch, ref_tokenizer, apply_chat_template_kwargs):   # P2's prompt: records the env it ran under
        seen["think"] = os.environ.get("OPD_REF_THINK"); batch.batch["ref_input_ids"] = torch.full((2, 4), 7); return batch
    mod = types.ModuleType("verl.trainer.ppo.ref_input_utils"); mod.prepare_ref_model_inputs = prepare
    for name in ("verl", "verl.trainer", "verl.trainer.ppo"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["verl.trainer.ppo.ref_input_utils"] = mod

    class Tok:
        V = {1: "x", 2: "\\boxed{", 3: "4", 4: "}", 5: " ok"}
        def batch_decode(self, rows):
            return [self.V[r[0]] for r in rows]

    class Out:
        def __init__(self, t): self.batch = {"ref_log_prob": t}

    class WG:
        def compute_ref_log_prob(self, b):
            assert b.batch["ref_input_ids"].eq(7).all(); return Out(torch.full((2, 5), -0.25))

    class B:
        def __init__(self): self.batch = {}

    b = B()
    b.batch.update(responses=torch.tensor([[1, 2, 3, 4, 5], [1, 5, 5, 5, 5]]), response_mask=torch.ones(2, 5, dtype=torch.long),
                   ref_log_prob=torch.full((2, 5), -2.0), ref_input_ids=torch.full((2, 4), 3),
                   ref_attention_mask=torch.ones(2, 4, dtype=torch.long), ref_position_ids=torch.arange(4).repeat(2, 1))
    tr = types.SimpleNamespace(ref_tokenizer=Tok(), ref_in_actor=False, ref_policy_wg=WG(), actor_rollout_wg=None)
    os.environ["OPD_REF_THINK"] = "1"
    try:
        b, m = S.second_pass(tr, b, {})
        assert seen["think"] == "0" and os.environ["OPD_REF_THINK"] == "1"   # P2's prompt for the second pass only
    finally:
        os.environ.pop("OPD_REF_THINK", None)
    assert b.batch["ref_log_prob"][0].tolist() == [-2.0, -2.0, -2.0, -2.0, -0.25]   # the box closes at token 3
    assert b.batch["ref_log_prob"][1].tolist() == [-2.0] * 5 and b.batch["ref_input_ids"].eq(3).all()   # P5's inputs back
    assert m["tail/rows_box"] == 1.0 and m["tail/swapped"] == 1.0 and m["tail/violations"] == 0.0


def _bud_case(xp):   # [v2.5.9z73] (Q-061) six positions; rows: box at 1, no box, box at 4 (after the budget), padding
    th = xp.asarray([[-1.0] * 6, [-2.0] * 6, [-3.0] * 6, [-4.0] * 6])
    pl = xp.asarray([[-0.5] * 6, [-0.7] * 6, [-0.9] * 6, [-0.1] * 6])
    box = xp.asarray([1, -1, 4, -1])
    valid = xp.asarray([[1] * 6, [1] * 6, [1] * 6, [1, 1, 1, 0, 0, 0]])
    return th, pl, box, valid


def test_budget_switch_numpy():   # budget 3: tokens 0..2 thinking unless the box closes earlier
    th, pl, box, valid = _bud_case(np)
    out, st = S.swap(th, pl, box, valid, np, budget=3)
    assert out[0].tolist() == [-1.0, -1.0, -0.5, -0.5, -0.5, -0.5]     # the box (1) before the budget: as the tail swap
    assert out[1].tolist() == [-2.0, -2.0, -2.0, -0.7, -0.7, -0.7]     # no box: from the budget on
    assert out[2].tolist() == [-3.0, -3.0, -3.0, -0.9, -0.9, -0.9]     # a box after the budget: the budget wins
    assert out[3].tolist() == [-4.0] * 6                                 # three valid tokens, no box: nothing to switch
    assert (st["budget"], st["rows_switched"], st["rows_by_box"], st["rows_by_budget"], st["budget_late"], st["violations"]) == (3, 3, 1, 2, 0, 0)
    assert (st["switch_p10"], st["switch_p50"], st["switch_p90"], st["switch_max"]) == (2, 3, 3, 3)
    assert st["swapped"] == 4 + 3 + 3


def test_budget_zero_is_the_tail_swap():
    th, pl, box, valid = _bud_case(np)
    a, sa = S.swap(th, pl, box, valid, np)
    b, sb = S.swap(th, pl, box, valid, np, budget=0)
    assert a.tolist() == b.tolist() and sa == sb and "budget" not in sa
    assert a[1].tolist() == [-2.0] * 6 and a[2].tolist() == [-3.0] * 5 + [-0.9]


def test_budget_from_the_environment():
    old = os.environ.get("OPD_TAILSWAP_BUDGET")
    try:
        os.environ["OPD_TAILSWAP_BUDGET"] = "2048"; assert S.budget() == 2048
        os.environ["OPD_TAILSWAP_BUDGET"] = "x"; assert S.budget() == 0
        os.environ.pop("OPD_TAILSWAP_BUDGET"); assert S.budget() == 0
    finally:
        os.environ.pop("OPD_TAILSWAP_BUDGET", None) if old is None else os.environ.__setitem__("OPD_TAILSWAP_BUDGET", old)


def test_budget_switch_torch():
    try:
        import torch
    except ImportError:   # the Mac has no torch; PROBE-TAIL-BUD runs this on the node
        return
    th, pl, box, valid = _bud_case(torch)
    out, st = S.swap(th, pl, box, valid, torch, budget=3)
    ref, sref = S.swap(*_bud_case(np), np, budget=3)
    # [v2.5.9z80] float32 against float64 (PROBE-TAIL-BUD 10-24: -0.7 and -0.9 are not exact in float32), no exact compare
    assert max(abs(a - b) for ra, rb in zip(out.tolist(), ref.tolist()) for a, b in zip(ra, rb)) < 1e-6
    assert abs(st["delta_mean"] - sref["delta_mean"]) < 1e-5
    assert {k: v for k, v in st.items() if k != "delta_mean"} == {k: v for k, v in sref.items() if k != "delta_mean"}


def test_softmix_numpy():   # [v2.5.9z83] (Q-073) 0.5 / 0.5 on the valid positions, the rest thinking; the merge between and finite
    th = np.array([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]]); pl = np.array([[-3.0, -1.0, -3.0], [-0.1, -0.9, -2.0]])
    valid = np.array([[1, 1, 1], [1, 1, 0]])
    out, st = S.softmix(th, pl, valid, np, 0.5)
    assert out.tolist() == [[-2.0, -1.5, -3.0], [-0.3, -0.7, -0.5]]
    assert (st["valid"], st["softmix_w"], st["merged_between"], st["merged_finite"]) == (5, 0.5, 1, 1)
    assert abs(st["delta_mean"] - float(np.mean([-2.0, 1.0, 0.0, 0.4, -0.4]))) < 1e-12
    out, st = S.softmix(th, pl, valid, np, 0.25)                                   # w is the thinking reading's share
    assert abs(out[0, 0] - (0.25 * -1.0 + 0.75 * -3.0)) < 1e-12 and st["merged_between"] == 1
    pl2 = pl.copy(); pl2[0, 1] = -np.inf
    assert S.softmix(th, pl2, valid, np, 0.5)[1]["merged_finite"] == 0           # a non-finite reading shows in the merge


def test_softmix_from_the_environment():
    old = os.environ.get("OPD_TAILSWAP_SOFTMIX")
    try:
        for v, w in (("0.5", 0.5), ("0", 0.0), ("1", 0.0), ("x", 0.0), ("", 0.0)):
            os.environ["OPD_TAILSWAP_SOFTMIX"] = v; assert S.softmix_w() == w, v
        os.environ.pop("OPD_TAILSWAP_SOFTMIX"); assert S.softmix_w() == 0.0
    finally:
        os.environ.pop("OPD_TAILSWAP_SOFTMIX", None) if old is None else os.environ.__setitem__("OPD_TAILSWAP_SOFTMIX", old)


def test_softmix_torch():
    try:
        import torch
    except ImportError:   # the Mac has no torch; PROBE-SOFTMIX runs this on the node
        return
    th = torch.tensor([[-1.0, -2.0, -3.0]]); pl = torch.tensor([[-3.0, -1.0, -3.0]]); valid = torch.tensor([[1, 1, 0]])
    out, st = S.softmix(th, pl, valid, torch, 0.5)
    assert torch.allclose(out, torch.tensor([[-2.0, -1.5, -3.0]])) and st["merged_between"] == 1 and st["merged_finite"] == 1


def test_second_pass_softmix_with_stand_ins():   # [v2.5.9z83] (Q-073) the merge, no swap
    try:
        import torch
    except ImportError:
        return
    seen = {}

    def prepare(batch, ref_tokenizer, apply_chat_template_kwargs):   # P2's prompt: records the env it ran under
        seen["think"] = os.environ.get("OPD_REF_THINK"); batch.batch["ref_input_ids"] = torch.full((2, 4), 7); return batch
    mod = types.ModuleType("verl.trainer.ppo.ref_input_utils"); mod.prepare_ref_model_inputs = prepare
    for name in ("verl", "verl.trainer", "verl.trainer.ppo"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["verl.trainer.ppo.ref_input_utils"] = mod

    class Tok:
        V = {1: "x", 2: "\\boxed{", 3: "4", 4: "}", 5: " ok"}
        def batch_decode(self, rows):
            return [self.V[r[0]] for r in rows]

    class Out:
        def __init__(self, t): self.batch = {"ref_log_prob": t}

    class WG:
        def compute_ref_log_prob(self, b):
            assert b.batch["ref_input_ids"].eq(7).all(); return Out(torch.full((2, 5), -0.25))

    class B:
        def __init__(self): self.batch = {}

    b = B()
    b.batch.update(responses=torch.tensor([[1, 2, 3, 4, 5], [1, 5, 5, 5, 5]]), response_mask=torch.ones(2, 5, dtype=torch.long),
                   ref_log_prob=torch.full((2, 5), -2.0), ref_input_ids=torch.full((2, 4), 3),
                   ref_attention_mask=torch.ones(2, 4, dtype=torch.long), ref_position_ids=torch.arange(4).repeat(2, 1))
    tr = types.SimpleNamespace(ref_tokenizer=Tok(), ref_in_actor=False, ref_policy_wg=WG(), actor_rollout_wg=None)
    os.environ["OPD_REF_THINK"] = "1"; os.environ["OPD_TAILSWAP_SOFTMIX"] = "0.5"
    try:
        b, m = S.second_pass(tr, b, {})
        assert seen["think"] == "0" and os.environ["OPD_REF_THINK"] == "1"   # P2's prompt for the second pass only
    finally:
        os.environ.pop("OPD_REF_THINK", None); os.environ.pop("OPD_TAILSWAP_SOFTMIX", None)
    assert b.batch["ref_log_prob"].tolist() == [[-1.125] * 5, [-1.125] * 5]   # 0.5 * -2.0 + 0.5 * -0.25 everywhere
    assert m["tail/softmix_w"] == 0.5 and m["tail/merged_between"] == 1.0 and "tail/swapped" not in m
