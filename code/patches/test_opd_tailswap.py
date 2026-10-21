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
