"""[v2.5.9z89] (Q-080b) unit tests of opd_finished.py: which rows finished, the mask, the all-unfinished case, the alert (numpy; torch
when installed); on the node (verl importable) also verl's own token-mean aggregation: the loss with one row masked equals the loss
without that row, an all-zero mask aggregates to a finite value, and the actor's update reads response_mask from the batch.
Run: python -m pytest -q test_opd_finished.py, or any plain runner of the test_ functions (PROBE-VLM-TAIL runs them on the node)."""
import inspect
import os

import numpy as np

import opd_finished as F

EOS = [9]


def _case(xp):   # L = 5; rows: stopped early (n=3), hit the limit without an end token, hit the limit on an end token, empty
    r = xp.asarray([[1, 2, 9, 0, 0], [1, 2, 3, 4, 5], [1, 2, 3, 4, 9], [0, 0, 0, 0, 0]])
    v = xp.asarray([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
    return r, v


def test_finished_rows_numpy():
    r, v = _case(np)
    assert F.finished_rows(r, v, EOS, np).tolist() == [True, False, True, False]


def test_mask_numpy():
    r, v = _case(np)
    new, st = F.mask(r, v, EOS, np)
    assert new.tolist() == [[1, 1, 1, 0, 0], [0] * 5, [1] * 5, [0] * 5]
    assert st == {"rows": 4, "unfinished": 2, "unfinished_frac": 0.5, "all_unfinished": 0}
    assert v.tolist()[1] == [1] * 5                                                    # the input is not changed in place


def test_no_finished_row_masks_nothing():
    r = np.asarray([[1, 2, 3], [4, 5, 6]]); v = np.ones((2, 3), dtype=int)
    new, st = F.mask(r, v, EOS, np)
    assert new.tolist() == v.tolist() and st["all_unfinished"] == 1 and st["unfinished_frac"] == 1.0


def test_mask_torch():
    try:
        import torch
    except ImportError:
        return
    r, v = _case(torch)
    r, v = torch.as_tensor(r), torch.as_tensor(v)
    new, st = F.mask(r, v, EOS, torch)
    assert new.tolist() == [[1, 1, 1, 0, 0], [0] * 5, [1] * 5, [0] * 5] and new.dtype == v.dtype and st["unfinished"] == 2


def test_alert_after_three_high_steps_once():
    F._RUN.update(high=0, alerted=False)
    os.environ.pop("RUNBOOK", None)
    assert [F.alert(x, "T") for x in (0.95, 0.5, 0.91, 0.92, 0.93, 0.99)] == [False, False, False, False, True, False]


def test_end_ids():
    class Tok:
        eos_token_id = 9
        def convert_tokens_to_ids(self, t):
            return {"<|im_end|>": 9, "<|endoftext|>": 7}.get(t, -1)
    assert F.end_ids(Tok()) == [7, 9]


def _verl():
    try:
        from verl.trainer.ppo.core_algos import agg_loss
        import torch
        return agg_loss, torch
    except Exception:
        return None, None


def test_verl_token_mean_drops_a_masked_row():   # the node: the loss with a masked row = the loss without it (Fable's condition)
    agg, torch = _verl()
    if agg is None:
        return
    g = torch.Generator().manual_seed(0)
    loss = torch.rand(4, 6, generator=g); m = torch.ones(4, 6); m[0, 4:] = 0; m[2, 3:] = 0
    m2 = m.clone(); m2[1] = 0
    a = agg(loss_mat=loss, loss_mask=m2, loss_agg_mode="token-mean")
    b = agg(loss_mat=loss[[0, 2, 3]], loss_mask=m[[0, 2, 3]], loss_agg_mode="token-mean")
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6), (float(a), float(b))
    z = agg(loss_mat=loss, loss_mask=torch.zeros(4, 6), loss_agg_mode="token-mean")
    assert bool(torch.isfinite(z)), float(z)


def test_verl_an_all_masked_micro_batch_is_finite_with_zero_gradient():   # the node: Fable 23:2x UTC, the empty micro-batch guard
    agg, torch = _verl()
    if agg is None:
        return
    loss = torch.rand(3, 5, requires_grad=True)
    z = agg(loss_mat=loss, loss_mask=torch.zeros(3, 5), loss_agg_mode="token-mean")
    z.backward()
    assert bool(torch.isfinite(z)) and loss.grad is not None and bool((loss.grad == 0).all()), (float(z), loss.grad)


def test_empty_mb_frac_without_verl_or_a_budget_is_minus_one():
    os.environ.pop("OPD_FINISHED_MB_TOKENS", None)
    assert F.empty_mb_frac(None, np.zeros((2, 3))) == -1.0


def test_empty_mb_frac_from_verls_partition():   # the node (verl and torch): one all-masked micro-batch among the partition's
    try:
        import torch
        from tensordict import TensorDict
        from verl.utils.seqlen_balancing import rearrange_micro_batches  # noqa: F401
    except Exception:
        return
    am = torch.ones(4, 10, dtype=torch.long)
    td = TensorDict({"attention_mask": am}, batch_size=[4])
    os.environ["OPD_FINISHED_MB_TOKENS"] = "10"   # one row a micro-batch
    try:
        v = torch.ones(4, 6, dtype=torch.long); v[1] = 0
        assert F.empty_mb_frac(td, v) == 0.25
    finally:
        os.environ.pop("OPD_FINISHED_MB_TOKENS", None)


def test_verl_actor_reads_the_batch_response_mask():   # the node: the update uses the batch's response_mask, not the attention mask
    try:
        import verl.workers.actor.dp_actor as A
    except Exception:
        return
    src = inspect.getsource(A)
    assert '"response_mask"' in src or "'response_mask'" in src, "dp_actor does not read response_mask from the batch"
