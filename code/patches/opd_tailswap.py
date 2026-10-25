"""Tail-swap OPD (runbook [v2.5.9n], T4; the reviewer 12:5x / 14:1x UTC 10-21): the full form of the stop merge (Q-038).

P5's teacher (the 8B on the thinking template, '<think>\\n' appended) reads the whole non-thinking answer as if it were still
thinking; after the answer's first complete \\boxed{...} that reading keeps wanting more thought (it stops with </think>,
not with the answer's own end), so the student learns to keep writing after the box. Tail swap: up to and including the
first box the teacher's log-probs are P5's (thinking reading); after it they come from a second reference pass of the same
8B on the non-thinking template (the student's own template: P2's reading), which ends an answer the way an answer ends.
Rows without a complete box keep P5's reading everywhere.
[v2.5.9z73] (Q-061, budget-position routing) OPD_TAILSWAP_BUDGET=N: the switch at min(the first box, N): the first N response
tokens keep P5's reading unless a box closes earlier, so a row without a box (or with a late one) longer than N is read
non-thinking from token N on. Logged too: the rows switched by a box / by the budget, the switch positions (p10 / p50 / p90 /
max of each switched row's first swapped index) and the rows whose first swap lies beyond N (budget_late, 0 by construction).

Hooked into verl's RayPPOTrainer.fit right after the reference log-probs of the re-tokenized teacher prompt
(tailswap_patch.py); a no-op unless OPD_TAILSWAP=1. The second pass re-uses prepare_ref_model_inputs with OPD_REF_THINK=0
for the call (P2's prompt) and the same reference worker; P5's reference inputs are put back afterwards.
Logged per step (metrics tail/*): rows, rows with a box, positions swapped, the mean change of the teacher's log-prob there
(non-thinking - thinking), whether both readings are finite on every valid token, violations (swaps at or before a box);
the same per step as JSON lines in OPD_TAILSWAP_LOG (optional).
[v2.5.9z83] (Q-073, Fable 10-24 19:51 Beijing) OPD_TAILSWAP_SOFTMIX = w in (0, 1): no swap - at every valid response position the
reference log-prob is w * the thinking reading + (1 - w) * the non-thinking one (the unnormalized geometric mean of the two teachers:
per position its normalizer only shifts every sampled token's advantage by a constant, unbiased in expectation; a known approximation);
logged: the mean (non-thinking - thinking), whether the merge lies between the two readings and is finite on every valid token.
"""
import json
import os


def active():
    return os.environ.get("OPD_TAILSWAP") == "1"


def budget():
    """[v2.5.9z73] OPD_TAILSWAP_BUDGET (tokens; 0 or unset: the switch at the box only)."""
    try:
        return max(0, int(os.environ.get("OPD_TAILSWAP_BUDGET", "0") or 0))
    except ValueError:
        return 0


def softmix_w():
    """[v2.5.9z83] OPD_TAILSWAP_SOFTMIX as a weight in (0, 1); anything else (unset, 0, 1, not a number): 0.0, the swap."""
    try:
        w = float(os.environ.get("OPD_TAILSWAP_SOFTMIX", "0") or 0)
    except ValueError:
        return 0.0
    return w if 0.0 < w < 1.0 else 0.0


def softmix(lp_think, lp_plain, valid, xp, w=0.5):
    """[v2.5.9z83] (merged, stats): w * lp_think + (1 - w) * lp_plain on the valid positions (the rest keep the thinking reading)."""
    v = valid > 0
    out = xp.where(v, w * lp_think + (1.0 - w) * lp_plain, lp_think)
    nv = int(v.sum())
    lo, hi = xp.minimum(lp_think, lp_plain), xp.maximum(lp_think, lp_plain)
    fin = (lambda a: bool(xp.isfinite(a[v]).all())) if nv else (lambda a: True)
    between = bool(((out >= lo - 1e-6) & (out <= hi + 1e-6))[v].all()) if nv else True
    stats = {"rows": int(lp_think.shape[0]), "valid": nv, "softmix_w": float(w),
             "delta_mean": float((lp_plain - lp_think)[v].mean()) if nv else 0.0,
             "lp_think_finite": int(fin(lp_think)), "lp_plain_finite": int(fin(lp_plain)),
             "merged_finite": int(fin(out)), "merged_between": int(between)}
    return out, stats


def _q(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else -1


def swap(lp_think, lp_plain, box_end, valid, xp, budget=0):
    """(merged, stats). (B, L) log-probs of the two readings, box_end (B,) (-1: no box), valid (B, L) the response mask;
    xp = numpy or torch. Positions after the box (index > box_end >= 0) and valid take the non-thinking reading; [v2.5.9z73]
    with budget N > 0 after min(box_end, N - 1) (index N - 1 is the last one read thinking)."""
    B, L = lp_think.shape
    pos = xp.arange(L).reshape(1, L)
    be = box_end.reshape(B, 1)
    if budget > 0:   # [v2.5.9z73] the switch point: the box when it closes before the budget, else the budget
        cap = be * 0 + (budget - 1)
        by_box = (be >= 0) & (be < budget - 1)
        sw = xp.where(by_box, be, cap)
    else:
        by_box, sw = be >= 0, be
    m = (pos > sw) & (sw >= 0) & (valid > 0)
    out = xp.where(m, lp_plain, lp_think)
    v = valid > 0
    n = int(m.sum())
    fin = (lambda a: bool(xp.isfinite(a[v]).all())) if int(v.sum()) else (lambda a: True)
    stats = {"rows": int(B), "rows_box": int((box_end >= 0).sum()), "swapped": n, "valid": int(v.sum()),
             "swapped_frac": round(n / max(int(v.sum()), 1), 6),
             "delta_mean": float((lp_plain - lp_think)[m].mean()) if n else 0.0,
             "lp_think_finite": int(fin(lp_think)), "lp_plain_finite": int(fin(lp_plain)),
             "violations": int((m & ~((pos > sw) & (sw >= 0))).sum())}
    if budget > 0:   # [v2.5.9z73] who switched where
        rows_sw = m.sum(1) > 0
        far = (m * (L - pos)).max(1)
        first = [L - int(x) for x in (far if xp.__name__ == "numpy" else far.values).tolist()]
        hit = [f for f, r in zip(first, rows_sw.reshape(B).tolist()) if r]
        bb = by_box.reshape(B).tolist()
        stats.update({"budget": int(budget), "rows_switched": len(hit),
                      "rows_by_box": sum(1 for r, b in zip(rows_sw.reshape(B).tolist(), bb) if r and b),
                      "rows_by_budget": sum(1 for r, b in zip(rows_sw.reshape(B).tolist(), bb) if r and not b),
                      "switch_p10": _q(hit, 0.1), "switch_p50": _q(hit, 0.5), "switch_p90": _q(hit, 0.9), "switch_max": max(hit) if hit else -1,
                      "budget_late": sum(1 for f in hit if f > budget)})
    return out, stats


def second_pass(trainer, batch, apply_chat_template_kwargs):
    """The hook: (batch with the swapped ref_log_prob, metrics). trainer = the RayPPOTrainer (its ref tokenizer and workers)."""
    import torch
    from verl.trainer.ppo.ref_input_utils import prepare_ref_model_inputs
    import opd_stopmerge as _sm
    responses, valid = batch.batch["responses"], batch.batch["response_mask"]
    w = softmix_w()   # [v2.5.9z83] (Q-073) the soft mix needs no box
    box = None if w else torch.tensor(_sm.box_ends(responses, valid, trainer.ref_tokenizer), dtype=torch.long)
    lp_think = batch.batch["ref_log_prob"].clone()
    keep = {k: batch.batch[k].clone() for k in ("ref_input_ids", "ref_attention_mask", "ref_position_ids") if k in batch.batch.keys()}
    old = os.environ.get("OPD_REF_THINK"); os.environ["OPD_REF_THINK"] = "0"   # P2's prompt for this call only
    try:
        batch = prepare_ref_model_inputs(batch=batch, ref_tokenizer=trainer.ref_tokenizer, apply_chat_template_kwargs=apply_chat_template_kwargs)
    finally:
        os.environ.pop("OPD_REF_THINK") if old is None else os.environ.__setitem__("OPD_REF_THINK", old)
    wg = trainer.actor_rollout_wg if trainer.ref_in_actor else trainer.ref_policy_wg
    lp_plain = wg.compute_ref_log_prob(batch).batch["ref_log_prob"].to(lp_think.device, lp_think.dtype)
    for k, t in keep.items():   # P5's reference inputs back (nothing later reads them, but the batch stays as P5 leaves it)
        batch.batch[k] = t
    if w:   # [v2.5.9z83] (Q-073) every position: w * thinking + (1 - w) * non-thinking
        merged, stats = softmix(lp_think, lp_plain, valid, torch, w)
    else:
        merged, stats = swap(lp_think, lp_plain, box, valid, torch, budget=budget())   # [v2.5.9z73] 0 unless OPD_TAILSWAP_BUDGET
    batch.batch["ref_log_prob"] = merged   # assigned, not union-ed (union asserts equal values on a shared key)
    p = os.environ.get("OPD_TAILSWAP_LOG")
    if p:
        with open(p, "a") as f:
            f.write(json.dumps(stats) + "\n")
    return batch, {f"tail/{k}": float(v) for k, v in stats.items()}
