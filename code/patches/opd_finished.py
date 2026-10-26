"""Only finished rollouts count (runbook [v2.5.9z89], Q-080b E15-VLM-P5tail-lo; Fable's design ruling 10-26 23:0x UTC).

A row is finished when its response ended by itself: fewer valid tokens than the response limit (the rollout stopped at an end
token), or its last valid token is an end token (one that stopped exactly at the limit). An unfinished row (truncated at the
limit) gets its whole response_mask set to 0 before the advantages and the update: it adds nothing to the loss and, with verl's
token-mean aggregation, nothing to its denominator either (PROBE-VLM-TAIL checks that property on the node's verl: the loss of a
batch with one row masked equals the loss of the batch without that row). The learning rate is not rescaled. n = 1 rollout per
prompt, so no group statistic changes. If no row of a step finished, nothing is masked that step (a step without a single
finished row would aggregate an empty mask) and the step is logged as all_unfinished.
Logged per step (metrics vlm/*): rows, unfinished rows, unfinished_frac, all_unfinished; vlm/unfinished_frac >= 0.9 on 3 steps in a
row prints FINISHED_ONLY_ALERT and sends one vlm_unfinished_high event (an alert, the training goes on).
[Fable 10-26 23:2x UTC] a micro-batch whose rows are all masked: verl's token-mean then divides 0 by its eps-guarded token count, so the
micro-batch contributes 0 and no gradient (PROBE-VLM-TAIL's node test proves both on the installed verl, fail-closed); vlm/empty_mb_frac
= the share of the update's micro-batches with every row masked, from verl's own partition of this batch (rearrange_micro_batches
over the attention mask at OPD_FINISHED_MB_TOKENS, the actor's ppo_max_token_len_per_gpu; one GPU, the mini-batch = the batch);
-1 when that partition cannot be computed here.
Hooked from opd_tailswap.second_pass (the tail-swap hook, after the reference log-probs, before the advantages); a no-op unless
OPD_FINISHED_ONLY=1.
"""
import os
import subprocess
import sys

_RUN = {"high": 0, "alerted": False}


def active():
    return os.environ.get("OPD_FINISHED_ONLY") == "1"


def finished_rows(responses, valid, eos_ids, xp):
    """(B,) booleans: finished rows. responses / valid (B, L)."""
    B, L = valid.shape
    n = valid.sum(1)
    last = xp.maximum(n - 1, 0 * n)
    idx = xp.arange(B)
    tok = responses[idx, last]
    is_eos = tok == -1
    for e in eos_ids:
        is_eos = is_eos | (tok == int(e))
    return ((n < L) | is_eos) & (n > 0)


def mask(responses, valid, eos_ids, xp):
    """(new valid, stats): unfinished rows' masks set to 0; unchanged when no row finished (all_unfinished)."""
    fin = finished_rows(responses, valid, eos_ids, xp)
    B = int(valid.shape[0]); nf = int(fin.sum())
    stats = {"rows": B, "unfinished": B - nf, "unfinished_frac": round((B - nf) / max(B, 1), 6), "all_unfinished": int(nf == 0)}
    if nf == 0:
        return valid, stats
    w = fin.reshape(B, 1)
    return valid * (w.astype(valid.dtype) if xp.__name__ == "numpy" else w.to(valid.dtype)), stats


def end_ids(tokenizer):
    out = set()
    for t in ([getattr(tokenizer, "eos_token_id", None)] + [tokenizer.convert_tokens_to_ids(x) for x in ("<|im_end|>", "<|endoftext|>")]
              if hasattr(tokenizer, "convert_tokens_to_ids") else [getattr(tokenizer, "eos_token_id", None)]):
        if isinstance(t, int) and t >= 0:
            out.add(t)
    return sorted(out)


def alert(frac, task):
    """3 steps in a row at >= 0.9: one FINISHED_ONLY_ALERT line and one event; the training goes on."""
    _RUN["high"] = _RUN["high"] + 1 if frac >= float(os.environ.get("OPD_FINISHED_ALERT_FRAC", "0.9")) else 0
    if _RUN["high"] >= 3 and not _RUN["alerted"]:
        _RUN["alerted"] = True
        print(f"FINISHED_ONLY_ALERT vlm/unfinished_frac >= 0.9 on 3 steps in a row (last {frac:.3f}); the training goes on", flush=True)
        rb = os.environ.get("RUNBOOK")
        if rb:
            try:
                subprocess.run([sys.executable, os.path.join(rb, "scripts", "control.py"), "event", task or "node", "vlm_unfinished_high",
                                f"unfinished_frac >= 0.9 on 3 steps in a row (last {frac:.3f})"], timeout=30, capture_output=True)
            except Exception:
                pass
        return True
    return False


def empty_mb_frac(td, new_valid):
    """The share of the actor's micro-batches whose rows are all masked (see the docstring); -1.0 when it cannot be computed."""
    try:
        n = int(os.environ.get("OPD_FINISHED_MB_TOKENS", "0") or 0)
        if n <= 0:
            return -1.0
        from verl.utils.seqlen_balancing import rearrange_micro_batches
        _, idx = rearrange_micro_batches(batch=td.select("attention_mask"), max_token_len=n)
        alive = new_valid.sum(1) > 0
        groups = [list(g) for g in idx]
        empty = sum(1 for g in groups if g and not bool(alive[g].any()))
        return round(empty / max(len(groups), 1), 6)
    except Exception:
        return -1.0


def apply(trainer, batch):
    """The hook: (batch with the masked response_mask, metrics)."""
    import torch
    valid = batch.batch["response_mask"]
    new, stats = mask(batch.batch["responses"], valid, end_ids(trainer.tokenizer), torch)
    batch.batch["response_mask"] = new
    stats["empty_mb_frac"] = empty_mb_frac(batch.batch, new)   # [Fable 10-26 23:2x UTC]
    alert(stats["unfinished_frac"], os.environ.get("OPD_TASK", ""))
    return batch, {f"vlm/{k}": float(v) for k, v in stats.items()}
