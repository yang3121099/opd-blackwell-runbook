"""Stop-merged OPD (runbook [v2.5.9l], Q-038; the reviewer 14:1x UTC 10-21, after 2609.20511's fixes 2-4).

P5 distils a non-thinking student from the 8B teacher reading the student's answer in thinking mode. After the answer's
first complete \\boxed{...} that teacher puts its stopping mass on </think> (end the thinking, then answer), not on the
student's own stop token <|im_end|>, so the student's stop there gets a low teacher log-prob and the per-token advantage
(teacher log p - student log p) pushes it to keep writing (P5's tail after the first box: ~842 tokens; p_stop_last 0.12).

Fix: in the reference (teacher) pass only, at every response position AFTER the end of the first complete \\boxed{...}
where the student's token is a stop token (EOS_IDS), the teacher's log-prob is the log of its total mass on the stop set
(STOP_IDS: <|im_end|>, </think>, <|endoftext|>), log sum_s p_teacher(s), instead of log p_teacher(<|im_end|>). Every other
position, and every position up to and including the box, keeps the teacher's own log-prob (before the box the thinking
teacher also offers </think> to mean "go and answer", so merging there would teach the student to stop unanswered).

The full-vocabulary normaliser needs no second pass: log p(label) = logit(label) - logsumexp(logits), so
logsumexp(logits) = logit(label) - log p(label), exact in float32 from the same (bf16) logits the log-prob came from.

Environment: OPD_STOPMERGE=1 turns it on (apply_patches.py hooks prepare_ref_model_inputs for the box positions and the
reference forward pass for the merge); OPD_STOPMERGE_IDS (STOP_IDS), OPD_STOPMERGE_EOS (EOS_IDS); OPD_STOPMERGE_LOG
(JSON lines, one per reference micro-batch: rows, rows with a box, positions merged, the gain at them, violations).
"""
import json
import os

THINK_CLOSE, IM_END, ENDOFTEXT = 151668, 151645, 151643


def _ids(name, default):
    v = os.environ.get(name, "")
    return [int(x) for x in v.split(",") if x.strip()] if v else list(default)


def stop_ids():
    return _ids("OPD_STOPMERGE_IDS", (IM_END, THINK_CLOSE, ENDOFTEXT))


def eos_ids():
    return _ids("OPD_STOPMERGE_EOS", (IM_END, ENDOFTEXT))


def active():
    return os.environ.get("OPD_STOPMERGE") == "1"


def first_box_end(pieces):
    """Index of the token that closes the first complete \\boxed{...} (braces balanced), or -1. pieces = the decoded
    text of each response token in order (the marker and braces are ASCII, so per-token decoding keeps them whole)."""
    text, owner = [], []
    for i, p in enumerate(pieces):
        text.append(p); owner.extend([i] * len(p))
    s = "".join(text)
    k = s.find("\\boxed{")
    if k < 0:
        return -1
    depth = 0
    for j in range(k + len("\\boxed"), len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return owner[j]
    return -1   # the first box never closes: no complete box in this response


def box_ends(responses, response_mask, tokenizer):
    """Per row: first_box_end over its valid tokens (response_mask), from the tokenizer's per-token decoding."""
    out = []
    for ids, m in zip(responses.tolist(), response_mask.tolist()):
        n = int(sum(m)); ids = ids[:n]
        out.append(first_box_end(tokenizer.batch_decode([[t] for t in ids])) if n else -1)
    return out


def stop_logprob(logits, labels, log_probs, ids):
    """torch: log of the teacher's mass on the token set ids at every position. logits (..., V) (already divided by the
    temperature), labels (...), log_probs (...) = log_softmax(logits)[labels] as computed by verl."""
    import torch
    lse = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1).float() - log_probs.float()
    return torch.logsumexp(logits[..., ids].float() - lse.unsqueeze(-1), dim=-1)


def merge(log_probs, stop_lp, responses, box_end, eos, xp, valid=None):
    """(merged log-probs, stats). log_probs / stop_lp / responses (B, L); box_end (B,) with -1 for no box; eos the
    student's stop ids; xp = numpy or torch; valid (B, L) the response mask (<|endoftext|> is also Qwen's padding, so
    padding is never merged nor counted). Positions merged: valid, after the box (index > box_end, box_end >= 0), the
    student's token in eos."""
    B, L = responses.shape
    # [v2.5.9z4] torch: the positions on the responses' device (10-21 20:0x UTC the smoke failed in the reference pass: arange on the CPU,
    # opd_box_end on cuda:0, "Expected all tensors to be on the same device")
    pos = (xp.arange(L) if xp.__name__ == "numpy" else xp.arange(L, device=responses.device)).reshape(1, L)
    be = box_end.reshape(B, 1)
    after = (pos > be) & (be >= 0)
    if valid is not None:
        after = after & (valid > 0)
    is_eos = xp.zeros(responses.shape, dtype=bool) if xp.__name__ == "numpy" else xp.zeros_like(responses, dtype=xp.bool)
    for e in eos:
        is_eos = is_eos | (responses == e)
    if valid is not None:
        is_eos = is_eos & (valid > 0)
    m = after & is_eos
    out = xp.where(m, stop_lp.astype(log_probs.dtype) if xp.__name__ == "numpy" else stop_lp.to(log_probs.dtype), log_probs)
    n = int(m.sum())
    gain = (stop_lp - log_probs)[m]
    stats = {"rows": int(B), "rows_box": int((box_end >= 0).sum()), "merged": n,
             "eos_total": int(is_eos.sum()), "eos_before_or_no_box": int((is_eos & ~after).sum()),
             "gain_mean": float(gain.mean()) if n else None, "gain_min": float(gain.min()) if n else None,
             "violations": int((m & ~after).sum()),
             "stop_mass_after_box_mean": float(xp.exp(stop_lp[after]).mean()) if int(after.sum()) else None}
    return out, stats


def log_stats(stats):
    p = os.environ.get("OPD_STOPMERGE_LOG")
    if p:
        with open(p, "a") as f:
            f.write(json.dumps(stats) + "\n")
