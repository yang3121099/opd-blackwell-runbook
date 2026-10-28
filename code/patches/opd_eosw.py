"""Position-weighted OPD (runbook [v2.5.9z102], Q-087 E10b-P5-eosw; Fable 10-27 15:5x UTC, theory_20261028 (e) 1): the stop decision
weighs more in OPD's loss. In each response the OPD_EOSW_N (16) positions after the token that closes its LAST complete \\boxed{...}
- the distributions that choose tokens k_last+1 .. k_last+16, TERM4's p_stop_last_max16 window - get weight OPD_EOSW_W (64), every
other position 1; a response without a complete box keeps 1 everywhere.

The weight multiplies G-OPD's per-token advantage (-(log pi - log q_ref): that position's own log-ratio, no return-to-go) just before
the policy loss. A positive factor commutes with PPO's clip and dual clip (a max / min of terms all scaled by the same w), so this is
the per-token loss weighted by w; verl's token-mean keeps its unweighted denominator, so the window's tokens get w times the gradient.
At about 0.3 % of the tokens the step's loss grows about x1.2 (theory (e)), not the x10 of weighting every post-box position.

Environment: OPD_EOSW=1 on (eosw_patch.py's hooks are no-ops otherwise); OPD_EOSW_W (64); OPD_EOSW_N (16); OPD_EOSW_LOG (JSON lines,
one per prepare_ref_model_inputs call: rows, rows with a box, positions weighted and their share of the valid tokens).
"""
import json
import os
import time

BOX = "\\boxed{"


def active():
    return os.environ.get("OPD_EOSW") == "1"


def params():
    """(weight, window) from the environment."""
    return float(os.environ.get("OPD_EOSW_W", "64")), int(os.environ.get("OPD_EOSW_N", "16"))


def last_box_end(pieces):
    """Index of the token holding the closing brace of the LAST complete \\boxed{...} (braces balanced), or -1. pieces = each response
    token's decoded text (the marker and braces are ASCII, so per-token decoding keeps them whole, as opd_stopmerge.first_box_end).
    A box that never closes ends the search (a later marker would sit inside it)."""
    owner = [i for i, p in enumerate(pieces) for _ in p]
    s = "".join(pieces)
    end, k = -1, s.find(BOX)
    while k >= 0:
        depth, j = 1, k + len(BOX)
        while j < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[j], 0); j += 1
        if depth:
            break
        end = owner[j - 1]
        k = s.find(BOX, j)
    return end


def box_last_ends(responses, response_mask, tokenizer):
    """Per row: last_box_end over its valid tokens (response_mask), from the tokenizer's per-token decoding."""
    out = []
    for ids, m in zip(responses.tolist(), response_mask.tolist()):
        n = int(sum(m)); ids = ids[:n]
        out.append(last_box_end(tokenizer.batch_decode([[t] for t in ids])) if n else -1)
    return out


def weights(box_end, valid, w, n, xp):
    """((B, L) float32 weights, (B, L) bool window): w at the positions p with box_end < p <= box_end + n (box_end >= 0) that are
    valid, 1 elsewhere. box_end (B,), valid (B, L); xp = numpy or torch (the tensors' device kept)."""
    B, L = valid.shape
    npy = xp.__name__ == "numpy"
    pos = (xp.arange(L) if npy else xp.arange(L, device=valid.device)).reshape(1, L)
    be = box_end.reshape(B, 1) if npy else box_end.to(valid.device).reshape(B, 1)
    win = (pos > be) & (pos <= be + n) & (be >= 0) & (valid > 0)
    one = xp.ones((B, L), dtype=xp.float32) if npy else xp.ones((B, L), dtype=xp.float32, device=valid.device)
    return xp.where(win, one * w, one), win


def stats(box_end, win, valid):
    nv = int(valid.sum())
    return {"rows": int(box_end.shape[0]), "rows_box": int((box_end >= 0).sum()), "weighted": int(win.sum()),
            "weighted_frac": round(int(win.sum()) / nv, 6) if nv else None, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def log_stats(st):
    p = os.environ.get("OPD_EOSW_LOG")
    if p:
        with open(p, "a") as f:
            f.write(json.dumps(st) + "\n")


def make(responses, response_mask, tokenizer):
    """torch: the (B, L) weight tensor for the batch (prepare_ref_model_inputs' hook), its call logged."""
    import torch
    w, n = params()
    be = torch.tensor(box_last_ends(responses, response_mask, tokenizer), dtype=torch.long)
    wt, win = weights(be, response_mask, w, n, torch)
    st = stats(be, win, response_mask); st.update({"w": w, "n": n})
    log_stats(st)
    return wt
