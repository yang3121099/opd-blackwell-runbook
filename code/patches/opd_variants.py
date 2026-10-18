"""Advantage variants for E10 (P2-P5), applied at G-OPD verl dp_actor.py where it computes
`advantages = -(old_log_prob - ref_log_prob)` (single-teacher, lambda=1 branch).

All tensors are (batch, response_len); response_mask is 1 on real response tokens.
`ref_log_prob` is the teacher's log-prob of the sampled token at each position.
Returns (advantages, loss_mask): loss_mask is used ONLY for the policy loss; entropy/KL metrics
keep the full response_mask (R-001 item 2).
"""
import os

import torch

_GENS = {}


def _gen(device):
    """Per-device generator seeded from OPD_SEED; advances deterministically call by call (R-001 item 10)."""
    key = str(device)
    if key not in _GENS:
        g = torch.Generator(device=device)
        g.manual_seed(int(os.environ.get("OPD_SEED", "1234")))
        _GENS[key] = g
    return _GENS[key]


def opd_k1(old_log_prob, ref_log_prob, response_mask):
    """P2: standard OPD, A_t = log q_T - log p_theta."""
    return (ref_log_prob - old_log_prob) * response_mask


def bin_permuted_teacher(old_log_prob, ref_log_prob, response_mask, n_bins=10, generator=None):
    """P3: within each response, bin tokens by the student's own log-prob into n_bins quantiles and
    permute the teacher log-prob only among tokens of the same bin. Keeps E[log q | log p] and the
    self term in place; removes per-token teacher information beyond the student's confidence."""
    permuted = ref_log_prob.clone()
    for b in range(old_log_prob.shape[0]):
        idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        lp = old_log_prob[b, idx]
        order = torch.argsort(lp)
        bins = torch.chunk(order, min(n_bins, idx.numel()))
        for bin_ids in bins:
            if bin_ids.numel() < 2:
                continue
            pos = idx[bin_ids]
            perm = pos[torch.randperm(pos.numel(), generator=generator, device=pos.device)]
            permuted[b, pos] = ref_log_prob[b, perm]
    return (permuted - old_log_prob) * response_mask


def bin_mean_teacher(old_log_prob, ref_log_prob, response_mask, n_bins=10):
    """P3b (reviewer A3, 2026-10-15): same per-response student-log-prob bins as P3, but the teacher
    log-prob is replaced by its bin mean E[log q | bin] instead of being permuted. Keeps the
    confidence-conditional teacher statistics without P3's permutation noise; removes all per-token
    teacher information beyond the student's own confidence bin."""
    meaned = ref_log_prob.clone()
    for b in range(old_log_prob.shape[0]):
        idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        order = torch.argsort(old_log_prob[b, idx])
        for bin_ids in torch.chunk(order, min(n_bins, idx.numel())):
            pos = idx[bin_ids]
            meaned[b, pos] = ref_log_prob[b, pos].mean()
    return (meaned - old_log_prob) * response_mask


def fixed_negative(old_log_prob, response_mask, frac=0.2, value=-0.5):
    """P4: A=-0.5 on the lowest-`frac` student log-prob tokens of each response, 0 elsewhere
    (2608.31046 Sec. 3.2). Returns (adv, selected_mask); the loss is averaged over selected tokens only."""
    adv = torch.zeros_like(old_log_prob)
    sel = torch.zeros_like(response_mask)
    for b in range(old_log_prob.shape[0]):
        idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        k = max(1, int(round(frac * idx.numel())))
        low = idx[torch.argsort(old_log_prob[b, idx])[:k]]
        adv[b, low] = value
        sel[b, low] = 1
    return adv, sel


def sparse_topk(adv, response_mask, frac=0.2, keep="top"):
    """P6/P6' (R-003 Sec. 3.2): keep the advantage only on the top-`frac` |A_t| tokens of each response
    (keep='top', P6) or only on the remaining 1-frac (keep='rest', P6'); the rest are zeroed.
    The loss mask stays the full response mask, so the denominator matches P2 (R-001 item 4)."""
    out = torch.zeros_like(adv)
    for b in range(adv.shape[0]):
        idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        k = max(1, int(round(frac * idx.numel())))
        order = torch.argsort(adv[b, idx].abs(), descending=True)
        chosen = idx[order[:k]] if keep == "top" else idx[order[k:]]
        out[b, chosen] = adv[b, chosen]
    return out


EOS_IDS = (151645, 151643)   # <|im_end|>, <|endoftext|>
THINK_CLOSE = 151668          # </think>


def final_eos_pos(responses, response_mask, b):
    """Index of the last valid token of row b if it is an EOS (a terminated response), else None."""
    idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
    if idx.numel() == 0 or responses is None:
        return None
    last = int(idx[-1])
    return last if int(responses[b, last]) in EOS_IDS else None


def first_term_pos(responses, response_mask, b):
    """First termination marker of row b: the first </think> or EOS among the valid tokens, else None
    (reviewer, 2026-10-16 00:30: in collapsed P4 the stop decision sits on a spontaneous </think>)."""
    idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
    if idx.numel() == 0 or responses is None:
        return None
    toks = responses[b, idx]
    hit = torch.nonzero((toks == THINK_CLOSE) | torch.isin(toks, torch.tensor(EOS_IDS, device=toks.device)),
                        as_tuple=False).squeeze(-1)
    return int(idx[hit[0]]) if hit.numel() else None


def zero_eos(adv, responses, response_mask):
    """P2-noEOS (wave 1b): OPD advantage with the final EOS position of terminated responses set to 0."""
    out = adv.clone()
    for b in range(adv.shape[0]):
        p = final_eos_pos(responses, response_mask, b)
        if p is not None:
            out[b, p] = 0
    return out


def stop_only(adv, responses, response_mask, k=8):
    """P-stop (wave 1b): keep the OPD advantage only on the final EOS of terminated responses and the k tokens
    before it; everything else (and every token of a truncated response) is 0. Loss mask stays the full mask."""
    out = torch.zeros_like(adv)
    for b in range(adv.shape[0]):
        p = final_eos_pos(responses, response_mask, b)
        if p is None:
            continue
        idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
        keep = idx[(idx >= p - k) & (idx <= p)]
        out[b, keep] = adv[b, keep]
    return out


def random_subset(adv, response_mask, frac=0.2, generator=None):
    """P6-rand (wave 1b): keep the OPD advantage on a uniformly random `frac` of each response's tokens."""
    out = torch.zeros_like(adv)
    for b in range(adv.shape[0]):
        idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        k = max(1, int(round(frac * idx.numel())))
        pick = idx[torch.randperm(idx.numel(), generator=generator, device=idx.device)[:k]]
        out[b, pick] = adv[b, pick]
    return out


def fixed_negative_termpen(old_log_prob, response_mask, responses, frac=0.2, value=-0.5):
    """P4-EOSpen (wave 1b): P4, plus the first termination marker (</think> or EOS) forced into the penalised set."""
    adv, sel = fixed_negative(old_log_prob, response_mask, frac, value)
    for b in range(adv.shape[0]):
        p = first_term_pos(responses, response_mask, b)
        if p is not None:
            adv[b, p] = value
            sel[b, p] = 1
    return adv, sel


# v2.5.7h (reviewer 2026-10-18 19:3x UTC, lab Q-014): token-selective OPD with the P5 teacher prompt. The installed
# dp_actor patch hands only the variants of its list to compute(); a new arm is routed through the listed hook P6 with
# OPD_VARIANT=P6 and OPD_VARIANT_OVERRIDE=<arm> (run_arm.sh sets both, and clears the override for every other arm), so
# the node's G-OPD checkout need not be patched again while arms run. The override applies only with its own hook.
OVERRIDE_VIA = {"P5top10": "P6", "P5rand10m": "P6"}
_LAST = {}   # the last call's selection, for adv_stats (same process, called right after compute)


def resolve_variant(variant):
    ov = os.environ.get("OPD_VARIANT_OVERRIDE", "")
    return ov if ov and OVERRIDE_VIA.get(ov) == variant else variant


def top_frac(adv, response_mask, frac=0.10):
    """P5top10: the OPD advantage only on the top-`frac` |A_t| tokens of each response, the rest 0 (sparse_topk)."""
    return sparse_topk(adv, response_mask, frac, "top")


RAND_CAP = 100.0


def rand_matched(adv, response_mask, frac=0.10, generator=None, cap=RAND_CAP):
    """P5rand10m: a uniformly random `frac` of each response's tokens keeps its advantage, scaled per response so that
    the sum of |A| over the random set equals the sum of |A| over that response's top-`frac` |A| set (scale capped at
    `cap`; returns the advantages and the per-response scales)."""
    out, scales = torch.zeros_like(adv), []
    for b in range(adv.shape[0]):
        idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        k = max(1, int(round(frac * idx.numel())))
        a = adv[b, idx]
        s_top = a.abs().sort(descending=True).values[:k].sum()
        pick = torch.randperm(idx.numel(), generator=generator, device=idx.device)[:k]
        s_rand = a[pick].abs().sum()
        sc = torch.clamp(s_top / s_rand.clamp(min=1e-12), max=cap) if s_rand > 0 else adv.new_tensor(1.0)
        out[b, idx[pick]] = a[pick] * sc
        scales.append(float(sc))
    return out, scales


def compute(variant, old_log_prob, ref_log_prob, response_mask, generator=None, responses=None):
    """Returns (advantages, loss_mask). P5 uses opd_k1; its difference is the teacher prompt
    (think template + '<think>\\n'), handled in ref input preparation, not here."""
    variant = resolve_variant(variant)
    if variant == "P5top10":
        a = top_frac(opd_k1(old_log_prob, ref_log_prob, response_mask), response_mask, 0.10)
        _LAST.update(variant=variant, sel=(a != 0) & (response_mask > 0), scales=None)
        return a, response_mask
    if variant == "P5rand10m":
        g = generator if generator is not None else _gen(old_log_prob.device)
        a, sc = rand_matched(opd_k1(old_log_prob, ref_log_prob, response_mask), response_mask, 0.10, g)
        _LAST.update(variant=variant, sel=(a != 0) & (response_mask > 0), scales=sc)
        return a, response_mask
    if variant in ("P2", "P5"):
        return opd_k1(old_log_prob, ref_log_prob, response_mask), response_mask
    if variant == "P2noEOS":
        return zero_eos(opd_k1(old_log_prob, ref_log_prob, response_mask), responses, response_mask), response_mask
    if variant == "Pstop":
        return stop_only(opd_k1(old_log_prob, ref_log_prob, response_mask), responses, response_mask), response_mask
    if variant == "P6rand":
        g = generator if generator is not None else _gen(old_log_prob.device)
        return random_subset(opd_k1(old_log_prob, ref_log_prob, response_mask), response_mask, 0.2, g), response_mask
    if variant == "P6top5":
        return sparse_topk(opd_k1(old_log_prob, ref_log_prob, response_mask), response_mask, 0.05, "top"), response_mask
    if variant == "P4EOSpen":
        return fixed_negative_termpen(old_log_prob, response_mask, responses)
    if variant == "P3":
        g = generator if generator is not None else _gen(old_log_prob.device)
        return bin_permuted_teacher(old_log_prob, ref_log_prob, response_mask, generator=g), response_mask
    if variant == "P3b":
        return bin_mean_teacher(old_log_prob, ref_log_prob, response_mask), response_mask
    if variant == "P4":
        return fixed_negative(old_log_prob, response_mask)
    if variant in ("P6", "P6r"):
        a = opd_k1(old_log_prob, ref_log_prob, response_mask)
        return sparse_topk(a, response_mask, 0.2, "top" if variant == "P6" else "rest"), response_mask
    raise ValueError(variant)


@torch.no_grad()
def _topk_overlap(score_a, score_b, response_mask, frac=0.2):
    """Mean per-response overlap |top_a ∩ top_b| / k between the top-`frac` tokens of two scores
    (random baseline = frac)."""
    vals = []
    for b in range(score_a.shape[0]):
        idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
        if idx.numel() < 5:
            continue
        k = max(1, int(round(frac * idx.numel())))
        ta = set(idx[torch.argsort(score_a[b, idx], descending=True)[:k]].tolist())
        tb = set(idx[torch.argsort(score_b[b, idx], descending=True)[:k]].tolist())
        vals.append(len(ta & tb) / k)
    return sum(vals) / len(vals) if vals else float("nan")


_SHADOW_GEN = {}

# v2.5.7h (Q-014): token classes of the selected set. EOS = <|im_end|> / <|endoftext|>; discourse = re-check, hedging and
# connective words (DISC, the token's text stripped and lower-cased); math = a digit, an operator / bracket / LaTeX
# character, or a LaTeX command name (MATH_WORDS); other = the rest (prose, whitespace, single letters). Built once per
# process from the student's tokenizer (OPD_TOKENIZER) when OPD_TOKCAT=1; no effect on training.
DISC = {"wait", "alternatively", "however", "hmm", "actually", "but", "let", "check", "verify", "double", "recheck", "re",
        "confirm", "maybe", "perhaps", "okay", "ok", "so", "therefore", "thus", "hence", "now", "first", "next", "then",
        "again", "mistake", "sure", "think", "seems", "hold", "oh", "ah", "right", "yes", "no", "well"}
MATH_WORDS = {"frac", "dfrac", "sqrt", "cdot", "times", "pi", "boxed", "sin", "cos", "tan", "log", "ln", "le", "ge", "leq",
              "geq", "neq", "infty", "sum", "prod", "int", "left", "right", "text", "mod", "pmod", "binom", "circ", "angle",
              "triangle", "equiv", "approx"}
CATS = ("eos", "disc", "math", "other")
_CATS = {}


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs, n = bs[:], 0
    for x in range(256):
        if x not in bs:
            bs.append(x); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


def token_class(tid, piece):
    """0 eos, 1 discourse, 2 math, 3 other for one token (piece: its text)."""
    import re as _re
    if tid in EOS_IDS:
        return 0
    t = piece.strip().lower()
    if t in DISC:
        return 1
    if _re.search(r"[0-9=+\-*/^_{}\\|<>$()\[\]]", t) or t.lstrip("\\") in MATH_WORDS:
        return 2
    return 3


def token_categories():
    """Class per vocabulary id (a CPU LongTensor), or None when OPD_TOKCAT is not 1 or the tokenizer cannot be read."""
    if "v" in _CATS:
        return _CATS["v"]
    _CATS["v"] = None
    if os.environ.get("OPD_TOKCAT") != "1" or not os.environ.get("OPD_TOKENIZER"):
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.environ["OPD_TOKENIZER"])
        u2b = {v: k for k, v in _bytes_to_unicode().items()}
        toks = tok.convert_ids_to_tokens(list(range(len(tok))))
        cls = []
        for i, t in enumerate(toks):
            t = t or ""
            piece = bytes(u2b[c] for c in t).decode("utf-8", "replace") if all(c in u2b for c in t) else t
            cls.append(token_class(i, piece))
        _CATS["v"] = torch.tensor(cls, dtype=torch.long)
    except Exception:
        pass
    return _CATS["v"]


def selection_stats(variant, response_mask, responses):
    """Diagnostics of the token-selective arms (Q-014): kept share, the matched scale of P5rand10m, the share of
    terminated responses whose EOS is selected, and the token classes of the selected set and of all tokens."""
    out = {}
    if not _LAST.get("variant") or resolve_variant(variant) != _LAST["variant"]:
        return out
    rm = response_mask > 0
    sel = _LAST["sel"] & rm
    n, k = max(rm.sum().item(), 1), max(sel.sum().item(), 1)
    out["opd/sel_kept_frac"] = sel.sum().item() / n
    if _LAST.get("scales"):   # P5rand10m: the matched scale per response (reviewer A1 19:4x UTC: its distribution and the cap)
        sc = sorted(_LAST["scales"]); q = lambda f: sc[min(len(sc) - 1, int(f * len(sc)))]
        out.update({"opd/rand_scale_mean": sum(sc) / len(sc), "opd/rand_scale_max": sc[-1], "opd/rand_scale_p10": q(0.1),
                    "opd/rand_scale_p50": q(0.5), "opd/rand_scale_p90": q(0.9),
                    "opd/rand_scale_cap_frac": sum(x >= RAND_CAP for x in sc) / len(sc)})
    if responses is not None:
        rows = [b for b in range(responses.shape[0]) if final_eos_pos(responses, response_mask, b) is not None]
        if rows:
            out["opd/sel_eos_kept_frac"] = sum(bool(sel[b, final_eos_pos(responses, response_mask, b)]) for b in rows) / len(rows)
        cats = token_categories()
        if cats is not None:
            c = cats.to(responses.device)[responses.clamp(min=0, max=len(cats) - 1)]
            for j, name in enumerate(CATS):
                out[f"opd/sel_{name}_frac"] = ((c == j) & sel).sum().item() / k
                out[f"opd/all_{name}_frac"] = ((c == j) & rm).sum().item() / n
    _LAST.clear()
    return out


@torch.no_grad()
def adv_stats(variant, adv_orig, adv_used, response_mask, loss_mask, entropy=None, old_log_prob=None, ref_log_prob=None,
              responses=None, eos_ids=(151645, 151643), tail=32):
    """Per-step diagnostics (R-001 items 3-4): advantage scale on the loss tokens, and for P3 how much
    of the original OPD advantage survives the within-bin permutation."""
    lm = loss_mask > 0
    rm = response_mask > 0
    au = adv_used[lm].float()
    out = {
        "opd/adv_abs_mean": au.abs().mean().item() if au.numel() else 0.0,
        "opd/adv_std": au.std().item() if au.numel() > 1 else 0.0,
        "opd/adv_nonzero_frac": ((adv_used != 0) & rm).sum().item() / max(rm.sum().item(), 1),
        "opd/loss_token_frac": lm.sum().item() / max(rm.sum().item(), 1),
    }
    if variant in ("P3", "P3b") and rm.sum() > 2:
        x, y = adv_orig[rm].float(), adv_used[rm].float()
        c = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
        out.update({"opd/p3_corr": c, "opd/p3_r2": c * c,
                    "opd/p3_var_ratio": (y.var() / x.var().clamp(min=1e-12)).item()})
    # Diagnostics only (no effect on training): what P3 would keep, measured on P2/P5 rollouts, with a
    # separate generator so the P3 arm's own permutation stream is untouched.
    if variant in ("P2", "P5") and old_log_prob is not None and ref_log_prob is not None and rm.sum() > 2:
        key = str(adv_orig.device)
        if key not in _SHADOW_GEN:
            _SHADOW_GEN[key] = torch.Generator(device=adv_orig.device); _SHADOW_GEN[key].manual_seed(999)
        sh = bin_permuted_teacher(old_log_prob, ref_log_prob, response_mask, generator=_SHADOW_GEN[key])
        x, y = adv_orig[rm].float(), sh[rm].float()
        c = torch.corrcoef(torch.stack([x, y]))[0, 1].item()
        out.update({"opd/p3shadow_r2": c * c})
    # P6 pre-check (R-003): do the largest-|A| tokens coincide with high-entropy (forking) tokens?
    if entropy is not None and rm.sum() > 2:
        out["opd/topA_vs_topH_overlap"] = _topk_overlap(adv_orig.abs().float(), entropy.float(), response_mask)
    # "EOS is penalised" test (R-003 follow-up, A3): uses the OPD advantage adv_orig = log q_T - log p_theta
    # for every arm. Terminated = last valid token is EOS; truncated = no EOS at the last valid position.
    if responses is not None:
        eos = torch.tensor(eos_ids, device=responses.device)
        a_eos, a_tail, n_term, n_trunc = [], [], 0, 0
        eos_low, eos_lp = [], []
        for b in range(responses.shape[0]):
            idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            last = idx[-1]
            if torch.isin(responses[b, last], eos):
                n_term += 1
                a_eos.append(adv_orig[b, last].float())
                # P4 mechanism test (reviewer, 2026-10-15 ~21:10 UTC): is the EOS among the lowest-20% student
                # log-prob tokens, i.e. the set P4 penalises? Logged for every arm; no effect on training.
                if old_log_prob is not None:
                    k = max(1, int(round(0.2 * idx.numel())))
                    low = idx[torch.argsort(old_log_prob[b, idx])[:k]]
                    eos_low.append(float(bool((low == last).any())))
                    eos_lp.append(old_log_prob[b, last].float())
            else:
                n_trunc += 1
                a_tail.append(adv_orig[b, idx[-tail:]].float().mean())
        n = max(n_term + n_trunc, 1)
        out["opd/terminated_frac"] = n_term / n
        if a_eos:
            out["opd/eos_adv_mean"] = torch.stack(a_eos).mean().item()
        if eos_low:
            out["opd/eos_in_low20_frac"] = sum(eos_low) / len(eos_low)
            out["opd/eos_student_logp_mean"] = torch.stack(eos_lp).mean().item()
        if a_tail:
            out["opd/trunc_tail_adv_mean"] = torch.stack(a_tail).mean().item()
        # Termination-marker test (wave 1b, reviewer 00:30): the first </think> or EOS may carry the stop decision
        # (collapsed P4 stops via a spontaneous </think>). Share of responses containing </think>, and how often
        # the first marker falls inside the lowest-20% student log-prob set. Diagnostics only.
        n_close, term_low = 0, []
        for b in range(responses.shape[0]):
            idx = torch.nonzero(response_mask[b] > 0, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            n_close += int(bool((responses[b, idx] == THINK_CLOSE).any()))
            p = first_term_pos(responses, response_mask, b)
            if p is not None and old_log_prob is not None:
                k = max(1, int(round(0.2 * idx.numel())))
                low = idx[torch.argsort(old_log_prob[b, idx])[:k]]
                term_low.append(float(bool((low == p).any())))
        out["opd/think_close_frac"] = n_close / max(responses.shape[0], 1)
        if term_low:
            out["opd/term_in_low20_frac"] = sum(term_low) / len(term_low)
    out.update(selection_stats(variant, response_mask, responses))   # v2.5.7h: P5top10 / P5rand10m only
    return out
