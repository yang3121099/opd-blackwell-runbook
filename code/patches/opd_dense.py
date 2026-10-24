"""Dense OPD (runbook [v2.5.9z76], Q-065; Runming 10-22 via the reviewer: every OPD arm so far uses a SAMPLED estimate of the
per-token reverse KL, advantage = log q(y) - log p(y) at the sampled token y; the reference pass sees the teacher's whole
vocabulary, so the per-token KL can be taken over the vocabulary instead).

Per response position, with the student's temperature-scaled logits z (log-softmax lp, probs p) and the teacher's top-K
(OPD_DENSE_K, 64) log-probs t on ids I (taken in the reference pass), the teacher's remaining mass r = 1 - sum exp(t) spread
evenly over the other V - K tokens (log q_rest = log r - log(V - K)):
  RKL = KL(p || q) = sum_v p lp - sum_I p_I t - (1 - P_I) log q_rest
  FKL = KL(q || p) = sum_I q_I (t - lp_I) + r log q_rest - (r / (V - K)) (sum_v lp - sum_I lp_I)
  AKL (Wu et al. 2024, "Rethinking KL divergence in knowledge distillation for LLMs"; the definition to be confirmed by the
      literature reader 13): the teacher's head H = its smallest top set with mass >= mu (OPD_DENSE_MU, 0.5), the gaps
      g_head = sum_H |q - p|, g_tail = sum_{V \\ H} |q - p| (the rest of V at q_rest), weights taken without gradient:
      AKL = g_head / (g_head + g_tail) FKL + g_tail / (g_head + g_tail) RKL.
OPD_DENSE=rkl | akl turns it on: the reference pass keeps the teacher's top-K at each response position (batch keys
ref_topk_ids / ref_topk_lp, carried like ref_log_prob), and the actor's update replaces the policy-gradient loss by the
token mean of the dense divergence (gradient through the student's logits only, in chunks of OPD_DENSE_CHUNK positions under
activation checkpointing). Logged per update: dense/rkl, dense/fkl, dense/akl_w_head, dense/mass_topk (the teacher's mass in
its top-K), dense/corr_sampled (Pearson r over the valid tokens between the dense RKL and the sampled one, log p(y) - log q(y)),
dense/rkl_sampled (the sampled one's token mean: both estimate the same expectation, so their ratio catches sign / index errors).
dense_terms() works on numpy (the Mac tests) and torch (the trainer) alike.
"""
import os

DEFAULT_K, DEFAULT_MU, DEFAULT_CHUNK = 64, 0.5, 2048


def mode():
    m = os.environ.get("OPD_DENSE", "")
    return m if m in ("rkl", "akl") else ""


def active():
    return bool(mode())


def topk_k():
    try:
        return max(1, int(os.environ.get("OPD_DENSE_K", DEFAULT_K)))
    except ValueError:
        return DEFAULT_K


def mu():
    try:
        return float(os.environ.get("OPD_DENSE_MU", DEFAULT_MU))
    except ValueError:
        return DEFAULT_MU


class _NP:
    """the few ops dense_terms needs, on numpy"""
    def __init__(self):
        import numpy as np
        self.np = np

    def lse(self, z):
        m = z.max(-1, keepdims=True)
        return (m + self.np.log(self.np.exp(z - m).sum(-1, keepdims=True)))[..., 0]

    def gather(self, a, idx):
        return self.np.take_along_axis(a, idx, -1)

    def exp(self, a):
        return self.np.exp(a)

    def log(self, a):
        return self.np.log(a)

    def abs(self, a):
        return self.np.abs(a)

    def clip_min(self, a, lo):
        return self.np.maximum(a, lo)

    def cumsum(self, a):
        return self.np.cumsum(a, -1)

    def detach(self, a):
        return a


class _TORCH:
    def __init__(self):
        import torch
        self.t = torch

    def lse(self, z):
        return self.t.logsumexp(z, -1)

    def gather(self, a, idx):
        return a.gather(-1, idx)

    def exp(self, a):
        return a.exp()

    def log(self, a):
        return a.log()

    def abs(self, a):
        return a.abs()

    def clip_min(self, a, lo):
        return a.clamp_min(lo)

    def cumsum(self, a):
        return a.cumsum(-1)

    def detach(self, a):
        return a.detach()


def backend(x):
    return _NP() if type(x).__module__.startswith("numpy") else _TORCH()


def dense_terms(z, t_ids, t_lp, mu_=DEFAULT_MU, eps=1e-8):
    """z (N, V) student logits (already / temperature, float32), t_ids (N, K) the teacher's top-K ids (descending), t_lp (N, K)
    their log-probs. -> dict of (N,) tensors: rkl, fkl, akl, w_head (AKL's FKL weight), mass_topk."""
    B = backend(z)
    V, K = z.shape[-1], t_ids.shape[-1]
    lp = z - B.lse(z)[..., None]
    p = B.exp(lp)
    lpI = B.gather(lp, t_ids)
    pI = B.exp(lpI)
    qI = B.exp(t_lp)
    mass = qI.sum(-1)
    r = B.clip_min(1.0 - mass, eps)
    lq_rest = B.log(r) - float(__import__("math").log(max(V - K, 1)))
    PI = pI.sum(-1)
    negent = (p * lp).sum(-1)
    rkl = negent - (pI * t_lp).sum(-1) - (1.0 - PI) * lq_rest
    fkl = (qI * (t_lp - lpI)).sum(-1) + r * lq_rest - (r / max(V - K, 1)) * (lp.sum(-1) - lpI.sum(-1))
    # AKL: the head = the shortest prefix of the (descending) top-K with cumulative mass >= mu (always at least one token)
    cum = B.cumsum(qI)
    head = (cum - qI) < mu_                                        # a token is in the head when the mass before it is < mu
    gapI = B.abs(qI - pI)
    g_head = (gapI * head).sum(-1)
    q_rest = B.exp(lq_rest)
    tail_rest = B.abs(q_rest[..., None] - p).sum(-1) - B.abs(q_rest[..., None] - pI).sum(-1)   # V \\ I at q_rest
    g_tail = (gapI * (1 - head * 1)).sum(-1) + tail_rest
    w_head = B.detach(g_head / B.clip_min(g_head + g_tail, eps))
    akl = w_head * fkl + (1.0 - w_head) * rkl
    return {"rkl": rkl, "fkl": fkl, "akl": akl, "w_head": w_head, "mass_topk": mass}


def topk_logprobs(logits, k):
    """torch: (N, V) logits (already / temperature) -> (ids (N, k) descending, their log-probs (N, k) float32)."""
    import torch
    lse = torch.logsumexp(logits.float(), -1, keepdim=True)
    v, i = torch.topk(logits.float(), k, dim=-1)
    return i, v - lse


def dense_loss_rows(logits, t_ids, t_lp, which, chunk=None, mu_=None):
    """torch: the per-position dense divergence for (N, V) logits with gradient, in chunks under activation checkpointing;
    -> (loss (N,), stats dict of (N,) detached: rkl, fkl, w_head, mass_topk)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    chunk = chunk or int(os.environ.get("OPD_DENSE_CHUNK", DEFAULT_CHUNK)); mu_ = mu() if mu_ is None else mu_
    out, st = [], {"rkl": [], "fkl": [], "w_head": [], "mass_topk": []}

    def one(zc, ic, lc):
        d = dense_terms(zc.float(), ic, lc.float(), mu_)
        return d[which], d["rkl"].detach(), d["fkl"].detach(), d["w_head"], d["mass_topk"].detach()
    for s in range(0, logits.shape[0], chunk):
        a = checkpoint(one, logits[s:s + chunk], t_ids[s:s + chunk], t_lp[s:s + chunk], use_reentrant=False)
        out.append(a[0])
        for k, v in zip(("rkl", "fkl", "w_head", "mass_topk"), a[1:]):
            st[k].append(v)
    return torch.cat(out), {k: torch.cat(v) for k, v in st.items()}


def pearson(a, b):
    """torch or numpy 1-d: Pearson r (None for fewer than 3 points or a constant side)."""
    B = backend(a)
    n = a.shape[0]
    if n < 3:
        return None
    am, bm = a - a.mean(), b - b.mean()
    den = float(((am * am).sum() * (bm * bm).sum()) ** 0.5)
    return float((am * bm).sum()) / den if den > 0 else None


# ---------------------------------------------------------------------------------------------------------- the trainer's hooks
# (dense_patch.py puts the calls into G-OPD's dp_actor.py / fsdp_workers.py; one process holds one worker, so module state is fine)
REF_TOPK = []        # the reference pass's micro-batches: (ids (b, R, K) int32, log-probs (b, R, K) float16)
LAST_REF = None      # their concatenation in the batch's order (compute_log_prob -> compute_ref_log_prob)
ACT = None           # the actor micro-batch's (dense loss (b, R) with gradient, {stat: (b, R)})


def _rows_loss(get, rows, t_ids, t_lp, which, chunk=None):
    """the dense rows with gradient, each chunk's logits fetched by get(rows chunk) inside the checkpoint (no full copy)."""
    import torch
    from torch.utils.checkpoint import checkpoint
    chunk = chunk or int(os.environ.get("OPD_DENSE_CHUNK", DEFAULT_CHUNK)); m = mu()
    out, st = [], {"rkl": [], "fkl": [], "w_head": [], "mass_topk": []}

    def one(rc, ic, lc):
        d = dense_terms(get(rc).float(), ic, lc.float(), m)
        return d[which], d["rkl"].detach(), d["fkl"].detach(), d["w_head"], d["mass_topk"].detach()
    for s in range(0, rows.shape[0], chunk):
        a = checkpoint(one, rows[s:s + chunk], t_ids[s:s + chunk], t_lp[s:s + chunk], use_reentrant=False)
        out.append(a[0])
        for k, v in zip(("rkl", "fkl", "w_head", "mass_topk"), a[1:]):
            st[k].append(v)
    if not out:
        z = torch.zeros(0, device=rows.device)
        return z, {k: z for k in st}
    return torch.cat(out), {k: torch.cat(v) for k, v in st.items()}


def rmpad_hook(logits_rmpad, mb, indices, bsz, seqlen, R, ref, pad_input, index_first_axis, rearrange):
    """the remove-padding forward: ref -> the teacher's top-K at the response positions to REF_TOPK; else the dense loss -> ACT."""
    import torch
    global ACT
    if ref:
        ids, lps = topk_logprobs(logits_rmpad, topk_k())
        fi = pad_input(hidden_states=ids, indices=indices, batch=bsz, seqlen=seqlen)[:, -R - 1:-1]
        fl = pad_input(hidden_states=lps, indices=indices, batch=bsz, seqlen=seqlen)[:, -R - 1:-1]
        REF_TOPK.append((fi.to(torch.int32).cpu(), fl.to(torch.float16).cpu()))
        return
    dev, K = logits_rmpad.device, mb["ref_topk_ids"].shape[-1]
    ti = torch.zeros(bsz, seqlen, K, dtype=torch.long, device=dev); tl = torch.zeros(bsz, seqlen, K, dtype=torch.float32, device=dev)
    va = torch.zeros(bsz, seqlen, 1, dtype=torch.float32, device=dev)
    ti[:, -R - 1:-1] = mb["ref_topk_ids"].to(dev).long(); tl[:, -R - 1:-1] = mb["ref_topk_lp"].to(dev).float()
    va[:, -R - 1:-1, 0] = mb["response_mask"].to(dev).float()
    flat = lambda x: index_first_axis(rearrange(x, "b s k -> (b s) k"), indices)
    ti_r, tl_r, v_r = flat(ti), flat(tl), flat(va)[:, 0]
    rows = (v_r > 0).nonzero().squeeze(-1)
    loss, st = _rows_loss(lambda rc: logits_rmpad.index_select(0, rc), rows, ti_r.index_select(0, rows), tl_r.index_select(0, rows), mode())
    nnz = logits_rmpad.shape[0]
    back = lambda v: pad_input(hidden_states=torch.zeros(nnz, dtype=v.dtype, device=dev).index_add(0, rows, v).unsqueeze(-1),
                               indices=indices, batch=bsz, seqlen=seqlen).squeeze(-1)[:, -R - 1:-1]
    ACT = (back(loss), {k: back(v) for k, v in st.items()})


def padded_hook(logits, mb, ref):
    """the padded forward: logits (b, R, V) already sliced to the response positions."""
    import torch
    global ACT
    if ref:
        b, R, V = logits.shape
        ids, lps = topk_logprobs(logits.reshape(-1, V) if logits.is_contiguous() else logits.contiguous().reshape(-1, V), topk_k())
        REF_TOPK.append((ids.reshape(b, R, -1).to(torch.int32).cpu(), lps.reshape(b, R, -1).to(torch.float16).cpu()))
        return
    b, R, V = logits.shape; dev = logits.device
    valid = mb["response_mask"].to(dev).reshape(-1) > 0
    rows = valid.nonzero().squeeze(-1)
    ti = mb["ref_topk_ids"].to(dev).long().reshape(b * R, -1).index_select(0, rows)
    tl = mb["ref_topk_lp"].to(dev).float().reshape(b * R, -1).index_select(0, rows)
    loss, st = _rows_loss(lambda rc: logits[rc // R, rc % R], rows, ti, tl, mode())
    back = lambda v: torch.zeros(b * R, dtype=v.dtype, device=dev).index_add(0, rows, v).reshape(b, R)
    ACT = (back(loss), {k: back(v) for k, v in st.items()})


def finish_ref(batch_idx_list, restore):
    """compute_log_prob's end: the reference pass's top-K in the batch's order (restore = restore_dynamic_batch)."""
    import torch
    global LAST_REF
    if not REF_TOPK:
        return
    ids, lps = torch.cat([x[0] for x in REF_TOPK]), torch.cat([x[1] for x in REF_TOPK])
    REF_TOPK.clear()
    if batch_idx_list is not None:
        ids, lps = restore(ids, batch_idx_list), restore(lps, batch_idx_list)
    LAST_REF = (ids, lps)


def take_ref():
    """compute_ref_log_prob: the top-K tensors for the output batch ({} when none)."""
    global LAST_REF
    if LAST_REF is None:
        return {}
    ids, lps = LAST_REF; LAST_REF = None
    return {"ref_topk_ids": ids, "ref_topk_lp": lps}


def take_act_loss(agg_loss, response_mask, loss_agg_mode, log_prob, ref_log_prob):
    """update_policy: (the dense loss aggregated like the policy loss, metrics dense/*); raises without a hooked forward."""
    global ACT
    if ACT is None:
        raise RuntimeError("dense OPD: no teacher top-K in this micro-batch (the reference pass did not keep it, or fused kernels)")
    mat, st = ACT; ACT = None
    loss = agg_loss(loss_mat=mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    m = response_mask > 0
    met = {f"dense/{k}": float(v[m].mean()) if int(m.sum()) else 0.0 for k, v in st.items()}
    met[f"dense/{mode()}"] = float(mat.detach()[m].mean()) if int(m.sum()) else 0.0
    if ref_log_prob is not None and int(m.sum()) >= 3:
        smp = (log_prob.detach() - ref_log_prob.to(log_prob.device))[m].double()
        r = pearson(st["rkl"][m].double(), smp)
        met["dense/corr_sampled"] = r if r is not None else 0.0
        met["dense/rkl_sampled"] = float(smp.mean())   # [Fable's A1] the sampled estimate's token mean: the same expectation as dense/rkl
    return loss, met
