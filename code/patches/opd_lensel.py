"""Length-selected OPD (runbook [v2.5.8m], Q-030; Runming's question 4, GFPO's filter as precedent).

Each prompt is rolled out n times (16); before the old / reference log-probs are computed, only k (4) of its responses are
kept: the shortest (OPD_LENSEL=short), the longest (long) or k at random (rand, the control). The update then sees
prompts x k sequences (32 x 4 = 128, the size of P5's update). Ties in length (e.g. several truncated responses) are broken
at random, so 'long' does not favour the first rollouts of a group.

Hooked into verl's RayPPOTrainer.fit right after `batch = batch.union(gen_batch_output)` (lensel_patch.py); a no-op unless
OPD_LENSEL is set. Logged per step (metrics lensel/*): length, truncation, EOS and score of all n and of the k kept;
per prompt (OPD_LENSEL_LOG, JSON lines): the n lengths, truncation / EOS flags, scores and the kept indices.

Environment: OPD_LENSEL short|long|rand; OPD_LENSEL_K (4); OPD_LENSEL_EOS (token ids, default 151645,151643 = Qwen3's
<|im_end|>, <|endoftext|>); OPD_SEED (1234); OPD_LENSEL_LOG (path, optional).
"""
import json
import os

import numpy as np

MODES = ("short", "long", "rand")


def choose(lengths, groups, mode, k, rng):
    """Indices kept: for each group (a list of row indices of one prompt, in rollout order) k rows by length.

    lengths: array of response lengths (one per row); ties broken by a random key drawn from rng."""
    if mode not in MODES:
        raise ValueError(f"OPD_LENSEL must be one of {MODES}, got {mode!r}")
    keep = []
    for g in groups:
        g = list(g)
        if len(g) <= k:
            keep.extend(g)
            continue
        tie = rng.random(len(g))
        if mode == "rand":
            order = np.argsort(tie, kind="stable")
        elif mode == "short":
            order = np.lexsort((tie, np.asarray([lengths[i] for i in g], dtype=float)))
        else:
            order = np.lexsort((tie, -np.asarray([lengths[i] for i in g], dtype=float)))
        keep.extend(sorted(g[j] for j in order[:k]))
    return sorted(keep)


def group_rows(uids):
    """Row indices grouped by uid, groups in order of first appearance."""
    groups, where = [], {}
    for i, u in enumerate(uids):
        if u not in where:
            where[u] = len(groups)
            groups.append([])
        groups[where[u]].append(i)
    return groups


def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def response_stats(responses, attention_mask, eos_ids):
    """(lengths, truncated, eos) per row from the padded response ids and the full attention mask."""
    resp = _np(responses)
    rl = resp.shape[1]
    mask = _np(attention_mask)[:, -rl:]
    lengths = mask.sum(-1).astype(int)
    last = np.array([resp[i, n - 1] if n > 0 else -1 for i, n in enumerate(lengths)])
    eos = np.isin(last, list(eos_ids))
    truncated = (lengths >= rl) & ~eos
    return lengths, truncated, eos


def _scores(batch, reward_fn):
    if reward_fn is None:
        return None
    try:
        from verl.trainer.ppo.reward import compute_reward
        rt, _ = compute_reward(batch, reward_fn)
        return _np(rt.sum(-1)).astype(float)
    except Exception as e:  # logging only: the selection does not depend on it
        print(f"[opd_lensel] scores of all rollouts not computed: {e}")
        return None


def select_by_length(batch, reward_fn=None, step=0):
    """The hook: returns (the batch with k rows per prompt, metrics)."""
    mode = os.environ.get("OPD_LENSEL", "")
    k = int(os.environ.get("OPD_LENSEL_K", "4"))
    eos_ids = [int(x) for x in os.environ.get("OPD_LENSEL_EOS", "151645,151643").split(",") if x.strip()]
    rng = np.random.default_rng(int(os.environ.get("OPD_SEED", "1234")) * 100003 + int(step))
    lengths, trunc, eos = response_stats(batch.batch["responses"], batch.batch["attention_mask"], eos_ids)
    groups = group_rows(list(batch.non_tensor_batch["uid"]))
    keep = choose(lengths, groups, mode, k, rng)
    scores = _scores(batch, reward_fn)
    sel = np.zeros(len(lengths), dtype=bool)
    sel[keep] = True
    m = {"lensel/n_all": float(len(lengths)), "lensel/n_selected": float(len(keep)), "lensel/groups": float(len(groups)),
         "lensel/all_len_mean": float(lengths.mean()), "lensel/sel_len_mean": float(lengths[sel].mean()),
         "lensel/all_clip": float(trunc.mean()), "lensel/sel_clip": float(trunc[sel].mean()),
         "lensel/all_eos": float(eos.mean()), "lensel/sel_eos": float(eos[sel].mean())}
    if scores is not None:
        m["lensel/all_score"] = float(scores.mean())
        m["lensel/sel_score"] = float(scores[sel].mean())
    log = os.environ.get("OPD_LENSEL_LOG")
    if log:
        with open(log, "a") as f:
            for gi, g in enumerate(groups):
                f.write(json.dumps({"step": int(step), "group": gi, "mode": mode, "len": [int(lengths[i]) for i in g],
                                    "trunc": [int(trunc[i]) for i in g], "eos": [int(eos[i]) for i in g],
                                    "score": None if scores is None else [float(scores[i]) for i in g],
                                    "kept": [j for j, i in enumerate(g) if sel[i]]}) + "\n")
    return batch.select_idxs(np.asarray(keep, dtype=np.int64)), m
