"""Zero-variance groups dropped before the update, with oversampling (runbook [v2.5.9w], Q-039 r3; the reviewer's ruling
16:30 UTC 10-21 and its revision 16:4x UTC: rows removed, not masked).

E12b-P5-r3 draws 128 prompts x 8 rollouts a step (twice E12b-P5's 64). Right after the rollouts are joined to the batch, and
before the old log-probs, every rollout is scored with the trainer's own reward function; a group (the 8 rollouts of one
prompt) whose scores are all equal carries no signal and is dropped. Of the groups with a spread, at most K (64) are kept:
all of them when there are K or fewer, otherwise K drawn at random with a seeded generator (OPD_SEED x 100003 + step, so a
step's choice can be reproduced). The kept rows go on (batch.select_idxs, as opd_lensel does), so the advantage, its batch
whitening (reinforce_plus_plus_baseline), the token-mean loss and the old log-probs see only kept rows: a dropped group gets
no gradient and does not enter the whitening statistics. If no group has a spread, the first group is kept: its advantages
are 0 after the group mean, and stay 0 after whitening, so the step makes no update (logged as r3/empty).

Logged per step (metrics r3/*): groups, groups with a spread, kept groups and rows, generated tokens (all rollouts) and kept
tokens, the score of all rollouts and of the kept ones, the zero-variance share, the step's generator seed; after the
advantage (adv_stats): the masked mean and variance of the advantages over the kept tokens (0 and 1 if the whitening ran on
the kept rows only), the rows and prompts of the update and its PPO mini-batch count. Per group (OPD_VARDROP_LOG, JSON
lines): scores, lengths, spread, kept or not.

Environment: OPD_VARDROP (1 = on); OPD_VARDROP_K (64); OPD_SEED (1234); OPD_VARDROP_LOG (path, optional);
OPD_VARDROP_EOS (token ids, default 151645,151643).
"""
import json
import math
import os

import numpy as np

from opd_lensel import group_rows, response_stats

EPS = 1e-9


def choose_groups(scores, groups, k, rng):
    """(kept row indices, kept group indices, groups with a spread, empty): at most k groups whose scores are not all equal."""
    spread = [gi for gi, g in enumerate(groups) if np.ptp(np.asarray([scores[i] for i in g], dtype=float)) > EPS]
    if len(spread) > k:
        chosen = sorted(int(x) for x in rng.choice(np.asarray(spread), size=k, replace=False))
    else:
        chosen = list(spread)
    empty = not chosen
    if empty:
        chosen = [0]
    rows = sorted(i for gi in chosen for i in groups[gi])
    return rows, chosen, spread, empty


def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _scores(batch, reward_fn):
    """Each rollout's score from the trainer's reward function. Unlike opd_lensel's (logging only), the selection depends
    on it, so a failure stops the step."""
    if reward_fn is None:
        raise RuntimeError("opd_vardrop: no reward function, groups cannot be scored")
    from verl.trainer.ppo.reward import compute_reward
    rt, _ = compute_reward(batch, reward_fn)
    return _np(rt.sum(-1)).astype(float)


def seed_of(step):
    return int(os.environ.get("OPD_SEED", "1234")) * 100003 + int(step)


def select_groups(batch, reward_fn=None, step=0, scores=None):
    """The hook: returns (the batch with the kept groups' rows, metrics). scores: given by the tests instead of reward_fn."""
    k = int(os.environ.get("OPD_VARDROP_K", "64"))
    eos_ids = [int(x) for x in os.environ.get("OPD_VARDROP_EOS", "151645,151643").split(",") if x.strip()]
    seed = seed_of(step)
    rng = np.random.default_rng(seed)
    if scores is None:
        scores = _scores(batch, reward_fn)
    scores = np.asarray(scores, dtype=float)
    lengths, trunc, eos = response_stats(batch.batch["responses"], batch.batch["attention_mask"], eos_ids)
    groups = group_rows(list(batch.non_tensor_batch["uid"]))
    rows, chosen, spread, empty = choose_groups(scores, groups, k, rng)
    sel = np.zeros(len(lengths), dtype=bool)
    sel[rows] = True
    m = {"r3/groups": float(len(groups)), "r3/nonzero_groups": float(len(spread)), "r3/kept_groups": float(len(chosen)),
         "r3/kept_rows": float(len(rows)), "r3/gen_tokens": float(lengths.sum()), "r3/kept_tokens": float(lengths[sel].sum()),
         "r3/all_score": float(scores.mean()), "r3/kept_score": float(scores[sel].mean()),
         "r3/zero_var_frac": float(1 - len(spread) / max(1, len(groups))), "r3/empty": float(empty),
         "r3/all_clip": float(trunc.mean()), "r3/kept_clip": float(trunc[sel].mean()), "r3/rng_seed": float(seed)}
    log = os.environ.get("OPD_VARDROP_LOG")
    if log:
        cs, sp = set(chosen), set(spread)
        with open(log, "a") as f:
            for gi, g in enumerate(groups):
                f.write(json.dumps({"step": int(step), "group": gi, "rng_seed": seed, "score": [float(scores[i]) for i in g],
                                    "len": [int(lengths[i]) for i in g], "trunc": [int(trunc[i]) for i in g],
                                    "spread": gi in sp, "kept": gi in cs, "empty_step": bool(empty)}) + "\n")
    return batch.select_idxs(np.asarray(rows, dtype=np.int64)), m


def adv_stats(batch, mini_batch_prompts, n):
    """After compute_advantage: the advantages' masked mean / variance over the kept tokens (verl's masked_whiten: unbiased
    variance), the update's rows and prompts, and its PPO mini-batch count (mini_batch_prompts x n rows each)."""
    adv = _np(batch.batch["advantages"]).astype(float)
    mask = _np(batch.batch["response_mask"]).astype(float)
    cnt = mask.sum()
    mean = float((adv * mask).sum() / max(cnt, 1.0))
    var = float((((adv - mean) ** 2) * mask).sum() / max(cnt - 1, 1.0))
    rows = adv.shape[0]
    return {"r3/adv_mean": mean, "r3/adv_var": var, "r3/update_rows": float(rows),
            "r3/update_prompts": float(len(set(batch.non_tensor_batch["uid"].tolist()))),
            "r3/ppo_mini_batches": float(math.ceil(rows / max(1, mini_batch_prompts * n)))}
