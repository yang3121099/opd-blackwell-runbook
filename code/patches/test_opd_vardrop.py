"""[v2.5.9w] (Q-039 r3) unit tests of the zero-variance group drop: the choice, its statistics, the advantage statistics and
the trainer hooks (numpy only). Run: python -m pytest -q test_opd_vardrop.py, or any plain runner of the test_ functions
(no fixtures; PROBE-E12B-R3 runs them on the node); GOPD_RAY_TRAINER=<G-OPD's ray_trainer.py> also patches a copy of the
real file."""
import contextlib
import json
import os
import shutil
import tempfile

import numpy as np

import lensel_patch as LP
import opd_vardrop as V
import vardrop_patch as VP


def raises(exc, f, *a, **kw):
    try:
        f(*a, **kw)
    except exc:
        return True
    return False


@contextlib.contextmanager
def env(**kw):   # set (value) or remove (None) environment variables for the block
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


class FakeBatch:   # the parts of verl's DataProto the hooks use
    def __init__(self, batch, non_tensor_batch):
        self.batch, self.non_tensor_batch = batch, non_tensor_batch

    def select_idxs(self, idxs):
        return FakeBatch({k: v[idxs] for k, v in self.batch.items()}, {k: v[idxs] for k, v in self.non_tensor_batch.items()})


def _batch(n_groups, n=8, rl=8, prompt=3, eos=151645, length=lambda g, j: 1 + (g + j) % 7):
    """Interleaved rows (as verl's repeat(interleave=True)): n rollouts per group, the response lengths from length()."""
    am, resp, uids = [], [], []
    for g in range(n_groups):
        for j in range(n):
            k = length(g, j)
            r = [7] * k + [0] * (rl - k)
            if 0 < k < rl:
                r[k - 1] = eos
            am.append([1] * prompt + [1] * k + [0] * (rl - k)); resp.append(r); uids.append(f"u{g}")
    return FakeBatch({"attention_mask": np.array(am), "responses": np.array(resp), "row": np.arange(len(am))},
                     {"uid": np.array(uids, dtype=object)})


def _scores(spread_groups, n_groups, n=8):
    """Binary scores: groups in spread_groups have 3 of n correct, the others all correct (even) or all wrong (odd)."""
    s = []
    for g in range(n_groups):
        s += ([1.0] * 3 + [0.0] * (n - 3)) if g in spread_groups else [float(g % 2 == 0)] * n
    return np.array(s)


def test_choose_groups_caps_at_k_and_skips_zero_variance():
    groups = [list(range(g * 4, g * 4 + 4)) for g in range(10)]
    scores = np.array([1, 0, 1, 1] * 6 + [1, 1, 1, 1] * 2 + [0, 0, 0, 0] * 2, dtype=float)   # groups 0-5 have a spread
    rows, chosen, spread, empty = V.choose_groups(scores, groups, 4, np.random.default_rng(0))
    assert spread == [0, 1, 2, 3, 4, 5] and len(chosen) == 4 and set(chosen) <= set(spread) and not empty
    assert rows == sorted(i for g in chosen for i in groups[g]) and len(rows) == 16
    rows, chosen, spread, empty = V.choose_groups(scores, groups, 64, np.random.default_rng(0))
    assert chosen == spread and len(rows) == 24 and not empty                       # fewer than k: all of them
    # draws differ with the seed, never pick a group without a spread
    picks = {tuple(V.choose_groups(scores, groups, 3, np.random.default_rng(s))[1]) for s in range(30)}
    assert len(picks) > 3 and all(set(p) <= set(spread) for p in picks)


def test_choose_groups_empty_keeps_one_group():
    groups = [[0, 1], [2, 3]]
    rows, chosen, spread, empty = V.choose_groups(np.array([1.0, 1.0, 0.0, 0.0]), groups, 64, np.random.default_rng(0))
    assert empty and chosen == [0] and rows == [0, 1] and spread == []


def test_choose_groups_continuous_scores():
    # stop2-like scores: 1.47 vs 1.46 is a spread, exact ties are not
    groups = [[0, 1], [2, 3]]
    rows, chosen, spread, _ = V.choose_groups(np.array([1.47, 1.46, 1.2, 1.2]), groups, 64, np.random.default_rng(0))
    assert spread == [0] and rows == [0, 1]


def test_select_groups_rows_metrics_log():
    log = os.path.join(tempfile.mkdtemp(), "vardrop.jsonl")
    b = _batch(20)
    sc = _scores({1, 4, 6, 9, 12, 15, 18}, 20)
    with env(OPD_VARDROP="1", OPD_VARDROP_K="5", OPD_VARDROP_LOG=log, OPD_SEED="7"):
        out, m = V.select_groups(b, step=3, scores=sc)
    uids = out.non_tensor_batch["uid"].tolist()
    kept = sorted({int(u[1:]) for u in uids})
    assert len(uids) == 40 and len(kept) == 5 and set(kept) <= {1, 4, 6, 9, 12, 15, 18}
    assert out.batch["row"].tolist() == sorted(g * 8 + j for g in kept for j in range(8))        # whole groups, in order
    assert m["r3/groups"] == 20 and m["r3/nonzero_groups"] == 7 and m["r3/kept_groups"] == 5 and m["r3/kept_rows"] == 40
    assert m["r3/zero_var_frac"] == 13 / 20 and m["r3/empty"] == 0 and m["r3/rng_seed"] == 7 * 100003 + 3
    assert m["r3/gen_tokens"] == b.batch["attention_mask"][:, 3:].sum() and m["r3/kept_tokens"] < m["r3/gen_tokens"]
    assert m["r3/all_score"] == sc.mean() and m["r3/kept_score"] == 3 / 8
    rows = [json.loads(x) for x in open(log)]
    assert len(rows) == 20 and sum(r["kept"] for r in rows) == 5 and sum(r["spread"] for r in rows) == 7
    assert all(r["spread"] for r in rows if r["kept"]) and all(r["step"] == 3 and r["rng_seed"] == 7 * 100003 + 3 for r in rows)
    assert rows[1]["score"] == [1.0] * 3 + [0.0] * 5 and len(rows[0]["len"]) == 8


def test_select_groups_reproducible_by_step_and_seed():
    b = _batch(30); sc = _scores(set(range(0, 30, 2)), 30)
    with env(OPD_VARDROP_K="4", OPD_VARDROP_LOG=None, OPD_SEED="1234"):
        a1 = V.select_groups(b, step=5, scores=sc)[0].batch["row"].tolist()
        a2 = V.select_groups(b, step=5, scores=sc)[0].batch["row"].tolist()
        a3 = V.select_groups(b, step=6, scores=sc)[0].batch["row"].tolist()
    with env(OPD_VARDROP_K="4", OPD_VARDROP_LOG=None, OPD_SEED="99"):
        a4 = V.select_groups(b, step=5, scores=sc)[0].batch["row"].tolist()
    assert a1 == a2 and a1 != a3 and a1 != a4 and len(a1) == 32


def test_select_groups_needs_a_reward():
    assert raises(RuntimeError, V.select_groups, _batch(2), None, 1)


def test_adv_stats_after_whitening_on_kept_rows():
    # rpp-baseline on the kept rows: group-mean subtraction, then verl's masked_whiten (unbiased variance) -> mean 0, var 1
    rng = np.random.default_rng(0)
    b = _batch(6, rl=8)
    resp_mask = (b.batch["attention_mask"][:, 3:] > 0).astype(float)
    sc = rng.integers(0, 2, size=48).astype(float)
    for g in range(6):
        sc[g * 8:(g + 1) * 8] -= sc[g * 8:(g + 1) * 8].mean()
    a = sc[:, None] * resp_mask
    cnt = resp_mask.sum(); mu = (a * resp_mask).sum() / cnt
    var = (((a - mu) ** 2) * resp_mask).sum() / cnt * cnt / (cnt - 1)
    w = (a - mu) / np.sqrt(var + 1e-8) * resp_mask
    b.batch["advantages"], b.batch["response_mask"] = w, resp_mask
    m = V.adv_stats(b, 16, 8)
    assert abs(m["r3/adv_mean"]) < 1e-9 and abs(m["r3/adv_var"] - 1) < 1e-6
    assert m["r3/update_rows"] == 48 and m["r3/update_prompts"] == 6 and m["r3/ppo_mini_batches"] == 1
    big = _batch(40); big.batch["advantages"] = np.zeros((320, 8)); big.batch["response_mask"] = np.ones((320, 8))
    assert V.adv_stats(big, 16, 8)["r3/ppo_mini_batches"] == 3   # 320 rows / 128 a mini-batch


SNIPPET = ("        for epoch in range(3):\n            for batch_dict in []:\n                metrics = {}\n"
           "                with marked_timer('step', timing_raw):\n"
           "                    batch = batch.repeat(repeat_times=4, interleave=True)\n"
           "                    batch = batch.union(gen_batch_output)\n\n"
           "                    if \"response_mask\" not in batch.batch.keys():\n                        pass\n"
           "                    with marked_timer('adv', timing_raw):\n"
           "                        batch = compute_advantage(\n"
           "                            batch,\n"
           "                            config=self.config.algorithm,\n"
           "                        )\n\n"
           "                    if self.use_critic:\n                        pass\n")


def test_patch_text_idempotent_and_both_hooks():
    s, ch = VP.patch_text(SNIPPET)
    assert ch and s.count(VP.MARK) == 2
    i = s.index(VP.ANCHOR) + len(VP.ANCHOR)
    assert s[i:].startswith("                    if __import__('os').environ.get('OPD_VARDROP'):")
    assert "batch, _opd_vd_metrics = _opd_vardrop(batch, self.reward_fn, self.global_steps)" in s
    j = s.index(VP.ANCHOR2) + len(VP.ANCHOR2)
    assert s[j:].startswith("                        if __import__('os').environ.get('OPD_VARDROP'):")
    assert VP.patch_text(s) == (s, False)
    assert raises(AssertionError, VP.patch_text, SNIPPET + VP.ANCHOR) and raises(AssertionError, VP.patch_text, "nothing here\n")
    assert raises(AssertionError, VP.patch_text, SNIPPET.replace(VP.ANCHOR2, ""))
    compile(s.replace("        for", "for", 1).replace("\n        ", "\n"), "x", "exec")


def test_coexists_with_the_lensel_hook_either_order():
    a = VP.patch_text(LP.patch_text(SNIPPET)[0])[0]
    b = LP.patch_text(VP.patch_text(SNIPPET)[0])[0]
    for s in (a, b):
        assert s.count(VP.MARK) == 2 and s.count(LP.MARK) == 1 and s.count(VP.ANCHOR) == 1
        compile(s.replace("        for", "for", 1).replace("\n        ", "\n"), "x", "exec")


def test_patch_file_atomic_and_orig():
    d = tempfile.mkdtemp(); p = os.path.join(d, "ray_trainer.py"); open(p, "w").write(SNIPPET)
    assert VP.patch_file(p) and open(p + ".orig").read() == SNIPPET and VP.MARK in open(p).read()
    assert not VP.patch_file(p) and not os.path.exists(p + ".opd_tmp")
    real = os.environ.get("GOPD_RAY_TRAINER")
    if real:   # the node's G-OPD file: both anchors exist once and the patched text compiles
        q = os.path.join(d, "real.py"); shutil.copy2(real, q)
        done = VP.MARK in open(q).read()   # on the node the installed file may carry the hooks already (a rerun)
        assert VP.patch_file(q, keep_orig=False) != done and open(q).read().count(VP.MARK) == 2
        compile(open(q).read(), q, "exec")
