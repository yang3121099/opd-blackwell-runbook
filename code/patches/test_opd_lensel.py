"""[v2.5.8m] (Q-030) unit tests of length-selected OPD: the selection, its statistics and the trainer hook (numpy only).
Run: python -m pytest -q test_opd_lensel.py, or any plain runner of the test_ functions (no fixtures; PROBE-LENSEL);
GOPD_RAY_TRAINER=<G-OPD's ray_trainer.py> also patches a copy of the real file."""
import contextlib
import json
import os
import shutil
import tempfile

import numpy as np

import lensel_patch as LP
import opd_lensel as L


def raises(exc, f, *a):
    try:
        f(*a)
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


class FakeBatch:   # the parts of verl's DataProto the hook uses
    def __init__(self, batch, non_tensor_batch):
        self.batch, self.non_tensor_batch, self.meta_info = batch, non_tensor_batch, {"m": 1}

    def select_idxs(self, idxs):
        return FakeBatch({k: v[idxs] for k, v in self.batch.items()}, {k: v[idxs] for k, v in self.non_tensor_batch.items()})


def _batch(lengths_per_group, rl=8, prompt=3, eos=151645, pad=0, no_eos=()):
    """Interleaved rows (as verl's repeat(interleave=True)): group g holds len(lengths_per_group[g]) responses."""
    rows, uids = [], []
    for g, ls in enumerate(lengths_per_group):
        for n in ls:
            resp = [7] * n + [pad] * (rl - n)
            if 0 < n and (g, n) not in no_eos and n < rl:
                resp[n - 1] = eos
            rows.append(([1] * prompt + [1] * n + [0] * (rl - n), resp))
            uids.append(f"u{g}")
    am = np.array([r[0] for r in rows]); resp = np.array([r[1] for r in rows])
    return FakeBatch({"attention_mask": am, "responses": resp, "row": np.arange(len(rows))}, {"uid": np.array(uids, dtype=object)})


def test_choose_short_long_rand():
    lengths = [5, 1, 4, 2, 8, 3, 7, 6]
    g = [list(range(8))]
    rng = np.random.default_rng(0)
    assert L.choose(lengths, g, "short", 3, rng) == [1, 3, 5]
    assert L.choose(lengths, g, "long", 3, rng) == [4, 6, 7]
    r = L.choose(lengths, g, "rand", 3, rng)
    assert len(r) == 3 and len(set(r)) == 3 and set(r) <= set(range(8))
    assert raises(ValueError, L.choose, lengths, g, "median", 3, rng)


def test_choose_ties_random_and_groups():
    # every response truncated at the same length: 'long' must not always take the first rollouts of the group
    lengths = [8] * 16
    picks = {tuple(L.choose(lengths, [list(range(16))], "long", 4, np.random.default_rng(s))) for s in range(20)}
    assert len(picks) > 1 and (0, 1, 2, 3) not in picks or len(picks) > 5
    # two groups, a short group keeps all its rows
    assert L.choose([3, 1, 2, 9, 9], [[0, 1, 2], [3, 4]], "short", 2, np.random.default_rng(1)) == [1, 2, 3, 4]


def test_group_rows_order():
    assert L.group_rows(["b", "b", "a", "a", "b"]) == [[0, 1, 4], [2, 3]]


def test_response_stats():
    b = _batch([[3, 8, 0, 5]], rl=8, no_eos={(0, 5)})
    lengths, trunc, eos = L.response_stats(b.batch["responses"], b.batch["attention_mask"], [151645, 151643])
    assert lengths.tolist() == [3, 8, 0, 5]
    assert eos.tolist() == [True, False, False, False]          # the full-length one ends without EOS
    assert trunc.tolist() == [False, True, False, False]        # truncated = full length and no EOS
    # a response that ends with EOS exactly at the limit is not truncated
    resp = np.array([[7] * 7 + [151645]]); am = np.ones((1, 11), dtype=int)
    l2, t2, e2 = L.response_stats(resp, am, [151645])
    assert l2.tolist() == [8] and t2.tolist() == [False] and e2.tolist() == [True]


def test_select_by_length():
    log = os.path.join(tempfile.mkdtemp(), "lensel.jsonl")
    b = _batch([[6, 2, 8, 4], [1, 8, 8, 3]], rl=8)
    with env(OPD_LENSEL="short", OPD_LENSEL_K="2", OPD_LENSEL_LOG=log):
        out, m = L.select_by_length(b, reward_fn=None, step=3)
    assert out.batch["row"].tolist() == [1, 3, 4, 7] and out.non_tensor_batch["uid"].tolist() == ["u0", "u0", "u1", "u1"]
    assert m["lensel/n_all"] == 8 and m["lensel/n_selected"] == 4 and m["lensel/groups"] == 2
    assert m["lensel/sel_len_mean"] == 2.5 and m["lensel/all_len_mean"] == 5.0
    assert m["lensel/all_clip"] == 3 / 8 and m["lensel/sel_clip"] == 0
    assert "lensel/all_score" not in m
    rows = [json.loads(x) for x in open(log)]
    assert len(rows) == 2 and rows[0]["len"] == [6, 2, 8, 4] and rows[0]["kept"] == [1, 3] and rows[1]["kept"] == [0, 3]
    assert rows[0]["step"] == 3 and rows[0]["mode"] == "short" and rows[0]["score"] is None
    with env(OPD_LENSEL="long", OPD_LENSEL_K="2", OPD_LENSEL_LOG=None):
        out, m = L.select_by_length(b, step=3)
    assert out.batch["row"].tolist()[:2] == [0, 2] and m["lensel/sel_len_mean"] >= m["lensel/all_len_mean"]


def test_select_is_reproducible():
    b = _batch([[i % 8 for i in range(16)]] * 3, rl=8)
    with env(OPD_LENSEL="rand", OPD_LENSEL_K="4", OPD_LENSEL_LOG=None):
        a1 = L.select_by_length(b, step=5)[0].batch["row"].tolist()
        a2 = L.select_by_length(b, step=5)[0].batch["row"].tolist()
        a3 = L.select_by_length(b, step=6)[0].batch["row"].tolist()
    assert a1 == a2 and a1 != a3 and len(a1) == 12


SNIPPET = ("        for epoch in range(3):\n            for batch_dict in []:\n                metrics = {}\n"
           "                with marked_timer('step', timing_raw):\n"
           "                    batch = batch.repeat(repeat_times=4, interleave=True)\n"
           "                    batch = batch.union(gen_batch_output)\n\n"
           "                    if \"response_mask\" not in batch.batch.keys():\n                        pass\n")


def test_patch_text_idempotent():
    s, ch = LP.patch_text(SNIPPET)
    assert ch and s.count(LP.MARK) == 1
    i = s.index(LP.ANCHOR) + len(LP.ANCHOR)
    assert s[i:].startswith("                    if __import__('os').environ.get('OPD_LENSEL'):")
    assert "batch, _opd_ls_metrics = _opd_lensel(batch, self.reward_fn, self.global_steps)" in s
    assert LP.patch_text(s) == (s, False)
    assert raises(AssertionError, LP.patch_text, SNIPPET + LP.ANCHOR) and raises(AssertionError, LP.patch_text, "nothing here\n")
    compile(s.replace("        for", "for", 1).replace("\n        ", "\n"), "x", "exec")


def test_patch_file_atomic():
    p = os.path.join(tempfile.mkdtemp(), "ray_trainer.py"); open(p, "w").write(SNIPPET)
    assert LP.patch_file(p) is True and LP.patch_file(p) is False
    assert not os.path.exists(p + ".opd_tmp") and open(p).read().count(LP.MARK) == 1


def test_patch_real_trainer():   # the anchor is found once in G-OPD's ray_trainer.py (GOPD_RAY_TRAINER) and the result compiles
    if not os.environ.get("GOPD_RAY_TRAINER"):
        return
    p = os.path.join(tempfile.mkdtemp(), "ray_trainer.py"); shutil.copy(os.environ["GOPD_RAY_TRAINER"], p)
    s0 = open(p).read()
    if LP.MARK in s0:   # the node's installed copy may carry the hook already: check it compiles and holds it once
        assert s0.count(LP.MARK) == 1
    else:
        assert LP.patch_file(p) is True
    compile(open(p).read(), p, "exec")
