"""[v2.5.9x] (Q-041 / Q-042) unit tests of the per-prompt teacher template: keys, route / mix decisions, the log, the hooks in
ref_input_utils.py (alone and with stopmerge's), and the prompt order of diff_labels (numpy / torch-free parts only; the order
test needs torch and returns early without it). Run: python -m pytest -q test_opd_route.py, or any plain runner of the test_
functions (PROBE-ROUTE runs them on the node, GOPD_REF_INPUT_UTILS = the installed ref_input_utils.py)."""
import contextlib
import importlib
import json
import os
import shutil
import sys
import tempfile

import opd_route as R
import route_patch as RP
import stopmerge_patch as SP

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def msgs(t):
    return [{"role": "user", "content": t}]


def labels_file(d, rows):
    p = os.path.join(d, "labels_17.jsonl")
    with open(p, "w") as f:
        for k, r, split in rows:
            f.write(json.dumps({"split": split, "pos": 0, "key": k, "k": int(r * 8), "n": 8, "r": r, "mean_len": 1.0, "trunc": 0}) + "\n")
    return p


def fresh():
    importlib.reload(R)


def test_key_matches_diff_labels():
    import diff_labels as D
    for m in (msgs("What is 1+1?"), [{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]):
        assert R.key(m) == D.key(m) and len(R.key(m)) == 16
    assert R.key(msgs("a")) != R.key(msgs("b"))


def test_route_decisions_by_threshold_and_unknown():
    lab = {R.key(msgs("easy")): 0.75, R.key(msgs("edge")): 0.5, R.key(msgs("hard")): 0.125}
    assert R.decide(R.key(msgs("easy")), "route", lab) == (False, 0.75)     # easy -> non-thinking teacher
    assert R.decide(R.key(msgs("edge")), "route", lab) == (False, 0.5)      # r >= 0.5 is easy
    assert R.decide(R.key(msgs("hard")), "route", lab) == (True, 0.125)     # hard -> thinking teacher
    assert R.decide(R.key(msgs("new")), "route", lab) == (True, None)       # unknown -> thinking, counted
    try:
        R.decide("x", "median", lab); assert False
    except ValueError:
        pass


def test_mix_coin_is_fixed_half_and_label_free():
    ks = [R.key(msgs(f"p{i}")) for i in range(4000)]
    share = sum(R.coin(k, 1234) for k in ks) / len(ks)
    assert 0.47 < share < 0.53
    assert [R.coin(k, 1234) for k in ks[:50]] == [R.coin(k, 1234) for k in ks[:50]]   # fixed per prompt
    assert [R.coin(k, 1234) for k in ks[:200]] != [R.coin(k, 99) for k in ks[:200]]   # the seed matters
    assert R.decide(ks[0], "mix", {}) == (R.coin(ks[0], 1234), None)


def test_think_for_and_flush_log():
    d = tempfile.mkdtemp(); log = os.path.join(d, "route.jsonl")
    lab = labels_file(d, [(R.key(msgs("easy")), 0.875, "train"), (R.key(msgs("hard")), 0.0, "train"), (R.key(msgs("m500")), 1.0, "math500")])
    with env(OPD_ROUTE="route", OPD_ROUTE_LABELS=lab, OPD_ROUTE_LOG=log):
        fresh()
        got = [R.think_for(msgs(t)) for t in ("easy", "hard", "m500", "hard")]
        rec = R.flush(4)
    assert got == [False, True, True, True]                       # a MATH-500 key is not a training label: unknown
    assert rec["rows"] == 4 and rec["think"] == 3 and rec["plain"] == 1 and rec["unknown"] == 1 and rec["batch"] == 4
    line = json.loads(open(log).read().splitlines()[0])
    assert line["call"] == 1 and line["decisions"][0] == [R.key(msgs("easy")), 0.875, False]
    with env(OPD_ROUTE="mix", OPD_ROUTE_LABELS=lab, OPD_ROUTE_LOG=log, OPD_SEED="7"):
        fresh()
        R.think_for(msgs("easy")); rec = R.flush(1)
    assert rec["unknown"] == 0 and rec["decisions"][0][2] == R.coin(R.key(msgs("easy")), 7)



def test_count_mode_keeps_the_template():   # [v2.5.9z9] (Q-053) PROBE-T32: P5's own choice (OPD_REF_THINK), the rows only counted
    d = tempfile.mkdtemp(); log = os.path.join(d, "route.jsonl")
    for ref_think, want in (("1", True), ("0", False)):
        with env(OPD_ROUTE="count", OPD_ROUTE_LABELS=None, OPD_ROUTE_LOG=log, OPD_REF_THINK=ref_think):
            fresh()
            got = [R.think_for(msgs(t)) for t in ("easy", "hard", "x")]
            rec = R.flush(3)
        assert got == [want] * 3 and rec["mode"] == "count" and rec["rows"] == rec["batch"] == 3 and rec["unknown"] == 0
        assert rec["think"] == (3 if want else 0) and rec["plain"] == (0 if want else 3)
    lines = [json.loads(x) for x in open(log).read().splitlines()]
    assert len(lines) == 2 and [x["think"] for x in lines] == [3, 0]
    try:
        R.decide("k", "other", {}); raise RuntimeError("an unknown mode passed")
    except ValueError:
        pass


SNIPPET = (
    "import torch\n\n\ndef prepare_ref_model_inputs(\n    batch,\n    ref_tokenizer,\n    apply_chat_template_kwargs=None,\n):\n"
    "    batch_size = len(batch)\n    for i in range(batch_size):\n        messages = list(batch[i])\n"
    "        # [opd-variants patch v2] P5: teacher scores in think mode\n        import os as _os\n"
    "        _think = _os.environ.get('OPD_REF_THINK') == '1'\n"
    "        _kw = dict(apply_chat_template_kwargs, enable_thinking=True) if _think else apply_chat_template_kwargs\n"
    "    responses = None\n    response_attention_mask = None\n    return batch\n\n\ndef other():\n    return 1\n")


def test_patch_text_both_hooks_idempotent():
    s, ch = RP.patch_text(SNIPPET)
    assert ch and s.count(RP.MARK) == 2
    i = s.index(RP.ANCHOR) + len(RP.ANCHOR)
    assert s[i:].startswith("        if _os.environ.get('OPD_ROUTE'):") and "_think = _rt.think_for(messages)" in s
    assert s.index("_rt.flush(batch_size)") < s.index("    return batch\n") < s.index("def other")
    assert RP.patch_text(s) == (s, False)
    compile(s, "x", "exec")
    try:
        RP.patch_text(SNIPPET.replace(RP.ANCHOR, "")); assert False
    except AssertionError:
        pass


def test_with_stopmerge_either_order():
    for s in (RP.patch_text(SP.patch_ref_text(SNIPPET)[0])[0], SP.patch_ref_text(RP.patch_text(SNIPPET)[0])[0]):
        assert s.count(RP.MARK) == 2 and s.count(SP.MARK) >= 1
        compile(s, "x", "exec")


def test_patch_file_and_the_real_file():
    d = tempfile.mkdtemp(); p = os.path.join(d, "ref_input_utils.py"); open(p, "w").write(SNIPPET)
    assert RP.patch_file(p) and open(p + ".orig").read() == SNIPPET and not RP.patch_file(p)
    real = os.environ.get("GOPD_REF_INPUT_UTILS")
    if real:   # the node's installed file (P5 patch in, perhaps stopmerge's and ours): the patched copy compiles, two marks
        q = os.path.join(d, "real.py"); shutil.copy2(real, q)
        done = RP.MARK in open(q).read()
        assert RP.patch_file(q, keep_orig=False) != done and open(q).read().count(RP.MARK) == 2
        compile(open(q).read(), q, "exec")


def test_sampler_order_matches_randperm():
    try:
        import torch
    except ImportError:
        return
    import diff_labels as D
    g = torch.Generator(); g.manual_seed(42)
    want = list(torch.utils.data.RandomSampler(list(range(1000)), generator=g))
    assert D.sampler_order(1000, 42) == want and D.sampler_order(1000, 43) != want


def test_mix_coin_with_p():   # [v2.5.9z38] (Q-060) a threshold coin at p; unset p keeps E10b-P5mix's parity coin
    import hashlib
    ks = [R.key(msgs(f"p{i}")) for i in range(6400)]
    for p in (0.75, 0.9):
        share = sum(R.coin(k, 1234, p) for k in ks) / len(ks)
        assert abs(share - p) < 0.02, (p, share)
    assert all(R.coin(k, 1234, 0.9) for k in ks if R.coin(k, 1234, 0.75))   # nested: thinking at 0.75 -> thinking at 0.9
    assert [R.coin(k, 1234) for k in ks[:300]] == [int(hashlib.sha1(f"1234:{k}".encode()).hexdigest()[:8], 16) % 2 == 0 for k in ks[:300]]   # parity as run
    assert R.decide(ks[0], "mix", {}, p=0.9) == (R.coin(ks[0], 1234, 0.9), None)


def test_think_for_reads_route_p():   # [v2.5.9z38] OPD_ROUTE_P reaches the coin (no pytest fixtures: PROBE-ROUTE calls tests directly)
    import os
    keep = {k: os.environ.get(k) for k in ("OPD_ROUTE", "OPD_ROUTE_LABELS", "OPD_ROUTE_P")}; old = R._labels
    try:
        ks = [msgs(f"q{i}") for i in range(400)]
        os.environ.update(OPD_ROUTE="mix", OPD_ROUTE_LABELS="", OPD_ROUTE_P="0.9"); R._labels = None
        share = sum(R.think_for(m) for m in ks) / len(ks)
        assert 0.84 < share < 0.96
        del os.environ["OPD_ROUTE_P"]
        assert [R.think_for(m) for m in ks[:50]] == [R.coin(R.key(m), 1234) for m in ks[:50]]
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        R._labels = old
