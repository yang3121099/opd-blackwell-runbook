"""[v2.5.9l] (Q-038) unit tests of stop-merged OPD: the box position, the merge (numpy; torch when installed) and the
stop-set log-prob (torch). Run: python -m pytest -q test_opd_stopmerge.py, or any plain runner of the test_ functions
(PROBE-STOPMERGE runs them on the node, with torch)."""
import json
import math
import os
import unittest

import numpy as np

import opd_stopmerge as S


def test_first_box_end():
    assert S.first_box_end(["So ", "\\boxed{", "4", "2", "}", ". Done"]) == 4
    assert S.first_box_end(["\\bo", "xed{\\frac{1}", "{2}", "} then \\boxed{3}"]) == 3   # marker split, nested braces, first box
    assert S.first_box_end(["no box here"]) == -1
    assert S.first_box_end(["\\boxed{", "1 + {2"]) == -1   # never closed
    assert S.first_box_end([]) == -1


class Tok:   # per-token decoding of a toy vocabulary
    V = {1: "x = ", 2: "\\boxed{", 3: "7", 4: "}", 5: " ok", 9: ""}
    def batch_decode(self, rows):
        return [self.V[r[0]] for r in rows]


def test_box_ends_reads_valid_tokens_only():
    resp = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 9, 9], [2, 3, 4, 9, 9]])
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    assert S.box_ends(resp, mask, Tok()) == [3, -1, -1]   # row 2's box closes in the padding: not counted


def _case(xp):
    IM, TH, PAD = S.IM_END, S.THINK_CLOSE, S.ENDOFTEXT
    responses = xp.asarray([[11, 12, IM, 13, IM], [11, IM, 12, 13, PAD], [11, 12, 13, IM, PAD]])
    log_probs = xp.asarray([[-1.0] * 5, [-2.0] * 5, [-3.0] * 5])
    stop_lp = xp.asarray([[-0.1] * 5, [-0.2] * 5, [-0.3] * 5])
    box_end = xp.asarray([1, -1, 2])   # row 0's box closes at 1, row 1 has none, row 2's at 2
    return responses, log_probs, stop_lp, box_end


def test_merge_numpy():
    r, lp, st, be = _case(np)
    valid = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])   # the last token of rows 1-2 is padding
    out, stats = S.merge(lp, st, r, be, S.eos_ids(), np, valid)
    assert out[0].tolist() == [-1.0, -1.0, -0.1, -1.0, -0.1]   # both stops after the box, nothing else
    assert out[1].tolist() == [-2.0] * 5                        # no box: unchanged (an early stop keeps its own log-prob)
    assert out[2].tolist() == [-3.0, -3.0, -3.0, -0.3, -3.0]   # the stop after the box; the padding after it untouched
    assert stats["merged"] == 3 and stats["rows_box"] == 2 and stats["violations"] == 0 and stats["eos_total"] == 4
    assert math.isclose(stats["gain_mean"], (0.9 * 2 + 2.7) / 3) and stats["eos_before_or_no_box"] == 1
    out2, stats2 = S.merge(lp, st, r, be, S.eos_ids(), np)   # without a mask the padding after a box counts too
    assert stats2["merged"] == 4



def test_after_box_detail_numpy():   # [v2.5.9z10] logging only: offset bins, each boxed row's max stop mass and first stop after the box
    r, lp, st, be = _case(np)
    valid = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    out, stats = S.merge(lp, st, r, be, S.eos_ids(), np, valid)
    assert out[0].tolist() == [-1.0, -1.0, -0.1, -1.0, -0.1]                     # the merge itself unchanged
    mean15, k15 = stats["stop_mass_by_offset"]["1-5"]
    assert k15 == 4 and math.isclose(mean15, (3 * math.exp(-0.1) + math.exp(-0.3)) / 4)   # row 0: offsets 1-3, row 2: offset 1 (4 is padding)
    assert stats["stop_mass_by_offset"]["6-50"] == [None, 0] and stats["stop_mass_by_offset"][">50"] == [None, 0]
    assert stats["row_max_stop_mass"] == [round(math.exp(-0.1), 5), round(math.exp(-0.3), 5)]   # rows 0 and 2 (row 1: no box)
    assert stats["eos_offset_after_box"] == [1, 1] and stats["utc"].endswith("Z")
    json.dumps(stats)   # one JSON line


def test_after_box_detail_far_offsets_no_stop_box_at_the_end():
    L = 60
    r = np.full((3, L), 11); r[0, 55] = S.IM_END                                  # row 0: box at 0, its stop at offset 55
    st = np.full((3, L), -2.0); st[0, 1:6] = -0.5; st[1, 30] = -0.05              # row 1: box at 2, no stop after it; a peak at 30
    out, stats = S.merge(np.full((3, L), -9.0), st, r, np.array([0, 2, L - 1]), S.eos_ids(), np)   # row 2: box on the last token
    b = stats["stop_mass_by_offset"]
    assert b["1-5"][1] == 5 + 5 and math.isclose(b["1-5"][0], (5 * math.exp(-0.5) + 5 * math.exp(-2.0)) / 10)
    assert b["6-50"][1] == 45 + 45 and b[">50"][1] == 9 + 7                         # offsets 51-59 (row 0), 51-57 (row 1)
    assert stats["eos_offset_after_box"] == [55, -1, -1] and stats["row_max_stop_mass"] == [round(math.exp(-0.5), 5), round(math.exp(-0.05), 5), None]


def test_merge_at_the_box_token_itself_is_not_merged():
    r = np.array([[S.IM_END, S.IM_END]]); lp = np.array([[-5.0, -5.0]]); st = np.array([[-1.0, -1.0]])
    out, stats = S.merge(lp, st, r, np.array([0]), S.eos_ids(), np)
    assert out.tolist() == [[-5.0, -1.0]] and stats["merged"] == 1


def test_ids_from_environment(monkeypatch=None):
    import os
    old = os.environ.get("OPD_STOPMERGE_IDS")
    os.environ["OPD_STOPMERGE_IDS"] = "7, 8"
    try:
        assert S.stop_ids() == [7, 8]
    finally:
        os.environ.pop("OPD_STOPMERGE_IDS") if old is None else os.environ.__setitem__("OPD_STOPMERGE_IDS", old)
    assert S.stop_ids() == [S.IM_END, S.THINK_CLOSE, S.ENDOFTEXT] and S.eos_ids() == [S.IM_END, S.ENDOFTEXT]


def test_torch_merge_and_stop_logprob():
    try:
        import torch
    except ImportError:   # the Mac has no torch; PROBE-STOPMERGE runs this on the node
        return
    r, lp, st, be = _case(torch)
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    out, stats = S.merge(lp, st, r, be, S.eos_ids(), torch, valid)
    # [v2.5.9z2] float32 (torch's default): -0.1 reads back as -0.10000000149..., so compared with a tolerance (the numpy test,
    # float64, keeps the exact one); this exact comparison failed PROBE-STOPMERGE on the node 10-21 19:24 UTC
    assert torch.allclose(out[0], torch.tensor([-1.0, -1.0, -0.1, -1.0, -0.1], dtype=out.dtype), atol=1e-6), out[0]
    assert stats["merged"] == 3 and stats["violations"] == 0
    g = torch.Generator().manual_seed(0)
    logits = (torch.randn(2, 3, 50, generator=g) * 3).to(torch.bfloat16)
    labels = torch.randint(0, 50, (2, 3), generator=g)
    ref = torch.log_softmax(logits.float(), -1)
    log_probs = ref.gather(-1, labels.unsqueeze(-1)).squeeze(-1)   # as verl's logprobs_from_logits gives it
    ids = [3, 7, 11]
    got = S.stop_logprob(logits, labels, log_probs, ids)
    want = torch.logsumexp(ref[..., ids], -1)
    assert torch.allclose(got, want, atol=1e-5), (got, want)
    one = S.stop_logprob(logits, labels, log_probs, [5])
    assert torch.allclose(one, ref[..., 5], atol=1e-5)


def test_torch_after_box_detail_matches_numpy():   # [v2.5.9z10] node only (torch): the same fields as the numpy path
    try:
        import torch
    except ImportError:
        return
    r, lp, st, be = _case(np)
    valid = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
    _, a = S.merge(lp, st, r, be, S.eos_ids(), np, valid)
    t = lambda x: torch.tensor(x)
    _, b = S.merge(t(lp).float(), t(st).float(), t(r), t(be), S.eos_ids(), torch, t(valid))
    assert a["eos_offset_after_box"] == b["eos_offset_after_box"] and a["stop_mass_by_offset"]["1-5"][1] == b["stop_mass_by_offset"]["1-5"][1]
    assert all(math.isclose(x, y, rel_tol=1e-5) for x, y in zip(a["row_max_stop_mass"], b["row_max_stop_mass"]))
    assert math.isclose(a["stop_mass_by_offset"]["1-5"][0], b["stop_mass_by_offset"]["1-5"][0], rel_tol=1e-5)


def test_torch_positions_on_the_responses_device():   # [v2.5.9z4] the unit tests have no GPU: the device argument is checked
    try:
        import torch
    except ImportError:   # the Mac has no torch; PROBE-STOPMERGE runs this on the node
        return
    seen, orig = [], torch.arange

    def rec(*a, **k):
        seen.append(k.get("device"))
        return orig(*a, **k)
    r, lp, st, be = _case(torch)
    torch.arange = rec
    try:
        S.merge(lp, st, r, be, S.eos_ids(), torch)
    finally:
        torch.arange = orig
    assert seen and all(d == r.device for d in seen), seen


def test_patch_texts_on_the_real_files():   # GOPD_VERL=<G-OPD's verl package dir> (the node: the installed one; PROBE-STOPMERGE)
    import os
    import stopmerge_patch as P
    d = os.environ.get("GOPD_VERL")
    if not d:   # [v2.5.9z6] a visible skip (pytest and the probe's runner count it), not a silent pass
        raise unittest.SkipTest("GOPD_VERL not set")
    for rel, fn, n in (("trainer/ppo/ref_input_utils.py", P.patch_ref_text, 1), ("workers/actor/dp_actor.py", P.patch_actor_text, 6)):
        _check_real_file(os.path.join(d, rel), rel, fn, n, P.MARK)


def _check_real_file(path, rel, fn, n, mark):
    """[v2.5.9z6] the installed file as it is, and its rollback copy <file>.orig when there is one: an unpatched text gets
    both hooks once, compiles and is patched idempotently; one already patched (an earlier smoke patches the installed
    file: PROBE-STOPMERGE's third run 10-21 20:20 UTC failed on it) has the hooks n times, compiles and is left unchanged."""
    texts = [open(path).read()] + ([open(path + ".orig").read()] if os.path.exists(path + ".orig") else [])
    for src in texts:
        if mark in src:
            assert src.count(mark) == n, (rel, "already patched", src.count(mark))
            compile(src, rel, "exec")
            assert fn(src) == (src, False), (rel, "already patched, changed again")
            continue
        new, changed = fn(src)
        assert changed and new.count(mark) == n, (rel, new.count(mark))
        compile(new, rel, "exec")
        again, changed2 = fn(new)
        assert not changed2 and again == new   # idempotent


def test_real_file_check_on_patched_installed_files():
    """[v2.5.9z6] G-OPD's two files (GOPD_SRC: the source tree's verl dir; on the node its files may be patched already, then
    their .orig is the unpatched text), checked unpatched, then patched in place with a .orig (as the smoke leaves the
    installed package), and a wrong hook count must fail. Skipped (visibly) without GOPD_SRC or without an unpatched text."""
    import shutil
    import tempfile
    import stopmerge_patch as P
    src = os.environ.get("GOPD_SRC")
    if not src:
        raise unittest.SkipTest("GOPD_SRC not set")
    d = tempfile.mkdtemp()
    try:
        for rel, fn, n in (("trainer/ppo/ref_input_utils.py", P.patch_ref_text, 1), ("workers/actor/dp_actor.py", P.patch_actor_text, 6)):
            f = os.path.join(src, rel)
            if P.MARK in open(f).read():
                f += ".orig"
                if not os.path.exists(f) or P.MARK in open(f).read():
                    raise unittest.SkipTest(f"GOPD_SRC's {rel} is patched and has no unpatched .orig")
            p = os.path.join(d, rel); os.makedirs(os.path.dirname(p), exist_ok=True); shutil.copyfile(f, p)
            _check_real_file(p, rel, fn, n, P.MARK)                        # unpatched, no .orig
            assert P.patch_file(p, fn) and os.path.exists(p + ".orig")      # the smoke's patch
            _check_real_file(p, rel, fn, n, P.MARK)                        # patched + its .orig: the third run's case
            try:
                _check_real_file(p, rel, fn, n + 1, P.MARK); raise RuntimeError("a wrong hook count passed")
            except AssertionError:
                pass
    finally:
        shutil.rmtree(d)


def test_patch_file_atomic_with_rollback_copy():
    import os
    import tempfile
    import stopmerge_patch as P
    d = tempfile.mkdtemp(); p = os.path.join(d, "ref_input_utils.py")
    src = ("import torch\ndef prepare_ref_model_inputs(\n    batch, ref_tokenizer, apply_chat_template_kwargs=None):\n"
           "    responses = response_attention_mask = None\n    return batch\n\ndef other():\n    return batch\n")
    open(p, "w").write(src)
    assert P.patch_file(p, P.patch_ref_text) and open(p + ".orig").read() == src
    s = open(p).read()
    assert s.count(P.MARK) == 1 and s.index(P.MARK) < s.index("    return batch\n\ndef other")   # in the right function
    assert not P.patch_file(p, P.patch_ref_text) and open(p).read() == s   # second run: unchanged
