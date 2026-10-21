"""[v2.5.9z] (Q-051) unit tests of the VLM teacher's inputs: ids / mask / 3-D and 1-D positions (numpy ops, so they run anywhere),
the torch path and the hook (PROBE-VLM runs them on the node, GOPD_REF_INPUT_UTILS = the installed ref_input_utils.py)."""
import os
import shutil
import tempfile

import numpy as np

import opd_vlmref as V
import vlmref_patch as VP
import route_patch as RP

NP = {"cat": lambda xs, d: np.concatenate(xs, axis=d), "full": lambda shape, v, like: np.full(shape, v, dtype=like.dtype),
      "arange": lambda k, like: np.arange(k, dtype=like.dtype)}


def sample():
    """Two rows, prompt 5 (left padded: row 0 has 2 pads), response 3 (row 1: 1 pad); 4 position channels (text + t / h / w)."""
    ids = np.array([[0, 0, 11, 12, 13, 21, 22, 0], [14, 15, 16, 17, 18, 23, 24, 25]])
    am = np.array([[0, 0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
    pos1 = np.array([[0, 0, 0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7]])
    pos3 = np.stack([pos1, pos1, pos1 + 10, pos1 + 20], axis=1)   # (2, 4, 8); an image would make channels differ in the prompt
    resp = ids[:, 5:]; rm = am[:, 5:]
    return ids, am, pos1, pos3, resp, rm


def test_ids_mask_and_3d_positions():
    ids, am, pos1, pos3, resp, rm = sample()
    rid, ram, rpos = V.build(ids, am, pos3, resp, rm, [151667, 198], NP)
    assert rid.tolist() == [[0, 0, 11, 12, 13, 151667, 198, 21, 22, 0], [14, 15, 16, 17, 18, 151667, 198, 23, 24, 25]]
    assert ram.tolist() == [[0, 0, 1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    assert rpos.shape == (2, 4, 10)
    assert rpos[1, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]                   # text channel: +1, +2, then the response + 2
    assert rpos[1, 2].tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]          # every rope channel continues from its own last value
    assert rpos[0, 3, 5:7].tolist() == [23, 24] and rpos[0, 3, 7:].tolist() == [25, 26, 27]


def test_1d_positions_and_empty_suffix():
    ids, am, pos1, pos3, resp, rm = sample()
    rid, ram, rpos = V.build(ids, am, pos1, resp, rm, [151667, 198], NP)
    assert rpos.tolist()[1] == list(range(10)) and rpos.shape == (2, 10)
    rid, ram, rpos = V.build(ids, am, pos3, resp, rm, [], NP)                    # an Instruct teacher: the student's inputs as they are
    assert rid.tolist() == ids.tolist() and ram.tolist() == am.tolist() and rpos.tolist() == pos3.tolist()


def test_suffix_ids_from_the_tokenizer():
    class Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [151667, 198] if s == "<think>\n" else [7] * len(s)}
    assert V.suffix_ids(Tok()) == [151667, 198] and V.suffix_ids(Tok(), "") == []


def test_torch_path():
    try:
        import torch
    except ImportError:
        return
    ids, am, pos1, pos3, resp, rm = sample()
    t = lambda x: torch.tensor(x)
    rid, ram, rpos = V.build(t(ids), t(am), t(pos3), t(resp), t(rm), [151667, 198], V.torch_ops())
    nid, nam, npos = V.build(ids, am, pos3, resp, rm, [151667, 198], NP)
    assert rid.tolist() == nid.tolist() and ram.tolist() == nam.tolist() and rpos.tolist() == npos.tolist()


SNIPPET = ("def prepare_ref_model_inputs(\n    batch,\n    ref_tokenizer,\n    apply_chat_template_kwargs=None,\n):\n"
           "    if apply_chat_template_kwargs is None:\n        apply_chat_template_kwargs = {}\n    batch_size = len(batch)\n"
           "    for i in range(batch_size):\n        messages = list(batch[i])\n        import os as _os\n"
           "        _think = _os.environ.get('OPD_REF_THINK') == '1'\n    return batch\n\n\n"
           "def other(apply_chat_template_kwargs=None):\n    if apply_chat_template_kwargs is None:\n        apply_chat_template_kwargs = {}\n    return 1\n")


def test_patch_text_and_with_the_route_hook():
    s, ch = VP.patch_text(SNIPPET)
    assert ch and s.count(VP.MARK) == 1 and VP.patch_text(s) == (s, False)
    assert s.index(VP.MARK) < s.index("    batch_size = len(batch)") and s.index("def other") > s.index(VP.MARK)
    for x in (VP.patch_text(RP.patch_text(SNIPPET)[0])[0], RP.patch_text(VP.patch_text(SNIPPET)[0])[0]):
        compile(x, "x", "exec"); assert x.count(VP.MARK) == 1 and x.count(RP.MARK) == 2


def test_patch_file_and_the_real_file():
    d = tempfile.mkdtemp(); p = os.path.join(d, "ref_input_utils.py"); open(p, "w").write(SNIPPET)
    assert VP.patch_file(p) and open(p + ".orig").read() == SNIPPET and not VP.patch_file(p)
    real = os.environ.get("GOPD_REF_INPUT_UTILS")
    if real:
        q = os.path.join(d, "real.py"); shutil.copy2(real, q)
        done = VP.MARK in open(q).read()
        assert VP.patch_file(q, keep_orig=False) != done and open(q).read().count(VP.MARK) == 1
        compile(open(q).read(), q, "exec")
