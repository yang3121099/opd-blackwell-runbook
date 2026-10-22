"""[v2.5.9z46] opd_x5 / x5_patch (Q-057 E16-X5): one '<think>\\n' at the teacher prompt's end (or an empty thought), the terminal stop id
mapped to the teacher's EOS, the other special ids out of the loss, the hooks in place. No pytest fixtures: PROBE-X5 calls these functions
directly with the python the arm trains in (torch there: the map_ids test runs; here it returns early)."""
import json
import os
import shutil
import tempfile

import opd_x5 as X
import route_patch as RP
import x5_patch as XP

P = "<｜begin▁of▁sentence｜><｜User｜>What is 1+1?<｜Assistant｜>"


def test_teacher_prompt_one_think_or_an_empty_thought():
    assert X.teacher_prompt(P + "<think>\n<think>\n", empty=False) == P + "<think>\n"     # R1's own + the P5 patch's
    assert X.teacher_prompt(P + "<think>\n", empty=False) == P + "<think>\n"
    assert X.teacher_prompt(P, empty=False) == P + "<think>\n"
    assert X.teacher_prompt(P + "<think>\n<think>\n", empty=True) == P + "<think>\n\n</think>\n\n"
    assert X._tail == (P + "<think>\n\n</think>\n\n")[-40:]
    old = os.environ.get("OPD_X5_EMPTY_THINK")
    try:
        os.environ["OPD_X5_EMPTY_THINK"] = "1"; assert X.teacher_prompt(P + "<think>\n").endswith("</think>\n\n")
        os.environ["OPD_X5_EMPTY_THINK"] = "0"; assert X.teacher_prompt(P + "<think>\n").endswith("<｜Assistant｜><think>\n")
    finally:
        os.environ.pop("OPD_X5_EMPTY_THINK", None) if old is None else os.environ.__setitem__("OPD_X5_EMPTY_THINK", old)


def test_plan_and_apply_rows():
    a, b, c = 11, 22, 33
    rows = [[a, b, 151645, 151643, 151643],      # stops with <|im_end|>, then padding (mask 0): mapped at 2
            [a, b, c, 151643, 151643],           # stops with <|endoftext|>: mapped at 3 (a teacher EOS already; counted)
            [a, b, c, a, b],                     # truncated: nothing mapped
            [a, 151644, 151648, 151645, 151645],  # special ids inside: 1, 2, 3 out of the loss; the end mapped at 4
            [151643, 151643, 151643, 151643, 151643]]   # an empty response (mask 0 everywhere)
    masks = [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    assert X.plan_row(rows[0], masks[0]) == (151645, 2, [])
    assert X.plan_row(rows[3], masks[3]) == (151645, 4, [1, 2, 3])
    assert X.plan_row(rows[4], masks[4]) == (None, None, [])
    ids, mask, cnt = X.apply_rows(rows, masks)
    assert ids[0] == [a, b, 151643, 151643, 151643] and mask[0] == masks[0]
    assert ids[1] == rows[1] and ids[2] == rows[2] and mask[2] == masks[2]
    assert ids[3] == [a, 151644, 151648, 151645, 151643] and mask[3] == [1, 0, 0, 0, 1]
    assert cnt == {"rows": 5, "terminated": 3, "term_151645": 2, "term_151643": 1, "mapped": 3, "truncated": 1, "empty": 1,
                   "masked": {"151644": 1, "151648": 1, "151645": 1}}
    assert rows[0][2] == 151645 and masks[3] == [1, 1, 1, 1, 1]   # the inputs untouched


SNIPPET = (
    "import torch\n\n\ndef prepare_ref_model_inputs(\n    batch,\n    ref_tokenizer,\n    apply_chat_template_kwargs=None,\n):\n"
    "    batch_size = len(batch)\n    response_length = 5\n    for i in range(batch_size):\n        messages = list(batch[i])\n"
    "        # [opd-variants patch v2] P5: teacher scores in think mode\n        import os as _os\n"
    "        _think = _os.environ.get('OPD_REF_THINK') == '1'\n"
    "        _kw = dict(apply_chat_template_kwargs, enable_thinking=True) if _think else apply_chat_template_kwargs\n"
    "        ref_prompt_str = ref_tokenizer.apply_chat_template(\n            messages, \n            add_generation_prompt=True, \n"
    "            tokenize=False,\n            **_kw\n        )\n"
    "        if _think:\n            ref_prompt_str = ref_prompt_str + '<think>\\n'\n"
    "    responses = None\n    return batch\n\n\ndef other():\n    return 1\n")


def test_patch_text_hooks_idempotent_and_with_route():
    s, ch = XP.patch_text(SNIPPET)
    assert ch and s.count(XP.MARK) == 2
    i = s.index(XP.ANCHOR) + len(XP.ANCHOR)
    assert s[i:].startswith("        if _os.environ.get('OPD_X5') == '1':") and "ref_prompt_str = _x5.teacher_prompt(ref_prompt_str)" in s
    assert s.index("_x5.map_ids(batch, response_length)") < s.index("    return batch\n") < s.index("def other")
    assert XP.patch_text(s) == (s, False)
    compile(s, "x", "exec")
    for t in (XP.patch_text(RP.patch_text(SNIPPET)[0])[0], RP.patch_text(XP.patch_text(SNIPPET)[0])[0]):   # either order with route's
        assert t.count(XP.MARK) == 2 and t.count(RP.MARK) == 2
        compile(t, "x", "exec")
    try:
        XP.patch_text(SNIPPET.replace(XP.ANCHOR, "")); assert False
    except AssertionError:
        pass


def test_patch_file_and_the_real_file():
    d = tempfile.mkdtemp(); p = os.path.join(d, "ref_input_utils.py"); open(p, "w").write(SNIPPET)
    assert XP.patch_file(p) and open(p + ".orig").read() == SNIPPET and not XP.patch_file(p)
    real = os.environ.get("GOPD_REF_INPUT_UTILS")
    if real:   # the node's installed file (P5 patch in, perhaps route's and ours): the patched copy compiles, two marks
        q = os.path.join(d, "real.py"); shutil.copy2(real, q)
        done = XP.MARK in open(q).read()
        assert XP.patch_file(q, keep_orig=False) != done and open(q).read().count(XP.MARK) == 2
        compile(open(q).read(), q, "exec")


def test_map_ids_on_tensors():   # the node (torch)
    try:
        import torch
    except ImportError:
        return

    class B:
        pass
    b = B(); prompt = [[7, 8, 9], [7, 8, 9]]
    resp = [[11, 22, 151645, 151643, 151643], [11, 151646, 22, 33, 44]]
    b.batch = {"ref_input_ids": torch.tensor([p + r for p, r in zip(prompt, resp)]), "response_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])}
    d = tempfile.mkdtemp(); log = os.path.join(d, "x5.jsonl"); old = os.environ.get("OPD_X5_LOG")
    try:
        os.environ["OPD_X5_LOG"] = log
        rec = X.map_ids(b, 5)
    finally:
        os.environ.pop("OPD_X5_LOG", None) if old is None else os.environ.__setitem__("OPD_X5_LOG", old)
    assert b.batch["ref_input_ids"].tolist() == [[7, 8, 9, 11, 22, 151643, 151643, 151643], [7, 8, 9, 11, 151646, 22, 33, 44]]
    assert b.batch["response_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 0, 1, 1, 1]] and b.batch["response_mask"].dtype == torch.int64
    assert rec["mapped"] == 1 and rec["truncated"] == 1 and rec["masked"] == {"151646": 1}
    line = json.loads(open(log).read().splitlines()[-1])
    assert line["call"] == rec["call"] and line["term_151645"] == 1
