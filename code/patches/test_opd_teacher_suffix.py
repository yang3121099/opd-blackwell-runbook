"""[z163] (lab/registry/Q-101.md E10b-P5selfp2) unit tests of the teacher-side prompt suffix: opd_route.teacher_suffix (empty: the same
object, the teacher input byte for byte as before; set: the suffix on its own line at the end of the last user message, in a new list /
dict, the batch's rows - the student's prompt - untouched), the per-call log, teacher_suffix_patch.py (the hook before the P5 patch's
_kw line, the flush before the return, idempotent, either order with route_patch.py / mode_patch.py), and an end-to-end run on a stand-in
prepare_ref_model_inputs. No torch. Run: python -m pytest -q test_opd_teacher_suffix.py"""
import contextlib
import importlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mode_patch as MP  # noqa: E402
import opd_route as R  # noqa: E402
import route_patch as RP  # noqa: E402
import teacher_suffix_patch as TP  # noqa: E402

P2 = open(os.path.join(HERE, "..", "train", "teacher_suffix", "p2.txt")).read()


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


def msgs(t, system=None):
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": t}]


# the stand-in of G-OPD's prepare_ref_model_inputs after apply_patches.py's P5 patch (the real _kw line); SEEN = what the teacher renders
SNIPPET = ("SEEN = []\n\n\ndef prepare_ref_model_inputs(\n    batch,\n    ref_tokenizer,\n    apply_chat_template_kwargs=None,\n):\n"
           "    batch_size = len(batch)\n    for i in range(batch_size):\n        messages = list(batch[i])\n"
           "        # [opd-variants patch v2] P5: teacher scores in think mode\n        import os as _os\n"
           "        _think = _os.environ.get('OPD_REF_THINK') == '1'\n"
           "        _kw = dict(apply_chat_template_kwargs, enable_thinking=True) if _think else apply_chat_template_kwargs\n"
           "        SEEN.append([dict(m) for m in messages])\n"
           "    responses = None\n    return batch\n\n\ndef other():\n    return 1\n")


def test_p2_text_verbatim():
    assert P2 == "Be concise. Use as few reasoning steps as possible, then give the final answer."   # Q-098 / Q-101's P2, no newline


def test_empty_suffix_is_the_same_object():
    m = msgs("What is 1+1?", system="s")
    for v in (None, ""):
        with env(OPD_TEACHER_SUFFIX=v):
            assert R.teacher_suffix(m) is m


def test_suffix_last_user_message_only_and_a_copy():
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "q2"}]
    keep = json.dumps(m)
    with env(OPD_TEACHER_SUFFIX=P2):
        out = R.teacher_suffix(m)
    assert out[-1]["content"] == "q2\n" + P2 and [x["content"] for x in out[:3]] == ["s", "q1", "a"]
    assert json.dumps(m) == keep and out is not m and out[-1] is not m[-1] and out[0] is m[0]   # the input (the student's) untouched
    with env(OPD_TEACHER_SUFFIX=P2):
        assert R.teacher_suffix([{"role": "system", "content": "s"}])[0]["content"] == "s"     # no user message: unchanged


def test_flush_counts_and_log():
    log = os.path.join(tempfile.mkdtemp(), "ts.jsonl")
    with env(OPD_TEACHER_SUFFIX=P2, OPD_TEACHER_SUFFIX_LOG=log):
        importlib.reload(R)
        for t in ("a", "b", "c"):
            R.teacher_suffix(msgs(t))
        rec = R.teacher_suffix_flush(3)
    assert rec["rows"] == rec["suffixed"] == rec["batch"] == 3 and rec["suffix_chars"] == len(P2)
    assert json.loads(open(log).read().splitlines()[0])["suffixed"] == 3


def test_patch_text_and_idempotent_either_order():
    s, ch = TP.patch_text(SNIPPET)
    assert ch and s.count(TP.MARK) == 2   # the hook and the flush
    i = s.index(TP.HOOK)
    assert s[i + len(TP.HOOK):].startswith(TP.ANCHOR_PREFIX)   # right before the _kw line
    assert TP.patch_text(s) == (s, False)
    a = TP.patch_text(MP.patch_text(RP.patch_text(SNIPPET)[0])[0])[0]   # after route + mode
    b = MP.patch_text(RP.patch_text(TP.patch_text(SNIPPET)[0])[0])[0]   # before them
    for x in (a, b):
        compile(x, "ref_input_utils.py", "exec")
        assert x.count(TP.MARK) == 2 and x.count(RP.MARK) >= 1 and x.count(MP.MARK) == 1
    try:
        TP.patch_text(SNIPPET.replace("        _kw = dict(apply_chat_template_kwargs", "        _kw2 = dict(apply_chat_template_kwargs")); assert False
    except AssertionError as e:
        assert "anchor found 0 times" in str(e)


def test_end_to_end_teacher_only():
    s = TP.patch_text(SNIPPET)[0]
    batch = [msgs("a"), msgs("b", system="s")]
    keep = json.dumps(batch)
    for suf, want in (("", ["a", "b"]), (None, ["a", "b"]), (P2, ["a\n" + P2, "b\n" + P2])):
        ns = {}; log = os.path.join(tempfile.mkdtemp(), "ts.jsonl")
        with env(OPD_TEACHER_SUFFIX=suf, OPD_TEACHER_SUFFIX_LOG=log, OPD_REF_THINK="1", OPD_ROUTE=None):
            importlib.reload(R); sys.modules["opd_route"] = R
            exec(compile(s, "ref_input_utils.py", "exec"), ns)
            ns["prepare_ref_model_inputs"](batch, None, {})
        assert [x[-1]["content"] for x in ns["SEEN"]] == want
        if suf:
            rec = json.loads(open(log).read().splitlines()[-1])
            assert rec["rows"] == rec["suffixed"] == 2
        else:
            assert not os.path.exists(log)   # no suffix: no log line, the teacher input as before
    assert json.dumps(batch) == keep   # the batch's rows (the student's prompt) never change
