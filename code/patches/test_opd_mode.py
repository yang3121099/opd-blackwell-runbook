"""[v2.5.9z108] (Q-083 E10b-P5mode) unit tests of the mode token: opd_route's prefix mode (the token decides the teacher, the key and the
log are of the prompt without it), teacher_view / strip_messages (a new list, the batch untouched), mode_patch.py's line after route's
hook (alone, idempotent, and run end to end on a stand-in prepare_ref_model_inputs), and mode_data.py's prompts (mix's coin, the
token in front, refused twice). No torch, no pandas. Run: python -m pytest -q test_opd_mode.py (PROBE-MODE runs them on the node,
GOPD_REF_INPUT_UTILS = the installed ref_input_utils.py)."""
import contextlib
import importlib
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import mode_patch as MP  # noqa: E402
import opd_route as R  # noqa: E402
import route_patch as RP  # noqa: E402


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


# a stand-in for G-OPD's prepare_ref_model_inputs after the P5 patch: what the teacher renders is recorded in SEEN
SNIPPET = ("SEEN = []\n\n\ndef prepare_ref_model_inputs(\n    batch,\n    ref_tokenizer,\n    apply_chat_template_kwargs=None,\n):\n"
           "    batch_size = len(batch)\n    for i in range(batch_size):\n        messages = list(batch[i])\n"
           "        # [opd-variants patch v2] P5: teacher scores in think mode\n        import os as _os\n"
           "        _think = _os.environ.get('OPD_REF_THINK') == '1'\n"
           "        _kw = dict(apply_chat_template_kwargs or {}, enable_thinking=True) if _think else apply_chat_template_kwargs\n"
           "        SEEN.append((messages[-1]['content'], _think))\n"
           "    responses = None\n    response_attention_mask = None\n    return batch\n\n\ndef other():\n    return 1\n")


def test_split_and_prefixed():
    assert R.split_prefix("[long] What is 1+1?") == ("[long]", "What is 1+1?")
    assert R.split_prefix("[short] x") == ("[short]", "x")
    for t in ("[long]x", " [long] x", "[longer] x", "x [long] y", "[LONG] x", ""):
        assert R.split_prefix(t) == (None, t)                          # only the exact token and one space at the very start
    assert R.prefixed("q", True) == "[long] q" and R.prefixed("q", False) == "[short] q"
    for think in (True, False):
        tok, rest = R.split_prefix(R.prefixed("q", think))
        assert rest == "q" and R.TOKENS[tok] is think


def test_strip_messages_is_a_copy():
    m = msgs("[long] q", system="s")
    out = R.strip_messages(m)
    assert out == msgs("q", system="s") and m == msgs("[long] q", system="s")   # the input untouched
    assert out is not m and out[1] is not m[1] and out[0] is m[0]
    plain = msgs("q")
    assert R.strip_messages(plain) == plain
    with env(OPD_ROUTE="mix", OPD_MODE_STRIP=None):
        assert R.teacher_view(m) is m                                  # other modes: as before
    with env(OPD_ROUTE="prefix", OPD_MODE_STRIP=None):
        assert R.teacher_view(m) == msgs("q", system="s")
    with env(OPD_ROUTE="prefix", OPD_MODE_STRIP="0"):
        assert R.teacher_view(m) is m


def test_prefix_decisions_key_and_log():
    d = tempfile.mkdtemp(); log = os.path.join(d, "route.jsonl")
    with env(OPD_ROUTE="prefix", OPD_ROUTE_LABELS=os.path.join(d, "none.jsonl"), OPD_ROUTE_LOG=log):
        importlib.reload(R)
        got = [R.think_for(msgs(t)) for t in ("[long] a", "[short] b", "c", "[short] a")]
        rec = R.flush(4)
    assert got == [True, False, True, False]                           # no token: the thinking teacher, counted unknown
    assert rec["mode"] == "prefix" and rec["rows"] == 4 and rec["think"] == 2 and rec["plain"] == 2 and rec["unknown"] == 1
    assert rec["decisions"][0] == [R.key(msgs("a")), "[long]", True]   # the key of the prompt without the token
    assert rec["decisions"][3][0] == R.key(msgs("a")) and rec["decisions"][2] == [R.key(msgs("c")), None, True]
    assert json.loads(open(log).read().splitlines()[0])["unknown"] == 1


def test_mode_patch_text():
    s0 = RP.patch_text(SNIPPET)[0]
    s, ch = MP.patch_text(s0)
    assert ch and s.count(MP.MARK) == 1
    i = s.index(MP.ANCHOR) + len(MP.ANCHOR)
    assert s[i:].startswith("            messages = _rt.teacher_view(messages)")
    assert MP.patch_text(s) == (s, False)
    try:
        MP.patch_text(SNIPPET); assert False                           # route's hook first
    except AssertionError as e:
        assert "route_patch" in str(e)


def test_mode_patch_end_to_end():
    s = MP.patch_text(RP.patch_text(SNIPPET)[0])[0]
    batch = [msgs("[long] a"), msgs("[short] b"), msgs("[long] c", system="s")]
    keep = json.dumps(batch)
    for strip, want in ((None, ["a", "b", "c"]), ("0", ["[long] a", "[short] b", "[long] c"])):
        ns = {}; log = os.path.join(tempfile.mkdtemp(), "route.jsonl")
        with env(OPD_ROUTE="prefix", OPD_MODE_STRIP=strip, OPD_ROUTE_LOG=log, OPD_REF_THINK="1", OPD_ROUTE_LABELS=None):
            importlib.reload(R); sys.modules["opd_route"] = R
            exec(compile(s, "ref_input_utils.py", "exec"), ns)
            ns["prepare_ref_model_inputs"](batch, None, {})   # route's hook flushes the call's log line itself
        rec = json.loads(open(log).read().splitlines()[-1])
        assert [x[0] for x in ns["SEEN"]] == want and [x[1] for x in ns["SEEN"]] == [True, False, True]
        assert rec["stripped"] == (3 if strip is None else 0) and rec["rows"] == 3 and rec["unknown"] == 0
    assert json.dumps(batch) == keep                                   # the batch's rows untouched
    ns = {}
    with env(OPD_ROUTE=None, OPD_REF_THINK="1"):                       # without OPD_ROUTE: P5 exactly
        exec(compile(s, "ref_input_utils.py", "exec"), ns)
        ns["prepare_ref_model_inputs"](batch, None, {})
    assert [x[0] for x in ns["SEEN"]] == ["[long] a", "[short] b", "[long] c"] and all(x[1] for x in ns["SEEN"])


def test_mode_patch_file_and_the_real_file():
    d = tempfile.mkdtemp(); p = os.path.join(d, "ref_input_utils.py"); open(p, "w").write(RP.patch_text(SNIPPET)[0])
    assert MP.patch_file(p) and os.path.exists(p + ".orig") and not MP.patch_file(p)
    real = os.environ.get("GOPD_REF_INPUT_UTILS")
    if real:   # the node's installed file: route's hook then ours, compiles, one mark each
        q = os.path.join(d, "real.py"); shutil.copy2(real, q)
        RP.patch_file(q, keep_orig=False); MP.patch_file(q, keep_orig=False)
        t = open(q).read()
        assert t.count(MP.MARK) == 1 and t.count(RP.MARK) == 2
        compile(t, q, "exec")


def test_mode_data_prompts_follow_mix():
    import mode_data as MD
    ps = [msgs(f"problem {i}") for i in range(2000)]
    out = [MD.mode_prompt(m, 1234) for m in ps]
    share = sum(t for _, t in out) / len(out)
    assert 0.46 < share < 0.54
    for m, (n, think) in zip(ps, out):
        assert think == R.coin(R.key(m), 1234)                          # mix's coin on the plain prompt
        assert n[-1]["content"] == R.prefixed(m[-1]["content"], think) and m[-1]["content"].startswith("problem")
        assert R.strip_messages(n) == m and R.key(R.strip_messages(n)) == R.key(m)
    try:
        MD.mode_prompt(out[0][0], 1234); assert False
    except ValueError:
        pass
    sys_first = MD.mode_prompt(msgs("q", system="s"), 1234)[0]
    assert sys_first[0] == {"role": "system", "content": "s"} and R.split_prefix(sys_first[1]["content"])[0] is not None
