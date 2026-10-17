"""gen_eval --chunk: a run killed between blocks and rerun gives the same .jsonl, line by line, as an uninterrupted run,
and the same metrics (gen_seconds aside); a chunked run equals a one-pass run; a block cut by the kill is regenerated;
a .part written with other arguments is refused. No GPU: vLLM, the tokenizer, the benchmark and the grader are stubbed
(the stub LLM is deterministic per prompt and seed, as vLLM is with a fixed per-request seed)."""
import hashlib
import json
import os
import sys
import tempfile
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_eval  # noqa: E402

CALLS = {"n": 0, "crash_after": None}


class _Out:
    def __init__(self, text, n_tok, fin):
        self.text, self.token_ids, self.finish_reason = text, list(range(n_tok)), fin


class _Req:
    def __init__(self, outs):
        self.outputs = outs


class FakeLLM:
    def __init__(self, **kw):
        pass

    def generate(self, prompts, sp):
        if CALLS["crash_after"] is not None and CALLS["n"] >= CALLS["crash_after"]:
            raise RuntimeError("simulated preemption")
        CALLS["n"] += 1
        res = []
        for p in prompts:
            outs = []
            for i in range(sp.n):
                h = int(hashlib.md5(f"{p}|{i}|{sp.seed}".encode()).hexdigest(), 16)
                outs.append(_Out(f"ans {h % 7} \\boxed{{{h % 3}}}", 10 + h % 90, "length" if h % 11 == 0 else "stop"))
            res.append(_Req(outs))
        return res


class FakeSP:
    def __init__(self, n, temperature, top_p, top_k, max_tokens, seed):
        self.n, self.seed = n, seed


class FakeTok:
    @staticmethod
    def from_pretrained(m):
        return FakeTok()

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
        return "PROMPT:" + msgs[0]["content"]


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSP))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeTok))
    rows = [{"pid": f"math500-{i}", "problem": f"problem {i}", "answer": str(i % 3), "level": 1 + i % 3, "source": "stub"}
            for i in range(11)]
    monkeypatch.setattr(gen_eval, "load_bench", lambda name, limit=None: rows[:limit] if limit else rows)
    monkeypatch.setattr(gen_eval, "grade", lambda text, gold: (text.endswith("{" + gold + "}"), gold))
    monkeypatch.setattr(gen_eval, "first_correct_pos", lambda text, gold, loose=False: None)
    CALLS.update(n=0, crash_after=None)


def run(out, *extra, tag="t"):
    argv = ["gen_eval.py", "--model", "m", "--bench", "math500", "--mode", "no_think", "--n", "2", "--seed", "1",
            "--tag", tag, "--out", out, *extra]
    old = sys.argv
    sys.argv = argv
    try:
        gen_eval.main()
    finally:
        sys.argv = old
    stem = os.path.join(out, f"{tag}_math500_no_think_t0.6")
    m = json.load(open(stem + ".metrics.json"))
    m.pop("gen_seconds")
    return open(stem + ".jsonl").read().splitlines(), m


def test_chunked_equals_one_pass():
    a, ma = run(tempfile.mkdtemp())
    b, mb = run(tempfile.mkdtemp(), "--chunk", "3")
    assert a == b and ma == mb and len(a) == 22


def test_killed_then_resumed_equals_uninterrupted():
    ref, mref = run(tempfile.mkdtemp(), "--chunk", "3")
    d = tempfile.mkdtemp()
    CALLS.update(n=0, crash_after=2)
    with pytest.raises(RuntimeError):
        run(d, "--chunk", "3")
    part = os.path.join(d, "t_math500_no_think_t0.6.jsonl.part")
    assert len(open(part).read().splitlines()) == 12   # two blocks of 3 problems x 2 samples survived the kill
    CALLS.update(n=0, crash_after=None)
    got, mgot = run(d, "--chunk", "3")
    assert got == ref and mgot == mref
    assert CALLS["n"] == 2   # only the two missing blocks (problems 6-8, 9-10) were generated again
    assert not os.path.exists(part) and not os.path.exists(part + ".meta")


def test_cut_line_and_partial_block_are_regenerated():
    ref, _ = run(tempfile.mkdtemp(), "--chunk", "3")
    d = tempfile.mkdtemp()
    CALLS.update(n=0, crash_after=2)
    with pytest.raises(RuntimeError):
        run(d, "--chunk", "3")
    part = os.path.join(d, "t_math500_no_think_t0.6.jsonl.part")
    lines = open(part).read().splitlines()
    with open(part, "w") as f:   # second block only partly on disk, and its last line cut mid-way
        f.write("\n".join(lines[:9]) + "\n" + lines[9][:25])
    CALLS.update(n=0, crash_after=None)
    got, _ = run(d, "--chunk", "3")
    assert got == ref and CALLS["n"] == 3   # block 2 regenerated whole, then blocks 3 and 4


def test_other_arguments_refused():
    d = tempfile.mkdtemp()
    CALLS.update(n=0, crash_after=1)
    with pytest.raises(RuntimeError):
        run(d, "--chunk", "3")
    CALLS.update(n=0, crash_after=None)
    with pytest.raises(SystemExit, match="RESUME_REFUSED"):
        run(d, "--chunk", "4")
