"""[v2.5.9z16] the ifeval bench and its grading.

Run anywhere:
- gen_eval's loader: rows from the #fetch copy, WAIT_INPUTS without it;
- the prompt as it is (no suffix) and records / metrics left ungraded, through gen_eval.main with vLLM and the tokenizer stubbed;
- ifeval_grade.py's pure logic: the answer after </think>, coverage, the four accuracies, pass@k, the fold into the metrics.

Run only with IFEVAL_TEST_PY=<a python with nltk 3.9.1 + punkt_tab in its nltk_data, langdetect, absl-py, immutabledict>
(the node's venv-ifeval, or a Mac copy); skipped otherwise:
- Google's checker on known responses, strict and loose;
- the whole path gen_eval -> ifeval_grade.py -> .ifeval.json + metrics;
- Google's own unit tests (--selftest).
The prompts below are written for these tests (IFEval's instruction ids and kwargs, not its prompts)."""
import json
import os
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_eval as G  # noqa: E402
import ifeval_grade as IG  # noqa: E402

IFPY = os.environ.get("IFEVAL_TEST_PY", "")
needs_checker = pytest.mark.skipif(not IFPY, reason="IFEVAL_TEST_PY not set (node-only: venv-ifeval)")
END = "Is there anything else I can help with?"
DATA = [
    {"key": 1, "prompt": "Describe a lighthouse in two sentences. Do not use any commas.",
     "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_sentences"], "kwargs": [{}, {"relation": "less than", "num_sentences": 3}]},
    {"key": 2, "prompt": f"Highlight two points with markdown, e.g. *point*, and end with the exact phrase: {END}",
     "instruction_id_list": ["detectable_format:number_highlighted_sections", "startend:end_checker"], "kwargs": [{"num_highlights": 2}, {"end_phrase": END}]},
    {"key": 3, "prompt": "Give your whole answer as JSON.", "instruction_id_list": ["detectable_format:json_format"], "kwargs": [{"x": None}]},
]
ANSWERS = {   # strict: 1 yes, 2 no (the closing "**" line; loose drops it), 3 yes; loose: all three
    1: "The lighthouse stands alone. Its beam sweeps the dark sea.",
    2: f"Here is *one point* and *another point*.\n{END}\n**",
    3: '```json\n{"answer": 42}\n```',
}


def _fetch_copy(root, rows=DATA):
    d = os.path.join(root, "data", "hf", "google__IFEval"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "ifeval_input_data.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n\n")   # a trailing blank line is skipped
    return os.path.join(d, "ifeval_input_data.jsonl")


# ---------- gen_eval: the bench ----------
def test_load_ifeval_waits_without_the_fetch_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("OPD_ROOT", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        G.load_bench("ifeval")
    assert str(e.value) == "WAIT_INPUTS ifeval: google__IFEval/ifeval_input_data.jsonl is not on the node (its #fetch line)"


def test_load_ifeval_rows_as_given(tmp_path, monkeypatch):
    monkeypatch.setenv("OPD_ROOT", str(tmp_path)); _fetch_copy(str(tmp_path))
    rows = G.load_bench("ifeval")
    assert [r["pid"] for r in rows] == ["ifeval-1", "ifeval-2", "ifeval-3"]
    assert rows[0]["problem"] == DATA[0]["prompt"] and rows[0]["answer"] is None and rows[0]["level"] is None
    assert rows[0]["source"] == "google/IFEval" and rows[0]["later"] == "ifeval" and len(G.load_bench("ifeval", limit=2)) == 2


class _Out:
    def __init__(self, text):
        self.text, self.token_ids, self.finish_reason = text, [1] * (5 + len(text) // 4), "stop"


class _LLM:
    seen = []

    def __init__(self, **kw):
        pass

    def generate(self, prompts, sp):
        _LLM.seen += prompts
        return [types.SimpleNamespace(outputs=[_Out(ANSWERS[int(p.split("|")[0][len("PROMPT:"):])])]) for p in prompts]


class _Tok:
    keys = {}

    @staticmethod
    def from_pretrained(m):
        return _Tok()

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
        c = msgs[0]["content"]
        return f"PROMPT:{_Tok.keys.get(c, 0)}|{c}"


def _gen(tmp_path, monkeypatch, bench="ifeval", mode="no_think"):
    monkeypatch.setenv("OPD_ROOT", str(tmp_path)); _fetch_copy(str(tmp_path))
    _Tok.keys = {r["prompt"]: r["key"] for r in DATA}; _LLM.seen = []
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=_LLM, SamplingParams=lambda **kw: types.SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=_Tok))
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["gen_eval.py", "--model", "m", "--bench", bench, "--mode", mode, "--n", "1", "--temperature", "0",
                                      "--top_p", "1.0", "--top_k", "0", "--max_tokens", "4096", "--chunk", "2", "--tag", "t", "--out", out])
    G.main()
    stem = os.path.join(out, f"t_{bench}_{mode}_t0.0")
    return stem, [json.loads(l) for l in open(stem + ".jsonl")], json.load(open(stem + ".metrics.json"))


def test_gen_eval_ifeval_prompt_as_is_and_ungraded(tmp_path, monkeypatch):
    stem, recs, m = _gen(tmp_path, monkeypatch)
    assert [p.split("|", 1)[1] for p in _LLM.seen] == [r["prompt"] for r in DATA]   # no suffix, no \boxed
    assert all(r["correct"] is None and r["pred"] is None and r["gold"] is None and r["first_correct_pos"] is None for r in recs)
    assert [r["pid"] for r in recs] == ["ifeval-1", "ifeval-2", "ifeval-3"] and recs[1]["text"] == ANSWERS[2]
    assert m["avg_acc"] is None and m["pass_at_k"] == {} and m["ifeval"] is None and m["by_level"] == {} and m["num_problems"] == 3
    assert m["bench"] == "ifeval" and m["temperature"] == 0.0 and "unparsed_rate" not in m


def test_other_benches_unchanged_by_the_later_flag():
    r = {"pid": "gpqa-0", "problem": "p", "answer": "B", "letters": "ABCD", "level": "Physics", "source": "s"}
    c = types.SimpleNamespace(text="\\boxed{B}", token_ids=[1, 2], finish_reason="stop")
    rec = G.make_record(r, 0, c, "no_think")
    m = G.compute_metrics(types.SimpleNamespace(model="m", bench="gpqa", mode="no_think", n=1, temperature=0.0, top_p=1.0, top_k=0, max_tokens=4096),
                          [r], [rec], 1.0)
    assert rec["correct"] is True and m["avg_acc"] == 100.0 and m["pass_at_k"] == {1: 100.0} and "ifeval" not in m


# ---------- ifeval_grade.py: pure logic ----------
@pytest.mark.parametrize("text,mode,want", [
    ("plain answer", "no_think", ("plain answer", False)),
    ("  spaced \n", "no_think", ("  spaced \n", False)),                          # no thinking block: the text as it is
    ("<think>\nsteps, with commas\n</think>\n\nThe answer.", "think", ("The answer.", False)),
    ("a </think> b </think>\n\n final ", "no_think", ("final", False)),           # the last </think>
    ("<think>\nnever closed, cut at the budget", "think", ("", True))])
def test_response_of(text, mode, want):
    assert IG.response_of(text, mode) == want


def _inputs():
    return [types.SimpleNamespace(key=r["key"]) for r in DATA]


def test_coverage():
    recs = [{"pid": f"ifeval-{k}"} for k in (1, 2, 3)]
    assert IG.coverage(_inputs(), recs) is None
    assert IG.coverage(_inputs(), recs[:2]).startswith("1 prompts missing (['ifeval-3'])")
    assert "1 unknown (['ifeval-9'])" in IG.coverage(_inputs(), recs + [{"pid": "ifeval-9"}])
    assert IG.coverage(_inputs(), recs + [{"pid": "ifeval-1"}]) == "sample counts differ: [1, 2]"


def _row(pid, s, lo, ids, sample=0, unclosed=False):
    return {"pid": pid, "sample": sample, "strict": all(s), "loose": all(lo), "strict_list": s, "loose_list": lo,
            "instruction_id_list": ids, "think_unclosed": unclosed, "empty": unclosed, "truncated": False}


def test_summarize_four_accuracies_groups_and_pass_at_k():
    a, b = ["punctuation:no_comma", "length_constraints:number_sentences"], ["detectable_format:json_format"]
    rows = [_row("ifeval-1", [True, True], [True, True], a), _row("ifeval-1", [False, True], [True, True], a, 1),
            _row("ifeval-2", [False], [True], b), _row("ifeval-2", [False], [False], b, 1, unclosed=True)]
    s = IG.summarize(rows)
    assert (s["prompt_strict"], s["prompt_loose"], s["inst_strict"], s["inst_loose"]) == (25.0, 75.0, 50.0, 83.33)
    assert s["pass_at_k"] == {1: 25.0, 2: 50.0} and s["n_prompts"] == 2 and s["n_samples"] == 4 and s["n_instructions"] == 6
    assert s["by_instruction"]["punctuation:no_comma"] == {"n": 2, "strict": 50.0, "loose": 100.0}
    assert s["by_group"]["detectable_format"] == {"n": 2, "strict": 0.0, "loose": 50.0} and s["think_unclosed"] == 1


def test_fold_into_the_metrics(tmp_path):
    p = str(tmp_path / "m.json"); json.dump({"bench": "ifeval", "avg_acc": None, "pass_at_k": {}, "ifeval": None, "mean_tokens": 9.0}, open(p, "w"))
    s = {"prompt_strict": 66.67, "prompt_loose": 100.0, "inst_strict": 80.0, "inst_loose": 100.0, "pass_at_k": {1: 66.67}}
    IG.fold(p, s)
    m = json.load(open(p))
    assert m["avg_acc"] == 66.67 and m["pass_at_k"] == {"1": 66.67} and m["mean_tokens"] == 9.0
    assert m["ifeval"] == {"prompt_strict": 66.67, "prompt_loose": 100.0, "inst_strict": 80.0, "inst_loose": 100.0} and not os.path.exists(p + ".tmp")


def test_vendored_files_are_upstreams():   # SOURCE.md's git blob ids (no checker import: runs anywhere)
    import hashlib
    src = open(os.path.join(HERE, "ifeval", "SOURCE.md")).read()
    for line in src.splitlines():
        f = [x.strip() for x in line.strip("|").split("|")]
        if len(f) == 2 and f[0].endswith(".py"):
            b = open(os.path.join(HERE, "ifeval", "instruction_following_eval", f[0]), "rb").read()
            assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == f[1], f[0]
    assert "Apache License" in open(os.path.join(HERE, "ifeval", "LICENSE")).read() and IG.UPSTREAM.endswith("e6890f85757dd84e27ca6df2dd30651dafad28e0")


# ---------- Google's checker (IFEVAL_TEST_PY) ----------
def _grade(tmp_path, recs, mode):
    data = _fetch_copy(str(tmp_path / "d"))
    j, m, o = str(tmp_path / "r.jsonl"), str(tmp_path / "r.metrics.json"), str(tmp_path / "r.ifeval.json")
    open(j, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    json.dump({"bench": "ifeval", "mode": mode, "avg_acc": None, "pass_at_k": {}, "ifeval": None}, open(m, "w"))
    p = subprocess.run([IFPY, "-I", os.path.join(HERE, "ifeval_grade.py"), "--data", data, "--jsonl", j, "--metrics", m, "--out", o],
                       capture_output=True, text=True, timeout=300)
    return p, (json.load(open(o)) if os.path.exists(o) else None), json.load(open(m))


@needs_checker
def test_checker_known_responses_strict_and_loose(tmp_path):
    recs = [{"pid": f"ifeval-{k}", "sample": 0, "text": ANSWERS[k], "finish_reason": "stop"} for k in (1, 2, 3)]
    p, g, m = _grade(tmp_path, recs, "no_think")
    assert p.returncode == 0, p.stderr[-2000:]
    assert (g["prompt_strict"], g["prompt_loose"], g["inst_strict"], g["inst_loose"]) == (66.67, 100.0, 80.0, 100.0)
    v = {r["pid"]: (r["strict_list"], r["loose_list"]) for r in g["per_sample"]}
    assert v == {"ifeval-1": ([True, True], [True, True]), "ifeval-2": ([True, False], [True, True]), "ifeval-3": ([True], [True])}
    assert m["avg_acc"] == 66.67 and m["ifeval"]["inst_strict"] == 80.0 and m["pass_at_k"] == {"1": 66.67}
    assert g["versions"]["nltk"] == "3.9.1" and p.stdout.startswith("IFEVAL r prompt_strict=66.67 prompt_loose=100.0 inst_strict=80.0 inst_loose=100.0")


@needs_checker
def test_checker_thinking_answer_and_unclosed(tmp_path):
    recs = [{"pid": "ifeval-1", "sample": 0, "text": "<think>\nfirst, then, finally, done\n</think>\n\n" + ANSWERS[1], "finish_reason": "stop"},
            {"pid": "ifeval-2", "sample": 0, "text": "<think>\nhmm, " + ANSWERS[2], "finish_reason": "length"},
            {"pid": "ifeval-3", "sample": 0, "text": "<think></think>" + ANSWERS[3], "finish_reason": "stop"}]
    p, g, _ = _grade(tmp_path, recs, "think")
    assert p.returncode == 0, p.stderr[-2000:]
    assert [r["strict"] for r in g["per_sample"]] == [True, False, True] and g["think_unclosed"] == 1 and g["truncated"] == 1
    assert g["per_sample"][1]["loose_list"] == [False, False] and g["empty_responses"] == 1   # no answer: nothing followed


@needs_checker
def test_checker_refuses_a_reading_that_does_not_cover_the_data(tmp_path):
    p, g, m = _grade(tmp_path, [{"pid": "ifeval-1", "sample": 0, "text": "x", "finish_reason": "stop"}], "no_think")
    assert p.returncode == 2 and "IFEVAL_COVERAGE r.jsonl: 2 prompts missing" in p.stderr and g is None and m["avg_acc"] is None


@needs_checker
def test_gen_eval_then_grade_end_to_end(tmp_path, monkeypatch):
    stem, _, _ = _gen(tmp_path, monkeypatch)
    p = subprocess.run([IFPY, "-I", os.path.join(HERE, "ifeval_grade.py"), "--data", str(tmp_path / "data/hf/google__IFEval/ifeval_input_data.jsonl"),
                        "--jsonl", stem + ".jsonl", "--metrics", stem + ".metrics.json", "--out", stem + ".ifeval.json"], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    m = json.load(open(stem + ".metrics.json"))
    assert m["avg_acc"] == 66.67 and m["ifeval"] == {"prompt_strict": 66.67, "prompt_loose": 100.0, "inst_strict": 80.0, "inst_loose": 100.0}
    assert m["mean_tokens"] > 0 and json.load(open(stem + ".ifeval.json"))["n_prompts"] == 3


@needs_checker
def test_selftest_google_unit_tests_and_known():
    p = subprocess.run([IFPY, "-I", os.path.join(HERE, "ifeval_grade.py"), "--selftest"], capture_output=True, text=True, timeout=900)
    r = json.loads(p.stdout.split("IFEVAL_SELFTEST ", 1)[1])
    assert p.returncode == 0 and r["ok"] and r["punkt_tab_in_venv"] and r["count_sentences_3"] == 3, r
    assert r["google_tests"]["instructions_test"]["summary"].startswith("Ran 41 tests") and all(k["ok"] for k in r["known"])
