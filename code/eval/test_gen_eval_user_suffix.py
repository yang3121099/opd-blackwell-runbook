"""[z161] (Fable 11-15, lab/registry/Q-098.md prompt baselines) gen_eval --user_suffix: empty (the default) leaves every user message
exactly as before (character for character, every bench kind); a suffix is appended on its own line after the standard suffix; the
metrics name it; a reading started before z161 (its .meta without the key) still resumes with the default and is refused with a suffix.
No GPU, no tokenizer: the user message is the only thing the suffix changes (the chat template then renders it as before)."""
import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_eval  # noqa: E402

P1 = "Think briefly: keep your reasoning short and stop as soon as you are confident in the answer."
P2 = "Be concise. Use as few reasoning steps as possible, then give the final answer."


def old_content(bench, problem):   # the expression main() used before z161, verbatim
    suffix = gen_eval.MC_SUFFIX if bench in gen_eval.MC_SOURCES else "" if bench == "ifeval" else gen_eval.PROMPT_SUFFIX
    return gen_eval.MODE_PREFIX.get(bench, "") + problem + suffix


@pytest.mark.parametrize("bench", ["math500", "aime24", "olympiadbench", "math500L", "math500S", "mmlupro1k", "gpqa", "ifeval"])
def test_empty_suffix_unchanged(bench):
    for prob in ("What is $1+1$?", "Line one\nline two\n", ""):
        assert gen_eval.user_content(bench, prob) == old_content(bench, prob)
        assert gen_eval.user_content(bench, prob, "") == old_content(bench, prob)


@pytest.mark.parametrize("suffix", [P1, P2])
def test_suffix_on_its_own_line(suffix):
    c = gen_eval.user_content("math500", "Compute $2^{10}$.", suffix)
    assert c == "Compute $2^{10}$." + gen_eval.PROMPT_SUFFIX + "\n" + suffix
    assert c.endswith("\\boxed{}.\n" + suffix)


def test_cli_metrics_and_resume(tmp_path):
    src = open(os.path.join(HERE, "gen_eval.py")).read()
    assert 'ap.add_argument("--user_suffix", default=""' in src
    assert "user_content(args.bench, r[\"problem\"], args.user_suffix)" in src
    assert "**({\"user_suffix\": args.user_suffix} if getattr(args, \"user_suffix\", \"\") else {})" in src
    assert "user_suffix" in gen_eval.RESUME_KEYS and gen_eval.RESUME_DEFAULTS["user_suffix"] == ""
    # a .part / .meta written before z161 (no user_suffix key) resumes under the default, is refused with a suffix
    rows = [{"pid": f"p{i}"} for i in range(4)]
    base = {k: None for k in gen_eval.RESUME_KEYS if k not in ("user_suffix", "prefill", "seed_mode")}
    base.update({"n": 1, "chunk": 2})
    part, meta = str(tmp_path / "x.jsonl.part"), str(tmp_path / "x.jsonl.part.meta")
    with open(part, "w") as f:
        for r in rows[:2]:
            f.write(json.dumps({"pid": r["pid"], "sample": 0}) + "\n")
    json.dump({"args": dict(base, prefill="", seed_mode="shared"), "gen_seconds_done": 1.0, "problems_done": 2}, open(meta, "w"))
    a = types.SimpleNamespace(**dict(base, prefill="", seed_mode="shared", user_suffix=""))
    recs, _ = gen_eval.resume_state(part, meta, a, rows, 2)
    assert len(recs) == 2
    b = types.SimpleNamespace(**dict(base, prefill="", seed_mode="shared", user_suffix=P1))
    with pytest.raises(SystemExit):
        gen_eval.resume_state(part, meta, b, rows, 2)


def test_meta_omits_empty_suffix():   # [z161] an older gen_eval can resume a reading this version wrote without a suffix
    base = {k: 1 for k in gen_eval.RESUME_KEYS}
    m0 = gen_eval.meta_args(types.SimpleNamespace(**dict(base, user_suffix="")))
    m1 = gen_eval.meta_args(types.SimpleNamespace(**dict(base, user_suffix=P1)))
    assert "user_suffix" not in m0 and m1["user_suffix"] == P1
    assert {k: v for k, v in m1.items() if k != "user_suffix"} == m0
    m2 = gen_eval.meta_args(types.SimpleNamespace(**{k: 1 for k in gen_eval.RESUME_KEYS if k != "user_suffix"}))   # an args object without the key
    assert m2 == m0
