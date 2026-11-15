"""[z166] unit tests of deer_eval.py (no GPU, no model): DEER's confidence, the state machine on a scripted fake engine (an exit after a
failed trial, a natural </think>, the trial limit, the thinking budget), the token counts (main = every generated token), Q-098's
calibration split, and the lambda selection rule. Run: python -m pytest -q test_deer_eval.py"""
import math
import os
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deer_eval as D  # noqa: E402

A = SimpleNamespace(points="wait", max_tokens=1000, think_ratio=0.8, temperature=0.6, top_p=0.95, top_k=0, max_judge_steps=10,
                    prob_check_max_tokens=20, threshold=0.95, policy="avg2", seed=0)
tok_len = len   # one "token" per character in these tests


def out(text, stop=None, finish="stop", probs=()):
    return SimpleNamespace(text=text, n=len(text), finish_reason=finish, stop_reason=stop, probs=list(probs))


class Fake:
    """Answers each request from a per-kind script, recording the prompts and params it saw."""
    def __init__(self, think, probe, answer):
        self.s = {"think": list(think), "probe": list(probe), "answer": list(answer)}; self.seen = []

    def generate(self, prompts, params):
        res = []
        for p, q in zip(prompts, params):
            kind = "probe" if p.endswith(D.ANSWER_PROMPT) else "answer" if p.endswith("\n" + D.THINK_END + "\n\n") else "think"
            self.seen.append((kind, p, q)); res.append(self.s[kind].pop(0))
        return res


def test_confidence():
    assert D.confidence([0.1, 0.9, 0.99], closed=True) == math.exp((math.log(0.9) + math.log(0.99)) / 2)   # from the 2nd token, geometric
    assert D.confidence([0.1, 0.9, 0.99], closed=False) == 0.0                                            # no </think> in the trial: 0
    assert D.confidence([0.5], closed=True) == 0.0                                                        # a 1-token trial: 0 (DEER divides by 0)
    assert D.confidence([0.1, 0.5, 0.9], True, "min") == 0.5 and abs(D.confidence([0.1, 0.5, 0.9], True, "avg1") - 0.7) < 1e-12


def test_exit_after_a_failed_trial():
    f = Fake(think=[out("aaa", "Wait"), out("bbb", "Wait")], probe=[out("1}", D.THINK_END, probs=[0.9, 0.5, 0.5]), out("2}", D.THINK_END, probs=[0.9, 0.99, 0.99])],
             answer=[out("The answer is \\boxed{2}.")])
    st = D.run_deer([D.new_state("P|", "p0", 0)], f, tok_len, A)[0]
    assert st["exit"] == "high_prob" and st["trials"] == [0.5, 0.99] and st["steps"] == 2
    assert D.final_text(st) == "aaaWaitbbb\n</think>\n\nThe answer is \\boxed{2}."             # trials never enter the kept text
    assert st["gen_tokens"] == 3 + 2 + 3 + 2 + len("The answer is \\boxed{2}.") and st["probe_tokens"] == 4   # main count: trials included
    kinds = [k for k, _, _ in f.seen]
    assert kinds == ["think", "probe", "think", "probe", "answer"]
    pr = [q for k, _, q in f.seen if k == "probe"][0]
    assert pr["max_tokens"] == 20 and pr["stop"] == [D.THINK_END] and pr["logprobs"] == 1 and pr["temperature"] == 1.0   # DEER's probe
    th = [q for k, _, q in f.seen if k == "think"][0]
    assert th["stop"] == ["Wait", D.THINK_END] and th["temperature"] == 0.6 and th["max_tokens"] == 800                  # think budget 0.8 x 1000


def test_natural_end_no_trial():
    f = Fake(think=[out("abc", D.THINK_END)], probe=[], answer=[out("ans")])
    st = D.run_deer([D.new_state("P|", "p0", 0)], f, tok_len, A)[0]
    assert st["exit"] == "regular_end" and st["trials"] == [] and D.final_text(st) == "abc\n</think>\n\nans" and st["gen_tokens"] == 6


def test_trial_limit_then_only_think_end():
    a = SimpleNamespace(**dict(vars(A), max_judge_steps=1))
    f = Fake(think=[out("x", "Wait"), out("y", D.THINK_END)], probe=[out("1}", D.THINK_END, probs=[0.9, 0.1])], answer=[out("z")])
    st = D.run_deer([D.new_state("P|", "p0", 0)], f, tok_len, a)[0]
    stops = [q["stop"] for k, _, q in f.seen if k == "think"]
    assert stops == [["Wait", D.THINK_END], [D.THINK_END]] and st["exit"] == "regular_end" and D.final_text(st).startswith("xWaity")


def test_thinking_budget():
    a = SimpleNamespace(**dict(vars(A), max_tokens=100, think_ratio=0.8))   # think limit 80; a trial within 50 of it ends the thinking
    f = Fake(think=[out("t" * 40, "Wait")], probe=[out("1}", D.THINK_END, probs=[0.9, 0.1])], answer=[out("ans")])
    st = D.run_deer([D.new_state("P|", "p0", 0)], f, tok_len, a)[0]
    assert st["exit"] == "too_long" and st["trials"] == [0.1]
    f2 = Fake(think=[out("t" * 80, None, "length")], probe=[out("1}", D.THINK_END, probs=[0.9, 0.1])], answer=[out("a")])
    assert D.run_deer([D.new_state("P|", "p0", 0)], f2, tok_len, a)[0]["exit"] == "too_long"


def test_several_states_and_seeds():
    f = Fake(think=[out("a", D.THINK_END), out("b", D.THINK_END)], probe=[], answer=[out("1"), out("2")])
    sts = D.run_deer([D.new_state("P|", "p0", 0), D.new_state("P|", "p0", 1)], f, tok_len, A)
    assert [D.final_text(s) for s in sts] == ["a\n</think>\n\n1", "b\n</think>\n\n2"]
    seeds = [q["seed"] for k, _, q in f.seen if k == "think"]
    assert len(set(seeds)) == 2   # the two samples of a problem draw their own streams


def test_split_is_q098s():
    cal = D.calib_indices()
    assert cal == set(random.Random(20261115).sample(range(500), 100)) and len(cal) == 100


def test_select_rule():
    base = {"avg_acc": 88.0}
    runs = [{"deer_threshold": 0.90, "avg_acc": 85.0, "mean_tokens": 2000}, {"deer_threshold": 0.95, "avg_acc": 87.2, "mean_tokens": 3000},
            {"deer_threshold": 0.99, "avg_acc": 88.5, "mean_tokens": 4000}]
    assert D.select(base, runs)["deer_threshold"] == 0.95                  # within 1 point of the base, then the shortest
    assert D.select({"avg_acc": 95.0}, runs) is None


def test_record_and_metrics_are_gen_evals():   # the reading's files read as any gen_eval reading (outcomes, TERM4, forced passes)
    import pytest
    pytest.importorskip("math_verify")   # gen_eval's grader: a node-only test (math_verify is not on the Mac)
    import gen_eval as G
    st = D.new_state("P|", "math500-0", 1); st.update(think="so 2Wait sure", answer="The answer is \\boxed{2}.", gen_tokens=500, probe_tokens=40,
                                                   trials=[0.5, 0.97], exit="high_prob", finish_reason="stop", phase="done")
    r = {"pid": "math500-0", "problem": "1+1?", "answer": "2", "level": 1, "source": "x"}
    rec = D.record_of(G, r, st, lambda t: list(range(len(t.split()))))
    assert rec["correct"] is True and rec["pred"] == "2" and rec["sample"] == 1 and rec["think_closed"] is True
    assert rec["n_tokens"] == 500 and rec["n_tokens_kept"] == len(D.final_text(st).split()) and rec["deer"]["n_trials"] == 2
    a = SimpleNamespace(model="m", bench="math500", mode="think", n=1, temperature=0.6, top_p=0.95, top_k=0, max_tokens=16384, dtype="auto",
                        limit=None, seed=0, chunk=100, prefill="", seed_mode="shared", user_suffix="", tp=1)
    m = G.compute_metrics(a, [r], [rec], 1.0)
    assert m["avg_acc"] == 100.0 and m["mean_tokens"] == 500.0   # the main count drives mean_tokens


def test_exit_reaps_the_engine_z169():   # [z169] the process ends with gen_eval's reaper and os._exit, as gen_eval / forced_stop do
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deer_eval.py")).read()
    tail = src[src.index('if __name__ == "__main__":'):]
    assert "from gen_eval import _reap_children" in tail and "_reap_children()" in tail and "os._exit(rc or 0)" in tail
    assert tail.index("_reap_children()") < tail.index("os._exit(rc or 0)")
