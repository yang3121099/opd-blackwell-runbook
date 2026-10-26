"""[v2.5.9z89] (Q-080a) PROBE-VLM-COMPLETE's numbers from vlm_teacher_check.py: the completion rate over the finished answers, the
threshold, the notes; the end token's probability and argmax from vLLM-shaped prompt_logprobs. No GPU, vLLM or model."""
import json
import math
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "analysis"))
import vlm_teacher_check as C  # noqa: E402

END = 9


def _write(d, name, rows):
    with open(os.path.join(d, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _dir(n=10, fin=8, tops=(True,) * 8, self_p=0.6, think=True):
    d = tempfile.mkdtemp()
    _write(d, "student.jsonl", [{"i": i, "finish_reason": "stop" if i < fin else "length", "n_tokens": 100} for i in range(n)])
    _write(d, "complete_teacher.jsonl", [{"i": i, "p_stop": 0.8, "argmax_end": t} for i, t in enumerate(tops)])
    _write(d, "complete_self.jsonl", [{"i": i, "p_stop": self_p, "argmax_end": True} for i in range(fin)])
    if think:
        _write(d, "complete_think.jsonl", [{"i": i, "p_stop": 0.3, "argmax_end": False} for i in range(fin)])
    return d


def test_pass_at_the_threshold_and_the_readings():
    r = C.step_complete_verdict(types.SimpleNamespace(out=_dir(tops=(True,) * 7 + (False,))))
    assert r["completion_rate"] == 0.875 and not r["pass"] and r["scored"] == 8          # 7 of 8 < 0.9
    r = C.step_complete_verdict(types.SimpleNamespace(out=_dir(n=10, fin=10, tops=(True,) * 9 + (False,))))
    assert r["completion_rate"] == 0.9 and r["pass"]                                      # 0.9 passes
    assert r["teacher_p_stop_last_mean"] == 0.8 and r["student_p_stop_last_mean"] == 0.6 and r["student_finished"] == 1.0
    assert r["thinking_teacher_nonthinking"] == {"p_stop_last_mean": 0.3, "argmax_end": 0.0} and r["note"] == ""


def test_a_low_student_finished_share_is_noted_not_a_fail():
    r = C.step_complete_verdict(types.SimpleNamespace(out=_dir(n=10, fin=4, tops=(True,) * 4, think=False)))
    assert r["student_finished"] == 0.4 and r["pass"] and "noted" in r["note"] and r["thinking_teacher_nonthinking"] is None


def test_nothing_scored_is_a_fail():
    r = C.step_complete_verdict(types.SimpleNamespace(out=_dir(n=3, fin=0, tops=(), think=False)))
    assert r["completion_rate"] is None and not r["pass"]


def test_p_last_top():
    LP = lambda v: types.SimpleNamespace(logprob=v)
    o = types.SimpleNamespace(prompt_token_ids=[1, 2, END], prompt_logprobs=[None, {}, {END: LP(math.log(0.7)), 4: LP(math.log(0.2))}])
    p, top = C.p_last_top(o, END)
    assert abs(p - 0.7) < 1e-9 and top
    o2 = types.SimpleNamespace(prompt_token_ids=[1, 2, END], prompt_logprobs=[None, {}, {END: LP(math.log(0.3)), 4: LP(math.log(0.6))}])
    assert C.p_last_top(o2, END)[1] is False
    o3 = types.SimpleNamespace(prompt_token_ids=[1, 2, 5], prompt_logprobs=[None, {}, {5: LP(-0.1)}])
    assert C.p_last_top(o3, END) == (None, None)                                            # the prompt does not end on the end token
