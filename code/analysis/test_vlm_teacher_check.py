"""[v2.5.9z] vlm_teacher_check.py without models: the verdict's numbers and thresholds, the config fields compared, the prompt text."""
import argparse
import json
import os
import tempfile

import vlm_teacher_check as C


def write(d, name, rows):
    with open(os.path.join(d, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def args(d):
    return argparse.Namespace(out=d, student="s", teacher="t", data="", n=4)


def test_verdict_two_suffixes(monkeypatch):
    d = tempfile.mkdtemp()
    write(d, "student.jsonl", [{"i": i, "finish_reason": "stop" if i < 3 else "length", "correct": i < 2, "n_tokens": 100} for i in range(4)])
    write(d, "teacher.jsonl", [{"i": i, "finish_reason": "stop" if i < 3 else "length", "correct": i < 3, "n_tokens": 900} for i in range(4)])
    write(d, "pstop.jsonl", [{"i": 0, "p_stop_think": 0.9, "p_stop_empty": 0.2}, {"i": 1, "p_stop_think": 0.6, "p_stop_empty": 0.3}, {"i": 2, "p_stop_think": None, "p_stop_empty": 0.4}])
    r = C.step_verdict(args(d))
    assert r["student_finished"] == 0.75 and r["teacher_finished"] == 0.75 and r["teacher_acc"] == 0.75
    assert r["pstop"]["think"] == {"n": 2, "mean": 0.75, "median": 0.9} and r["pstop"]["empty"]["n"] == 3
    assert r["pass"] and r["suffix"] == "think" and r["suffix_text"] == "<think>\n"
    write(d, "pstop.jsonl", [{"i": 0, "p_stop_think": 0.1, "p_stop_empty": 0.8}, {"i": 1, "p_stop_think": 0.2, "p_stop_empty": 0.7}])
    r = C.step_verdict(args(d))
    assert r["pass"] and r["suffix"] == "empty" and r["suffix_text"] == "<think>\n\n</think>\n\n"   # E15-VLM-P5e
    write(d, "pstop.jsonl", [{"i": 0, "p_stop_think": 0.1, "p_stop_empty": 0.2}])
    r = C.step_verdict(args(d))
    assert not r["pass"] and r["suffix"] is None
    write(d, "pstop.jsonl", [{"i": 0, "p_stop_think": 0.9, "p_stop_empty": 0.9}])
    monkeypatch.setenv("PVL_FINISH_MIN", "0.8")
    assert not C.step_verdict(args(d))["pass"]                       # the teacher finishing 75% < 80%


def test_base_prompt_strips_the_think_opening():
    class P:
        def __init__(self, tail):
            self.tail = tail

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "U|A:" + self.tail
    r = {"prompt": [{"role": "user", "content": "<image>x"}]}
    assert C.base_prompt(P("<think>\n"), r) == "U|A:" and C.base_prompt(P(""), r) == "U|A:"


def test_cfg_fields_and_text():
    d = tempfile.mkdtemp()
    json.dump({"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "image_mean": [0.5] * 3, "size": {"shortest_edge": 65536}, "other": 1},
              open(os.path.join(d, "preprocessor_config.json"), "w"))
    f = C.cfg_fields(d)
    assert f == {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "image_mean": [0.5] * 3, "size": {"shortest_edge": 65536}}
    assert C.text_of({"prompt": [{"role": "user", "content": "<image>Find x.\nPlease reason"}]}) == "Find x.\nPlease reason"

    class P:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "U:" + "".join(c.get("text", "[IMG]") for c in msgs[0]["content"]) + "|A:"
    assert C.chat(P(), {"prompt": [{"role": "user", "content": "<image>Find x."}]}) == "U:[IMG]Find x.|A:"
