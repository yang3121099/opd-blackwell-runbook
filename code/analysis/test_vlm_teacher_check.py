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


def test_verdict_pass_and_fail(monkeypatch):
    d = tempfile.mkdtemp()
    write(d, "student.jsonl", [{"i": i, "finish_reason": "stop" if i < 3 else "length", "correct": i < 2, "n_tokens": 100} for i in range(4)])
    write(d, "teacher.jsonl", [{"i": i, "finish_reason": "stop" if i < 3 else "length", "correct": i < 3, "n_tokens": 900} for i in range(4)])
    write(d, "pstop.jsonl", [{"i": 0, "p_stop": 0.9}, {"i": 1, "p_stop": 0.6}, {"i": 2, "p_stop": None}])
    r = C.step_verdict(args(d))
    assert r["student_finished"] == 0.75 and r["student_acc"] == 0.5 and r["teacher_finished"] == 0.75 and r["teacher_acc"] == 0.75
    assert r["pstop_n"] == 2 and r["pstop_mean"] == 0.75 and r["pass"]
    monkeypatch.setenv("PVL_FINISH_MIN", "0.8")
    assert not C.step_verdict(args(d))["pass"]
    monkeypatch.setenv("PVL_FINISH_MIN", "0.7")
    write(d, "pstop.jsonl", [{"i": 0, "p_stop": 0.2}, {"i": 1, "p_stop": 0.3}])
    assert not C.step_verdict(args(d))["pass"]


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
