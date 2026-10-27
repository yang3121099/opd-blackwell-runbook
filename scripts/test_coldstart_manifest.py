"""[v2.5.9z100] coldstart_manifest.py on a small queue: open stages only, the models with their tasks, the weights (#eval / #term
rows and E12 starts by run_e12.sh's naming), the pair references; the committed tasks/coldstart.json is in the current format."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coldstart_manifest as C  # noqa: E402

Q = """b1   open   E12d-P5tail-8k-s43:PROBE-E12B-8K EVAL-a:E12d-P5tail-8k-s43 E12c-P5tail-8k E15-VLM-P2lo EVAL-x5-base
b2   done   EVAL-old
#eval EVAL-a a_s50_aime24_T0.6 result:runs/E12d-P5tail-8k-s43/global_step_50/actor/huggingface aime24 16 0.6 32768
#pair EVAL-a a_s50_aime24_T0.6 EVAL-b/b_aime24.jsonl EVAL-c/c_aime24.jsonl
#budget EVAL-a budget_curve EVAL-base/base.jsonl a_s50_aime24_T0.6
#eval EVAL-x5-base x5 Qwen/Qwen2.5-Math-1.5B math500 4 0.6
#eval EVAL-old o result:runs/E10b-OLD/global_step_150/actor/huggingface math500 4 0.6
#term PROBE-TERM4-4B 4B_x result:runs/E10d-P5tail/global_step_50/actor/huggingface EVAL-t/t.jsonl
#strict STRICT-C1 EVAL-s/s.jsonl EVAL-r/r.jsonl
"""


def test_manifest_from_a_small_queue():
    m = C.build(Q)
    names = {x["name"]: x for x in m["models"]}
    assert {"Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"} <= set(names) and names["Qwen3-8B"]["by"] == "overnight.sh"
    assert names["Qwen2.5-Math-1.5B"]["hf"] == "Qwen/Qwen2.5-Math-1.5B" and names["Qwen2.5-Math-1.5B"]["tasks"] == ["EVAL-x5-base"]
    assert names["Qwen3-VL-2B-Instruct"]["tasks"] == ["E15-VLM-P2lo"] and "Qwen3-32B" not in names
    w = {x["path"]: x["tasks"] for x in m["weights"]}
    assert w["runs/E12d-P5tail-8k-s43/global_step_50/actor/huggingface"] == ["EVAL-a"]                 # a reading's model
    assert w["runs/E10d-P5tail-s43/global_step_50/actor/huggingface"] == ["E12d-P5tail-8k-s43"]         # an E12 arm's start
    assert w["runs/E10c-P5tail-c50/global_step_30/actor/huggingface"] == ["E12c-P5tail-8k"]             # the 0.6B exception
    assert "runs/E10d-P5tail/global_step_50/actor/huggingface" in w and "runs/E10b-OLD/global_step_150/actor/huggingface" not in w
    pairs = {x["path"] for x in m["pairs"]}
    assert {"EVAL-b/b_aime24.jsonl", "EVAL-c/c_aime24.jsonl", "EVAL-base/base.jsonl", "EVAL-t/t.jsonl", "EVAL-s/s.jsonl"} <= pairs
    assert m["open_tasks"] == 5 and [e["name"] for e in m["envs"]] == ["venv", "venv-fa", "venv-vl", "venv-ifeval"]


def test_e12_start_naming():
    assert C.e12_start("E12b-P5tail-16k")[0] == "runs/E10b-P5tail/global_step_50/actor/huggingface"
    assert C.e12_start("E12e-P5self-8k")[0] == "runs/E10e-P5self/global_step_50/actor/huggingface"
    assert C.e12_start("E12d-P5-16k-s43")[0] == "runs/E10d-P5-s43/global_step_150/actor/huggingface"
    assert C.e12_start("E12c-P5-8k")[0] == "runs/E10c-P5-fix4/global_step_150/actor/huggingface"
    assert C.e12_start("E12d-P0") == (None, "base model") and C.e12_start("E10b-P5") == (None, "")



def test_committed_manifest_is_current_format():   # cold_start.sh regenerates it from the queue it pulls; the committed one is the first version
    have = json.load(open(os.path.join(HERE, "..", "tasks", "coldstart.json")))
    assert have["schema"] == "coldstart/1" and {"models", "envs", "data", "artifacts", "weights", "pairs"} <= set(have)
