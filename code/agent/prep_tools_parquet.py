"""Tool-call samples -> verl parquet for tool-call OPD (runbook [v2.5.9b], research_program §1.7).

The tool schemas are rendered with the student's own chat template (Qwen3's "# Tools ... <tools>...</tools> ..." system
block, exactly as apply_chat_template(..., tools=...) writes it) and stored as an explicit system message. The student
(verl's RLHFDataset, data.apply_chat_template_kwargs) and the teacher (G-OPD's prepare_ref_model_inputs, its own tokenizer)
then render byte-identical prompts in the native tool format without any change to verl (CoworkEnv-Lessons L04: the
teacher and student prefixes must agree byte for byte). This script checks that for both tokenizers: the system-message
form renders exactly like the native tools= form, the student without thinking and the teacher with enable_thinking=True
(the P5 setting), and fails otherwise.

Input: JSON lines {"id", "tools": [OpenAI-style function schemas], "query", "ref_calls", "rules", "forbidden"}
(code/agent/data/synthetic_tools.jsonl for the smoke; convert_toolrl.py and convert_claweval.py write the same shape).
[v2.5.9z19] Every sample is checked before the tokenizers load (check_sample: tool_grader.validate, i.e. known rule types
only, well-formed rules / reference calls, something to grade, no rule or reference call on a tool the prompt does not
offer; tools well-formed with distinct names); the first bad sample stops the run with its id.
Output: <out>_train.parquet / <out>_val.parquet with data_source "tool_call", prompt [system, user], reward_model
{"style": "rule", "ground_truth": <rules, forbidden, ref_calls and [v2.5.9z19] schemas (tool_grader.schemas) as JSON>},
extra_info {index, id, tools (JSON)}; and <out>_sample.txt, one fully rendered prompt (student and teacher) with its token
count (L03: look at one sample).

Usage: python prep_tools_parquet.py IN.jsonl OUT_PREFIX STUDENT_TOKENIZER TEACHER_TOKENIZER [--repeat N] [--val K]
"""
import argparse
import json

import tool_grader as G

SYS_OPEN, END = "<|im_start|>system\n", "<|im_end|>"


def render(tok, messages, tools=None, thinking=False):
    kw = {"tools": tools} if tools else {}
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking, **kw)


def tool_system(tok, query, tools):
    """The system block the template writes for these tools (the text between <|im_start|>system\\n and <|im_end|>)."""
    native = render(tok, [{"role": "user", "content": query}], tools)
    if not native.startswith(SYS_OPEN) or END not in native:
        raise ValueError("the template did not open with a tools system block")
    return native[len(SYS_OPEN):native.index(END)]


def check_sample(i, s):
    """[v2.5.9z19] Raises ValueError (naming the sample) when the sample is malformed; returns its ground truth."""
    where = f"sample {s.get('id', i)}" if isinstance(s, dict) else f"line {i + 1}"
    if not isinstance(s, dict) or not isinstance(s.get("query"), str) or not s["query"].strip():
        raise ValueError(f"{where}: no query")
    tools = s.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{where}: no tools")
    names = []
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if not isinstance(f, dict) or not isinstance(f.get("name"), str) or not isinstance(f.get("parameters", {}), dict):
            raise ValueError(f"{where}: a tool is not an OpenAI-style function schema: {json.dumps(t)[:200]}")
        names.append(f["name"])
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: duplicate tool names")
    gt = {"rules": s.get("rules", []), "forbidden": s.get("forbidden", []), "ref_calls": s.get("ref_calls", []),
          "schemas": G.schemas(tools)}
    G.validate(gt, names, where)
    return gt


def build_row(i, s, stok, ttok):
    """One parquet row; raises when the sample is malformed or the system-message form differs from the native tools= form
    for either tokenizer."""
    gt = check_sample(i, s)
    q, tools = s["query"], s["tools"]
    sysmsg = tool_system(stok, q, tools)
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": q}]
    for name, tok, think in (("student", stok, False), ("teacher", ttok, True)):
        a, b = render(tok, msgs, None, think), render(tok, [{"role": "user", "content": q}], tools, think)
        if a != b:
            raise AssertionError(f"sample {s.get('id', i)}: the {name} renders the system-message form differently from tools=")
    return {"data_source": "tool_call", "prompt": msgs, "ability": "tool",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"index": i, "id": s.get("id", str(i)), "tools": json.dumps(tools)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out"); ap.add_argument("student"); ap.add_argument("teacher")
    ap.add_argument("--repeat", type=int, default=1, help="repeat the training rows (a smoke needs a full batch)")
    ap.add_argument("--val", type=int, default=0, help="rows kept apart as the validation file (0: validate on all)")
    a = ap.parse_args()
    samples = [json.loads(x) for x in open(a.inp) if x.strip()]
    for i, s in enumerate(samples):   # [v2.5.9z19] all samples checked before the tokenizers load
        check_sample(i, s)
    import pandas as pd
    from transformers import AutoTokenizer
    stok, ttok = AutoTokenizer.from_pretrained(a.student), AutoTokenizer.from_pretrained(a.teacher)
    rows = [build_row(i, s, stok, ttok) for i, s in enumerate(samples)]
    val = rows[:a.val] if a.val else rows
    train = (rows[a.val:] if a.val else rows) * a.repeat
    pd.DataFrame(train).to_parquet(a.out + "_train.parquet"); pd.DataFrame(val).to_parquet(a.out + "_val.parquet")
    r0 = rows[0]["prompt"]
    st, tt = render(stok, r0), render(ttok, r0, thinking=True)
    with open(a.out + "_sample.txt", "w") as f:
        f.write(f"student prompt ({len(stok(st, add_special_tokens=False)['input_ids'])} tokens):\n{st}\n\n")
        f.write(f"teacher prompt, thinking, before the P5 '<think>\\n' ({len(ttok(tt, add_special_tokens=False)['input_ids'])} tokens):\n{tt}\n")
    print(f"TOOLS_PREP_OK {len(samples)} samples, train {len(train)} rows, val {len(val)} rows")


if __name__ == "__main__":
    main()
