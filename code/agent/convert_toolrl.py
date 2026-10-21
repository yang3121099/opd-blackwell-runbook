"""ToolRL (qiancheng0/ToolRL, dataset/rlla_4k, Apache-2.0) -> the tool-call JSON lines of prep_tools_parquet.py (runbook
[v2.5.9z19]; research_program §1.7).

rlla_4k is verl parquet in ToolRL's own text template: the system message lists the tools as
    N. Name: <name>\\nDescription: <text>\\nParameters: {<json>}
between its own instructions (a persona, then "**Steps for Each Turn**" and an output format of <think>, <tool_call> lines
with "parameters", <response>); the user message is the dialogue so far ("**Dialogue Records History**", earlier turns with
their <tool_call> and <obs>); reward_model.ground_truth is the reference answer in that format, and the rows have no rules.
Each row whose answer calls tools becomes one sample:
  - tools: the listed tools as OpenAI-style schemas; ToolRL's parameter types mapped to JSON-schema types (str / string,
    int, float, bool, list / List[...] / Tuple[...] / set, dict; a list of types takes its first non-null one), the original
    type appended to the description when the mapping does not say it all (e.g. List[List[int]], Callable[...]);
    "<type>, optional" or a non-empty default keeps the parameter out of required; a non-empty default is kept;
  - query: the user message as it is (ToolRL's system instructions are dropped: prep renders the student's native tool
    block, whose call format is Qwen3's <tool_call> with "arguments");
  - ref_calls: the reference answer's calls ("parameters" -> "arguments"); rules [] and forbidden [], so tool_grader
    scores them with ast_match 1 / 0 (numbers and numeric strings compare equal: the references often quote numbers).
Skipped and counted: rows whose answer has no tool call (a text <response> only, which this grader does not score), whose
tool block or reference calls do not parse, with two tools of one name, whose reference names a tool not listed, or whose
reference passes an argument the tool does not list (112 of rlla_4k's rows: broken references such as train-10's, a
JSON list cut at its commas into argument names; ast_match could never score them 1). Every sample is checked as
prep_tools_parquet.py checks it.

Usage: python convert_toolrl.py IN [IN ...] OUT.jsonl
  IN: rlla_4k's train.parquet / test.parquet (pandas), or JSON lines of the same rows (code/agent/data/toolrl_rows.jsonl).
  The sample id is toolrl-<split>-<index> (extra_info.split / index, else the file name and the row number).
"""
import json
import os
import re
import sys
from collections import Counter

import prep_tools_parquet as P

TOOLS_HEAD = "**Available Tools**\nIn your response, you can use the following tools:\n"
TOOLS_END = "\n\n**Steps for Each Turn**"
TOOL_RE = re.compile(r"^\d+\. Name: (.*?)\nDescription: (.*?)\nParameters: (.*)$", re.S)
SCALARS = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
           "number": "number", "double": "number", "bool": "boolean", "boolean": "boolean"}
ARRAYS = ("list", "tuple", "set", "array")
OBJECTS = ("dict", "object", "mapping")


def json_type(t):
    """(JSON schema of a ToolRL type, optional, whether the schema says all the type says)."""
    if isinstance(t, list):   # ["integer", "null"]
        rest = [x for x in t if x != "null"]
        s, _, full = json_type(rest[0]) if rest else ({}, False, False)
        return s, "null" in t, full and len(rest) == 1
    t = str(t or "").strip()
    optional = t.lower().endswith(", optional")
    if optional:
        t = t[:-len(", optional")].strip()
    low = t.lower()
    if low in SCALARS:
        return {"type": SCALARS[low]}, optional, True
    head, _, inner = low.partition("[")
    inner = inner[:-1] if inner.endswith("]") else inner
    if head in ARRAYS:
        if inner in SCALARS:
            return {"type": "array", "items": {"type": SCALARS[inner]}}, optional, head in ("list", "array")
        return {"type": "array"}, optional, not inner and head in ("list", "array")
    if head in OBJECTS:
        return {"type": "object"}, optional, not inner
    return {}, optional, False


def parse_tools(system):
    """OpenAI-style tool schemas from ToolRL's system message; raises ValueError when the block does not parse."""
    if TOOLS_HEAD not in system:
        raise ValueError("no tool block")
    block = system.split(TOOLS_HEAD, 1)[1].split(TOOLS_END, 1)[0]
    tools = []
    for item in re.split(r"\n(?=\d+\. Name: )", block.strip()):
        m = TOOL_RE.match(item)
        if not m:
            raise ValueError(f"tool entry does not parse: {item[:80]!r}")
        params = json.loads(m.group(3))
        if not isinstance(params, dict):
            raise ValueError(f"parameters of {m.group(1)!r} are not an object")
        props, required = {}, []
        for k, v in params.items():
            v = v if isinstance(v, dict) else {"description": str(v)}
            s, optional, full = json_type(v.get("type"))
            desc = str(v.get("description", "")).strip()
            if not full and v.get("type"):
                desc = f"{desc} (type: {v['type'] if isinstance(v['type'], str) else json.dumps(v['type'])})".strip()
            prop = dict(s)
            if desc:
                prop["description"] = desc
            if v.get("default") not in ("", None):
                prop["default"] = v["default"]
            props[k] = prop
            if not optional and v.get("default") in ("", None):
                required.append(k)
        tools.append({"type": "function", "function": {"name": m.group(1).strip(), "description": m.group(2).strip(),
                                                       "parameters": {"type": "object", "properties": props, "required": required}}})
    return tools


def parse_answer(ans):
    """The reference answer's calls [{"name", "arguments"}] ([] for a text-only answer); raises ValueError when a line of
    its <tool_call> block is not a call."""
    if "<tool_call>" not in ans:
        return []
    body = ans.split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
    calls = []
    for line in body.split("\n"):
        c = json.loads(line)
        args = c.get("parameters", c.get("arguments", {})) if isinstance(c, dict) else None
        if not isinstance(args, dict) or not isinstance(c.get("name"), str):
            raise ValueError(f"not a call: {line[:80]!r}")
        calls.append({"name": c["name"], "arguments": args})
    return calls


def convert_row(r, split, index):
    """(sample or None, reason it was skipped)."""
    msgs = list(r["prompt"])
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    try:
        tools = parse_tools(system)
    except ValueError:
        return None, "tool block does not parse"
    names = [t["function"]["name"] for t in tools]
    if len(set(names)) != len(names):
        return None, "two tools of one name"
    try:
        calls = parse_answer(r["reward_model"]["ground_truth"])
    except ValueError:
        return None, "reference calls do not parse"
    if not calls:
        return None, "answer without a tool call"
    if any(c["name"] not in names for c in calls):
        return None, "reference names a tool not listed"
    props = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in tools}
    if any(set(c["arguments"]) - set(props[c["name"]]) for c in calls):   # e.g. train-10: a list argument split into keys
        return None, "reference argument not in the tool's parameters"
    return {"id": f"toolrl-{split}-{index}", "source": "ToolRL/rlla_4k", "tools": tools, "query": user.strip(),
            "ref_calls": calls, "rules": [], "forbidden": []}, None


def _plain(v):   # numpy arrays / pandas objects of a parquet row -> lists and dicts
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def read_rows(path):
    if path.endswith(".parquet"):
        import pandas as pd
        return [_plain(r) for r in pd.read_parquet(path).to_dict("records")]
    return [json.loads(x) for x in open(path) if x.strip()]


def convert(paths):
    """(samples, stats)."""
    out, st = [], Counter()
    for path in paths:
        default_split = os.path.basename(path).split(".")[0]
        for i, r in enumerate(read_rows(path)):
            st["rows"] += 1
            ei = r.get("extra_info") or {}
            s, why = convert_row(r, ei.get("split", default_split), ei.get("index", i))
            if s is None:
                st["skipped: " + why] += 1
                continue
            P.check_sample(len(out), s)
            st["samples"] += 1
            st["calls"] += len(s["ref_calls"])
            st["multi-turn queries"] += "<obs>" in s["query"] or s["query"].count("<user>") > 1
            out.append(s)
    if len({s["id"] for s in out}) != len(out):
        raise ValueError("duplicate sample ids (two inputs of one split?)")
    return out, dict(st)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    samples, st = convert(sys.argv[1:-1])
    with open(sys.argv[-1], "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("TOOLRL_CONVERT_OK", json.dumps(st))


if __name__ == "__main__":
    main()
