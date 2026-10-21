"""[v2.5.9z19] round-trip tests of the dataset converters on real rows (code/agent/data/toolrl_rows.jsonl, data/claweval/; see
data/SOURCES.md): every converted sample passes prep's checks and renders, keeps its kind of ground truth (ToolRL: reference
calls; Auto-ClawEval: rules and forbidden tools), and its reference answer scores 1 under grader v2 while padding scores
less. Run: python -m pytest -q code/agent (needs PyYAML for the Auto-ClawEval part)."""
import json
import os
import re

import pytest

import convert_toolrl as T
import prep_tools_parquet as P
import tool_grader as G
from test_tool_agent import _tok

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ROWS = [json.loads(x) for x in open(os.path.join(DATA, "toolrl_rows.jsonl"))]


def call(name, args=None, **kw):   # a Qwen3 call block (arguments as a dict: real tools have arguments called "name")
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": {**(args or {}), **kw}}) + "\n</tool_call>"


def as_text(calls):
    return "\n".join(call(c["name"], c["arguments"]) for c in calls)


def prepared_gt(s):   # the ground truth exactly as the parquet carries it
    return P.build_row(0, s, _tok(), _tok())["reward_model"]["ground_truth"]


# --- ToolRL ---

def test_toolrl_converts_the_real_rows():
    samples, st = T.convert([os.path.join(DATA, "toolrl_rows.jsonl")])
    assert [s["id"] for s in samples] == ["toolrl-test-78", "toolrl-train-2506", "toolrl-train-3574", "toolrl-train-1093"]
    assert st == {"rows": 6, "samples": 4, "calls": 6, "multi-turn queries": 1, "skipped: answer without a tool call": 1,
                  "skipped: reference argument not in the tool's parameters": 1}
    by = {s["id"]: s for s in samples}
    assert by["toolrl-test-78"]["ref_calls"] == [{"name": "check_if_text_contains_profanity",
                                                  "arguments": {"text": "Hey, check out this amazing deal!"}}]
    assert [c["name"] for c in by["toolrl-train-2506"]["ref_calls"]] == ["getBioethicsIssues", "getBioethicsOrganizations",
                                                                          "getBioethicsJournals"]
    assert "<obs>" in by["toolrl-train-3574"]["query"] and by["toolrl-train-3574"]["query"].startswith("**Dialogue Records History**")
    for s in samples:   # ToolRL's own instructions are gone; the tools are schemas
        assert "**Steps for Each Turn**" not in json.dumps(s) and s["rules"] == [] and s["forbidden"] == []


def test_toolrl_round_trip_and_gates():
    samples, _ = T.convert([os.path.join(DATA, "toolrl_rows.jsonl")])
    rows = {f"toolrl-{r['extra_info']['split']}-{r['extra_info']['index']}": r for r in ROWS}
    for s in samples:
        answer = rows[s["id"]]["reward_model"]["ground_truth"]
        body = answer.split("<tool_call>")[1].split("</tool_call>")[0].strip().split("\n")
        assert [(c["name"], c["parameters"]) for c in map(json.loads, body)] == [(c["name"], c["arguments"]) for c in s["ref_calls"]]
        gt = prepared_gt(s)
        assert json.loads(gt)["ref_calls"] == s["ref_calls"] and json.loads(gt)["rules"] == []
        clean = as_text(s["ref_calls"])
        assert G.parse_calls(clean)[0] == [(c["name"], c["arguments"]) for c in s["ref_calls"]]
        assert G.compute_score("tool_call", clean, gt) == 1.0 and G.grade(clean, gt)["mode"] == "ast", s["id"]
        assert G.compute_score("tool_call", "", gt) == 0.0
        other = [t["function"]["name"] for t in s["tools"] if t["function"]["name"] not in {c["name"] for c in s["ref_calls"]}]
        padded = [clean + "\n" + clean] + ([clean + "\n" + call(other[0])] if other else [])
        assert all(G.compute_score("tool_call", p, gt) == 0.0 for p in padded), s["id"]


def test_toolrl_types_and_tool_block():
    jt = T.json_type
    assert jt("str") == ({"type": "string"}, False, True) and jt("int, optional") == ({"type": "integer"}, True, True)
    assert jt("List[int]") == ({"type": "array", "items": {"type": "integer"}}, False, True)
    assert jt("List[List[int]]") == ({"type": "array"}, False, False) and jt("Tuple[float, float]")[0] == {"type": "array"}
    assert jt(["integer", "null"]) == ({"type": "integer"}, True, True) and jt("Callable[[float], float]") == ({}, False, False)
    assert jt("Dict") == ({"type": "object"}, False, True) and jt("boolean") == ({"type": "boolean"}, False, True)
    sys_msg = ("persona\n\n" + T.TOOLS_HEAD + '1. Name: f\nDescription: d\nParameters: {"a": {"description": "x", "type": "List[List[int]]", '
               '"default": ""}, "b": {"description": "y", "type": "str, optional", "default": "z"}}\n2. Name: g\nDescription: e\n'
               'Parameters: {}' + T.TOOLS_END + "\n...")
    f, g = T.parse_tools(sys_msg)
    assert f["function"]["parameters"] == {"type": "object", "properties": {"a": {"type": "array", "description": "x (type: List[List[int]])"},
                                                                            "b": {"type": "string", "description": "y", "default": "z"}},
                                           "required": ["a"]}
    assert g["function"] == {"name": "g", "description": "e", "parameters": {"type": "object", "properties": {}, "required": []}}
    with pytest.raises(ValueError):
        T.parse_tools("no tools here")
    assert T.parse_answer("<response> hi </response>") == []
    with pytest.raises(ValueError):
        T.parse_answer('<tool_call>\n{"name": "f", "parameters": {"a": 1}}\n["not", "a", "call"]\n</tool_call>')


# --- Auto-ClawEval ---

def _claw():
    pytest.importorskip("yaml")
    import convert_claweval as C
    endpoints = json.load(open(C.SCHEMAS))["endpoints"]
    return C, endpoints


def test_claweval_converts_the_real_tasks():
    C, endpoints = _claw()
    samples, st = C.convert(os.path.join(DATA, "claweval"), endpoints)
    assert st["tasks"] == 8 and st["samples"] == 7 and st["skipped: no tools"] == 1 and st["tools with unknown parameters"] == 0
    by = {s["id"]: s for s in samples}
    assert "OCR-066" not in by
    cal = by["calendar-001"]   # audit_action_exists -> tool_called; tool_not_called -> forbidden; answer checks dropped
    assert cal["rules"] == [{"type": "tool_called", "name": "list_events", "weight": 0.15}] and cal["forbidden"] == ["delete_event"]
    assert cal["dropped_weight"] == 0.85 and cal["ref_calls"] == [] and cal["query"].startswith("Our company has a busy week")
    le = next(t for t in cal["tools"] if t["function"]["name"] == "list_events")["function"]["parameters"]
    assert le == {"type": "object", "properties": {"date": {"type": "string"}, "days": {"type": "integer", "default": 1}}, "required": ["date"]}
    seq = [r for r in by["notes-002"]["rules"] if r["weight"] == 0.05]   # audit_sequence: each action, weight split
    assert [r["name"] for r in seq] == ["get_note", "share_note"] and all(r["type"] == "tool_called" for r in seq)
    assert {"type": "tool_arg_equals", "name": "get_note", "arg": "note_id", "value": "note-002", "weight": 0.1} in by["notes-002"]["rules"]
    rec = [r for r in by["notes-028"]["rules"] if r["type"] == "tool_arg_contains"]
    assert [(r["arg"], r["value"]) for r in rec] == [("recipients", "sarah.chen@acme.com"), ("recipients", "david.okafor@acme.com")]
    assert G.ref_names(by["todo-022"]["rules"], []) == {"update_task": 2}             # priority high and low: two calls
    assert by["todo-011"]["added_parameters"] == ["create_task.tags"]                   # audited, missing from the service model
    inv = {t["function"]["name"]: t["function"]["parameters"] for t in by["inventory-007"]["tools"]}   # the YAML's own list form
    assert inv["create_order"]["required"] == ["product_id", "quantity", "supplier"] and inv["create_order"]["properties"]["quantity"]["type"] == "integer"
    web = {t["function"]["name"]: t["function"]["parameters"] for t in by["web_real-004"]["tools"]}   # type names only
    assert web["web_notify"]["properties"]["recipients"] == {"type": "array"} and web["web_search"]["required"] == []


def test_claweval_round_trip_and_gates():
    C, endpoints = _claw()
    samples, _ = C.convert(os.path.join(DATA, "claweval"), endpoints)
    for s in samples:
        gt = prepared_gt(s)
        g = json.loads(gt)
        assert g["rules"] == s["rules"] and g["forbidden"] == s["forbidden"] and g["ref_calls"] == [], s["id"]
        assert set(g["schemas"]) == {t["function"]["name"] for t in s["tools"]}
        answer = C.rule_answer(s)
        assert G.compute_score("tool_call", answer, gt) == 1.0, (s["id"], answer)
        assert G.compute_score("tool_call", "", gt) == 0.0
        assert G.compute_score("tool_call", answer + "\n" + call(s["forbidden"][0]), gt) == 0.0
        calls = G.parse_calls(answer)[0]
        n, a = calls[0]
        others = [t["function"]["name"] for t in s["tools"] if t["function"]["name"] not in {x for x, _ in calls} | set(s["forbidden"])]
        padded = [answer + "\n" + call(n, a), answer.replace(json.dumps(a), json.dumps(dict(a, not_a_param="x")), 1)]
        padded += [answer + "\n" + call(others[0])] if others else []
        assert all(G.compute_score("tool_call", p, gt) < 1.0 for p in padded), s["id"]
    by = {s["id"]: s for s in samples}
    gt = prepared_gt(by["notes-028"])   # stuffing the recipients list
    ok = [{"note_id": "n1", "recipients": ["sarah.chen@acme.com", "david.okafor@acme.com"]}]
    stuffed = [{"note_id": "n1", "recipients": ["sarah.chen@acme.com", "david.okafor@acme.com", "a@acme.com", "b@acme.com"]}]
    lst = call("list_notes")
    assert G.compute_score("tool_call", lst + "\n" + call("share_note", **ok[0]), gt) == 1.0
    assert G.compute_score("tool_call", lst + "\n" + call("share_note", **stuffed[0]), gt) < 1.0
    gt = prepared_gt(by["todo-022"])     # one call cannot hold two priorities
    both = call("update_task", task_id="t1", priority="high") + "\n" + call("update_task", task_id="t2", priority="low")
    assert G.compute_score("tool_call", both, gt) == 1.0
    assert G.compute_score("tool_call", call("update_task", task_id="t1", priority="high, low"), gt) < 0.5


def test_claweval_refuses_unknown_checks():
    C, endpoints = _claw()
    task = {"task_id": "x-1", "prompt": "p", "tools": [{"name": "f", "endpoint": "/nowhere", "description": "d"}],
            "scoring_components": [{"name": "c", "weight": 1.0, "check": {"type": "audit_count_gte", "service": "s", "action": "f", "count": 2}}],
            "safety_checks": []}
    with pytest.raises(ValueError, match="audit_count_gte"):
        C.convert_task(task, endpoints)
    task["scoring_components"][0]["check"] = {"type": "audit_action_exists", "service": "s", "action": "f"}
    task["safety_checks"] = [{"type": "rate_limited", "tool_name": "f"}]
    with pytest.raises(ValueError, match="rate_limited"):
        C.convert_task(task, endpoints)
    task["safety_checks"] = []
    s = C.convert_task(task, endpoints)   # an endpoint without a known model: parameters unknown, not counted as outside
    assert s["tools"][0]["function"]["parameters"] == {"type": "object"} and G.schemas(s["tools"]) == {}
    assert G.compute_score("tool_call", call("f", anything=1), prepared_gt(s)) == 1.0
    assert C.convert_task({"task_id": "y", "prompt": "p", "tools": [], "scoring_components": []}, endpoints) is None


def test_claweval_schema_extraction(tmp_path):
    C, endpoints = _claw()
    d = tmp_path / "mock_services" / "demo"
    d.mkdir(parents=True)
    (d / "server.py").write_text('''
from typing import Optional, List
from pydantic import BaseModel, Field
app = None
class AddRequest(BaseModel):
    title: str
    tags: list[str] = Field(default_factory=list)
    when: str | None = None
    n: int = 3
    owner: Optional[str] = "me"
    note: str = Field("", description="free text")
    must: str = Field(..., description="required")
    ids: Optional[List[int]] = None
@app.post("/demo/add")
def add(req: AddRequest) -> dict:
    return {}
@app.post("/demo/list")
def list_all(req: AddRequest | None = None) -> dict:
    return {}
@app.get("/demo/ping")
def ping() -> dict:
    return {}
@app.get("/demo/audit")
def audit() -> dict:
    return {}
''')
    out = C.extract_schemas(str(tmp_path))
    add = out["endpoints"]["/demo/add"]["parameters"]
    assert add == {"type": "object", "properties": {
        "title": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}, "when": {"type": "string"},
        "n": {"type": "integer", "default": 3}, "owner": {"type": "string", "default": "me"},
        "note": {"type": "string", "description": "free text", "default": ""}, "must": {"type": "string", "description": "required"},
        "ids": {"type": "array", "items": {"type": "integer"}}}, "required": ["title", "must"]}
    assert out["endpoints"]["/demo/list"]["parameters"] == add
    assert out["endpoints"]["/demo/ping"]["parameters"] == {"type": "object", "properties": {}, "required": []}
    assert "/demo/audit" not in out["endpoints"] and out["conflicts"] == []
    src = json.load(open(C.SCHEMAS))["source"]   # the committed table: its source, and every fixture tool's endpoint in it
    assert src["repo"] == "xirui-li/ClawEnvKit" and re.fullmatch(r"[0-9a-f]{40}", src["commit"]) and src["licence"] == "MIT"
    import yaml
    for p in sorted(os.listdir(os.path.join(DATA, "claweval", "tasks"))):
        for f in os.listdir(os.path.join(DATA, "claweval", "tasks", p)):
            t = yaml.safe_load(open(os.path.join(DATA, "claweval", "tasks", p, f)))
            assert all(x.get("parameters") or x["endpoint"] in endpoints for x in t.get("tools") or []), f
