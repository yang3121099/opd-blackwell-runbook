"""Auto-ClawEval / Auto-ClawEval-mini (AIcell, ClawEnvKit tasks, MIT) -> the tool-call JSON lines of prep_tools_parquet.py
(runbook [v2.5.9z19]; research_program §1.7).

Each task YAML (tasks/<category>/<task_id>.yaml) is an agentic task against mock services: a prompt, tools (name,
description, service, endpoint; parameters for only a few), weighted scoring components and safety checks. One task becomes
one single-turn sample: the prompt as the query, the tools as OpenAI-style schemas, and the components that read tool calls
as rules, with their weights (semantics of ClawEnvKit's evaluate/engine.py):
    audit_action_exists    -> tool_called
    audit_field_equals     -> tool_arg_equals (ClawEnvKit compares exactly, tool_grader after normalisation)
    audit_field_contains   -> tool_arg_contains (a case-insensitive substring in both)
    audit_sequence         -> tool_called for each action, the weight split evenly (one turn has no order)
    safety tool_not_called -> forbidden
Components that read the final answer or the workspace (keywords_present / keywords_absent, llm_judge, file_exists,
exit_code; safety keywords_not_in_output) are dropped; each sample keeps their total weight as dropped_weight. Any other
check type stops the run (its semantics are unknown here). Tasks without tools (file, OCR, terminal tasks) are skipped.
There are no reference calls: the samples are graded by their rules (tool_grader: rules x Jaccard x repeat x schema
penalty). Tool parameters: the YAML's own when given, else the request model of the tool's endpoint in ClawEnvKit's mock
services (data/clawenvkit_schemas.json, made by --extract-schemas from a ClawEnvKit checkout; read with ast, nothing in it
is run), else unknown ({"type": "object"} without properties; the grader then counts none of that tool's arguments as
outside its schema); an audited argument a known schema lacks is added to it (added_parameters). in_query is the share of the sample's equals / contains values found in the prompt (case-insensitive):
the others come from what an earlier call returns in the agentic task, so --min-in-query 1 keeps the samples one turn can
fully solve. Every sample is checked as prep_tools_parquet.py checks it, and rule_answer (the calls the rules ask for)
must score 1.

Usage: python convert_claweval.py DATASET_DIR OUT.jsonl [--schemas JSON] [--min-in-query F]
       python convert_claweval.py --extract-schemas CLAWENVKIT_DIR --commit SHA > data/clawenvkit_schemas.json
"""
import argparse
import ast
import glob
import json
import os
import sys
from collections import Counter

import prep_tools_parquet as P
import tool_grader as G

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMAS = os.path.join(HERE, "data", "clawenvkit_schemas.json")
OUTPUT_CHECKS = ("keywords_present", "keywords_absent", "llm_judge", "file_exists", "exit_code")
SCALARS = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
YAML_TYPES = {**SCALARS, "list": "array", "dict": "object"}


# --- ClawEnvKit mock services -> request schemas (static: ast only) ---

def _annotation_schema(node):
    """(JSON schema, nullable) of a pydantic field annotation."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):   # X | None
        sides = [s for s in (node.left, node.right) if not (isinstance(s, ast.Constant) and s.value is None)]
        return (_annotation_schema(sides[0])[0] if len(sides) == 1 else {}), True
    if isinstance(node, ast.Subscript):
        head = ast.unparse(node.value)
        inner = node.slice.value if type(node.slice).__name__ == "Index" else node.slice   # Python 3.8 wraps it in Index
        if head == "Optional":
            return _annotation_schema(inner)[0], True
        if head in ("list", "List", "set", "Set", "tuple", "Tuple"):
            item = inner.elts[0] if isinstance(inner, ast.Tuple) else inner
            s = _annotation_schema(item)[0]
            return ({"type": "array", "items": s} if s else {"type": "array"}), False
        if head in ("dict", "Dict"):
            return {"type": "object"}, False
        return {}, False
    name = ast.unparse(node)
    if name in SCALARS:
        return {"type": SCALARS[name]}, False
    if name in ("list", "List"):
        return {"type": "array"}, False
    if name in ("dict", "Dict"):
        return {"type": "object"}, False
    return {}, False


def _model_schema(cls):
    props, required = {}, []
    for st in cls.body:
        if not (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)):
            continue
        s, _ = _annotation_schema(st.annotation)
        s = dict(s)
        req, default = st.value is None, None
        if isinstance(st.value, ast.Call) and ast.unparse(st.value.func) == "Field":
            kw = {k.arg: k.value for k in st.value.keywords}
            first = st.value.args[0] if st.value.args else kw.get("default")
            req = (first is None and "default_factory" not in kw) or (isinstance(first, ast.Constant) and first.value is Ellipsis)
            if "description" in kw and isinstance(kw["description"], ast.Constant):
                s["description"] = kw["description"].value
            default = first if not req else None
        elif st.value is not None:
            default = st.value
        if default is not None:
            try:
                v = ast.literal_eval(default)
                if v is not None and v is not Ellipsis:
                    s["default"] = v
            except ValueError:
                pass
        props[st.target.id] = s
        if req:
            required.append(st.target.id)
    return {"type": "object", "properties": props, "required": required}


def extract_schemas(root):
    """{"endpoints": {path: {"services": [...], "parameters": JSON schema}}} from mock_services/*/server.py: each route's
    request model (its BaseModel fields; a route without a body has no parameters; the harness's audit / reset / health
    routes left out). A path whose services disagree is dropped and listed under "conflicts"."""
    seen = {}
    for path in sorted(glob.glob(os.path.join(root, "mock_services", "*", "server.py"))):
        service = os.path.basename(os.path.dirname(path))
        tree = ast.parse(open(path).read())
        models = {n.name: _model_schema(n) for n in tree.body if isinstance(n, ast.ClassDef)
                  and any(ast.unparse(b) == "BaseModel" for b in n.bases)}
        for fn in tree.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for d in fn.decorator_list:
                if not (isinstance(d, ast.Call) and ast.unparse(d.func) in ("app.post", "app.get") and d.args
                        and isinstance(d.args[0], ast.Constant)):
                    continue
                route, params = d.args[0].value, {"type": "object", "properties": {}, "required": []}
                if route.rstrip("/").rsplit("/", 1)[-1] in ("audit", "reset", "health"):   # the harness's own routes
                    continue
                args = fn.args.args
                if args and args[0].annotation is not None:
                    ann = args[0].annotation
                    if isinstance(ann, ast.BinOp):
                        ann = ann.left
                    params = models.get(ast.unparse(ann), params)
                seen.setdefault(route, []).append((service, params))
    endpoints, conflicts = {}, []
    for route, found in sorted(seen.items()):
        if any(json.dumps(p, sort_keys=True) != json.dumps(found[0][1], sort_keys=True) for _, p in found):
            conflicts.append(route)
            continue
        endpoints[route] = {"services": sorted({s for s, _ in found}), "parameters": found[0][1]}
    return {"endpoints": endpoints, "conflicts": conflicts}


# --- task YAML -> sample ---

def _yaml_property(v):
    """One parameter as the YAML gives it: a schema, or only its type name ("integer", "list", ...)."""
    v = {"type": v} if isinstance(v, str) else {k: x for k, x in v.items() if k in ("type", "description", "items", "enum")}
    if isinstance(v.get("type"), str):
        v["type"] = YAML_TYPES.get(v["type"], v["type"])
    if isinstance(v.get("items"), str):
        v["items"] = {"type": YAML_TYPES.get(v["items"], v["items"])}
    return v


def _yaml_parameters(p):
    """The YAML's own parameters: a list of {name, type, required, description}, or {name: schema or type name}."""
    if isinstance(p, list):
        return {"type": "object", "properties": {x["name"]: _yaml_property(x) for x in p},
                "required": [x["name"] for x in p if x.get("required")]}
    return {"type": "object", "properties": {k: _yaml_property(v) for k, v in p.items()}, "required": []}


def tool_schema(t, endpoints):
    if t.get("parameters"):
        params = _yaml_parameters(t["parameters"])
    elif t.get("endpoint") in endpoints:
        params = endpoints[t["endpoint"]]["parameters"]
    else:
        params = {"type": "object"}   # unknown
    return {"type": "function", "function": {"name": t["name"], "description": " ".join(str(t.get("description", "")).split()),
                                             "parameters": params}}


def convert_task(t, endpoints):
    """One sample, or None for a task without tools. Raises ValueError on a check of an unknown kind."""
    if not t.get("tools"):
        return None
    tid, rules, forbidden, dropped = t["task_id"], [], [], 0.0
    for c in t.get("scoring_components") or []:
        ck, w = c["check"], float(c.get("weight", 1.0))
        ty = ck["type"]
        if ty == "audit_action_exists" and not ck.get("field_match"):
            rules.append({"type": "tool_called", "name": ck["action"], "weight": w})
        elif ty in ("audit_field_equals", "audit_field_contains"):
            rules.append({"type": "tool_arg_equals" if ty == "audit_field_equals" else "tool_arg_contains",
                          "name": ck["action"], "arg": ck["field"], "value": ck["value"], "weight": w})
        elif ty == "audit_sequence" and not any(isinstance(a, dict) and a.get("field_match") for a in ck["actions"]):
            acts = [a["action"] if isinstance(a, dict) else a for a in ck["actions"]]
            rules += [{"type": "tool_called", "name": a, "weight": w / len(acts)} for a in acts]
        elif ty in OUTPUT_CHECKS:
            dropped += w
        else:
            raise ValueError(f"{tid}: check type {ty!r} with fields {sorted(ck)} has no rule here")
    for s in t.get("safety_checks") or []:
        if s["type"] == "tool_not_called":
            forbidden.append(s.get("tool_name") or s["tool"])
        elif s["type"] != "keywords_not_in_output":
            raise ValueError(f"{tid}: safety check {s['type']!r} has no rule here")
    q = str(t["prompt"]).strip()
    tools = [tool_schema(x, endpoints) for x in t["tools"]]
    added = []
    for r in rules:   # an audited argument the schema lacks (todo-011: create_task.tags) is added, typed by its value
        props = next((x["function"]["parameters"].get("properties") for x in tools if x["function"]["name"] == r["name"]), None)
        if r["type"] in G.ARG_RULES and props is not None and r["arg"].split(".")[0] not in props:
            props[r["arg"].split(".")[0]] = {"type": _value_type(r["value"])}
            added.append(f"{r['name']}.{r['arg']}")
    vals = [str(r["value"]) for r in rules if r["type"] in ("tool_arg_equals", "tool_arg_contains")]
    return {"id": tid, "source": "Auto-ClawEval", "tools": tools, "query": q, "ref_calls": [], "rules": rules,
            "forbidden": sorted(set(forbidden)),
            "in_query": round(sum(v.casefold() in q.casefold() for v in vals) / len(vals), 4) if vals else 1.0,
            "dropped_weight": round(dropped, 6), "added_parameters": added}


def _value_type(v):
    return ("boolean" if isinstance(v, bool) else "number" if isinstance(v, (int, float)) else "array" if isinstance(v, list)
            else "object" if isinstance(v, dict) else "string")


def rule_answer(s):
    """The calls the rules ask for, as Qwen3 <tool_call> blocks: each tool a positive rule names, as many calls as
    tool_grader.ref_names counts; the i-th distinct equals / number_near value of an argument in the i-th call, contains
    values joined by spaces when no equals value holds them (a list for an array parameter)."""
    types = {t["function"]["name"]: t["function"]["parameters"].get("properties", {}) for t in s["tools"]}
    out = []
    for name, k in G.ref_names(s["rules"], s.get("ref_calls", [])).items():
        calls = [{} for _ in range(k)]
        mine = [r for r in s["rules"] if r.get("name") == name and r["type"] in G.ARG_RULES and r["type"] != "tool_arg_excludes"]
        for arg in dict.fromkeys(r["arg"] for r in mine):
            exact = list(dict.fromkeys(json.dumps(r["value"]) for r in mine if r["arg"] == arg and r["type"] != "tool_arg_contains"))
            needles = [r["value"] for r in mine if r["arg"] == arg and r["type"] == "tool_arg_contains"]
            is_list = types.get(name, {}).get(arg, {}).get("type") == "array"
            for i, c in enumerate(calls):
                if exact:
                    c[arg] = json.loads(exact[min(i, len(exact) - 1)])
                else:
                    c[arg] = needles if is_list else " ".join(map(str, needles))
        out += [(name, c) for c in calls]
    return "\n".join("<tool_call>\n" + json.dumps({"name": n, "arguments": a}) + "\n</tool_call>" for n, a in out)


def load_tasks(root):
    import yaml
    for p in sorted(glob.glob(os.path.join(root, "tasks", "*", "*.yaml"))):
        with open(p) as f:
            yield yaml.safe_load(f)


def convert(root, endpoints, min_in_query=0.0):
    """(samples, stats)."""
    out, st = [], Counter()
    for t in load_tasks(root):
        st["tasks"] += 1
        s = convert_task(t, endpoints)
        if s is None:
            st["skipped: no tools"] += 1
            continue
        if s["in_query"] < min_in_query:
            st["skipped: in_query below the minimum"] += 1
            continue
        P.check_sample(len(out), s)
        st["tools with unknown parameters"] += sum("properties" not in x["function"]["parameters"] for x in s["tools"])
        st["parameters added from rules"] += len(s["added_parameters"])
        if G.compute_score("tool_call", rule_answer(s), json.dumps({**s, "schemas": G.schemas(s["tools"])})) < 1.0:
            st["rule answer below 1 (kept)"] += 1
        st["samples"] += 1
        st["rules"] += len(s["rules"])
        st["samples with forbidden tools"] += bool(s["forbidden"])
        st["samples fully in the query"] += s["in_query"] == 1.0
        out.append(s)
    return out, dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--schemas", default=SCHEMAS); ap.add_argument("--min-in-query", type=float, default=0.0)
    ap.add_argument("--extract-schemas", metavar="CLAWENVKIT_DIR")
    ap.add_argument("--repo", default="xirui-li/ClawEnvKit"); ap.add_argument("--commit", help="recorded with --extract-schemas")
    a = ap.parse_args()
    if a.extract_schemas:   # one endpoint per line
        d = extract_schemas(a.extract_schemas)
        print('{"source": ' + json.dumps({"repo": a.repo, "commit": a.commit, "licence": "MIT", "files": "mock_services/*/server.py"})
              + ',\n "conflicts": ' + json.dumps(d["conflicts"]) + ',\n "endpoints": {')
        print(",\n".join(f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in d["endpoints"].items()) + "\n }\n}")
        return
    if not (a.inp and a.out):
        ap.error("DATASET_DIR and OUT.jsonl are needed")
    endpoints = json.load(open(a.schemas))["endpoints"]
    samples, st = convert(a.inp, endpoints, a.min_in_query)
    with open(a.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("CLAWEVAL_CONVERT_OK", json.dumps(st))


if __name__ == "__main__":
    main()
