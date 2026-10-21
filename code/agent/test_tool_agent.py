"""[v2.5.9b] unit tests of the tool-call grader, the BFCL-style match and the parquet prep (no GPU, no transformers).
Run: python -m pytest -q test_tool_agent.py, or a plain runner of the test_ functions (PROBE-TOOL). QWEN_TEMPLATE=<a Qwen3
tokenizer_config.json> renders the prep checks with the real chat template (jinja2); otherwise a minimal template of the
same shape is used."""
import glob
import json
import os

import prep_tools_parquet as P
import tool_grader as G

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = [json.loads(x) for x in open(os.path.join(HERE, "data", "synthetic_tools.jsonl"))]


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def gt(rid):
    s = next(x for x in SAMPLES if x["id"] == rid)
    return json.dumps({"rules": s["rules"], "forbidden": s["forbidden"], "ref_calls": s["ref_calls"]})


def test_parse_calls():
    t = "<think>\nmaybe " + call("x", a=1) + "\n</think>\nSure. " + call("get_weather", city="Paris") + " <tool_call>{bad json</tool_call>"
    calls, n = G.parse_calls(t)
    assert calls == [("get_weather", {"city": "Paris"})] and n == 2      # the call inside <think> is ignored
    s = '<tool_call>{"name": "f", "arguments": "{\\"k\\": 2}"}</tool_call>'
    assert G.parse_calls(s)[0] == [("f", {"k": 2})]                       # arguments given as a JSON string
    assert G.parse_calls("no calls")[0] == [] and G.parse_calls(None)[0] == []


def test_unclosed_think_is_stripped():   # reviewer A1: a thinking teacher cut at a small budget leaves <think> open
    assert G.parse_calls("<think>\ndraft " + call("get_weather", city="Paris"))[0] == []
    assert G.compute_score("tool_call", "<think>\ndraft " + call("get_weather", city="Paris", unit="celsius"), gt("w1")) == 0.0
    assert G.parse_calls("<think>a</think>" + call("f", k=1))[0] == [("f", {"k": 1})]   # a closed block still ends where it closes


def test_rules_and_score():
    full = call("get_weather", city="Paris", unit="celsius")
    assert G.compute_score("tool_call", full, gt("w1")) == 1.0
    part = call("get_weather", city="paris")                                # case-insensitive; unit missing (weight 0.5)
    assert abs(G.compute_score("tool_call", part, gt("w1")) - 2 / 2.5) < 1e-9
    assert G.compute_score("tool_call", "It is sunny in Paris.", gt("w1")) == 0.0   # no call
    assert G.compute_score("tool_call", call("calculate", expression="1+1"), gt("w1")) == 0.0
    near = call("convert_units", value=26.205, from_unit="miles", to_unit="km")
    far = call("convert_units", value=42, from_unit="miles", to_unit="km")
    assert G.compute_score("tool_call", near, gt("c1")) == 1.0 and G.compute_score("tool_call", far, gt("c1")) == 0.75
    lst = call("create_event", title="Budget review", date="2026-11-03", time="14:30", attendees=["alice@example.com", "bob@example.com"])
    assert G.compute_score("tool_call", lst, gt("e1")) == 1.0               # list arguments: any element contains
    assert G.compute_score("tool_call", call("search_files", keyword="quarterly report", folder="../etc"), gt("p1")) < 1.0
    two = call("get_weather", city="Tokyo") + "\n" + call("get_weather", city="Osaka")
    assert G.compute_score("tool_call", two, gt("w2")) == 1.0 and G.compute_score("tool_call", call("get_weather", city="Tokyo"), gt("w2")) == 0.5


def test_forbidden_gate_and_legal():
    ok = call("send_email", to="carol@example.com", subject="Draft ready", body="It is in the shared folder.")
    assert G.compute_score("tool_call", ok, gt("s1")) == 1.0
    g = G.grade(ok + "\n" + call("delete_email", email_id="42"), gt("s1"))
    assert g["score"] == 0.0 and g["forbidden"] == ["delete_email"]
    assert G.grade(ok + "<tool_call>oops</tool_call>", gt("s1"))["legal"] is False and G.grade(ok, gt("s1"))["legal"] is True
    assert G.compute_score("tool_call", ok, "not json") == 0.0             # a broken ground truth scores 0, no exception


def test_ast_match():
    ref = [{"name": "search_flights", "arguments": {"origin": "SFO", "destination": "JFK", "date": "2026-12-20", "passengers": 3}}]
    good = [("search_flights", {"origin": "sfo", "destination": "JFK ", "date": "2026-12-20", "passengers": 3.0})]
    assert G.ast_match(good, ref)
    assert not G.ast_match([("search_flights", {"origin": "SFO", "destination": "JFK", "date": "2026-12-20"})], ref)   # missing
    assert not G.ast_match(good + good, ref)                                                                              # extra call
    schemas = {"search_flights": ["origin", "destination", "date", "passengers"]}
    assert not G.ast_match([("search_flights", dict(good[0][1], cabin="first"))], ref, schemas)                        # unknown arg
    two = [{"name": "get_weather", "arguments": {"city": "Tokyo"}}, {"name": "get_weather", "arguments": {"city": "Osaka"}}]
    assert G.ast_match([("get_weather", {"city": "Osaka"}), ("get_weather", {"city": "Tokyo"})], two)                   # any order


def test_every_sample_is_solvable():   # each reference call scores 1 and matches itself
    for s in SAMPLES:
        text = "\n".join(call(c["name"], **c["arguments"]) for c in s["ref_calls"])
        assert G.compute_score("tool_call", text, gt(s["id"])) == 1.0, s["id"]
        assert G.ast_match(G.parse_calls(text)[0], s["ref_calls"]), s["id"]


# --- [v2.5.9z19] grader v2: the reviewer's gate checks, then the parts ---

def gt2(rid, **over):   # the ground truth as prep writes it (with the tool schemas)
    s = next(x for x in SAMPLES if x["id"] == rid)
    return json.dumps({"rules": s["rules"], "forbidden": s["forbidden"], "ref_calls": s["ref_calls"],
                       "schemas": G.schemas(s["tools"]), **over})


def ref_text(s):
    return "\n".join(call(c["name"], **c["arguments"]) for c in s["ref_calls"])


def test_v2_gate_reference_scores_1_and_empty_scores_0():
    for s in SAMPLES:
        assert G.compute_score("tool_call", ref_text(s), gt2(s["id"])) == 1.0, s["id"]
        assert G.compute_score("tool_call", ref_text(s), gt2(s["id"], rules=[])) == 1.0, s["id"]   # the same calls by ast_match
        for empty in ("", "<think>\nlet me see\n</think>\n", "I cannot help.", "<tool_call></tool_call>"):
            assert G.compute_score("tool_call", empty, gt2(s["id"])) == 0.0, (s["id"], empty)
            assert G.compute_score("tool_call", empty, gt2(s["id"], rules=[])) == 0.0, (s["id"], empty)


def test_v2_gate_forbidden_tool_scores_0():
    ok = ref_text(next(x for x in SAMPLES if x["id"] == "s1"))
    assert G.compute_score("tool_call", ok, gt2("s1")) == 1.0
    for extra in (call("delete_email", email_id="42"), call("delete_email")):
        assert G.compute_score("tool_call", ok + "\n" + extra, gt2("s1")) == 0.0
        assert G.compute_score("tool_call", extra + "\n" + ok, gt2("s1", rules=[])) == 0.0


def test_v2_gate_padding_scores_below_the_clean_answer():
    for s in SAMPLES:
        clean = ref_text(s)
        other = next(t["function"]["name"] for t in s["tools"] if t["function"]["name"] not in {c["name"] for c in s["ref_calls"]})
        first = s["ref_calls"][0]
        padded = {
            "an extra tool": clean + "\n" + call(other),
            "a repeated call": clean + "\n" + call(first["name"], **first["arguments"]),
            "an argument outside the schema": "\n".join(call(c["name"], **c["arguments"], extra_arg="x") for c in s["ref_calls"]),
        }
        for kind, text in padded.items():
            for g in (gt2(s["id"]), gt2(s["id"], rules=[])):
                assert G.compute_score("tool_call", text, g) < 1.0, (s["id"], kind)
    # several values stuffed into an argument: a list, a comma list, "or"; free text is one value
    w1 = gt2("w1")
    for city in (["Paris", "London"], "Paris, London", "Paris or Lyon", "Paris; Lyon; Nice"):
        g = G.grade(call("get_weather", city=city, unit="celsius"), w1)
        assert g["passed"] == [True, True, True] and g["score"] < 1.0, city
    assert G.compute_score("tool_call", call("get_weather", city="Paris, London", unit="celsius"), w1) == (1 + 0.5 + 0.5) / 2.5
    assert G.compute_score("tool_call", call("get_weather", city="Paris", unit="celsius"), w1) == 1.0
    e1 = gt2("e1")
    att = ["alice@example.com", "bob@example.com"]
    base = dict(title="Budget review", date="2026-11-03", time="14:30")
    assert G.compute_score("tool_call", call("create_event", **base, attendees=att), e1) == 1.0
    assert G.compute_score("tool_call", call("create_event", **base, attendees=att + ["carol@example.com", "dan@example.com"]), e1) < 1.0
    s1 = gt2("s1")   # the body is free text: commas in it are not several values
    body = "Hi Carol, the draft is ready, it is in the shared folder, thanks."
    assert G.compute_score("tool_call", call("send_email", to="carol@example.com", subject="Draft ready", body=body), s1) == 1.0
    assert G.compute_score("tool_call", call("send_email", to="carol@example.com, eve@example.com", subject="Draft ready",
                                             body=body), s1) < 1.0


def test_v2_jaccard_repeat_schema_penalty():
    w1, w2 = gt2("w1"), gt2("w2")
    one = call("get_weather", city="Paris", unit="celsius")
    g = G.grade(one + "\n" + call("calculate", expression="1"), w1)
    assert g["jaccard"] == 0.5 and g["repeat"] == 1.0 and g["score"] == 0.5
    g = G.grade(one + "\n" + one, w1)
    assert g["jaccard"] == 1.0 and g["repeat"] == 0.5 and g["score"] == 0.5
    cities = [call("get_weather", city=c) for c in ("Tokyo", "Osaka", "Kyoto", "Nagoya", "Sapporo")]
    assert G.grade("\n".join(cities[:2]), w2)["score"] == 1.0
    assert abs(G.grade("\n".join(cities), w2)["score"] - 0.4) < 1e-9          # 2 of 5 calls within the reference count
    assert G.grade(cities[0], w2)["score"] == 0.5                            # a missing call costs only its rule
    g = G.grade(call("get_weather", city="Paris", unit="celsius", country="FR", lang="fr"), w1)
    assert g["unknown_args"] == 2 and g["score"] == 0.25                     # 0.5 per argument outside the schema
    assert G.grade(call("get_weather", city="Paris", unit="celsius", country="FR"), gt2("w1", schemas={}))["score"] == 1.0
    # without reference calls the reference names come from the positive rules: distinct equals values = distinct calls
    rules = [{"type": "tool_called", "name": "update_task"},
             {"type": "tool_arg_equals", "name": "update_task", "arg": "priority", "value": "high"},
             {"type": "tool_arg_equals", "name": "update_task", "arg": "priority", "value": "low"},
             {"type": "tool_arg_excludes", "name": "delete_task", "arg": "task_id", "value": "t1"}]
    assert G.ref_names(rules, []) == {"update_task": 2}
    gt = json.dumps({"rules": rules})
    two = call("update_task", task_id="t1", priority="High") + "\n" + call("update_task", task_id="t2", priority=" low")
    assert G.compute_score("tool_call", two, gt) == 1.0
    assert G.compute_score("tool_call", call("update_task", task_id="t1", priority="high"), gt) == 0.75
    assert G.compute_score("tool_call", two + "\n" + call("update_task", task_id="t3", priority="high"), gt) < 1.0


def test_v2_equals_rule_and_normalisation():
    calls = [("f", {"a": " Paris ", "n": "3", "m": 3.0, "l": ["x", "y"], "b": True})]
    eq = lambda arg, v: G.rule_credit({"type": "tool_arg_equals", "name": "f", "arg": arg, "value": v}, calls)
    assert eq("a", "paris") == 1.0 and eq("a", "Paris, France") == 0.0 and eq("a", "Par") == 0.0
    assert eq("n", 3) == 1.0 and eq("m", "3") == 1.0 and eq("n", "3.0") == 1.0 and eq("n", 4) == 0.0
    assert eq("l", ["X", "y"]) == 1.0 and eq("l", "x") == 0.0                 # a list equals only a list
    assert eq("b", True) == 1.0 and eq("b", "true") == 0.0 and eq("zz", None) == 0.0
    try:
        G.rule_credit({"type": "tool_arg_regex", "name": "f", "arg": "a", "value": "P.*"}, calls)
    except ValueError as e:
        assert "tool_arg_regex" in str(e)
    else:
        raise AssertionError("an unknown rule type was graded")
    assert G.compute_score("tool_call", call("f", a="x"), json.dumps({"rules": [{"type": "tool_arg_regex", "name": "f", "arg": "a",
                                                                                  "value": "x"}]})) == 0.0


def test_v2_ast_accept_optional_and_matching():
    # BFCL's possible answers: every acceptable value of an argument; "" in BFCL's list = the argument may be left out
    ref = [{"name": "get_weather", "arguments": {"city": "San Francisco"}, "accept": {"city": ["San Francisco", "SF", "San Francisco, CA"],
                                                                                       "unit": ["fahrenheit", "f"]},
            "optional": ["unit"]}]
    sch = {"get_weather": ["city", "unit"]}
    ok = lambda *cs: G.ast_match(list(cs), ref, sch)
    assert ok(("get_weather", {"city": "sf"})) and ok(("get_weather", {"city": "San Francisco, CA", "unit": "F"}))
    assert not ok(("get_weather", {"city": "Los Angeles"})) and not ok(("get_weather", {"city": "SF", "unit": "celsius"}))
    assert not ok(("get_weather", {"city": "SF, LA"})) and not ok(("get_weather", {"city": ["SF", "LA"]}))   # stuffing
    assert not ok(("get_weather", {"city": "SF", "country": "US"}))                                          # outside the schema
    assert not ok(("get_weather", {"city": "SF"}), ("get_weather", {"city": "SF"}))                           # an extra call
    gt = json.dumps({"rules": [], "ref_calls": ref, "schemas": sch})
    assert G.compute_score("tool_call", call("get_weather", city="SF", unit="f"), gt) == 1.0
    assert G.compute_score("tool_call", call("get_weather", city="SF", unit="kelvin"), gt) == 0.0
    # matching in any order, with backtracking: a greedy first pick would take the wider reference for the first call
    two = [{"name": "f", "arguments": {"x": 1}, "accept": {"x": [1, 2]}}, {"name": "f", "arguments": {"x": 1}}]
    assert G.ast_match([("f", {"x": 1}), ("f", {"x": 2})], two) and G.ast_match([("f", {"x": 2}), ("f", {"x": "1"})], two)
    assert not G.ast_match([("f", {"x": 2}), ("f", {"x": 2})], two)
    assert G.compute_score("tool_call", call("w..0LUDsu", qdBYgo=20),                                       # ToolRL quotes numbers
                           json.dumps({"ref_calls": [{"name": "w..0LUDsu", "arguments": {"qdBYgo": "20"}}]})) == 1.0


def test_v2_validate_refuses_malformed_ground_truth():
    good = {"rules": [{"type": "tool_called", "name": "f"}], "ref_calls": [], "forbidden": []}
    G.validate(good, ["f"])
    G.validate({"rules": [], "ref_calls": [{"name": "f", "arguments": {"x": 1}, "accept": {"x": [1, 2]}, "optional": []}]}, ["f"])
    bad = {
        "unknown rule type": {"rules": [{"type": "tool_arg_regex", "name": "f", "arg": "a", "value": "x"}]},
        "no arg": {"rules": [{"type": "tool_arg_contains", "name": "f", "value": "x"}]},
        "no value": {"rules": [{"type": "tool_arg_equals", "name": "f", "arg": "a"}]},
        "no name": {"rules": [{"type": "tool_called"}]},
        "non-numeric near": {"rules": [{"type": "tool_arg_number_near", "name": "f", "arg": "a", "value": "abc"}]},
        "negative weight": {"rules": [{"type": "tool_called", "name": "f", "weight": -1}]},
        "zero weights": {"rules": [{"type": "tool_called", "name": "f", "weight": 0}]},
        "only a negative rule": {"rules": [{"type": "tool_arg_excludes", "name": "f", "arg": "a", "value": "x"}]},
        "nothing to grade": {"rules": [], "ref_calls": []},
        "accept not a list": {"ref_calls": [{"name": "f", "arguments": {}, "accept": {"x": 1}}]},
        "a tool not offered": {"rules": [{"type": "tool_called", "name": "g"}]},
        "a forbidden reference": {"ref_calls": [{"name": "f", "arguments": {}}], "forbidden": ["f"]},
        "a forbidden rule": {"rules": [{"type": "tool_called", "name": "f"}], "forbidden": ["f"]},
        "a rule off the references": {"rules": [{"type": "tool_called", "name": "g"}], "ref_calls": [{"name": "f", "arguments": {}}]},
        "rules not a list": {"rules": {"type": "tool_called", "name": "f"}},
    }
    for why, g in bad.items():
        try:
            G.validate(g, ["f", "g"] if why == "a rule off the references" else ["f"], "sample z")
        except ValueError as e:
            assert str(e).startswith("sample z: "), why
        else:
            raise AssertionError(f"accepted: {why}")


def test_v2_prep_checks_samples_and_keeps_both_ground_truths():
    tok = _tok()
    s = dict(SAMPLES[0])                                          # rules and ref_calls: both survive, with the schemas
    g = json.loads(P.build_row(0, s, tok, tok)["reward_model"]["ground_truth"])
    assert g["rules"] == s["rules"] and g["ref_calls"] == s["ref_calls"] and g["schemas"] == {"get_weather": ["city", "unit"],
                                                                                              "calculate": ["expression"]}
    for why, change in {"unknown rule type": {"rules": [{"type": "tool_arg_regex", "name": "get_weather", "arg": "city", "value": "P"}]},
                        "duplicate tool names": {"tools": s["tools"] + s["tools"][:1]},
                        "no query": {"query": " "},
                        "a rule on a tool not offered": {"rules": [{"type": "tool_called", "name": "book_table"}]}}.items():
        try:
            P.build_row(3, dict(s, **change), tok, tok)
        except ValueError as e:
            assert f"sample {s['id']}" in str(e), why
        else:
            raise AssertionError(f"prep accepted: {why}")
    loose = dict(s, tools=[{"type": "function", "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object"}}}],
                 rules=[], ref_calls=[{"name": "get_weather", "arguments": {"city": "Paris", "anything": 1}}])
    assert json.loads(P.build_row(0, loose, tok, tok)["reward_model"]["ground_truth"])["schemas"] == {}   # unknown parameters


class JinjaTok:   # apply_chat_template as transformers renders it (jinja2, tojson without HTML escaping)
    MINI = ("{%- if tools %}{{- '<|im_start|>system\\n' }}{%- if messages[0].role == 'system' %}{{- messages[0].content + '\\n\\n' }}{%- endif %}"
            "{{- '# Tools\\n<tools>' }}{%- for tool in tools %}{{- '\\n' }}{{- tool | tojson }}{%- endfor %}{{- '\\n</tools><|im_end|>\\n' }}"
            "{%- else %}{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}{%- endif %}{%- endif %}"
            "{%- for m in messages %}{%- if m.role != 'system' %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- if enable_thinking is defined and enable_thinking is false %}"
            "{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}")

    def __init__(self, template=None):
        import jinja2
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        self.t = env.from_string(template or self.MINI)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, tools=None, **kw):
        return self.t.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kw)


def _tok():
    p = os.environ.get("QWEN_TEMPLATE") or next(iter(sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3-*/snapshots/*/tokenizer_config.json")))), None)
    return JinjaTok(json.load(open(p))["chat_template"] if p else None)


def test_prep_rows_render_like_native_tools():
    tok = _tok()
    for i, s in enumerate(SAMPLES):
        r = P.build_row(i, s, tok, tok)
        sysmsg = r["prompt"][0]["content"]
        assert r["prompt"][0]["role"] == "system" and "<tools>" in sysmsg and s["tools"][0]["function"]["name"] in sysmsg
        assert P.render(tok, r["prompt"]) == P.render(tok, [{"role": "user", "content": s["query"]}], s["tools"])
        g = json.loads(r["reward_model"]["ground_truth"])
        assert g["ref_calls"] == s["ref_calls"] and json.loads(r["extra_info"]["tools"]) == s["tools"]
        assert r["data_source"] == "tool_call"


def test_prep_refuses_a_template_mismatch():
    tok = _tok()

    class Drift(JinjaTok):   # a teacher whose template renders the system message differently
        def apply_chat_template(self, messages, **kw):
            out = super().apply_chat_template(messages, **kw)
            return out.replace("<|im_start|>system\n", "<|im_start|>system\n ", 1) if kw.get("tools") is None else out
    drift = Drift()
    drift.t = tok.t
    try:
        P.build_row(0, SAMPLES[0], tok, drift)
    except AssertionError as e:
        assert "teacher" in str(e)
    else:
        raise AssertionError("a template mismatch was not refused")


def test_eval_budget_curve():   # tool_eval.evaluate: a call that only completes after 6 "tokens" counts from budget 6 on
    import tool_eval as E
    words = ("I will check . " + call("get_weather", city="Paris", unit="celsius")).split(" ")
    rec = [{"id": "w1", "ground_truth": gt("w1"), "ids": words, "schemas": {"get_weather": ["city", "unit"]}},
           {"id": "w1", "ground_truth": gt("w1"), "ids": "no tool here".split(" "), "schemas": None}]
    cut = lambda ids, b: " ".join(ids[:b])
    c = E.evaluate(rec, [2, len(words)], cut)
    assert c["2"]["score"] == 0.0 and c["2"]["legal"] == 0.0 and c["2"]["mean_tokens"] == 2.0
    full = c[str(len(words))]
    assert full["score"] == 50.0 and full["legal"] == 50.0 and full["ast"] == 50.0 and full["forbidden"] == 0.0 and full["n"] == 2
    assert full["n_ast"] == 2
    rules_only = json.dumps({"rules": [{"type": "tool_called", "name": "get_weather"}]})   # [v2.5.9z19] no reference calls: not in ast
    c = E.evaluate(rec + [{"id": "r", "ground_truth": rules_only, "ids": words, "schemas": None}], [len(words)], cut)[str(len(words))]
    assert c["n_ast"] == 2 and c["ast"] == 50.0 and c["n"] == 3 and abs(c["score"] - 200 / 3) < 0.01
    assert E.evaluate([rec[1]], [2], cut)["2"]["ast"] == 0.0
    assert E.evaluate([{"id": "r", "ground_truth": rules_only, "ids": words, "schemas": None}], [2], cut)["2"]["ast"] is None
