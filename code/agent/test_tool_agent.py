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
