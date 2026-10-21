"""Single-turn tool-call grading (runbook [v2.5.9b], v2 [v2.5.9z19]; tool-call OPD, research_program §1.7, CoworkEnv-style rule
checks).

The model's calls are Qwen3's format: one or more blocks
    <tool_call>
    {"name": "<function>", "arguments": {...}}
    </tool_call>
(text inside a <think>...</think> block is ignored, an unclosed one to the end of the text). The ground truth is a JSON object
    {"rules": [{"type": ..., "name": ..., "arg": ..., "value": ..., "tol": ..., "weight": w}, ...],
     "forbidden": ["<tool>", ...],
     "ref_calls": [{"name": ..., "arguments": {...}, "accept": {arg: [values]}, "optional": [arg, ...]}, ...],
     "schemas": {"<tool>": [parameter names]}}
Rule types (RULE_TYPES; anything else is refused by validate): tool_called (some call to `name`), tool_arg_contains (a call
to `name` whose argument `arg` contains `value`, case-insensitive; lists: any element), tool_arg_equals [v2] (a call whose
`arg` equals `value` after normalisation: strings stripped and case-folded, numbers and numeric strings as numbers),
tool_arg_excludes (no call to `name` has `arg` containing `value`), tool_arg_number_near (a call whose `arg` is a number
within `tol`, default 1e-6, of `value`). `arg` may be a dotted path into nested objects. A reference call's `arguments` is
one correct call; `accept` lists every acceptable value of an argument (BFCL's possible answers), `optional` the arguments
that may be left out. `schemas` (written by prep_tools_parquet.py; tools whose parameters are unknown are absent) lists each
tool's parameters.

Score (v2):
  - 0 when no call parses or any call names a forbidden tool (the safety gate);
  - with rules: rules score x Jaccard x repeat x 0.5 ** (arguments outside their tool's schema).
    The rules score is the weighted mean of the rules' credits. A contains rule's credit is the share of the values in the
    matched argument that the sample asks for (contains / equals values and reference values of that argument): a list
    holds its elements; a string holds the parts between , ; | newlines or the word "or" when each part is a single token
    once the asked-for values are masked as one token each ("Paris, London" for "Paris" earns 1/2, "Paris, France" too;
    free text with spaces is one value, so values joined by spaces, as in "Paris Lyon", are not caught: the guard is
    for list-style stuffing, prose such as titles and bodies keeps full credit). Other rules credit 1 or 0. Jaccard is the Jaccard index of the set of called tool
    names and the set of reference names (extra tools lower it). repeat is the share of the calls to reference tools that
    are not beyond the reference count of that tool (padding with repeated calls lowers it; a missing call costs only its
    rules). The reference names and counts: the reference calls', or without them each tool a positive rule names, as many
    times as the most distinct equals / number_near values one of its arguments must take.
  - without rules (ToolRL, BFCL): ast_match 1 / 0, where an argument outside the schema is a mismatch.
compute_score is verl's custom reward interface. ast_match compares the calls with ref_calls as BFCL's AST checker does
(names, every reference argument present and equal to an acceptable value after normalisation unless optional, no
argument outside the schema, no missing or extra call; calls matched in any order).
"""
import json
import math
import re
from collections import Counter

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)   # an unclosed block (a thinking model cut at its budget) to the end
RULE_TYPES = ("tool_called", "tool_arg_contains", "tool_arg_equals", "tool_arg_excludes", "tool_arg_number_near")
ARG_RULES = RULE_TYPES[1:]
POSITIVE = ("tool_called", "tool_arg_contains", "tool_arg_equals", "tool_arg_number_near")
SCHEMA_PENALTY = 0.5        # per argument outside its tool's schema (rules samples; ast samples score 0)
SEP_RE = re.compile(r"[,;|\n]|\bor\b", re.I)
NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
MASK = "\0"


def parse_calls(text):
    """(calls, n_blocks): calls = [(name, arguments dict)] of the blocks that parse; n_blocks counts every block."""
    text = THINK_RE.sub("", text or "")
    blocks = CALL_RE.findall(text)
    calls = []
    for b in blocks:
        try:
            obj = json.loads(b)
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", {})
            if isinstance(args, str):   # some models emit the arguments as a JSON string
                try:
                    args = json.loads(args)
                except ValueError:
                    continue
            if isinstance(args, dict):
                calls.append((obj["name"], args))
    return calls, len(blocks)


def _get(args, path):
    cur = args
    for k in str(path).split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    return cur, True


def _contains(v, needle):
    if isinstance(v, (list, tuple)):
        return any(_contains(x, needle) for x in v)
    if isinstance(v, dict):
        return any(_contains(x, needle) for x in v.values())
    return str(needle).casefold() in str(v).casefold()


def _near(v, target, tol):
    try:
        return abs(float(v) - float(target)) <= float(tol)
    except (TypeError, ValueError):
        return False


def _norm(v):
    if isinstance(v, str):
        s = v.strip()
        if NUM_RE.match(s):   # [v2] "3" and 3 are the same argument (ToolRL's references often quote numbers)
            return round(float(s), 6)
        return s.casefold()
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6) if math.isfinite(v) else str(v)
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _eq(a, b):
    return _norm(a) == _norm(b)


def _flat(v):
    if isinstance(v, (list, tuple)):
        return [y for x in v for y in _flat(x)]
    if isinstance(v, dict):
        return [y for x in v.values() for y in _flat(x)]
    return [v]


def _parts(v, wanted):
    """The values an argument holds, each with the asked-for values masked (see the module docstring)."""
    out = []
    for x in _flat(v):
        s = x if isinstance(x, str) else json.dumps(x)
        for w in sorted({str(w) for w in wanted if str(w).strip()}, key=len, reverse=True):
            s = re.sub(re.escape(w), MASK, s, flags=re.I)
        ps = [p.strip() for p in SEP_RE.split(s) if p.strip()]
        out += ps if len(ps) > 1 and all(len(p.split()) == 1 for p in ps) else [s]
    return out


def _precision(v, wanted):
    ps = _parts(v, wanted)
    return sum(MASK in p for p in ps) / len(ps) if ps else 0.0


def _wanted(name, arg, rules, refs):
    """The values the sample asks for in `arg` of `name`: contains / equals values and the reference calls' values."""
    w = [r["value"] for r in rules if r.get("name") == name and r.get("arg") == arg
         and r.get("type") in ("tool_arg_contains", "tool_arg_equals")]
    for c in refs:
        if c.get("name") == name:
            v, ok = _get(c.get("arguments", {}), arg)
            w += _flat(v) if ok else []
            w += [y for x in c.get("accept", {}).get(arg, []) for y in _flat(x)]
    return [x for x in w if isinstance(x, (str, int, float)) and not isinstance(x, bool)]


def rule_credit(rule, calls, rules=(), refs=()):
    """The rule's credit in [0, 1] (see the module docstring); raises ValueError on an unknown rule type."""
    t, name = rule.get("type"), rule.get("name")
    mine = [a for n, a in calls if n == name]
    if t == "tool_called":
        return float(bool(mine))
    if t not in ARG_RULES:
        raise ValueError(f"unknown rule type {t!r}")
    vals = [v for v, ok in (_get(a, rule["arg"]) for a in mine) if ok]
    if t == "tool_arg_excludes":
        return float(not any(_contains(v, rule["value"]) for v in vals))
    if t == "tool_arg_equals":
        return float(any(_eq(v, rule["value"]) for v in vals))
    if t == "tool_arg_number_near":
        return float(any(_near(v, rule["value"], rule.get("tol", 1e-6)) for v in vals))
    wanted = _wanted(name, rule["arg"], rules, refs)
    return max([_precision(v, wanted) for v in vals if _contains(v, rule["value"])], default=0.0)


def ref_names(rules, refs):
    """Counter of the tool names a correct answer calls (see the module docstring)."""
    if refs:
        return Counter(c["name"] for c in refs)
    need = Counter()
    for r in rules:
        if r.get("type") in POSITIVE:
            need[r["name"]] = max(need[r["name"]], 1)
    for name in need:
        per_arg = {}
        for r in rules:
            if r.get("name") == name and r.get("type") in ("tool_arg_equals", "tool_arg_number_near"):
                per_arg.setdefault(r["arg"], set()).add(json.dumps(_norm(r["value"]), sort_keys=True))
        need[name] = max([need[name]] + [len(s) for s in per_arg.values()])
    return need


def jaccard(names, ref):
    a, b = set(names), set(ref)
    return len(a & b) / len(a | b) if a | b else 1.0


def repeat_share(names, ref):
    """The share of the calls to reference tools that are not beyond that tool's reference count."""
    a, b = Counter(names), Counter(ref)
    mine = sum(a[n] for n in b)
    return sum((a & b).values()) / mine if mine else 1.0


def unknown_args(calls, schemas):
    """Arguments outside their tool's schema (tools without a schema are not counted)."""
    if not schemas:
        return 0
    return sum(len(set(args) - set(schemas[n])) for n, args in calls if n in schemas)


def schemas(tools):
    """{name: [parameter names]} of OpenAI-style tool schemas; a tool whose parameters have no "properties" is left out
    (its parameters are unknown)."""
    out = {}
    for t in tools or []:
        f = t.get("function", t)
        p = f.get("parameters")
        if isinstance(p, dict) and isinstance(p.get("properties"), dict):
            out[f["name"]] = list(p["properties"])
    return out


def grade(text, gt):
    """dict: score, legal (at least one call parsed and no malformed block), n_calls, forbidden, passed and credit per rule,
    rules_score, jaccard, repeat, unknown_args, mode (rules / ast)."""
    gt = json.loads(gt) if isinstance(gt, str) else gt
    calls, n_blocks = parse_calls(text)
    forbidden = sorted({n for n, _ in calls} & set(gt.get("forbidden", [])))
    rules, refs = gt.get("rules", []), gt.get("ref_calls", [])
    credit = [rule_credit(r, calls, rules, refs) for r in rules]
    out = {"legal": bool(calls) and len(calls) == n_blocks, "n_calls": len(calls), "forbidden": forbidden,
           "passed": [c > 0 for c in credit], "credit": credit, "rules_score": None, "jaccard": None, "repeat": None,
           "unknown_args": unknown_args(calls, gt.get("schemas")), "mode": "rules" if rules else "ast"}
    score = 0.0
    if rules:
        w = [float(r.get("weight", 1.0)) for r in rules]
        out["rules_score"] = sum(wi * c for wi, c in zip(w, credit)) / sum(w) if sum(w) > 0 else 0.0
        names, ref = [n for n, _ in calls], ref_names(rules, refs)
        out["jaccard"], out["repeat"] = jaccard(names, ref), repeat_share(names, ref)
        score = out["rules_score"] * out["jaccard"] * out["repeat"] * SCHEMA_PENALTY ** out["unknown_args"]
    elif refs:
        score = float(ast_match(calls, refs, gt.get("schemas")))
    if not calls or forbidden:
        score = 0.0
    out["score"] = score
    return out


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl custom reward: the score in [0, 1]."""
    try:
        return float(grade(solution_str, ground_truth)["score"])
    except Exception:
        return 0.0


def _args_match(args, ref):
    a, acc, opt = ref.get("arguments", {}), ref.get("accept", {}), set(ref.get("optional", []))
    for k in set(a) | set(acc):
        if k not in args:
            if k in opt:
                continue
            return False
        ok = ([a[k]] if k in a else []) + list(acc.get(k, []))
        if not any(_eq(args[k], v) for v in ok):
            return False
    return True


def ast_match(calls, ref_calls, schemas=None):
    """BFCL-style: True when the calls equal the reference calls up to order: same names, every reference argument present
    and equal to an acceptable value after normalisation (unless optional), and no argument outside the function's schema
    (when schemas are given: {name: [params]})."""
    refs = [r if isinstance(r, dict) else {"name": r[0], "arguments": r[1]} for r in ref_calls]
    if len(calls) != len(refs):
        return False
    for name, args in calls:
        if schemas is not None and name in schemas and set(args) - set(schemas[name]):
            return False
    fits = [[j for j, r in enumerate(refs) if r["name"] == n and _args_match(a, r)] for n, a in calls]
    used = set()

    def assign(i):   # a matching of calls to reference calls (backtracking: an early greedy pick may block a later call)
        if i == len(fits):
            return True
        for j in fits[i]:
            if j not in used:
                used.add(j)
                if assign(i + 1):
                    return True
                used.discard(j)
        return False
    return assign(0)


def validate(gt, tool_names=None, where=""):
    """Raises ValueError when the ground truth is malformed: an unknown rule type, a rule without its fields, a negative
    weight, malformed forbidden / ref_calls / accept / optional, no positive rule and no reference call, a forbidden tool
    that a reference call or positive rule asks for, a positive rule on a tool the reference calls do not call, or (with
    the sample's tool names) a rule or reference call naming a tool the prompt does not offer."""
    def bad(msg):
        raise ValueError(f"{where}: {msg}" if where else msg)
    rules, refs, forb = gt.get("rules", []), gt.get("ref_calls", []), gt.get("forbidden", [])
    if not isinstance(rules, list) or not isinstance(refs, list) or not isinstance(forb, list):
        bad("rules, ref_calls and forbidden must be lists")
    for r in rules:
        if not isinstance(r, dict) or r.get("type") not in RULE_TYPES:
            bad(f"unknown rule type {r.get('type') if isinstance(r, dict) else r!r} (known: {', '.join(RULE_TYPES)})")
        if not isinstance(r.get("name"), str):
            bad(f"{r['type']} rule without a tool name")
        if r["type"] in ARG_RULES and (not isinstance(r.get("arg"), str) or "value" not in r):
            bad(f"{r['type']} rule on {r['name']} without arg / value")
        if r["type"] == "tool_arg_number_near" and not all(_near(x, 0, math.inf) for x in (r["value"], r.get("tol", 0))):
            bad(f"tool_arg_number_near on {r['name']}.{r['arg']} with a non-numeric value or tol")
        if not isinstance(r.get("weight", 1.0), (int, float)) or isinstance(r.get("weight"), bool) or r.get("weight", 1.0) < 0:
            bad(f"rule on {r['name']} with a weight that is not a number >= 0")
    if not all(isinstance(x, str) for x in forb):
        bad("forbidden must list tool names")
    for c in refs:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str) or not isinstance(c.get("arguments", {}), dict):
            bad(f"malformed reference call {c!r}")
        if not isinstance(c.get("accept", {}), dict) or not all(isinstance(v, list) for v in c.get("accept", {}).values()):
            bad(f"reference call {c['name']}: accept must map arguments to lists of values")
        if not isinstance(c.get("optional", []), list):
            bad(f"reference call {c['name']}: optional must be a list")
    if not refs and not any(r["type"] in POSITIVE for r in rules):
        bad("no reference call and no positive rule: nothing to grade")
    if rules and sum(float(r.get("weight", 1.0)) for r in rules) <= 0:
        bad("the rule weights sum to 0")
    if tool_names is not None:
        unknown = sorted(({r["name"] for r in rules if r["type"] in POSITIVE} | {c["name"] for c in refs}) - set(tool_names))
        if unknown:
            bad(f"rules or reference calls name tools the prompt does not offer: {unknown}")
    positive = {r["name"] for r in rules if r["type"] in POSITIVE}
    if set(forb) & (positive | {c["name"] for c in refs}):
        bad("a reference call or positive rule names a forbidden tool")
    if refs and positive - {c["name"] for c in refs}:
        bad(f"positive rules name tools the reference calls do not call: {sorted(positive - {c['name'] for c in refs})}")
