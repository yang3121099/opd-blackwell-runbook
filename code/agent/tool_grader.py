"""Single-turn tool-call grading (runbook [v2.5.9b]; tool-call OPD, research_program §1.7, CoworkEnv-style rule checks).

The model's calls are Qwen3's format: one or more blocks
    <tool_call>
    {"name": "<function>", "arguments": {...}}
    </tool_call>
(text inside a <think>...</think> block is ignored, an unclosed one to the end of the text). The ground truth is a JSON object
    {"rules": [{"type": ..., "name": ..., "arg": ..., "value": ..., "tol": ..., "weight": w}, ...],
     "forbidden": ["<tool>", ...], "ref_calls": [{"name": ..., "arguments": {...}}, ...]}
with four parameter-level rule types: tool_called (some call to `name`), tool_arg_contains (a call to `name` whose
argument `arg` contains `value`, case-insensitive; lists: any element), tool_arg_excludes (no call to `name` has `arg`
containing `value`), tool_arg_number_near (a call to `name` whose `arg` is a number within `tol`, default 1e-6, of
`value`). `arg` may be a dotted path into nested objects. Score = weighted share of rules passed; 0 when no call parses,
and 0 when any call names a forbidden tool (the safety gate). compute_score is verl's custom reward interface.
ast_match compares the calls with ref_calls as BFCL's AST checker does (names, every reference argument present and equal
after normalisation, no unknown extra argument, no missing or extra call).
"""
import json
import re

CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)   # an unclosed block (a thinking model cut at its budget) to the end


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


def rule_passes(rule, calls):
    t, name = rule.get("type"), rule.get("name")
    mine = [a for n, a in calls if n == name]
    if t == "tool_called":
        return bool(mine)
    if t == "tool_arg_contains":
        return any(ok and _contains(v, rule["value"]) for v, ok in (_get(a, rule["arg"]) for a in mine))
    if t == "tool_arg_excludes":
        return not any(ok and _contains(v, rule["value"]) for v, ok in (_get(a, rule["arg"]) for a in mine))
    if t == "tool_arg_number_near":
        return any(ok and _near(v, rule["value"], rule.get("tol", 1e-6)) for v, ok in (_get(a, rule["arg"]) for a in mine))
    raise ValueError(f"unknown rule type {t!r}")


def grade(text, gt):
    """dict: score, legal (at least one call parsed and no malformed block), n_calls, forbidden, passed per rule."""
    gt = json.loads(gt) if isinstance(gt, str) else gt
    calls, n_blocks = parse_calls(text)
    forbidden = sorted({n for n, _ in calls} & set(gt.get("forbidden", [])))
    rules = gt.get("rules", [])
    passed = [rule_passes(r, calls) for r in rules]
    w = [float(r.get("weight", 1.0)) for r in rules]
    score = 0.0
    if calls and not forbidden and sum(w) > 0:
        score = sum(wi for wi, p in zip(w, passed) if p) / sum(w)
    return {"score": score, "legal": bool(calls) and len(calls) == n_blocks, "n_calls": len(calls), "forbidden": forbidden,
            "passed": passed}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl custom reward: the rule score in [0, 1]."""
    try:
        return float(grade(solution_str, ground_truth)["score"])
    except Exception:
        return 0.0


def _norm(v):
    if isinstance(v, str):
        return v.strip().casefold()
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def ast_match(calls, ref_calls, schemas=None):
    """BFCL-style: True when the calls equal the reference calls up to order: same names, every reference argument present
    and equal after normalisation, and no argument outside the function's schema (when schemas are given: {name: [params]})."""
    pending = [(r["name"], r.get("arguments", {})) for r in ref_calls]
    if len(calls) != len(pending):
        return False
    for name, args in calls:
        if schemas is not None and name in schemas and set(args) - set(schemas[name]):
            return False
        hit = next((i for i, (rn, ra) in enumerate(pending)
                    if rn == name and all(k in args and _norm(args[k]) == _norm(v) for k, v in ra.items())), None)
        if hit is None:
            return False
        pending.pop(hit)
    return not pending
