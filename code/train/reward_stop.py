"""[v2.5.8c] verl custom reward of E12b-P5-stop (Q-021; reviewer ruling 2026-10-19 21:49 UTC):

    r = 1[the last \\boxed{} is correct]
      + 0.5 * 1[correct, every \\boxed{} has the same content, and the response ends within STOP_TAIL (64) student
                tokens after the end of the first \\boxed{...}, without being cut at the length limit]

No negative reward. Correctness is reward_mathverify.compute_score's (the same file, loaded by path, so E12b-P5 and
E12b-P5-stop grade alike). Box contents are compared without whitespace. Tokens are the student's: OPD_REWARD_TOKENIZER
(run_e12.sh sets it to the start). A response whose length is within OPD_REWARD_CUT_MARGIN (2) tokens of
OPD_REWARD_MAXLEN (data.max_response_length, 4096) counts as cut at the limit, not as ended."""
import importlib.util
import os
import re

STOP_TAIL = int(os.environ.get("OPD_STOP_TAIL", "64"))
BONUS = 0.5
_BASE = None
_TOK = None


def _base():
    global _BASE
    if _BASE is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_mathverify.py")
        spec = importlib.util.spec_from_file_location("reward_mathverify_for_stop", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _BASE = m.compute_score
    return _BASE


def _ntok(text):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(os.environ["OPD_REWARD_TOKENIZER"])
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


def boxes(s):
    """[(start, end, content)] of every closed \\boxed{...} in order, braces matched; an unclosed box ends the list."""
    out, i = [], 0
    while True:
        j = s.find("\\boxed{", i)
        if j < 0:
            return out
        k, depth = j + len("\\boxed{"), 1
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        if depth:
            return out
        out.append((j, k, re.sub(r"\s+", "", s[j + len("\\boxed{"):k - 1])))
        i = k


def stop_bonus(solution_str):
    """True when every box has the same content and the response ends within STOP_TAIL tokens after the first box."""
    b = boxes(solution_str)
    if not b or len({c for _, _, c in b}) != 1:
        return False
    if _ntok(solution_str) >= int(os.environ.get("OPD_REWARD_MAXLEN", "4096")) - int(os.environ.get("OPD_REWARD_CUT_MARGIN", "2")):
        return False
    return _ntok(solution_str[b[0][1]:]) <= STOP_TAIL


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    r = _base()(data_source, solution_str, ground_truth, extra_info)
    return r + BONUS if r >= 1.0 and stop_bonus(solution_str) else r
