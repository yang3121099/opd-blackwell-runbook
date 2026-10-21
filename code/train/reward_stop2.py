"""[v2.5.9f] verl custom reward of E12b-P5-stop2 (T1; reviewer 07:4x UTC 10-21): a continuous stop reward.

    r = c * (1 + 0.5 * clip(1 - tail / STOP2_SCALE, 0, 1))

c = 1[the last \\boxed{} is correct] (reward_mathverify.compute_score, the file next to this one, so E12b-P5 and its variants
grade alike); tail = the student's tokens after the end of the FIRST closed \\boxed{...}; a response cut at the length limit
(within OPD_REWARD_CUT_MARGIN (2) tokens of OPD_REWARD_MAXLEN (4096)) counts tail = STOP2_SCALE (1024): no bonus. No box:
tail -1 and r = 0. Q-021's stop reward (a 0 / 0.5 bonus within 64 tokens) was never paid, so the groups had no variance
and no gradient; this one differs between any two correct rollouts whose tails differ. Returns a dict (score, acc, tail,
cut, bonus) so verl's rollout dumps carry the parts (code/analysis/e12_rollout_stats.py reports them per step). Tokens are
the student's: OPD_REWARD_TOKENIZER (run_e12.sh sets it to the start)."""
import importlib.util
import os

SCALE = float(os.environ.get("OPD_STOP2_SCALE", "1024"))
BONUS = 0.5
_STOP = None


def _stop():   # reward_stop.py's box parser, token counter and base grader (loaded by path, as verl loads this file)
    global _STOP
    if _STOP is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_stop.py")
        spec = importlib.util.spec_from_file_location("reward_stop_for_stop2", p)
        _STOP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_STOP)
    return _STOP


def parts(solution_str, ground_truth, data_source="math", extra_info=None):
    s = _stop()
    acc = 1.0 if s._base()(data_source, solution_str, ground_truth, extra_info) >= 1.0 else 0.0
    b = s.boxes(solution_str)
    if not b:
        return {"score": 0.0, "acc": acc, "tail": -1, "cut": 0, "bonus": 0.0}
    cut = int(s._ntok(solution_str) >= int(os.environ.get("OPD_REWARD_MAXLEN", "4096")) - int(os.environ.get("OPD_REWARD_CUT_MARGIN", "2")))
    tail = int(SCALE) if cut else s._ntok(solution_str[b[0][1]:])
    bonus = BONUS * min(1.0, max(0.0, 1.0 - tail / SCALE))
    return {"score": acc * (1.0 + bonus), "acc": acc, "tail": tail, "cut": cut, "bonus": bonus if acc else 0.0}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return parts(solution_str, ground_truth, data_source, extra_info)
