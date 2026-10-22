"""A student whose vocabulary is the teacher's except seven special ids (runbook [v2.5.9z46]; Q-057 E16-X5 / E16-X5e, the reviewer's
design 10-22 14:4x Beijing and ruling 10:0x UTC): Qwen2.5-Math-1.5B-Instruct taught by DeepSeek-R1-Distill-Qwen-1.5B. TOKCHECK-X5:
both 151,936 rows, ids 0-151642 the same tokens, 151643-151649 different (the student's 151645 <|im_end|> is the teacher's
<｜Assistant｜>, its 151643 <|endoftext|> the teacher's <｜end▁of▁sentence｜>, the teacher's EOS). G-OPD appends the student's response
ids unchanged to the teacher's prompt (ref_input_utils.prepare_ref_model_inputs), so two hooks (x5_patch.py), no-ops unless OPD_X5=1:
  teacher_prompt(s): the teacher's prompt ends in exactly one '<think>\\n' (R1-Distill's template ends so already, and the P5 patch adds
    another); OPD_X5_EMPTY_THINK=1 (E16-X5e): it ends in '<think>\\n\\n</think>\\n\\n' instead (an empty thought: the teacher reads the
    student's response as the answer).
  map_ids(batch, response_length): per row of the teacher's copy of the response (ref_input_ids' last response_length columns), the
    student's stop id at the terminal position (the last token with response_mask 1: 151645 or 151643) becomes the teacher's EOS 151643,
    so the teacher's log-prob there is its stop probability; any other of the seven ids (151644, 151646-151649 anywhere, 151643 / 151645
    before the end) leaves the loss (response_mask 0 there; the teacher's attention mask is built before, so it keeps no hole).
One log line per call (OPD_X5_LOG): rows, terminated (a stop id at the end), term_151645, term_151643, mapped (= terminated), truncated
(no stop id at the end), masked per id (expected 0), the teacher prompt's tail, empty_think.
"""
import json
import os
import time

STUDENT_EOS, TEACHER_EOS = 151645, 151643
STOP_IDS = (151645, 151643)                                  # the student's two stop ids (its generation_config)
SPECIAL = (151643, 151644, 151645, 151646, 151647, 151648, 151649)
THINK, EMPTY = "<think>\n", "<think>\n\n</think>\n\n"
_tail = None
_calls = 0


def empty_think():
    return os.environ.get("OPD_X5_EMPTY_THINK") == "1"


def teacher_prompt(s, empty=None):
    """The teacher's prompt with one '<think>\\n' at its end, or ([E16-X5e]) an empty thought."""
    global _tail
    empty = empty_think() if empty is None else empty
    while s.endswith(THINK + THINK):
        s = s[:-len(THINK)]
    if not s.endswith(THINK):
        s = s + THINK
    if empty:
        s = s[:-len(THINK)] + EMPTY
    _tail = s[-40:]
    return s


def plan_row(ids, mask):
    """(terminal id or None, the position mapped to the teacher's EOS or None, positions leaving the loss) for one response row."""
    valid = [i for i, m in enumerate(mask) if m]
    if not valid:
        return None, None, []
    last = valid[-1]
    t = ids[last]
    mp = last if t in STOP_IDS else None
    return t, mp, [i for i in valid if ids[i] in SPECIAL and i != mp]


def apply_rows(rows_ids, rows_mask):
    """For lists of rows: (new ids rows, new mask rows, counts). Pure: the tests and map_ids share it."""
    out_ids, out_mask = [], []
    c = {"rows": len(rows_ids), "terminated": 0, "term_151645": 0, "term_151643": 0, "mapped": 0, "truncated": 0, "empty": 0,
         "masked": {}}
    for ids, mask in zip(rows_ids, rows_mask):
        ids, mask = list(ids), list(mask)
        t, mp, drop = plan_row(ids, mask)
        if t is None:
            c["empty"] += 1
        elif mp is None:
            c["truncated"] += 1
        else:
            c["terminated"] += 1; c["term_%d" % t] += 1; c["mapped"] += 1
            ids[mp] = TEACHER_EOS
        for i in drop:
            c["masked"][str(ids[i])] = c["masked"].get(str(ids[i]), 0) + 1
            mask[i] = 0
        out_ids.append(ids); out_mask.append(mask)
    return out_ids, out_mask, c


def map_ids(batch, response_length):
    """The hook before prepare_ref_model_inputs returns: the terminal stop id mapped in ref_input_ids, the other special ids out of
    response_mask; one log line."""
    global _calls
    import torch
    ids = batch.batch["ref_input_ids"]
    mask = batch.batch["response_mask"]
    resp = ids[:, -response_length:]
    new_ids, new_mask, c = apply_rows(resp.tolist(), mask.tolist())
    resp.copy_(torch.tensor(new_ids, dtype=resp.dtype, device=resp.device))
    if c["masked"]:
        batch.batch["response_mask"] = torch.tensor(new_mask, dtype=mask.dtype, device=mask.device)
    _calls += 1
    rec = dict(c, call=_calls, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), prompt_tail=_tail, empty_think=empty_think())
    log = os.environ.get("OPD_X5_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec
