"""The hooks of a student with the teacher's vocabulary but its own special ids (runbook [v2.5.9z46], Q-057 E16-X5 / -X5e) in G-OPD's
ref_input_utils.py, both no-ops unless OPD_X5=1:
1. in prepare_ref_model_inputs, right after the P5 patch's `if _think: ref_prompt_str = ref_prompt_str + '<think>\\n'` (apply_patches.py
   section 2): `ref_prompt_str = opd_x5.teacher_prompt(ref_prompt_str)` (one '<think>\\n' at the end; E16-X5e: an empty thought);
2. before prepare_ref_model_inputs returns: opd_x5.map_ids(batch, response_length) (the terminal stop id -> the teacher's EOS 151643,
   the other special ids out of response_mask; the call's log line).

run_arm.sh calls `python x5_patch.py` before E16-X5 / -X5e and their smoke (as route_patch.py): idempotent, the file verl imports,
written atomically; before the first change the file is kept once as ref_input_utils.py.orig next to it. Prints X5_HOOK_OK (already
there) or X5_HOOK_PATCHED. Usage: python x5_patch.py [<ref_input_utils.py>]
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9z46] X5: the teacher's vocabulary, the student's special ids"
ANCHOR = ("        if _think:\n"
          "            ref_prompt_str = ref_prompt_str + '<think>\\n'\n")
HOOK = (f"        if _os.environ.get('OPD_X5') == '1':  {MARK}\n"
        "            import opd_x5 as _x5\n"
        "            ref_prompt_str = _x5.teacher_prompt(ref_prompt_str)\n")
IDS = (f"    if __import__('os').environ.get('OPD_X5') == '1':  {MARK} (ids)\n"
       "        import opd_x5 as _x5\n"
       "        _x5.map_ids(batch, response_length)\n")


def patch_text(s):
    """(new text, changed): both hooks inside prepare_ref_model_inputs; unchanged when they are already there."""
    if MARK in s:
        return s, False
    i = s.index("def prepare_ref_model_inputs(\n"); j = s.find("\ndef ", i + 1); j = len(s) if j < 0 else j
    body = s[i:j]
    n = body.count(ANCHOR)
    if n != 1:
        raise AssertionError(f"X5 anchor found {n} times in prepare_ref_model_inputs (the P5 patch of apply_patches.py first)")
    k = body.rindex("    return batch\n")
    body = body[:k] + IDS + body[k:]
    body = body.replace(ANCHOR, ANCHOR + HOOK)
    return s[:i] + body + s[j:], True


def patch_file(path, keep_orig=True):
    s = open(path).read()
    new, changed = patch_text(s)
    if changed:
        if keep_orig and not os.path.exists(path + ".orig"):   # the rollback copy, kept once
            shutil.copy2(path, path + ".orig")
        tmp = path + ".opd_tmp"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    return changed


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else importlib.util.find_spec("verl.trainer.ppo.ref_input_utils").origin
    print("X5_HOOK_PATCHED" if patch_file(p) else "X5_HOOK_OK")
