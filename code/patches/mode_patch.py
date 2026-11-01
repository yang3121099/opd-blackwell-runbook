"""The teacher's view of the mode token (runbook [v2.5.9z108], Q-083 E10b-P5mode) in G-OPD's ref_input_utils.py: one line right after
route_patch.py's hook in prepare_ref_model_inputs,
    messages = _rt.teacher_view(messages)
so the teacher renders the prompt without the student's '[long] ' / '[short] ' token (opd_route.teacher_view: a new list, the batch's
rows untouched; a no-op unless OPD_ROUTE=prefix, and OPD_MODE_STRIP=0 keeps the token). Needs route_patch.py's hook first.

run_arm.sh calls `python mode_patch.py` after route_patch.py for E10b-P5mode and its smoke: idempotent, the file verl imports, the
patched text compiled before it is written, written atomically; before the first change the file is kept once as
ref_input_utils.py.orig next to it (route_patch.py's copy when it made one). Prints MODE_HOOK_OK (already there) or MODE_HOOK_PATCHED.
Usage: python mode_patch.py [<ref_input_utils.py>]
"""
import importlib.util
import os
import shutil
import sys

import route_patch as RP

MARK = "# [opd-variants patch v2] [v2.5.9z108] the teacher's view of the mode token"
ANCHOR = "            _think = _rt.think_for(messages)\n"
HOOK = f"            messages = _rt.teacher_view(messages)  {MARK}\n"


def patch_text(s):
    """(new text, changed): the line after route's hook; unchanged when it is already there."""
    if MARK in s:
        return s, False
    if RP.MARK not in s:
        raise AssertionError("route_patch.py's hook is not in the file (run route_patch.py first)")
    n = s.count(ANCHOR)
    if n != 1:
        raise AssertionError(f"mode anchor found {n} times (route_patch.py's think_for line)")
    new = s.replace(ANCHOR, ANCHOR + HOOK)
    compile(new, "ref_input_utils.py", "exec")
    return new, True


def patch_file(path, keep_orig=True):
    s = open(path).read()
    new, changed = patch_text(s)
    if changed:
        if keep_orig and not os.path.exists(path + ".orig"):   # the rollback copy, kept once
            shutil.copy2(path, path + ".orig")
        tmp = f"{path}.opd_tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    return changed


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else importlib.util.find_spec("verl.trainer.ppo.ref_input_utils").origin
    print("MODE_HOOK_PATCHED" if patch_file(p) else "MODE_HOOK_OK")
