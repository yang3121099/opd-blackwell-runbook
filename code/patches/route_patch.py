"""The hooks of the per-prompt teacher template (runbook [v2.5.9x], Q-041 / Q-042) in G-OPD's ref_input_utils.py, both no-ops
unless OPD_ROUTE is set:
1. in prepare_ref_model_inputs, right after the P5 patch's per-sample `_think = ... OPD_REF_THINK ...` line (apply_patches.py
   section 2) and before the template is chosen from it: `_think = opd_route.think_for(messages)`;
2. before prepare_ref_model_inputs returns: opd_route.flush(batch_size) writes the call's counts and decisions.

run_arm.sh calls `python route_patch.py` before E10b-P5route / -P5mix and their smoke (as with lensel_patch.py / stopmerge_patch.py):
idempotent, the file verl imports, written atomically; before the first change the file is kept once as ref_input_utils.py.orig
next to it. Prints ROUTE_HOOK_OK (already there) or ROUTE_HOOK_PATCHED. Usage: python route_patch.py [<ref_input_utils.py>]
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9x] per-prompt teacher template"
ANCHOR = "        _think = _os.environ.get('OPD_REF_THINK') == '1'\n"
HOOK = (f"        if _os.environ.get('OPD_ROUTE'):  {MARK}\n"
        "            import opd_route as _rt\n"
        "            _think = _rt.think_for(messages)\n")
FLUSH = (f"    if __import__('os').environ.get('OPD_ROUTE'):  {MARK} (the call's log)\n"
         "        import opd_route as _rt\n"
         "        _rt.flush(batch_size)\n")


def patch_text(s):
    """(new text, changed): both hooks inside prepare_ref_model_inputs; unchanged when they are already there."""
    if MARK in s:
        return s, False
    i = s.index("def prepare_ref_model_inputs(\n"); j = s.find("\ndef ", i + 1); j = len(s) if j < 0 else j
    body = s[i:j]
    n = body.count(ANCHOR)
    if n != 1:
        raise AssertionError(f"route anchor found {n} times in prepare_ref_model_inputs (the P5 patch of apply_patches.py first)")
    k = body.rindex("    return batch\n")
    body = body[:k] + FLUSH + body[k:]
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
    print("ROUTE_HOOK_PATCHED" if patch_file(p) else "ROUTE_HOOK_OK")
