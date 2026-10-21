"""The hook of the VLM teacher's inputs (runbook [v2.5.9z], Q-051; opd_vlmref.py) in G-OPD's ref_input_utils.py, a no-op unless
OPD_VLM_REF=1: at the start of prepare_ref_model_inputs, right after its `apply_chat_template_kwargs` default, the batch is returned
with the reference inputs built from the student's own ids (no text re-tokenization). Applied by run_vlm.sh before an E15 arm and its
smoke: idempotent, the file verl imports, written atomically, the file kept once as ref_input_utils.py.orig. Prints VLMREF_HOOK_OK or
VLMREF_HOOK_PATCHED. Usage: python vlmref_patch.py [<ref_input_utils.py>]
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9z] VLM teacher inputs"
ANCHOR = "    if apply_chat_template_kwargs is None:\n        apply_chat_template_kwargs = {}\n"
HOOK = (f"    if __import__('os').environ.get('OPD_VLM_REF') == '1':  {MARK}\n"
        "        import opd_vlmref as _vr\n"
        "        return _vr.build_batch(batch, ref_tokenizer)\n")


def patch_text(s):
    """(new text, changed): the hook after the default inside prepare_ref_model_inputs; unchanged when already there."""
    if MARK in s:
        return s, False
    i = s.index("def prepare_ref_model_inputs(\n"); j = s.find("\ndef ", i + 1); j = len(s) if j < 0 else j
    body = s[i:j]
    n = body.count(ANCHOR)
    if n != 1:
        raise AssertionError(f"vlmref anchor found {n} times in prepare_ref_model_inputs")
    return s[:i] + body.replace(ANCHOR, ANCHOR + HOOK) + s[j:], True


def patch_file(path, keep_orig=True):
    s = open(path).read()
    new, changed = patch_text(s)
    if changed:
        if keep_orig and not os.path.exists(path + ".orig"):
            shutil.copy2(path, path + ".orig")
        tmp = path + ".opd_tmp"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    return changed


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else importlib.util.find_spec("verl.trainer.ppo.ref_input_utils").origin
    print("VLMREF_HOOK_PATCHED" if patch_file(p) else "VLMREF_HOOK_OK")
