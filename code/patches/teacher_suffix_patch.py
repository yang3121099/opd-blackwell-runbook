"""The teacher-side prompt suffix (runbook [z163], lab/registry/Q-101.md E10b-P5selfp2) in G-OPD's ref_input_utils.py: in
prepare_ref_model_inputs, right before the teacher's template kwargs are chosen (the P5 patch's `_kw = dict(apply_chat_template_kwargs ...`
line, apply_patches.py section 2), `messages = opd_route.teacher_suffix(messages)`, and before the function returns the call's log line
(opd_route.teacher_suffix_flush). Both no-ops unless OPD_TEACHER_SUFFIX is set (non-empty). Only the teacher (ref) prompt passes through
prepare_ref_model_inputs; the student's prompt is the rollout's own tokens and never sees the suffix. Independent of route_patch.py /
mode_patch.py (either order).

run_arm.sh calls `python teacher_suffix_patch.py` for the E10b-P5selfp<k> arms and their smokes: idempotent, the file verl imports, the
patched text compiled before it is written, written atomically; before the first change the file is kept once as ref_input_utils.py.orig
next to it (another patch's copy when it made one). Prints TSUF_HOOK_OK (already there) or TSUF_HOOK_PATCHED.
Usage: python teacher_suffix_patch.py [<ref_input_utils.py>]
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [z163] teacher-side prompt suffix"
ANCHOR_PREFIX = "        _kw = dict(apply_chat_template_kwargs"
HOOK = (f"        if _os.environ.get('OPD_TEACHER_SUFFIX'):  {MARK}\n"
        "            import opd_route as _rt\n"
        "            messages = _rt.teacher_suffix(messages)\n")
FLUSH = (f"    if __import__('os').environ.get('OPD_TEACHER_SUFFIX'):  {MARK} (the call's log)\n"
         "        import opd_route as _rt\n"
         "        _rt.teacher_suffix_flush(batch_size)\n")


def patch_text(s):
    """(new text, changed): the hook before the teacher's template kwargs and the log before the return, inside prepare_ref_model_inputs;
    unchanged when they are already there."""
    if MARK in s:
        return s, False
    i = s.index("def prepare_ref_model_inputs(\n"); j = s.find("\ndef ", i + 1); j = len(s) if j < 0 else j
    body = s[i:j]
    lines = [l for l in body.splitlines(keepends=True) if l.startswith(ANCHOR_PREFIX)]
    if len(lines) != 1:
        raise AssertionError(f"teacher-suffix anchor found {len(lines)} times in prepare_ref_model_inputs (the P5 patch of apply_patches.py first)")
    k = body.rindex("    return batch\n")
    body = body[:k] + FLUSH + body[k:]
    body = body.replace(lines[0], HOOK + lines[0], 1)
    return s[:i] + body + s[j:], True


def patch_file(path, keep_orig=True):
    s = open(path).read()
    new, changed = patch_text(s)
    if changed:
        compile(new, path, "exec")
        if keep_orig and not os.path.exists(path + ".orig"):   # the rollback copy, kept once
            shutil.copy2(path, path + ".orig")
        tmp = f"{path}.opd_tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(new)
        os.replace(tmp, path)
    return changed


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else importlib.util.find_spec("verl.trainer.ppo.ref_input_utils").origin
    print("TSUF_HOOK_PATCHED" if patch_file(p) else "TSUF_HOOK_OK")
