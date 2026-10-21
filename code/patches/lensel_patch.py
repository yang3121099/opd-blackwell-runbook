"""The hook of length-selected OPD (runbook [v2.5.8m], Q-030) in verl's RayPPOTrainer.fit: right after the rollouts are
joined to the batch (`batch = batch.union(gen_batch_output)`), opd_lensel.select_by_length keeps k responses per prompt
before the old / reference log-probs. A no-op unless OPD_LENSEL is set, so every other arm runs the unchanged path.

apply_patches.py installs it with the other patches; run_arm.sh calls `python lensel_patch.py` before a length-selected arm
(the node's G-OPD was installed before this patch): idempotent, the file verl imports, written atomically (trainings that
are running keep the module they imported); before the first change the unpatched file is kept once as
ray_trainer.py.orig next to it (reviewer A1 08:2x UTC 10-21: a rollback copy). Prints LENSEL_HOOK_OK (already there) or
LENSEL_HOOK_PATCHED.
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] Q-030 length-selected OPD"
ANCHOR = "                    batch = batch.union(gen_batch_output)\n"
HOOK = (
    f"                    if __import__('os').environ.get('OPD_LENSEL'):  {MARK}\n"
    "                        from opd_lensel import select_by_length as _opd_lensel\n"
    "                        batch, _opd_ls_metrics = _opd_lensel(batch, self.reward_fn, self.global_steps)\n"
    "                        metrics.update(_opd_ls_metrics)\n"
)


def patch_text(s):
    """(new text, changed): the hook after the one anchor in fit; unchanged when the hook is already there."""
    if MARK in s:
        return s, False
    n = s.count(ANCHOR)
    if n != 1:
        raise AssertionError(f"lensel anchor found {n} times")
    return s.replace(ANCHOR, ANCHOR + HOOK), True


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
    p = sys.argv[1] if len(sys.argv) > 1 else importlib.util.find_spec("verl.trainer.ppo.ray_trainer").origin
    print("LENSEL_HOOK_PATCHED" if patch_file(p) else "LENSEL_HOOK_OK")
