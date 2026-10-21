"""The hooks of r3's group drop (runbook [v2.5.9w], Q-039) in verl's RayPPOTrainer.fit, both no-ops unless OPD_VARDROP is set:
1. right after the rollouts are joined to the batch (`batch = batch.union(gen_batch_output)`, lensel's anchor): opd_vardrop.
   select_groups keeps the rows of at most K groups whose scores differ, before the old log-probs;
2. right after compute_advantage: opd_vardrop.adv_stats logs the advantages' mean / variance over the kept tokens, the
   update's rows / prompts and its PPO mini-batch count.

run_e12.sh calls `python vardrop_patch.py` before E12b-P5-r3 and its smoke (as run_arm.sh does with lensel_patch.py):
idempotent, the file verl imports, written atomically (trainings that are running keep the module they imported); before the
first change the unpatched file is kept once as ray_trainer.py.orig next to it. Prints VARDROP_HOOK_OK (already there) or
VARDROP_HOOK_PATCHED. The lensel hook may sit at the same anchor; the two never run together and either order works.
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] Q-039 r3 group drop"
ANCHOR = "                    batch = batch.union(gen_batch_output)\n"
HOOK = (
    f"                    if __import__('os').environ.get('OPD_VARDROP'):  {MARK}\n"
    "                        from opd_vardrop import select_groups as _opd_vardrop\n"
    "                        batch, _opd_vd_metrics = _opd_vardrop(batch, self.reward_fn, self.global_steps)\n"
    "                        metrics.update(_opd_vd_metrics)\n"
)
ANCHOR2 = "                            config=self.config.algorithm,\n                        )\n"
HOOK2 = (
    f"                        if __import__('os').environ.get('OPD_VARDROP'):  {MARK} (advantage)\n"
    "                            from opd_vardrop import adv_stats as _opd_vd_adv\n"
    "                            metrics.update(_opd_vd_adv(batch, self.config.actor_rollout_ref.actor.ppo_mini_batch_size,\n"
    "                                                       self.config.actor_rollout_ref.rollout.n))\n"
)


def patch_text(s):
    """(new text, changed): both hooks after their one anchor each; unchanged when the hooks are already there."""
    if MARK in s:
        return s, False
    for a in (ANCHOR, ANCHOR2):
        n = s.count(a)
        if n != 1:
            raise AssertionError(f"vardrop anchor found {n} times: {a.strip()[:40]}")
    return s.replace(ANCHOR, ANCHOR + HOOK).replace(ANCHOR2, ANCHOR2 + HOOK2), True


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
    print("VARDROP_HOOK_PATCHED" if patch_file(p) else "VARDROP_HOOK_OK")
