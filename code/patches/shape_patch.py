"""The hook of the beyond arms' reward shaping (runbook [v2.5.9z118], research_program 3q; opd_variants.shape) in G-OPD's verl, a
no-op unless OPD_FRET=1 or OPD_TOPD_ALPHA is set:
  dp_actor.update_policy: the per-token advantage (P5's r_k = log q - log p) through opd_variants.shape - TOPD's bounded reward, then
    the suffix credit - just before the policy loss, and its shape/* metrics in the step's metrics.
The reference and the actor's log-prob passes are not touched.

run_arm.sh calls `python shape_patch.py` before E10b-P5fret / -P5topd / -P5fret-topd (a G-OPD installed before this patch), as
eosw_patch.py: idempotent, on the file verl imports, the patched text compiled before it is written, written atomically (running
trainings keep the modules they imported); the unpatched file kept once as <file>.orig. Prints SHAPE_HOOK_OK (already there) or
SHAPE_HOOK_PATCHED.
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9z118] fret / topd reward shaping"


def _once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise AssertionError(f"shape {what}: anchor found {n} times")
    return s.replace(old, new)


def patch_actor_text(s):
    """(new text, changed) for dp_actor.py (with or without the stop-merge, variant, dense and eosw hooks): the shaping right before
    the policy loss, after every other hook on the advantage."""
    if MARK in s:
        return s, False
    pl = "                    policy_loss_fn = get_policy_loss_fn(loss_mode)\n"
    s = _once(s, pl,
        f"                    if __import__('os').environ.get('OPD_FRET') == '1' or __import__('os').environ.get('OPD_TOPD_ALPHA'):  {MARK}\n"
        "                        import opd_variants as _osh\n"
        "                        advantages, _osh_m = _osh.shape(advantages, response_mask)\n"
        "                        micro_batch_metrics.update(_osh_m)\n"
        + pl, "policy loss")
    return s, True


def patch_file(path, fn, keep_orig=True):
    s = open(path).read()
    new, changed = fn(s)
    if changed:
        compile(new, path, "exec")   # a syntax error never reaches the file
        if keep_orig and not os.path.exists(path + ".orig"):   # the rollback copy, kept once
            shutil.copy2(path, path + ".orig")
        tmp = path + ".shape.tmp"
        with open(tmp, "w") as f:
            f.write(new)
        shutil.copymode(path, tmp); os.replace(tmp, path)
    return changed


def main():
    path = os.environ.get("GOPD_DP_ACTOR")
    if not path:
        spec = importlib.util.find_spec("verl.workers.actor.dp_actor")
        if spec is None or not spec.origin:
            print("SHAPE_HOOK_FAILED: verl.workers.actor.dp_actor not importable"); sys.exit(1)
        path = spec.origin
    print("SHAPE_HOOK_PATCHED" if patch_file(path, patch_actor_text) else "SHAPE_HOOK_OK")


if __name__ == "__main__":
    main()
