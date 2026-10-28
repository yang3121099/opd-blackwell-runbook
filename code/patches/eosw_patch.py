"""The hooks of position-weighted OPD (runbook [v2.5.9z102], Q-087; opd_eosw.py) in G-OPD's verl, all no-ops unless OPD_EOSW=1:
  ref_input_utils.prepare_ref_model_inputs: each response's weights (batch key opd_eosw_w), after the other hooks before its return;
  dp_actor.update_policy: the key selected with the others; the per-token advantage multiplied by it just before the policy loss,
    and eosw/weighted_frac (the window's share of the valid tokens) in the step's metrics.
The reference and the actor's log-prob passes select their own keys, so they run unchanged.

apply_patches.py installs them with the other patches (section 9); run_arm.sh calls `python eosw_patch.py` before E10b-P5-eosw (a
G-OPD installed before this patch), as dense_patch.py: idempotent, on the files verl imports, the patched text compiled before it is
written, written atomically (running trainings keep the modules they imported); the unpatched file kept once as <file>.orig.
Prints EOSW_HOOK_OK (already there) or EOSW_HOOK_PATCHED.
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9z102] position-weighted OPD"


def _once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise AssertionError(f"eosw {what}: anchor found {n} times")
    return s.replace(old, new)


def patch_ref_text(s):
    """(new text, changed) for ref_input_utils.py: the weights before prepare_ref_model_inputs returns (after the other hooks)."""
    if MARK in s:
        return s, False
    i = s.index("def prepare_ref_model_inputs(\n"); j = s.index("\ndef ", i + 1)
    k = s.rindex("    return batch\n", i, j)
    hook = (f"    {MARK}: each response's weights (the window after its last complete box)\n"
            "    if __import__('os').environ.get('OPD_EOSW') == '1':\n"
            "        import opd_eosw as _ew\n"
            "        batch.batch['opd_eosw_w'] = _ew.make(responses, response_attention_mask, ref_tokenizer)\n")
    return s[:k] + hook + s[k:], True


def patch_actor_text(s):
    """(new text, changed) for dp_actor.py (with or without the stop-merge, variant and dense hooks)."""
    if MARK in s:
        return s, False
    sel = '        if "base_ref_log_prob" in data.batch.keys():\n            select_keys.append("base_ref_log_prob")\n'
    s = _once(s, sel, sel +
        f'        if "opd_eosw_w" in data.batch.keys():  {MARK}: the weights into the update\n'
        '            select_keys.append("opd_eosw_w")\n',
        "select_keys")
    pl = "                    policy_loss_fn = get_policy_loss_fn(loss_mode)\n"
    s = _once(s, pl,
        f'                    if "opd_eosw_w" in model_inputs:  {MARK}: the window after the last box weighs w (a positive factor on the advantage)\n'
        '                        _ew_w = model_inputs["opd_eosw_w"].to(advantages.dtype)\n'
        "                        advantages = advantages * _ew_w\n"
        '                        micro_batch_metrics["eosw/weighted_frac"] = ((_ew_w > 1) & (response_mask > 0)).float().sum().item() / max(response_mask.sum().item(), 1)\n'
        + pl, "policy loss")
    return s, True


def patch_file(path, fn, keep_orig=True):
    s = open(path).read()
    new, changed = fn(s)
    if changed:
        compile(new, path, "exec")   # a syntax error never reaches the file
        if keep_orig and not os.path.exists(path + ".orig"):   # the rollback copy, kept once
            shutil.copy2(path, path + ".orig")
        tmp = path + ".eosw.tmp"
        with open(tmp, "w") as f:
            f.write(new)
        shutil.copymode(path, tmp); os.replace(tmp, path)
    return changed


def main():
    changed = False
    for mod, fn, env in (("verl.trainer.ppo.ref_input_utils", patch_ref_text, "GOPD_REF_INPUT_UTILS"),
                         ("verl.workers.actor.dp_actor", patch_actor_text, "GOPD_DP_ACTOR")):
        path = os.environ.get(env)
        if not path:
            spec = importlib.util.find_spec(mod)
            if spec is None or not spec.origin:
                print(f"EOSW_HOOK_FAILED: {mod} not importable"); sys.exit(1)
            path = spec.origin
        changed |= patch_file(path, fn)
    print("EOSW_HOOK_PATCHED" if changed else "EOSW_HOOK_OK")


if __name__ == "__main__":
    main()
