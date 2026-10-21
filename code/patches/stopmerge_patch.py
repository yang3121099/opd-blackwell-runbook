"""The hooks of stop-merged OPD (runbook [v2.5.9l], Q-038; opd_stopmerge.py) in G-OPD's verl, all no-ops unless OPD_STOPMERGE=1:
  ref_input_utils.prepare_ref_model_inputs: each response's first complete \\boxed{} (batch key opd_box_end);
  dp_actor.compute_log_prob: the key carried into the reference pass;
  dp_actor._forward_micro_batch: the teacher's log-mass on the stop set, padded back like the log-probs, and the merge.
The actor's own passes never carry opd_box_end (it is added after old_log_prob, and update_policy selects its keys), so they
run unchanged.

apply_patches.py installs them with the other patches; run_arm.sh calls `python stopmerge_patch.py` before a stop-merged arm
(the node's G-OPD was installed before this patch), as lensel_patch.py does: idempotent, on the files verl imports, the
patched text compiled before it is written, written atomically (trainings that are running keep the modules they
imported); before the first change the unpatched file is kept once as <file>.orig next to it (a rollback copy).
Prints STOPMERGE_HOOK_OK (already there) or STOPMERGE_HOOK_PATCHED.
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9l] stop-merged OPD"


def _once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise AssertionError(f"stopmerge {what}: anchor found {n} times")
    return s.replace(old, new)


def patch_ref_text(s):
    """(new text, changed) for ref_input_utils.py: the box positions before prepare_ref_model_inputs returns."""
    if MARK in s:
        return s, False
    i = s.index("def prepare_ref_model_inputs(\n"); j = s.index("\ndef ", i + 1)
    k = s.rindex("    return batch\n", i, j)
    hook = (f"    {MARK}: the first complete box of each response, for the reference pass\n"
            "    if __import__('os').environ.get('OPD_STOPMERGE') == '1':\n"
            "        import opd_stopmerge as _sm\n"
            "        batch.batch['opd_box_end'] = torch.tensor(_sm.box_ends(responses, response_attention_mask, ref_tokenizer), dtype=torch.long)\n")
    return s[:k] + hook + s[k:], True


def patch_actor_text(s):
    """(new text, changed) for dp_actor.py."""
    if MARK in s:
        return s, False
    s = _once(s,
        '        if has_ref_input_ids:\n            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])\n',
        '        if has_ref_input_ids:\n            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])\n'
        f'        if "opd_box_end" in data.batch.keys():  {MARK}: the reference pass\n'
        '            select_keys.append("opd_box_end")\n',
        "select_keys")
    s = _once(s,
        '        response_length = micro_batch["responses"].size(-1)\n        multi_modal_inputs = {}\n',
        '        response_length = micro_batch["responses"].size(-1)\n'
        f'        _sm_on, _sm_lp = "opd_box_end" in micro_batch.keys(), None  {MARK} (reference pass only)\n'
        '        multi_modal_inputs = {}\n',
        "flag")
    rm = ("                    log_probs = logprobs_from_logits(\n"
          "                        logits=logits_rmpad,\n"
          "                        labels=input_ids_rmpad_rolled,\n"
          "                        inplace_backward=inplace_backward,\n"
          "                    )\n")
    s = _once(s, rm, rm +
        f"                    if _sm_on:  {MARK}: the teacher's log-mass on the stop set, every position\n"
        "                        import opd_stopmerge as _sm\n"
        "                        assert not self.use_ulysses_sp, 'stop-merged OPD: no sequence parallelism'\n"
        "                        _sm_lp = _sm.stop_logprob(logits_rmpad, input_ids_rmpad_rolled, log_probs, _sm.stop_ids())\n",
        "rmpad log-probs")
    pb = "                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)\n"
    s = _once(s, pb, pb +
        f"                if _sm_lp is not None:  {MARK}: padded back like the log-probs\n"
        "                    _sm_lp = pad_input(hidden_states=_sm_lp.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -response_length - 1 : -1]\n",
        "pad back")
    pl = '                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])\n'
    s = _once(s, pl, pl +
        f"                    if _sm_on:  {MARK}\n"
        "                        import opd_stopmerge as _sm\n"
        '                        _sm_lp = _sm.stop_logprob(logits, micro_batch["responses"], log_probs, _sm.stop_ids())\n',
        "padded log-probs")
    s = _once(s, "            return entropy, log_probs\n",
        f"            if _sm_on:  {MARK}: student stops after the first box get the teacher's stop-set mass\n"
        "                import opd_stopmerge as _sm\n"
        "                if _sm_lp is None:\n"
        "                    raise RuntimeError('stop-merged OPD needs the logits: fused kernels are not supported')\n"
        '                _valid = micro_batch["ref_attention_mask"][:, -response_length:] if "ref_attention_mask" in micro_batch.keys() else None\n'
        '                log_probs, _sm_stats = _sm.merge(log_probs, _sm_lp, micro_batch["responses"], micro_batch["opd_box_end"], _sm.eos_ids(), torch, _valid)\n'
        "                _sm.log_stats(_sm_stats)\n"
        "            return entropy, log_probs\n",
        "merge")
    return s, True


def patch_file(path, fn, keep_orig=True):
    s = open(path).read()
    new, changed = fn(s)
    if changed:
        compile(new, path, "exec")   # a syntax error never reaches the file
        if keep_orig and not os.path.exists(path + ".orig"):   # the rollback copy, kept once
            shutil.copy2(path, path + ".orig")
        tmp = path + ".stopmerge.tmp"
        with open(tmp, "w") as f:
            f.write(new)
        shutil.copymode(path, tmp); os.replace(tmp, path)
    return changed


def main():
    changed = False
    for mod, fn in (("verl.trainer.ppo.ref_input_utils", patch_ref_text), ("verl.workers.actor.dp_actor", patch_actor_text)):
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.origin:
            print(f"STOPMERGE_HOOK_FAILED: {mod} not importable"); sys.exit(1)
        changed |= patch_file(spec.origin, fn)
    print("STOPMERGE_HOOK_PATCHED" if changed else "STOPMERGE_HOOK_OK")


if __name__ == "__main__":
    main()
