"""The hooks of dense OPD (runbook [v2.5.9z76], Q-065; opd_dense.py) in G-OPD's verl, all no-ops unless OPD_DENSE is rkl / akl:
  dp_actor._forward_micro_batch: in the reference pass (the micro-batch carries ref_input_ids) the teacher's top-K at the response
    positions (opd_dense.REF_TOPK); in the actor's update (it carries ref_topk_ids) the dense divergence per response position
    with gradient (opd_dense.ACT); the in-place cross-entropy backward off then (the dense loss reads the logits again);
    fused kernels refused (no logits);
  dp_actor.compute_log_prob: REF_TOPK cleared before the micro-batches, joined in the batch's order after them (LAST_REF);
  fsdp_workers.compute_ref_log_prob: LAST_REF's two tensors (ref_topk_ids, ref_topk_lp) into the output batch beside ref_log_prob;
  dp_actor.update_policy: the two keys selected (and ref_log_prob for the correlation); the policy loss replaced by the dense loss.
run_arm.sh calls `python dense_patch.py` before a dense arm: idempotent, on the files verl imports, the patched text compiled before
it is written, written atomically; the unpatched file kept once as <file>.orig. Prints DENSE_HOOK_OK / DENSE_HOOK_PATCHED.
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9z76] dense OPD"


def _once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise AssertionError(f"dense {what}: anchor found {n} times")
    return s.replace(old, new)


def patch_actor_text(s):
    """(new text, changed) for dp_actor.py (with or without the stop-merge hooks)."""
    if MARK in s:
        return s, False
    s = _once(s,
        '        multi_modal_inputs = {}\n        if "multi_modal_inputs" in micro_batch.keys():\n',
        f'        _dn_ref = _dn_act = False  {MARK}: the reference pass keeps the teacher top-K, the update takes the dense loss\n'
        '        if __import__("os").environ.get("OPD_DENSE") in ("rkl", "akl"):\n'
        '            import opd_dense as _dn\n'
        '            _dn_ref = "ref_input_ids" in micro_batch.keys()\n'
        '            _dn_act = "ref_topk_ids" in micro_batch.keys() and not _dn_ref\n'
        '            if (_dn_ref or _dn_act) and self.use_fused_kernels:\n'
        "                raise RuntimeError('dense OPD needs the logits: fused kernels are not supported')\n"
        '        multi_modal_inputs = {}\n        if "multi_modal_inputs" in micro_batch.keys():\n',
        "flags")
    s = _once(s,
        "                    inplace_backward = True\n                    if calculate_entropy:\n                        inplace_backward = False\n",
        "                    inplace_backward = True\n"
        f"                    if calculate_entropy or _dn_act:  {MARK}: the dense loss reads the logits in the backward\n"
        "                        inplace_backward = False\n",
        "inplace_backward")
    rm = ("                    log_probs = logprobs_from_logits(\n"
          "                        logits=logits_rmpad,\n"
          "                        labels=input_ids_rmpad_rolled,\n"
          "                        inplace_backward=inplace_backward,\n"
          "                    )\n")
    s = _once(s, rm, rm +
        f"                    if _dn_ref or _dn_act:  {MARK} (remove-padding)\n"
        "                        assert not self.use_ulysses_sp, 'dense OPD: no sequence parallelism'\n"
        "                        _dn.rmpad_hook(logits_rmpad, micro_batch, indices, batch_size, seqlen, response_length, _dn_ref, pad_input, index_first_axis, rearrange)\n",
        "rmpad log_probs")
    pd = '                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])\n'
    s = _once(s, pd, pd +
        f"                    if _dn_ref or _dn_act:  {MARK} (padded)\n"
        "                        _dn.padded_hook(logits, micro_batch, _dn_ref)\n",
        "padded log_probs")
    s = _once(s,
        "        log_probs_lst = []\n        entropy_lst = []\n",
        "        log_probs_lst = []\n        entropy_lst = []\n"
        f"        if __import__('os').environ.get('OPD_DENSE') in ('rkl', 'akl'):  {MARK}: this call's top-K only\n"
        "            __import__('opd_dense').REF_TOPK.clear()\n",
        "compute_log_prob lists")
    s = _once(s,
        "        return log_probs, entropys\n",
        f"        if __import__('os').environ.get('OPD_DENSE') in ('rkl', 'akl'):  {MARK}: the top-K in the batch's order\n"
        "            __import__('opd_dense').finish_ref(batch_idx_list if use_dynamic_bsz else None, restore_dynamic_batch)\n"
        "        return log_probs, entropys\n",
        "compute_log_prob return")
    s = _once(s,
        '        if "base_ref_log_prob" in data.batch.keys():\n            select_keys.append("base_ref_log_prob")\n',
        '        if "base_ref_log_prob" in data.batch.keys():\n            select_keys.append("base_ref_log_prob")\n'
        f'        if "ref_topk_ids" in data.batch.keys():  {MARK}: the teacher top-K for the dense loss\n'
        '            select_keys.extend(["ref_topk_ids", "ref_topk_lp"])\n'
        '            if "ref_log_prob" in data.batch.keys() and "ref_log_prob" not in select_keys:\n'
        '                select_keys.append("ref_log_prob")\n',
        "select_keys")
    s = _once(s,
        "                    micro_batch_metrics.update(pg_metrics)\n",
        "                    micro_batch_metrics.update(pg_metrics)\n"
        f"                    if __import__('os').environ.get('OPD_DENSE') in ('rkl', 'akl'):  {MARK}: the dense loss replaces the policy loss\n"
        "                        pg_loss, _dn_m = __import__('opd_dense').take_act_loss(agg_loss, response_mask, loss_agg_mode, log_prob, model_inputs.get('ref_log_prob'))\n"
        "                        micro_batch_metrics.update(_dn_m)\n",
        "pg_metrics")
    return s, True


def patch_workers_text(s):
    """(new text, changed) for fsdp_workers.py."""
    if MARK in s:
        return s, False
    s = _once(s,
        '            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)\n'
        '            output = DataProto.from_dict(tensors={"ref_log_prob": output})\n',
        '            output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)\n'
        f'            _dn_t = __import__("opd_dense").take_ref() if __import__("os").environ.get("OPD_DENSE") in ("rkl", "akl") else {{}}  {MARK}\n'
        '            output = DataProto.from_dict(tensors={"ref_log_prob": output, **_dn_t})\n',
        "compute_ref_log_prob")
    return s, True


def _write(path, text, keep_orig=True):
    compile(text, path, "exec")
    if keep_orig and not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")
    tmp = path + ".dense_tmp"
    open(tmp, "w").write(text); os.replace(tmp, path)


def main():
    act = os.environ.get("GOPD_DP_ACTOR") or importlib.util.find_spec("verl.workers.actor.dp_actor").origin
    wrk = os.environ.get("GOPD_FSDP_WORKERS") or importlib.util.find_spec("verl.workers.fsdp_workers").origin
    changed = False
    for path, fn in ((act, patch_actor_text), (wrk, patch_workers_text)):
        new, c = fn(open(path).read())
        if c:
            _write(path, new); changed = True
    print("DENSE_HOOK_PATCHED" if changed else "DENSE_HOOK_OK")


if __name__ == "__main__":
    main()
