"""The hook of tail-swap OPD (runbook [v2.5.9n], T4; opd_tailswap.py) in verl's RayPPOTrainer.fit: in the branch that
re-tokenizes the teacher prompt (use_ref_retokenization: P5's thinking reading), right after its reference log-probs are
joined to the batch, opd_tailswap.second_pass computes the non-thinking reading and swaps it in after each response's first
complete box. A no-op unless OPD_TAILSWAP=1, so every other arm runs the unchanged path.

apply_patches.py installs it with the other patches; run_arm.sh calls `python tailswap_patch.py` before a tail-swap arm
(the node's G-OPD was installed before this patch), as lensel_patch.py does: idempotent, on the file verl imports, the patched
text compiled before it is written, written atomically (trainings that are running keep the module they imported); before
the first change the unpatched file is kept once as ray_trainer.py.orig (lensel_patch.py may have kept it already: then that
one stays). Prints TAILSWAP_HOOK_OK (already there) or TAILSWAP_HOOK_PATCHED.
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9n] tail-swap OPD"
START = ("                                batch = prepare_ref_model_inputs(\n"
         "                                    batch=batch,\n"
         "                                    ref_tokenizer=self.ref_tokenizer,\n")
UNION = "                                batch = batch.union(ref_log_prob)\n"
HOOK = (f"                                if __import__('os').environ.get('OPD_TAILSWAP') == '1':  {MARK}\n"
        "                                    from opd_tailswap import second_pass as _opd_tail\n"
        "                                    batch, _opd_tail_metrics = _opd_tail(self, batch, apply_chat_template_kwargs)\n"
        "                                    metrics.update(_opd_tail_metrics)\n")


def patch_text(s):
    """(new text, changed): the hook after the reference log-probs of the re-tokenized branch; unchanged when already there."""
    if MARK in s:
        return s, False
    n = s.count(START)
    if n != 1:
        raise AssertionError(f"tailswap anchor found {n} times")
    i = s.index(START); j = s.index(UNION, i) + len(UNION)
    if "compute_ref_log_prob" not in s[i:j] or s[i:j].count("prepare_ref_model_inputs(") != 1:
        raise AssertionError("tailswap: the union after prepare_ref_model_inputs is not the branch's own")
    return s[:j] + HOOK + s[j:], True


def patch_file(path, keep_orig=True):
    s = open(path).read()
    new, changed = patch_text(s)
    if changed:
        compile(new, path, "exec")   # a syntax error never reaches the file
        if keep_orig and not os.path.exists(path + ".orig"):   # the rollback copy, kept once
            shutil.copy2(path, path + ".orig")
        tmp = path + ".tailswap.tmp"
        with open(tmp, "w") as f:
            f.write(new)
        shutil.copymode(path, tmp); os.replace(tmp, path)
    return changed


def main():
    spec = importlib.util.find_spec("verl.trainer.ppo.ray_trainer")
    if spec is None or not spec.origin:
        print("TAILSWAP_HOOK_FAILED: verl.trainer.ppo.ray_trainer not importable"); sys.exit(1)
    print("TAILSWAP_HOOK_PATCHED" if patch_file(spec.origin) else "TAILSWAP_HOOK_OK")


if __name__ == "__main__":
    main()
