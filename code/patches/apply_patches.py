"""Patch the G-OPD verl checkout for E10 arms. Always restores the pristine files from git first,
so re-running after a patch change is safe.

1) dp_actor.py
   - before the policy loss: `_opd_loss_mask = response_mask` (default for every algorithm)
   - after OPD computes `advantages = (- (reverse_kl))`: dispatch on env OPD_VARIANT
     (P3 bin-permuted teacher, P3b bin-mean teacher, P4 fixed negative on lowest-20% logp tokens) and log advantage
     diagnostics (opd/* metrics; P3 also logs corr/R^2/var-ratio vs the unpermuted advantage)
   - the policy loss uses `_opd_loss_mask`; entropy/KL keep the full response_mask (R-001 item 2)
2) ref_input_utils.prepare_ref_model_inputs: if env OPD_REF_THINK=1, render the teacher prompt in
   think mode and append '<think>\\n' (as in 2610.14270's think-likeness judge) -> arm P5.
4) ray_trainer.RayPPOTrainer.fit: [v2.5.8m] the length-selection hook (lensel_patch.py; a no-op unless OPD_LENSEL is set).
   [v2.5.9w] and r3's group drop (vardrop_patch.py; a no-op unless OPD_VARDROP is set).
Usage: python apply_patches.py $OPD_ROOT/G-OPD/verl
"""
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
repo = root.parent
FILES = ["verl/verl/workers/actor/dp_actor.py", "verl/verl/trainer/ppo/ref_input_utils.py", "verl/verl/trainer/fsdp_sft_trainer.py",
         "verl/verl/trainer/ppo/ray_trainer.py"]
subprocess.run(["git", "-C", str(repo), "checkout", "--", *FILES], check=True)
MARK = "# [opd-variants patch v2]"


def sub_once(s, old, new, what):
    assert s.count(old) == 1, f"{what}: anchor found {s.count(old)} times"
    return s.replace(old, new)


actor = root / "verl/workers/actor/dp_actor.py"
s = actor.read_text()
s = sub_once(s,
    "                    if entropy_coeff != 0:\n                        calculate_entropy = True\n                    entropy, log_prob = self._forward_micro_batch(\n",
    "                    if entropy_coeff != 0:\n                        calculate_entropy = True\n"
    f"                    if __import__('os').environ.get('OPD_DIAG', '1') == '1':  {MARK} entropy for diagnostics only (coef stays 0)\n"
    "                        calculate_entropy = True\n"
    "                    entropy, log_prob = self._forward_micro_batch(\n",
    "calculate_entropy")
s = sub_once(s,
    '                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)\n',
    '                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)\n'
    f'                    _opd_loss_mask = response_mask  {MARK} policy-loss mask (P4 narrows it)\n',
    "rollout_is_weights")
s = sub_once(s,
    "                        advantages = (- (reverse_kl))\n",
    "                        advantages = (- (reverse_kl))\n"
    f"                        {MARK} E10 advantage variants + diagnostics\n"
    "                        import os as _os\n"
    "                        from opd_variants import compute as _opd_compute, adv_stats as _opd_adv_stats\n"
    "                        _v = _os.environ.get('OPD_VARIANT', 'P2')\n"
    "                        _a0 = advantages\n"
    "                        if _v in ('P3', 'P3b', 'P4', 'P6', 'P6r', 'P2noEOS', 'Pstop', 'P6rand', 'P6top5', 'P4EOSpen'):\n"
    "                            advantages, _opd_loss_mask = _opd_compute(_v, old_log_prob, model_inputs['ref_log_prob'], response_mask, responses=model_inputs.get('responses'))\n"
    "                        micro_batch_metrics.update(_opd_adv_stats(_v, _a0, advantages, response_mask, _opd_loss_mask, entropy=entropy if calculate_entropy else None, old_log_prob=old_log_prob, ref_log_prob=model_inputs['ref_log_prob'], responses=model_inputs.get('responses')))\n",
    "advantages anchor")
s = sub_once(s,
    "                        advantages=advantages,\n"
    "                        response_mask=response_mask,\n"
    "                        loss_agg_mode=loss_agg_mode,\n",
    "                        advantages=advantages,\n"
    f"                        response_mask=_opd_loss_mask,  {MARK}\n"
    "                        loss_agg_mode=loss_agg_mode,\n",
    "policy_loss_fn call")
actor.write_text(s)
print("patched", actor)

ref = root / "verl/trainer/ppo/ref_input_utils.py"
s = ref.read_text()
old = (
    "        ref_prompt_str = ref_tokenizer.apply_chat_template(\n"
    "            messages, \n"
    "            add_generation_prompt=True, \n"
    "            tokenize=False,\n"
    "            **apply_chat_template_kwargs\n"
    "        )\n"
)
assert s.count(old) >= 1, "ref_input_utils anchor not found"
new = (
    f"        {MARK} P5: teacher scores in think mode\n"
    "        import os as _os\n"
    "        _think = _os.environ.get('OPD_REF_THINK') == '1'\n"
    "        _kw = dict(apply_chat_template_kwargs, enable_thinking=True) if _think else apply_chat_template_kwargs\n"
    "        ref_prompt_str = ref_tokenizer.apply_chat_template(\n"
    "            messages, \n"
    "            add_generation_prompt=True, \n"
    "            tokenize=False,\n"
    "            **_kw\n"
    "        )\n"
    "        if _think:\n"
    "            ref_prompt_str = ref_prompt_str + '<think>\\n'\n"
)
s = s.replace(old, new, 1)  # first occurrence = prepare_ref_model_inputs
ref.write_text(s)
print("patched", ref)

# 3. SFT trainer attention (Blackwell runbook): verl 0.6.1 hard-codes flash_attention_2 in fsdp_sft_trainer.py; read it
#    from OPD_ATTN instead (default unchanged), so the SFT step can run with SDPA where flash-attn is not installed.
sft = root / "verl/trainer/fsdp_sft_trainer.py"
s = sft.read_text()
s = sub_once(s, 'attn_implementation="flash_attention_2",',
             f'attn_implementation=__import__("os").environ.get("OPD_ATTN", "flash_attention_2"),  {MARK} SFT attention',
             "sft attn_implementation")
sft.write_text(s)
print("patched", sft)

# 4. [v2.5.8m] (Q-030) length-selected OPD: k of n rollouts per prompt kept before the log-probs (lensel_patch.py,
#    a no-op unless OPD_LENSEL is set)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lensel_patch import patch_file as _lensel_patch  # noqa: E402
_lensel_patch(str(root / "verl/trainer/ppo/ray_trainer.py"), keep_orig=False)   # the pristine file is in git
print("patched", root / "verl/trainer/ppo/ray_trainer.py")

# 5. [v2.5.9l] (Q-038) stop-merged OPD (stopmerge_patch.py; a no-op unless OPD_STOPMERGE=1): the box positions in
#    prepare_ref_model_inputs, the key into the reference pass, the teacher's stop-set mass and the merge there
from stopmerge_patch import patch_file as _sm_patch, patch_ref_text as _sm_ref, patch_actor_text as _sm_actor  # noqa: E402
_sm_patch(str(root / "verl/trainer/ppo/ref_input_utils.py"), _sm_ref, keep_orig=False)   # the pristine files are in git
_sm_patch(str(root / "verl/workers/actor/dp_actor.py"), _sm_actor, keep_orig=False)
print("patched", root / "verl/trainer/ppo/ref_input_utils.py", "and", root / "verl/workers/actor/dp_actor.py", "(stop-merged OPD)")

# 6. [v2.5.9n] (T4) tail-swap OPD (tailswap_patch.py; a no-op unless OPD_TAILSWAP=1): after the re-tokenized reference
#    log-probs, the non-thinking reading of the same teacher after each response's first complete box
from tailswap_patch import patch_file as _ts_patch  # noqa: E402
_ts_patch(str(root / "verl/trainer/ppo/ray_trainer.py"), keep_orig=False)   # the pristine file is in git
print("patched", root / "verl/trainer/ppo/ray_trainer.py", "(tail-swap OPD)")

# 7. [v2.5.9w] (Q-039) r3's group drop (vardrop_patch.py; a no-op unless OPD_VARDROP is set): after the rollouts are joined,
#    the rows of at most K groups whose scores differ go on; after compute_advantage, the kept tokens' advantage statistics
from vardrop_patch import patch_file as _vd_patch  # noqa: E402
_vd_patch(str(root / "verl/trainer/ppo/ray_trainer.py"), keep_orig=False)   # the pristine file is in git
print("patched", root / "verl/trainer/ppo/ray_trainer.py", "(r3 group drop)")
