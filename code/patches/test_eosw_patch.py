"""[v2.5.9z102] (Q-087) eosw_patch.py against G-OPD's ref_input_utils.py / dp_actor.py: GOPD_REF_INPUT_UTILS / GOPD_DP_ACTOR (the node:
the installed files after apply_patches.py; probe_eosw.sh passes them): every anchor found once, the patched text compiles, a second
application changes nothing (a file apply_patches.py hooked already is checked as installed), and the hooks sit where they belong -
the weights after prepare_ref_model_inputs' other hooks and before its return; the key selected in update_policy; the advantage
weighted after it is formed and just before the policy loss."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eosw_patch as P  # noqa: E402


def _texts(p):
    """(text to patch or None, installed text): the installed file; when it holds the hooks already (apply_patches.py section 9 keeps
    no copy, eosw_patch.py on demand keeps <file>.orig) the unpatched copy if there is one - the installed file must then be a fixed
    point of the patch and is checked as it is."""
    src = open(p).read()
    if P.MARK not in src:
        return src, None
    return (open(p + ".orig").read() if os.path.exists(p + ".orig") else None), src


def _patched(p, fn, marks):
    """The patched text to check: the patch applied to the unpatched text (changed, idempotent), or the installed one as it is."""
    src, installed = _texts(p)
    if installed is not None:
        assert fn(installed) == (installed, False) and installed.count(P.MARK) == marks
    if src is None:
        compile(installed, p, "exec"); return installed
    new, changed = fn(src)
    assert changed and new.count(P.MARK) == marks
    compile(new, p, "exec")
    assert fn(new) == (new, False)
    return new


def test_ref_text():
    p = os.environ.get("GOPD_REF_INPUT_UTILS")
    if not p:
        return
    new = _patched(p, P.patch_ref_text, 1)
    i = new.index("def prepare_ref_model_inputs("); j = new.index("\ndef ", i + 1); h = new.index("_ew.make(responses, response_attention_mask, ref_tokenizer)")
    assert i < h < j and new.index("    return batch\n", h) < j and "batch.batch['opd_eosw_w']" in new[h - 80:h]


def test_actor_text():
    p = os.environ.get("GOPD_DP_ACTOR")
    if not p:
        return
    new = _patched(p, P.patch_actor_text, 2)
    u = new.index("def update_policy")
    assert new.index('select_keys.append("opd_eosw_w")') > u
    k = new.index("advantages = advantages * _ew_w")
    assert u < k < new.index("policy_loss_fn = get_policy_loss_fn(loss_mode)", k) and "advantages = (- (reverse_kl))" in new[u:k]
    assert new.index("policy_loss_fn = get_policy_loss_fn(loss_mode)", k) - k < 400
