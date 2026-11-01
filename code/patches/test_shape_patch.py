"""[v2.5.9z118] shape_patch.py against G-OPD's dp_actor.py: GOPD_DP_ACTOR (the node: the installed file after apply_patches.py and any
eosw_patch.py; probe_fret.sh passes it): the anchor found once, the patched text compiles, a second application changes nothing, and
the hook sits after the advantage is formed (and after eosw's weighting when that hook is there) and just before the policy loss.
The synthetic text below runs everywhere (the Mac included)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shape_patch as P  # noqa: E402

PL = "policy_loss_fn = get_policy_loss_fn(loss_mode)"
SYN = ("class A:\n    def update_policy(self, data):\n        for micro in data:\n            if True:\n                if True:\n"
       "                    response_mask = micro['m']; micro_batch_metrics = {}; loss_mode = 'vanilla'\n"
       "                    if True:\n                        advantages = (- (reverse_kl))\n"
       "                    if \"opd_eosw_w\" in model_inputs:  # eosw\n                        advantages = advantages * 2\n"
       "                    " + PL + "\n                    pg = policy_loss_fn(advantages)\n")


def _check(new):
    u = new.index("def update_policy"); k = new.index("_osh.shape(advantages, response_mask)")
    assert u < new.index("advantages = (- (reverse_kl))") < k and new.index(PL, k) - k < 200
    if "opd_eosw_w" in new:
        assert new.index("advantages = advantages *", u) < k   # after eosw's weighting
    assert new.count(P.MARK) == 1 and "micro_batch_metrics.update(_osh_m)" in new[k:new.index(PL, k)]


def test_synthetic_text():
    new, changed = P.patch_actor_text(SYN)
    assert changed; compile(new, "dp_actor.py", "exec"); _check(new)
    assert P.patch_actor_text(new) == (new, False)                          # idempotent
    try:
        P.patch_actor_text(SYN.replace(PL, "x = 1"))
        assert False, "a missing anchor must fail"
    except AssertionError as e:
        assert "anchor found 0 times" in str(e)


def test_patch_file_writes_once(tmp_path):
    p = tmp_path / "dp_actor.py"; p.write_text(SYN)
    assert P.patch_file(str(p), P.patch_actor_text) is True and (tmp_path / "dp_actor.py.orig").read_text() == SYN
    assert P.patch_file(str(p), P.patch_actor_text) is False
    _check(p.read_text())


def test_actor_text():
    p = os.environ.get("GOPD_DP_ACTOR")
    if not p:
        return
    src = open(p).read()
    if P.MARK in src:                                                       # installed already: a fixed point, checked as it is
        assert P.patch_actor_text(src) == (src, False); compile(src, p, "exec"); _check(src); return
    new, changed = P.patch_actor_text(src)
    assert changed; compile(new, p, "exec"); _check(new)
    assert P.patch_actor_text(new) == (new, False)
