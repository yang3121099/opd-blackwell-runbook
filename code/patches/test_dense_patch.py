"""[v2.5.9z76] (Q-065) dense_patch.py against G-OPD's dp_actor.py / fsdp_workers.py: GOPD_DP_ACTOR / GOPD_FSDP_WORKERS (the node:
the installed files, after apply_patches.py and the stop-merge hooks): every anchor found once, the patched text compiles (the
workers' file only on python >= 3.10, its `match`), a second application changes nothing, the hooks sit where they belong."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dense_patch as P  # noqa: E402


def _compile(text, name):
    try:
        compile(text, name, "exec")
    except SyntaxError:
        if sys.version_info >= (3, 10):
            raise


def _unpatched(p):
    """[v2.5.9z88a] (PROBE-DENSE-4B 10-26 00:23Z: 8 passed, 2 failed) the text to patch: the installed file, or - once an arm has
    patched it in place (the dense / AKL arms do, in the shared venv) - the unpatched copy dense_patch kept as <file>.orig; the
    installed file must then be a fixed point of the patch. As test_opd_tailswap's ray_trainer.py.orig (z80)."""
    src = open(p).read()
    if P.MARK in src and os.path.exists(p + ".orig"):
        return open(p + ".orig").read(), src
    return src, None


def test_actor_text():
    p = os.environ.get("GOPD_DP_ACTOR")
    if not p:
        return
    src, installed = _unpatched(p)
    if installed is not None:
        assert P.patch_actor_text(installed) == (installed, False)
    new, changed = P.patch_actor_text(src)
    assert changed and new.count(P.MARK) == 8
    _compile(new, p)
    assert P.patch_actor_text(new) == (new, False)
    i = new.index("_dn.rmpad_hook("); assert "labels=input_ids_rmpad_rolled" in new[i - 400:i]
    i = new.index("_dn.padded_hook("); assert 'logprobs_from_logits(logits, micro_batch["responses"])' in new[i - 300:i]
    i = new.index("take_act_loss("); assert "micro_batch_metrics.update(pg_metrics)" in new[i - 400:i] and new.index("policy_loss = pg_loss", i) > i
    assert "if calculate_entropy or _dn_act:" in new


def test_workers_text():
    p = os.environ.get("GOPD_FSDP_WORKERS")
    if not p:
        return
    src, installed = _unpatched(p)
    if installed is not None:
        assert P.patch_workers_text(installed) == (installed, False)
    new, changed = P.patch_workers_text(src)
    assert changed and new.count(P.MARK) == 1
    _compile(new, p)
    assert P.patch_workers_text(new) == (new, False)
    assert 'tensors={"ref_log_prob": output, **_dn_t}' in new


def test_unpatched_prefers_orig_once_patched():   # [v2.5.9z88a] runs anywhere (no G-OPD needed)
    import tempfile
    d = tempfile.mkdtemp()
    f = os.path.join(str(d), "dp_actor.py")
    open(f, "w").write("x = 1\n")
    assert _unpatched(f) == ("x = 1\n", None)                                      # unpatched, no copy: itself
    open(f, "w").write("x = 1\n" + P.MARK + "\n")
    assert _unpatched(f) == ("x = 1\n" + P.MARK + "\n", None)                     # patched but no .orig: itself (fails loudly)
    open(f + ".orig", "w").write("x = 1\n")
    assert _unpatched(f) == ("x = 1\n", "x = 1\n" + P.MARK + "\n")               # patched in place: the kept copy
