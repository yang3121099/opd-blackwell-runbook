"""G2, permanent (v2.4): every OPD variant a task can run must
  (1) be a branch of opd_variants.compute,
  (2) be in the list of variants the dp_actor patch hands to opd_variants (apply_patches.py), unless it is P2 or P5,
      which use the unchanged OPD advantage, and
  (3) change the advantage or the loss mask of at least one token relative to plain OPD.
The variants come from scripts/task.sh (the ARMS whitelist) mapped through scripts/run_arm.sh (E10b-<name> -> ARM=...;
e.g. P4lowlr is P4 with a lower learning rate). A new arm that is whitelisted but not wired through fails here instead
of silently training as plain OPD (this happened once in the project's own copy and was caught before the run)."""
import inspect
import os
import re

import torch

from opd_variants import compute, opd_k1

HERE = os.path.dirname(os.path.abspath(__file__))
RUNBOOK = os.path.dirname(os.path.dirname(HERE))
DEFAULT = {"P2", "P5"}


def launched_variants():
    arms = re.search(r'^ARMS="([^"]+)"', open(os.path.join(RUNBOOK, "scripts/task.sh")).read(), re.M).group(1).split("|")
    mapping = dict(re.findall(r"E10b-([A-Za-z0-9]+)\) ARM=([A-Za-z0-9]+);", open(os.path.join(RUNBOOK, "scripts/run_arm.sh")).read()))
    missing = [a for a in arms if a not in mapping]
    assert not missing, f"whitelisted in task.sh but not mapped in run_arm.sh: {missing}"
    return {mapping[a] for a in arms}


def compute_branches():
    src = open(os.path.join(HERE, "opd_variants.py")).read()
    body = src[src.index("def compute("):]
    body = body[:body.index("\ndef ", 1)]
    names = set(re.findall(r'variant == "(P[0-9A-Za-z]+)"', body))
    for tup in re.findall(r"variant in \(([^)]*)\)", body):
        names |= set(re.findall(r'"(P[0-9A-Za-z]+)"', tup))
    return names


def dispatched():
    m = re.search(r"if _v in \(([^)]*)\):", open(os.path.join(HERE, "apply_patches.py")).read())
    assert m, "dp_actor dispatch list not found in apply_patches.py"
    return set(re.findall(r"'(P[0-9A-Za-z]+)'", m.group(1)))


def test_launched_variants_are_known():
    unknown = launched_variants() - compute_branches()
    assert not unknown, f"launched but not a branch of opd_variants.compute: {sorted(unknown)}"


def test_every_nondefault_variant_is_dispatched():
    missing = (launched_variants() | compute_branches()) - DEFAULT - dispatched()
    assert not missing, f"not in the dp_actor dispatch list (would train as plain OPD): {sorted(missing)}"
    stray = dispatched() - compute_branches()
    assert not stray, f"dispatched but unknown to opd_variants.compute: {sorted(stray)}"


def test_every_nondefault_variant_changes_the_advantage():
    g = torch.Generator().manual_seed(1)
    old = -torch.rand(4, 120, generator=g) * 5
    ref = old + 0.3 * torch.randn(4, 120, generator=g)
    mask = torch.ones(4, 120); mask[1, 90:] = 0
    resp = torch.full((4, 120), 11, dtype=torch.long); resp[0, 119] = 151645; resp[1, 89] = 151643; resp[2, 50] = 151668
    kw = {"ref_alt": ref + 0.3 * torch.randn(4, 120, generator=g)} if "ref_alt" in inspect.signature(compute).parameters else {}
    base = opd_k1(old, ref, mask)
    for v in sorted((launched_variants() | compute_branches()) - DEFAULT):
        a, lm = compute(v, old, ref, mask, responses=resp, **kw)
        assert bool(((a != base) & (mask > 0)).any()) or not torch.equal(lm, mask), f"{v}: identical to plain OPD"
