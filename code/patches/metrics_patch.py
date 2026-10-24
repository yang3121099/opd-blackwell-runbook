"""verl's reduce_metrics on ragged per-rank lists (runbook [v2.5.9z78]; PROBE-8B's first run 10-24 00:12 UTC, the first 4-GPU OPD arm):
DataProto.concat turns the ranks' metric dicts into {key: [rank0 value, rank1 value, ...]}, and a value that is itself a list of
per-micro-batch numbers can have a different length on each rank; np.mean of that ragged list raises "setting an array element with
a sequence ... inhomogeneous shape" at the end of the first step. The hook flattens such a list (every number of every rank) before
the mean / max / min; a rectangular list gives the same number as before, a list of plain numbers is not touched.

run_arm.sh calls `python metrics_patch.py` before every arm: idempotent, on the file verl imports (GOPD_METRIC_UTILS for the tests),
the patched text compiled before it is written, written atomically; the unpatched file kept once as <file>.orig.
Prints METRICS_HOOK_OK (already there) or METRICS_HOOK_PATCHED.
"""
import importlib.util
import os
import shutil
import sys

MARK = "# [opd-variants patch v2] [v2.5.9z78] ragged per-rank metric lists flattened"
OLD = """    for key, val in metrics.items():
        if "max" in key:"""
NEW = f"""    for key, val in metrics.items():
        if isinstance(val, (list, tuple)) and any(isinstance(x, (list, tuple, np.ndarray)) for x in val):  {MARK}
            val = np.concatenate([np.ravel(np.asarray(x, dtype=float)) for x in val])
        if "max" in key:"""


def patch_text(s):
    if MARK in s:
        return s, False
    if s.count(OLD) != 1:
        raise AssertionError(f"metrics anchor found {s.count(OLD)} times")
    return s.replace(OLD, NEW), True


def main():
    path = os.environ.get("GOPD_METRIC_UTILS") or importlib.util.find_spec("verl.utils.metric.utils").origin
    new, changed = patch_text(open(path).read())
    if changed:
        compile(new, path, "exec")
        if not os.path.exists(path + ".orig"):
            shutil.copy2(path, path + ".orig")
        tmp = path + ".metrics_tmp"
        open(tmp, "w").write(new); os.replace(tmp, path)
    print("METRICS_HOOK_PATCHED" if changed else "METRICS_HOOK_OK")


if __name__ == "__main__":
    main()
