"""verl's reduce_metrics on ragged per-rank lists (runbook [v2.5.9z78]; PROBE-8B's first run 10-24 00:12 UTC, the first 4-GPU OPD arm):
DataProto.concat turns the ranks' metric dicts into {key: [rank0 value, rank1 value, ...]}, and a value that is itself a list of
per-micro-batch numbers can have a different length on each rank; np.mean of that ragged list raises "setting an array element with
a sequence ... inhomogeneous shape" at the end of the first step. The hook flattens such a list (every number of every rank) before
the mean / max / min; a rectangular list gives the same number as before, a list of plain numbers is not touched.

[v2.5.9z81] (E10e-P5self 10-24 05:03 / 05:10 / 05:22 UTC, three attempts) DataProto.concat merges the ranks' metric dicts with
list_of_dict_to_dict_of_list, which asserts that every rank has the first rank's keys. opd_variants.adv_stats emits opd/eos_* only when a
rank's micro-batch holds an EOS-terminated response (and opd/trunc_tail_adv_mean only when it holds a truncated one): once nearly every
response hit the 4K cap, some ranks had none and the update ended in a bare AssertionError. The second hook merges the metrics with the
union of the ranks' keys (a key some ranks lack gets only the others' values; the same dict as before when the keys agree). Only the
metrics merge: the helper itself, which also merges non_tensor_batch, keeps its check.

run_arm.sh calls `python metrics_patch.py` before every arm: idempotent, on the files verl imports (GOPD_METRIC_UTILS / GOPD_PROTOCOL for
the tests), the patched text compiled before it is written, written atomically (a temporary name per process); the unpatched file kept
once as <file>.orig. Prints METRICS_HOOK_OK / METRICS_HOOK_PATCHED, then METRICS_CONCAT_HOOK_OK / METRICS_CONCAT_HOOK_PATCHED, or
METRICS_CONCAT_HOOK_SKIPPED when protocol.py has no such merge (one-GPU arms never merge ranks; multi-GPU arms run as before z81).
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


CMARK = "# [opd-variants patch v2] [v2.5.9z81] the union of the ranks' metric keys"
COLD = '                merged_meta_info["metrics"] = list_of_dict_to_dict_of_list(all_metrics)\n'
CNEW = (f'                merged_meta_info["metrics"] = {{}}  {CMARK}\n'
        '                for _m in all_metrics:  # a metric some ranks lack: opd/eos_* on a rank without an EOS-terminated response\n'
        '                    for _k, _v in _m.items():\n'
        '                        merged_meta_info["metrics"].setdefault(_k, []).append(_v)\n')


def patch_text(s):
    if MARK in s:
        return s, False
    if s.count(OLD) != 1:
        raise AssertionError(f"metrics anchor found {s.count(OLD)} times")
    return s.replace(OLD, NEW), True


def patch_concat_text(s):
    """[v2.5.9z81] (new text, changed) for verl/protocol.py."""
    if CMARK in s:
        return s, False
    if s.count(COLD) != 1:
        raise AssertionError(f"metrics merge anchor found {s.count(COLD)} times")
    return s.replace(COLD, CNEW), True


def _write(path, new):
    compile(new, path, "exec")
    if not os.path.exists(path + ".orig"):
        shutil.copy2(path, path + ".orig")
    tmp = f"{path}.metrics_tmp.{os.getpid()}"   # [v2.5.9z81] (z78b) two arms starting at once
    open(tmp, "w").write(new); os.replace(tmp, path)


def main():
    path = os.environ.get("GOPD_METRIC_UTILS") or importlib.util.find_spec("verl.utils.metric.utils").origin
    new, changed = patch_text(open(path).read())
    if changed:
        _write(path, new)
    print("METRICS_HOOK_PATCHED" if changed else "METRICS_HOOK_OK")
    proto = os.environ.get("GOPD_PROTOCOL") or importlib.util.find_spec("verl.protocol").origin
    try:
        new, changed = patch_concat_text(open(proto).read())
    except AssertionError as e:
        print(f"METRICS_CONCAT_HOOK_SKIPPED {e}"); return
    if changed:
        _write(proto, new)
    print("METRICS_CONCAT_HOOK_PATCHED" if changed else "METRICS_CONCAT_HOOK_OK")


if __name__ == "__main__":
    main()
