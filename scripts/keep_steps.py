"""[z168] (Fable board v22; executor B's keep_steps.sh on uk-8 11-15) the registered steps survive the local rotation. With CKPT_LOCAL the
trainer keeps only its newest 2 local checkpoints (verl max_actor_ckpt_to_keep=2) and the mirror keeps 2 on the shared disk, so with a short
save interval (SAVE=1) and uploads off an intermediate registered step (UW=25,50: s25) was deleted before anything copied it (c100 s25).
One pass, for each task under $CKPT_LOCAL: the steps to keep = its UPLOAD_WEIGHTS (the last UPLOAD_SETTINGS line upload_sync.sh wrote to
runs/<task>/upload.log; "all" = none) plus KEEP_STEPS (control.env, a comma list for every task). As soon as such a step's checkpoint is
complete (latest_checkpointed_iteration >= S, config.json + every weight file), its actor/huggingface (or huggingface_bf16) is HARD-LINKED
to <CKPT_LOCAL>.keep/<task>/global_step_S/actor/huggingface (instant; the rotation then frees nothing of it), then published to
$CKPT_ROOT/adhoc_models/result/runs/<task>/global_step_S/actor/huggingface - copied to .part, md5 of every file checked against the source,
fetch_model.py's .fetch_ok written (so a `result:` reading row finds it there without the Hub), renamed - and the local link removed.
All links first, then the copies, so a slow copy never delays a link. Stdlib only. Usage: python3 keep_steps.py [--once]"""
import datetime as dt
import glob
import hashlib
import json
import os
import re
import shutil
import sys

RX = re.compile(r"^global_step_(\d+)$")


def log(*a):
    print(dt.datetime.utcnow().strftime("%H:%M:%S"), *a, flush=True)


def cenv(root, key):
    v = None
    try:
        for l in open(os.path.join(root, "control.env")):
            m = re.match(rf"^(?:export )?{key}=(\S*)", l.strip())
            if m:
                v = m.group(1).strip("'\"")
    except OSError:
        pass
    return v if v is not None else os.environ.get(key, "")


def steps_to_keep(run_dir, extra):
    s = {int(x) for x in extra.split(",") if x.strip().isdigit()}
    try:
        lines = [l for l in open(os.path.join(run_dir, "upload.log"), errors="replace") if l.startswith("UPLOAD_SETTINGS ")]
    except OSError:
        lines = []
    if lines:
        m = re.search(r" UPLOAD_WEIGHTS=(\S+)", lines[-1])
        if m and m.group(1) != "all":
            s |= {int(x) for x in m.group(1).split(",") if x.isdigit()}
    return sorted(s)


def weights_dir(step_dir):
    """the step's complete HF weights dir (fp32 save, else upload_sync's bf16 copy), or None"""
    for n in ("huggingface", "huggingface_bf16"):
        d = os.path.join(step_dir, "actor", n)
        if not os.path.isfile(os.path.join(d, "config.json")):
            continue
        idx = os.path.join(d, "model.safetensors.index.json")
        if os.path.isfile(idx):
            try:
                need = set(json.load(open(idx))["weight_map"].values())
            except (ValueError, KeyError, OSError):
                continue
            if all(os.path.isfile(os.path.join(d, f)) for f in need):
                return d
        elif os.path.isfile(os.path.join(d, "model.safetensors")):
            return d
    return None


def latest(task_ckpt):
    try:
        return int("".join(c for c in open(os.path.join(task_ckpt, "latest_checkpointed_iteration.txt")).read() if c.isdigit()) or -1)
    except OSError:
        return -1


def link_tree(src, dst):
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            try:
                os.link(p, os.path.join(tmp, f))
            except OSError:   # another file system: a copy
                shutil.copy2(p, os.path.join(tmp, f))
    os.replace(tmp, dst)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def publish(src, pub, prefix, result_root):
    part = pub + ".part"
    shutil.rmtree(part, ignore_errors=True); os.makedirs(os.path.dirname(pub), exist_ok=True)
    shutil.copytree(src, part)
    files = []
    for f in sorted(os.listdir(src)):
        a, b = os.path.join(src, f), os.path.join(part, f)
        if not os.path.isfile(a):
            continue
        h = md5(a)
        if h != md5(b):
            raise OSError(f"md5 differs: {f}")
        files.append([f"{prefix}/{f}", os.path.getsize(b), h])
    json.dump({"repo": "local:keep_steps", "kind": "model", "prefix": prefix, "files": files, "hash": "md5",
               "utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}, open(os.path.join(part, ".fetch_ok"), "w"))
    if os.path.isdir(pub):
        shutil.rmtree(pub)
    os.replace(part, pub)


def one_pass(local_root, ckpt_root, opd_root):
    keep_root = local_root.rstrip("/") + ".keep"; result_root = os.path.join(ckpt_root, "adhoc_models", "result"); todo = []
    extra = cenv(opd_root, "KEEP_STEPS")
    for task in sorted(os.listdir(local_root)) if os.path.isdir(local_root) else []:
        ck = os.path.join(local_root, task, "ckpt")
        if not os.path.isdir(ck):
            continue
        for s in steps_to_keep(os.path.join(opd_root, "runs", task), extra):
            prefix = f"runs/{task}/global_step_{s}/actor/huggingface"; pub = os.path.join(result_root, prefix)
            k = os.path.join(keep_root, task, f"global_step_{s}", "actor", "huggingface")
            if os.path.isfile(os.path.join(pub, ".fetch_ok")):
                if os.path.isdir(k):
                    shutil.rmtree(os.path.join(keep_root, task, f"global_step_{s}"), ignore_errors=True)
                continue
            if not os.path.isdir(k):
                w = weights_dir(os.path.join(ck, f"global_step_{s}"))
                if w is None or latest(ck) < s:
                    continue
                try:
                    os.makedirs(os.path.dirname(k), exist_ok=True); link_tree(w, k); log("KEEP_OK", task, f"s{s}", os.path.basename(w))
                except OSError as e:
                    log("KEEP_FAIL", task, f"s{s}", type(e).__name__, e); continue
            todo.append((task, s, k, pub, prefix))
    for task, s, k, pub, prefix in todo:
        try:
            publish(k, pub, prefix, result_root); log("PUB_OK", task, f"s{s}", pub)
            shutil.rmtree(os.path.join(keep_root, task, f"global_step_{s}"), ignore_errors=True)
        except OSError as e:
            log("PUB_FAIL", task, f"s{s}", type(e).__name__, e)
    return todo


if __name__ == "__main__":
    lr, cr, orr = os.environ.get("CKPT_LOCAL", ""), os.environ.get("CKPT_ROOT", ""), os.environ.get("OPD_ROOT", "")
    if not (lr and cr and orr):
        log("KEEP_STEPS_OFF: CKPT_LOCAL / CKPT_ROOT / OPD_ROOT unset"); sys.exit(0)
    one_pass(lr, cr, orr)
