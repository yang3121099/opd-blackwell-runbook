#!/usr/bin/env bash
# VENV-VL-NVRTC (runbook [v2.5.9z31]; ops-lead's A1 10-22 06:3x UTC; Q-051): CUDA 12.9's NVRTC into $OPD_ROOT/venv-vl.
# PROBE-VLM's 1-step smoke (06:09 UTC) failed in the student's forward: transformers' Qwen3-VL rot_pos_emb calls torch.prod on the
# GPU, torch compiles that reduction at run time with NVRTC (its jiterator), and venv-vl's NVRTC is CUDA 12.8's (torch
# 2.8.0+cu128's nvidia-cuda-nvrtc-cu12), which does not know the B300's sm_103 ("nvrtc: error: invalid value for
# --gpu-architecture"). CUDA 12.9's NVRTC does; torch loads libnvrtc.so.12 from site-packages/nvidia/cuda_nvrtc, so the newer wheel
# in that place is enough. The CPU lane, niced, idle I/O; no GPU (PROBE-VLM checks NVRTC on its GPU first).
#   1. guards: the target is $OPD_ROOT/venv-vl (resolved, a plain directory with bin/python); while a venv-vl python runs:
#      WAIT_INPUTS (no attempt, launched again later). [ops-lead's A1 conditions] venv-vl's site-packages came from venv-fa by cp -a,
#      which copies links as links: (a) before anything is fetched, site-packages, nvidia/, nvidia/cuda_nvrtc/, every path below
#      it and the nvidia_cuda_nvrtc_cu12 dist-info must resolve (realpath) inside $OPD_ROOT/venv-vl, else FAIL naming the path,
#      nothing installed; (b) venv and venv-fa are only read: the sha256 of their nvidia/cuda_nvrtc/lib/libnvrtc.so.12* before
#      and after the install must be equal, else FAIL and an event venv_contamination. guard.json also records whether venv-fa's
#      and venv-vl's nvidia/cuda_nvrtc are links (lstat)
#   2. the pinned wheel (NVRTC_PIN, default nvidia-cuda-nvrtc-cu12==12.9.86) by wheel_fetch.py (PyPI's sha256 checked; [v2.5.9z34]
#      WHEEL_PAR_FIRST=1, NVRTC_CONNS (16) connections: the first try, 06:44 UTC, was its own one-connection probe at 0.069 MB/s
#      and timed out), installed offline with --no-index --no-deps: the resolver touches nothing else
#   3. checks (no GPU visible): torch still 2.8.0+cu128, flash-attn still 2.8.3, nvidia-cuda-nvrtc-cu12 at the pin, libnvrtc.so.12
#      in site-packages/nvidia/cuda_nvrtc/lib; `pip check` verbatim at the top of requirements.lock (a line that torch wants NVRTC
#      12.8 is expected: recorded, not a failure), then pip freeze
# Report runs/VENV-VL-NVRTC/{report.json, requirements.lock, manifest.json (the wheel's sha256), nvrtc.log} to
# logs/<node>/venv_vl_nvrtc/ (private); events venv_vl_nvrtc_pass / venv_vl_nvrtc_fail; .done after a PASS (a FAIL is retried, as
# BUILD-VENV-VL). Usage: bash venv_vl_nvrtc.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=VENV-VL-NVRTC; D=$OPD_ROOT/runs/$T; LOG=$D/nvrtc.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "VENV_VL_NVRTC $(date -u +%T) $*" | tee -a $LOG; }
ROOT=$(readlink -f "$OPD_ROOT"); V=$ROOT/venv-vl; PIN=${NVRTC_PIN:-nvidia-cuda-nvrtc-cu12==12.9.86}
NI="nice -n 19"; command -v ionice > /dev/null && NI="$NI ionice -c3"
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k in ("checks", "manifest", "guard"):
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/up/ 2> /dev/null; cp $D/requirements.lock $D/manifest.json $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/venv_vl_nvrtc >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event $T "venv_vl_nvrtc_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; [ $1 = FAIL ] || touch $D/.done; exit 0
}
# 1. guards
[ -d "$V" ] && [ ! -L "$V" ] && [ "$(readlink -f "$V")" = "$ROOT/venv-vl" ] && [ -x "$V/bin/python" ] || finish FAIL "no plain \$OPD_ROOT/venv-vl with bin/python"
pgrep -f "venv-vl/bin/python" > /dev/null && { echo "WAIT_INPUTS $T: a venv-vl python is running"; exit 1; }
PY=$V/bin/python; PIP="$PY -m pip"
$PIP --version > /dev/null 2>&1 || finish FAIL "no pip in venv-vl"
SPV=$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2> /dev/null)
[ -n "$SPV" ] && [ -d "$SPV" ] || finish FAIL "venv-vl's site-packages not found"
guard() {   # $1 = pre / post: (a) every path venv-vl's NVRTC install writes resolves inside venv-vl; (b) venv / venv-fa NVRTC sha256
  python3 - "$1" "$ROOT" "$SPV" "$D/guard.json" <<'PY'
import glob, hashlib, json, os, sys
when, root, spv, out = sys.argv[1:5]
vl = os.path.join(root, "venv-vl")
g = json.load(open(out)) if when == "post" and os.path.exists(out) else {}
def rel(p):   # where p really points, as $OPD_ROOT/... when it is under it
    r = os.path.realpath(p)
    return "$OPD_ROOT/" + os.path.relpath(r, root) if r.startswith(root + "/") else r
if when == "pre":
    paths = [spv, os.path.join(spv, "nvidia")] + glob.glob(os.path.join(spv, "nvidia_cuda_nvrtc_cu12-*.dist-info"))
    base = os.path.join(spv, "nvidia", "cuda_nvrtc")
    if os.path.lexists(base):
        paths.append(base)
        for dp, dns, fns in os.walk(base):   # links are not followed (walked as entries, resolved below)
            paths += [os.path.join(dp, x) for x in dns + fns]
    for d in glob.glob(os.path.join(spv, "nvidia_cuda_nvrtc_cu12-*.dist-info")):
        paths += [os.path.join(d, x) for x in os.listdir(d)]
    outside = [p for p in paths if os.path.lexists(p) and not (os.path.realpath(p) + "/").startswith(vl + "/")]
    g["outside"] = [os.path.relpath(p, root) + " -> " + rel(p) for p in outside]
    g["checked_paths"] = len(paths)
    g["links"] = {}
    for name in ("venv-fa", "venv-vl"):
        for sp in glob.glob(os.path.join(root, name, "lib", "python*", "site-packages")):
            c = os.path.join(sp, "nvidia", "cuda_nvrtc")
            g["links"][name] = {"nvidia": os.path.islink(os.path.join(sp, "nvidia")), "cuda_nvrtc": os.path.islink(c),
                                "files_linked": sum(os.path.islink(os.path.join(dp, f)) for dp, _, fs in os.walk(c) for f in fs) if os.path.isdir(c) else None}
sha = {}
for name in ("venv", "venv-fa"):
    for f in sorted(glob.glob(os.path.join(root, name, "lib", "python*", "site-packages", "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12*"))):
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        sha[os.path.relpath(f, root)] = h.hexdigest()
g["sha_" + when] = sha
if when == "post":
    g["others_unchanged"] = g.get("sha_pre") == sha
json.dump(g, open(out, "w"), indent=1)
print("GUARD", when, json.dumps({k: g[k] for k in g if k not in ("sha_pre", "sha_post")})[:600])
sys.exit(1 if (when == "pre" and g["outside"]) or (when == "post" and not g["others_unchanged"]) else 0)
PY
}
guard pre >> $LOG 2>&1 || finish FAIL "paths outside venv-vl, nothing installed: $(python3 -c "import json, sys; print('; '.join(json.load(open(sys.argv[1]))['outside'])[:400])" $D/guard.json 2>&1 | tail -1)"
[[ $PIN =~ ^nvidia-cuda-nvrtc-cu12==12\.9\.[0-9]+$ ]] || finish FAIL "NVRTC_PIN $PIN is not nvidia-cuda-nvrtc-cu12==12.9.<x>"
step "start: $PIN into \$OPD_ROOT/venv-vl"
# 2. the wheel, sha256-checked, installed offline without dependencies
WHEEL_PAR_FIRST=1 FETCH_CONNS=${NVRTC_CONNS:-16} $NI python3 $RUNBOOK/scripts/wheel_fetch.py $D/wheels $PIN >> $LOG 2>&1 || finish FAIL "wheel: $(grep -h '^WHEEL.*Error\|^WHEEL.*sha256 ' $LOG | tail -2 | tr '\n' ' ')"   # [v2.5.9z34]
cp $D/wheels/manifest.json $D/manifest.json 2> /dev/null
$NI $PIP install --no-index --find-links $D/wheels --no-deps "$PIN" >> $LOG 2>&1 || finish FAIL "offline install of $PIN failed"
if ! guard post >> $LOG 2>&1; then   # [ops-lead's A1 condition b]
  python3 $RUNBOOK/scripts/control.py event $T venv_contamination "libnvrtc sha256 in venv / venv-fa changed during the venv-vl install" > /dev/null 2>&1
  finish FAIL "venv / venv-fa libnvrtc sha256 changed during the install (venv_contamination)"
fi
# 3. checks
{ echo "# pip check ($(date -u +%FT%TZ); a line that torch wants nvidia-cuda-nvrtc-cu12 12.8 is expected)"; $PIP check 2>&1; echo "# pip freeze"; $PIP freeze 2>&1; } > $D/requirements.lock
CUDA_VISIBLE_DEVICES= $PY - "$D/checks.json" "${PIN#*==}" >> $LOG 2>&1 <<'PY'
import glob, json, os, sys, sysconfig
from importlib import metadata as M
out, want = sys.argv[1], sys.argv[2]
def ver(*names):
    for n in names:
        try:
            return M.version(n)
        except M.PackageNotFoundError:
            continue
    return None
sp = sysconfig.get_paths()["purelib"]
lib = sorted(glob.glob(os.path.join(sp, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12*")))
c = {"torch": ver("torch"), "flash_attn": ver("flash_attn", "flash-attn"), "nvrtc_wheel": ver("nvidia-cuda-nvrtc-cu12"),
     "libnvrtc": [os.path.relpath(x, sp) for x in lib]}
c["pass"] = c["torch"] == "2.8.0+cu128" and (c["flash_attn"] or "").startswith("2.8.3") and c["nvrtc_wheel"] == want and bool(lib)
json.dump(c, open(out, "w"), indent=1); print("NVRTC_CHECKS", json.dumps(c))
sys.exit(0 if c["pass"] else 1)
PY
[ $? = 0 ] || finish FAIL "checks: $(python3 -c "import json, sys; c = json.load(open(sys.argv[1])); print(', '.join(k + ' ' + str(c.get(k)) for k in ('torch', 'flash_attn', 'nvrtc_wheel', 'libnvrtc')))" $D/checks.json 2>&1 | tail -1)"
finish PASS "$PIN in venv-vl; torch $(python3 -c "import json, sys; c = json.load(open(sys.argv[1])); print(c['torch'] + ', flash-attn ' + c['flash_attn'] + ', ' + ' '.join(c['libnvrtc']))" $D/checks.json); pip check: $(sed -n 2p $D/requirements.lock | cut -c1-120)"
