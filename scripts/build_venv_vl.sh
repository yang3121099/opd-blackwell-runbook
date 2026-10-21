#!/usr/bin/env bash
# BUILD-VENV-VL (runbook v2.5.9y1; Q-051, ops-lead's A1 10-22 01:3x Beijing, Fable's requirements 01:4x): $OPD_ROOT/venv-vl for
# the Qwen3-VL arms only (the node's transformers 4.56.2 / vLLM 0.10.2 do not know Qwen3-VL; vLLM 0.11.0 is the first that does,
# still on torch 2.8.0, and verl 0.6.1's upper bound). The CPU lane, niced, idle I/O; no GPU, nothing that runs is touched.
#   1. guards: the target is $OPD_ROOT/venv-vl (absolute, resolved); an existing venv-vl is left alone (exit 0, "exists") unless
#      VENV_VL_REBUILD=1, which first renames it to venv-vl.tmp; every rm is of $OPD_ROOT/venv-vl.tmp only (a leftover of a
#      failed run is removed at the start); venv and venv-fa are only read.
#   2. a fresh venv on venv-fa's own base python, without pip; venv-fa's site-packages copied in (torch 2.8.0+cu128, flash-attn
#      2.8.3, xformers, torchvision / torchaudio, verl's editable G-OPD path, pip itself: nothing big downloaded again).
#   3. the pinned wheels (VL_PINS: vllm 0.11.0, transformers 4.57.1, qwen-vl-utils 0.0.14, mathruler 0.1.0) from PyPI by
#      wheel_fetch.py: the first timed over one connection (the proxy probe), a large one through par_fetch when that probe is
#      under 1 MB/s; every file checked against PyPI's sha256. Installed offline (--no-index, --no-deps) over the copied
#      versions inside venv-vl.tmp, then `pip install` of the same pins resolves only what they still lack (small packages,
#      from PyPI through the same proxy).
#   4. checks in venv-vl.tmp with no GPU visible: transformers / vllm / torch versions, transformers.models has qwen3_vl and
#      Qwen3VLForConditionalGeneration, vLLM's registry has Qwen3VLForConditionalGeneration, torch.version.cuda 12.8, verl's
#      qwen3_vl model patch imports, and Qwen3-VL-8B-Thinking's processor loads (when its .fetch_ok is there). Any miss = FAIL.
#   5. the venv's own scripts rewritten from venv-vl.tmp to venv-vl, then renamed into place (one rename).
# Report: runs/BUILD-VENV-VL/{report.json, requirements.lock (pip freeze), wheels/manifest.json (sha256 of each wheel), build.log}
# to logs/<node>/venv_vl/ (private); events venv_vl_built / venv_vl_failed; .done after a build. Usage: bash build_venv_vl.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=BUILD-VENV-VL; D=$OPD_ROOT/runs/$T; LOG=$D/build.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "BUILD_VENV_VL $(date -u +%T) $*" | tee -a $LOG; }
ROOT=$(readlink -f "$OPD_ROOT"); DST=$ROOT/venv-vl; TMP=$ROOT/venv-vl.tmp; SRCV=${VENV_VL_FROM:-$ROOT/venv-fa}
PINS=${VL_PINS:-vllm==0.11.0 transformers==4.57.1 qwen-vl-utils==0.0.14 mathruler==0.1.0}
NI="nice -n 19"; command -v ionice > /dev/null && NI="$NI ionice -c3"
finish() {   # $1 = PASS / FAIL / EXISTS, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k in ("checks", "wheels/manifest"):
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        r[k.split("/")[-1]] = json.load(open(p))
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/up/ 2> /dev/null; cp $D/requirements.lock $D/wheels/manifest.json $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/venv_vl >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  case $1 in PASS) ev=venv_vl_built;; EXISTS) ev=venv_vl_exists;; *) ev=venv_vl_failed;; esac
  python3 $RUNBOOK/scripts/control.py event $T $ev "$2" > /dev/null 2>&1
  step "$1 $2"; [ $1 = FAIL ] || touch $D/.done; exit 0
}
rm_tmp() {   # the only rm of this script: $OPD_ROOT/venv-vl.tmp, resolved, and nothing else
  [ -e "$TMP" ] || [ -L "$TMP" ] || return 0
  [ "$(readlink -f "$TMP")" = "$ROOT/venv-vl.tmp" ] && [ ! -L "$TMP" ] || { step "REFUSED: $TMP is not a plain directory at \$OPD_ROOT/venv-vl.tmp"; finish FAIL "venv-vl.tmp is a link or elsewhere"; }
  rm -rf -- "$ROOT/venv-vl.tmp"
}
# 1. guards
[ "$DST" = "$ROOT/venv-vl" ] && [ "$TMP" = "$ROOT/venv-vl.tmp" ] || finish FAIL "target paths not under \$OPD_ROOT"
if [ -e "$DST" ] || [ -L "$DST" ]; then
  [ "${VENV_VL_REBUILD:-0}" = 1 ] || finish EXISTS "venv-vl exists (set VENV_VL_REBUILD=1 to build it again)"
  [ -L "$DST" ] && finish FAIL "venv-vl is a link: not renamed"
  rm_tmp; mv -- "$DST" "$TMP" && step "rebuild: the old venv-vl moved to venv-vl.tmp"
fi
rm_tmp
[ -x "$SRCV/bin/python" ] || finish FAIL "no $(basename $SRCV) to start from"
step "start from=$(basename $SRCV) pins=[$PINS]"
# 2. a fresh venv on the source venv's base python, the source's site-packages copied in (read only)
BASEPY=$($SRCV/bin/python -c 'import sys; print(sys._base_executable)' 2>> $LOG); PYV=$($SRCV/bin/python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
SPS=$($SRCV/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>> $LOG)
[ -x "$BASEPY" ] && [ -d "$SPS" ] || finish FAIL "base python or site-packages of $(basename $SRCV) not found"
$BASEPY -m venv --without-pip --prompt venv-vl "$TMP" >> $LOG 2>&1 || finish FAIL "python -m venv failed"
SPT=$TMP/lib/python$PYV/site-packages; mkdir -p "$SPT"
$NI cp -a "$SPS/." "$SPT/" >> $LOG 2>&1 || finish FAIL "copy of $(basename $SRCV)'s site-packages failed"
PY=$TMP/bin/python
# pip: the copied one (venv-fa's), else ensurepip from the base python, else uv (a uv-built venv-fa may have no pip)
if $PY -m pip --version > /dev/null 2>&1; then PIP="$PY -m pip"
elif $PY -m ensurepip --default-pip >> $LOG 2>&1 && $PY -m pip --version > /dev/null 2>&1; then PIP="$PY -m pip"
elif command -v uv > /dev/null; then PIP="uv pip --python $PY"
else finish FAIL "no pip for venv-vl.tmp (none copied, no ensurepip, no uv)"; fi
step "venv on python $PYV, pip: ${PIP%% *}, $(du -sm "$SPT" 2> /dev/null | cut -f1) MB of site-packages copied; torch $($PY -c 'import importlib.metadata as m; print(m.version("torch"))' 2>> $LOG)"
# 3. the pinned wheels, sha256-checked, installed offline; then only the missing dependencies
$NI python3 $RUNBOOK/scripts/wheel_fetch.py $D/wheels $PINS >> $LOG 2>&1 || finish FAIL "wheels: $(grep -h '^WHEEL.*Error\|^WHEEL.*sha256 ' $LOG | tail -2 | tr '\n' ' ')"
step "wheels in: $(grep '^PROBE_MBPS' $LOG | tail -1); $(grep -c '^WHEEL' $LOG) lines"
$NI $PIP install --no-index --find-links $D/wheels --no-deps $PINS >> $LOG 2>&1 || finish FAIL "offline install of the pinned wheels failed"
$NI $PIP install $PINS >> $LOG 2>&1 || finish FAIL "dependencies of the pins could not be installed"
$PIP check >> $LOG 2>&1 || step "pip check reports conflicts (kept in build.log)"
$PIP freeze > $D/requirements.lock 2>> $LOG
# 4. checks, no GPU visible
CUDA_VISIBLE_DEVICES= $PY - "$D/checks.json" "$OPD_ROOT/models/Qwen3-VL-8B-Thinking" >> $LOG 2>&1 <<'PY'
import json, os, sys
out, vl8 = sys.argv[1], sys.argv[2]
c = {}
def chk(k, f):
    try:
        c[k] = f()
    except Exception as e:
        c[k] = f"ERROR {type(e).__name__}: {e}"[:300]
import torch, transformers
chk("torch", lambda: torch.__version__); chk("torch_cuda", lambda: torch.version.cuda)
chk("transformers", lambda: transformers.__version__)
chk("transformers_qwen3_vl", lambda: "qwen3_vl" in dir(transformers.models) and hasattr(transformers, "Qwen3VLForConditionalGeneration"))
import vllm
chk("vllm", lambda: vllm.__version__)
def reg():
    from vllm.model_executor.models.registry import ModelRegistry
    return "Qwen3VLForConditionalGeneration" in ModelRegistry.get_supported_archs()
chk("vllm_qwen3_vl", reg)
def verl_vl():
    import verl.models.transformers.qwen3_vl  # noqa: F401
    return True
chk("verl_qwen3_vl", verl_vl)
def proc():
    if not os.path.exists(os.path.join(vl8, ".fetch_ok")):
        return "skipped (Qwen3-VL-8B-Thinking not in)"
    from transformers import AutoProcessor
    return type(AutoProcessor.from_pretrained(vl8)).__name__
chk("processor_vl8b", proc)
json.dump(c, open(out, "w"), indent=1); print("CHECKS", json.dumps(c))
PY
python3 - "$D/checks.json" > $D/check_verdict.txt 2>> $LOG <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
bad = [k for k, v in c.items() if v is False or str(v).startswith("ERROR")]
if not str(c.get("vllm", "")).startswith("0.11.0"): bad.append("vllm version")
if not str(c.get("transformers", "")).startswith("4.57"): bad.append("transformers version")
if not str(c.get("torch", "")).startswith("2.8.0"): bad.append("torch version")
if str(c.get("torch_cuda")) != "12.8": bad.append("torch cuda")
print(" ".join(bad))
PY
[ -s $D/checks.json ] || finish FAIL "the checks did not run: $(tail -c 300 $LOG | tr '\n' ' ')"
BAD=$(cat $D/check_verdict.txt 2> /dev/null); [ -z "$BAD" ] || finish FAIL "checks: $BAD"
# 5. the venv's own scripts pointed at venv-vl, one rename into place
python3 - "$TMP" "$DST" <<'PY' >> $LOG 2>&1
import os, sys
old, new = sys.argv[1].encode(), sys.argv[2].encode()
paths = [os.path.join(sys.argv[1], "pyvenv.cfg")] + [os.path.join(sys.argv[1], "bin", f) for f in os.listdir(os.path.join(sys.argv[1], "bin"))]
for p in paths:   # text files only (activate scripts, console scripts' first lines, pyvenv.cfg); links and binaries left alone
    if os.path.islink(p) or not os.path.isfile(p):
        continue
    b = open(p, "rb").read()
    if b"\0" in b[:4096] or old not in b:
        continue
    open(p, "wb").write(b.replace(old, new)); print("REWRITTEN", os.path.basename(p))
PY
[ -z "$(grep -rIl -- "$TMP" "$TMP/bin" "$TMP/pyvenv.cfg" 2> /dev/null)" ] || finish FAIL "venv-vl.tmp still named in its scripts"
mv -- "$TMP" "$DST" || finish FAIL "rename to venv-vl failed"
V=$(CUDA_VISIBLE_DEVICES= $DST/bin/python -c 'import vllm, transformers; print(vllm.__version__, transformers.__version__)' 2>> $LOG) || finish FAIL "venv-vl does not start after the rename"
finish PASS "venv-vl built: vllm/transformers $V, torch $(python3 -c "import json; print(json.load(open('$D/checks.json'))['torch'])"), $(grep '^PROBE_MBPS' $LOG | tail -1)"
