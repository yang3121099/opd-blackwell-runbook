#!/usr/bin/env bash
# [v2.5.9y1] Stub tests of build_venv_vl.sh (no network, GPU or HF): a real throwaway venv-fa holding fake torch / transformers /
# vllm / verl / pip packages; the fake pip "upgrades" vllm and transformers inside the venv it runs in. Checked: a PASS builds
# venv-vl from a copy and leaves venv-fa as it was; an existing venv-vl is left alone; a failed check leaves no venv-vl; the only
# path removed is venv-vl.tmp (a link there is refused); VENV_VL_REBUILD=1 rebuilds. Usage: bash scripts/test_build_venv_vl.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); T=$(cd $T && pwd -P); mkdir -p $T/rb/scripts $T/root/runs $T/root/models
  cp $SRC/scripts/build_venv_vl.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  # wheel_fetch: a manifest and a probe line, or a failure
  printf 'import json, os, sys\nopen(os.environ["CALLS"], "a").write("wheel_fetch " + " ".join(sys.argv[1:]) + "\\n")\nos.makedirs(sys.argv[1], exist_ok=True)\nif os.environ.get("FAKE_WHEEL_FAIL"):\n    print("WHEEL vllm-0.11.0.whl 1 ab sha256 00 != PyPI 11"); sys.exit(1)\njson.dump({"wheels": [{"name": p.split("==")[0], "sha256": "ab" * 32} for p in sys.argv[2:]], "probe_MBps": 0.4, "ok": True}, open(os.path.join(sys.argv[1], "manifest.json"), "w"))\nprint("PROBE_MBPS 0.4")\n' > $T/rb/scripts/wheel_fetch.py
  python3 -m venv --without-pip $T/root/venv-fa
  SP=$($T/root/venv-fa/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  mkdir -p $SP/torch $SP/transformers/models $SP/vllm/model_executor/models $SP/verl/models/transformers $SP/pip $SP/torch-2.8.0.dist-info
  printf '__version__ = "2.8.0+cu128"\nfrom . import version\n' > $SP/torch/__init__.py; printf 'cuda = "12.8"\n' > $SP/torch/version.py
  printf 'Metadata-Version: 2.1\nName: torch\nVersion: 2.8.0+cu128\n' > $SP/torch-2.8.0.dist-info/METADATA
  printf '__version__ = "4.56.2"\nfrom . import models\n' > $SP/transformers/__init__.py; : > $SP/transformers/models/__init__.py
  printf '__version__ = "0.10.2"\n' > $SP/vllm/__init__.py; : > $SP/vllm/model_executor/__init__.py; : > $SP/vllm/model_executor/models/__init__.py
  printf 'class ModelRegistry:\n    @staticmethod\n    def get_supported_archs():\n        return ["Qwen2VLForConditionalGeneration"]\n' > $SP/vllm/model_executor/models/registry.py
  : > $SP/verl/__init__.py; : > $SP/verl/models/__init__.py; : > $SP/verl/models/transformers/__init__.py; : > $SP/verl/models/transformers/qwen3_vl.py
  : > $SP/pip/__init__.py; mkdir -p $SP/flash_attn; printf '__version__ = "2.8.3.post2"\n' > $SP/flash_attn/__init__.py
  cat > $SP/pip/__main__.py <<'PY'
import os, sys
open(os.environ["CALLS"], "a").write("pip " + " ".join(sys.argv[1:]) + " in " + sys.prefix + "\n")
a = sys.argv[1:]; sp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if a[:1] == ["--version"]:
    print("pip 24.0"); sys.exit(0)
if a[:1] == ["freeze"]:
    print("vllm==fake"); sys.exit(0)
if a[:1] == ["install"] and "--no-index" in a and not os.environ.get("FAKE_NO_UPGRADE"):
    open(os.path.join(sp, "vllm", "__init__.py"), "w").write('__version__ = "0.11.0"\n')
    open(os.path.join(sp, "vllm", "model_executor", "models", "registry.py"), "w").write(
        'class ModelRegistry:\n    @staticmethod\n    def get_supported_archs():\n        return ["Qwen3VLForConditionalGeneration"]\n')
    open(os.path.join(sp, "transformers", "__init__.py"), "w").write('__version__ = "4.57.1"\nfrom . import models\nQwen3VLForConditionalGeneration = object\n')
    open(os.path.join(sp, "transformers", "models", "__init__.py"), "w").write("qwen3_vl = None\n")
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(sp))), "bin"), exist_ok=True)
    open(os.path.join(sys.prefix, "bin", "vllm"), "w").write("#!" + sys.prefix + "/bin/python\nprint(1)\n")
sys.exit(0)
PY
  export CALLS=$T/calls; : > $CALLS
  FA_SUM=$(cd $SP && cat vllm/__init__.py transformers/__init__.py | cksum)
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/BUILD-VENV-VL/report.json "$1"; }
spsum() { (cd $SP && cat vllm/__init__.py transformers/__init__.py | cksum); }
VL=venv-vl
# 1. PASS: venv-vl from a copy of venv-fa's site-packages, upgraded there only, renamed into place, scripts pointing at venv-vl
setup
bash $T/rb/scripts/build_venv_vl.sh > /dev/null 2>&1
ok "PASS: venv-vl built (vllm 0.11.0, transformers 4.57.1), no venv-vl.tmp left, .done, event, report to logs/<node>/venv_vl" '[ "$(res result)" = PASS ] && [ -x $T/root/$VL/bin/python ] && [ ! -e $T/root/$VL.tmp ] && [ -f $T/root/runs/BUILD-VENV-VL/.done ] && grep -q "^control event BUILD-VENV-VL venv_vl_built" $CALLS && grep -q "^remote put $T/root/runs/BUILD-VENV-VL/up logs/abcd1234/venv_vl" $CALLS && [[ "$(res reason)" == *"0.11.0 4.57.1"* ]]'
ok "venv-fa untouched (its vllm / transformers as before); pip ran in venv-vl.tmp only" '[ "$(spsum)" = "$FA_SUM" ] && ! grep "^pip " $CALLS | grep -v " in $T/root/$VL.tmp$" | grep -q . && grep -q "^pip install --no-index --find-links $T/root/runs/BUILD-VENV-VL/wheels --no-deps vllm==0.11.0 transformers==4.57.1 qwen-vl-utils==0.0.14 mathruler==0.1.0 in" $CALLS'
ok "venv-vl's own scripts point at venv-vl (activate, a console script), none at venv-vl.tmp; the checks are in the report" '! grep -rIl "$VL.tmp" $T/root/$VL/bin $T/root/$VL/pyvenv.cfg 2> /dev/null | grep -q . && grep -q "$T/root/$VL" $T/root/$VL/bin/activate && head -1 $T/root/$VL/bin/vllm | grep -q "^#!$T/root/$VL/bin/python$" && [[ "$(res checks)" == *"Qwen"* || "$(res checks)" == *"True"* ]] && [ -s $T/root/runs/BUILD-VENV-VL/requirements.lock ]'
# 2. an existing venv-vl is left alone (EXISTS, nothing pulled)
rm -f $T/root/runs/BUILD-VENV-VL/.done; : > $CALLS; echo keep > $T/root/$VL/marker
bash $T/rb/scripts/build_venv_vl.sh > /dev/null 2>&1
ok "an existing venv-vl: EXISTS, untouched, no pull" '[ "$(res result)" = EXISTS ] && [ -f $T/root/$VL/marker ] && ! grep -q "^wheel_fetch" $CALLS'
# 3. VENV_VL_REBUILD=1: the old one moved to venv-vl.tmp, removed there, built again
rm -f $T/root/runs/BUILD-VENV-VL/.done; : > $CALLS
VENV_VL_REBUILD=1 bash $T/rb/scripts/build_venv_vl.sh > /dev/null 2>&1
ok "VENV_VL_REBUILD=1 builds again (the old marker gone)" '[ "$(res result)" = PASS ] && [ ! -f $T/root/$VL/marker ] && [ -x $T/root/$VL/bin/python ] && grep -q "^wheel_fetch" $CALLS'
cd $SRC; rm -rf $T
# 4. a check that fails (no upgrade: vllm stays 0.10.2): FAIL, no venv-vl, no .done
setup
FAKE_NO_UPGRADE=1 bash $T/rb/scripts/build_venv_vl.sh > /dev/null 2>&1
ok "vllm still 0.10.2 after the install: FAIL on the checks, no venv-vl, event venv_vl_failed" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"vllm version"* && "$(res reason)" == *"vllm_qwen3_vl"* ]] && [ ! -e $T/root/$VL ] && [ ! -f $T/root/runs/BUILD-VENV-VL/.done ] && grep -q "^control event BUILD-VENV-VL venv_vl_failed" $CALLS && [ "$(spsum)" = "$FA_SUM" ]'
# 5. a wheel that fails its sha256: FAIL before any install
: > $CALLS; FAKE_WHEEL_FAIL=1 bash $T/rb/scripts/build_venv_vl.sh > /dev/null 2>&1
ok "a wheel failing its sha256: FAIL, nothing installed; the leftover venv-vl.tmp of the run before was removed and rebuilt" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == wheels:* ]] && ! grep -q "^pip install" $CALLS'
cd $SRC; rm -rf $T
# [v2.5.9z3] no flash_attn in the copy: FAIL on the checks
setup; rm -rf $SP/flash_attn; bash $T/rb/scripts/build_venv_vl.sh > /dev/null 2>&1
ok "[v2.5.9z3] flash_attn missing: FAIL (flash_attn version)" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"flash_attn"* ]] && [ ! -e $T/root/$VL ]'
cd $SRC; rm -rf $T
# 6. venv-vl.tmp is a link to elsewhere: refused, the link target kept
setup; mkdir -p $T/elsewhere; echo precious > $T/elsewhere/f; ln -s $T/elsewhere $T/root/$VL.tmp
bash $T/rb/scripts/build_venv_vl.sh > /dev/null 2>&1
ok "venv-vl.tmp a link: FAIL, the target untouched" '[ "$(res result)" = FAIL ] && [ -f $T/elsewhere/f ] && [[ "$(res reason)" == *"link"* ]]'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "BUILD_VENV_VL_TEST PASS ($n checks)" || echo "BUILD_VENV_VL_TEST FAIL ($fails of $n)"
