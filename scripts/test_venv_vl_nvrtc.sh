#!/usr/bin/env bash
# [v2.5.9z31] Stub tests of venv_vl_nvrtc.sh (no network, GPU or HF), in 9y1's manner: a real throwaway venv-vl holding fake torch /
# flash-attn / nvidia-cuda-nvrtc-cu12 distributions and a fake pip that "installs" the NVRTC wheel inside the venv it runs in.
# Checked: a PASS installs only the pin, offline and without dependencies, keeps torch / flash-attn, writes `pip check` at the top of
# requirements.lock, uploads lock + manifest privately, leaves venv-fa as it was; a running venv-vl python waits; a changed torch, a
# failed wheel fetch, a linked venv-vl or a pin outside 12.9 are FAILs without .done. Usage: bash scripts/test_venv_vl_nvrtc.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); T=$(cd $T && pwd -P); mkdir -p $T/rb/scripts $T/root/runs $T/root/venv-fa/lib
  cp $SRC/scripts/venv_vl_nvrtc.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  printf 'import json, os, sys\nopen(os.environ["CALLS"], "a").write("wheel_fetch " + " ".join(sys.argv[1:]) + " PARFIRST=" + os.environ.get("WHEEL_PAR_FIRST", "") + " CONNS=" + os.environ.get("FETCH_CONNS", "") + "\\n")\nos.makedirs(sys.argv[1], exist_ok=True)\nif os.environ.get("FAKE_WHEEL_FAIL"):\n    print("WHEEL nvidia-cuda-nvrtc-cu12 sha256 mismatch"); sys.exit(1)\njson.dump({"nvidia-cuda-nvrtc-cu12": {"version": "12.9.86", "sha256": "210cf05005a447e2"}}, open(os.path.join(sys.argv[1], "manifest.json"), "w"))\n' > $T/rb/scripts/wheel_fetch.py
  echo "venv-fa stays" > $T/root/venv-fa/lib/marker
  # [ops-lead's A1 conditions] the live environments' NVRTC (read only; their sha256 before and after) and a "live" target for links
  for e in venv venv-fa; do mkdir -p $T/root/$e/lib/python3.x/site-packages/nvidia/cuda_nvrtc/lib; echo "$e nvrtc 12.8" > $T/root/$e/lib/python3.x/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12; done
  python3 -m venv --without-pip $T/root/venv-vl
  SP=$($T/root/venv-vl/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  dist() { mkdir -p $SP/$1-$2.dist-info; printf 'Metadata-Version: 2.1\nName: %s\nVersion: %s\n' $1 $2 > $SP/$1-$2.dist-info/METADATA; }
  dist torch 2.8.0+cu128; dist flash_attn 2.8.3; dist nvidia_cuda_nvrtc_cu12 12.8.93
  mkdir -p $SP/nvidia/cuda_nvrtc/lib $SP/pip; echo 12.8 > $SP/nvidia/cuda_nvrtc/lib/libnvrtc.so.12; : > $SP/pip/__init__.py
  cat > $SP/pip/__main__.py <<'PY'
import glob, os, shutil, sys
open(os.environ["CALLS"], "a").write("pip " + " ".join(sys.argv[1:]) + " in " + sys.prefix + "\n")
a = sys.argv[1:]; sp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def dist(name, ver):
    for d in glob.glob(os.path.join(sp, name + "-*.dist-info")):
        shutil.rmtree(d)
    os.makedirs(os.path.join(sp, f"{name}-{ver}.dist-info"))
    open(os.path.join(sp, f"{name}-{ver}.dist-info", "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
if a[:1] == ["--version"]:
    print("pip 24.0"); sys.exit(0)
if a[:1] == ["check"]:
    print("torch 2.8.0+cu128 has requirement nvidia-cuda-nvrtc-cu12==12.8.93; sys_platform == \"linux\", but you have nvidia-cuda-nvrtc-cu12 12.9.86."); sys.exit(1)
if a[:1] == ["freeze"]:
    print("nvidia-cuda-nvrtc-cu12==12.9.86\ntorch==2.8.0+cu128"); sys.exit(0)
if a[:1] == ["install"]:
    dist("nvidia_cuda_nvrtc_cu12", a[-1].split("==")[1]); open(os.path.join(sp, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12"), "w").write("12.9\n")
    if os.environ.get("FAKE_TORCH_BUMP"):
        dist("torch", "2.9.0+cu129")
    if os.environ.get("FAKE_CONTAMINATE"):   # a write through to venv-fa's NVRTC
        open(os.environ["FAKE_CONTAMINATE"], "a").write("12.9\n")
    sys.exit(0)
sys.exit(2)
PY
  export CALLS=$T/calls; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/runs/VENV-VL-NVRTC/report.json "$1"; }
D=runs/VENV-VL-NVRTC
# 1. PASS
setup; H0=$(cat $T/root/venv-fa/lib/marker); bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1
ok "PASS: the pinned wheel fetched (sha256 by wheel_fetch) and installed offline, --no-deps, in venv-vl only" '[ "$(res result)" = PASS ] && grep -q "^wheel_fetch $T/root/$D/wheels nvidia-cuda-nvrtc-cu12==12.9.86 PARFIRST=1 CONNS=16$" $CALLS && grep -q "^pip install --no-index --find-links $T/root/$D/wheels --no-deps nvidia-cuda-nvrtc-cu12==12.9.86 in $T/root/venv-vl$" $CALLS && [ "$(grep -c "^pip install" $CALLS)" = 1 ]'
ok "checks: torch 2.8.0+cu128, flash-attn 2.8.3, the wheel at 12.9.86, libnvrtc.so.12 (a relative path)" 'python3 -c "import json, sys; c = json.load(open(sys.argv[1]))[\"checks\"]; assert c[\"torch\"] == \"2.8.0+cu128\" and c[\"flash_attn\"] == \"2.8.3\" and c[\"nvrtc_wheel\"] == \"12.9.86\" and c[\"libnvrtc\"] == [\"nvidia/cuda_nvrtc/lib/libnvrtc.so.12\"] and c[\"pass\"], c" $T/root/$D/report.json'
ok "requirements.lock: pip check verbatim on top (the torch-wants-12.8 line kept, not a failure), then pip freeze; manifest with the sha256" 'head -1 $T/root/$D/requirements.lock | grep -q "^# pip check" && sed -n 2p $T/root/$D/requirements.lock | grep -q "has requirement nvidia-cuda-nvrtc-cu12==12.8.93" && grep -q "^# pip freeze" $T/root/$D/requirements.lock && grep -q 210cf05005a447e2 $T/root/$D/manifest.json'
ok "lock, manifest and report to logs/<node>/venv_vl_nvrtc (private), event pass, .done; venv-fa untouched" 'grep -q "^remote put $T/root/$D/up logs/abcd1234/venv_vl_nvrtc$" $CALLS && [ -f $T/root/$D/up/requirements.lock ] && [ -f $T/root/$D/up/manifest.json ] && grep -q "^control event VENV-VL-NVRTC venv_vl_nvrtc_pass" $CALLS && [ -f $T/root/$D/.done ] && [ "$(cat $T/root/venv-fa/lib/marker)" = "$H0" ] && ! grep -q "venv-fa" $CALLS'
rm -rf $T
# 2. a venv-vl python running: waiting, nothing installed
setup; ( exec -a "$T/root/venv-vl/bin/python" sleep 5 ) & SP=$!; sleep 0.3
bash $T/rb/scripts/venv_vl_nvrtc.sh > $T/o 2>&1; rc=$?; kill $SP 2> /dev/null; wait $SP 2> /dev/null
ok "a venv-vl python running: WAIT_INPUTS, exit 1, no fetch / install, no .done" '[ $rc = 1 ] && grep -q "^WAIT_INPUTS VENV-VL-NVRTC: a venv-vl python is running" $T/o && ! grep -q "^wheel_fetch\|^pip install" $CALLS && [ ! -f $T/root/$D/.done ]'
rm -rf $T
# 3. FAILs, no .done
check() { ok "$1" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"'"$2"'"* ]] && [ ! -f $T/root/$D/.done ]'; }
setup; FAKE_TORCH_BUMP=1 bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1; check "torch changed by the install is FAIL" "2.9.0+cu129"; rm -rf $T
setup; FAKE_WHEEL_FAIL=1 bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1; check "a failed wheel fetch is FAIL (and nothing installed)" "wheel: WHEEL nvidia-cuda-nvrtc-cu12 sha256 mismatch"; ok "no install after a failed fetch" '! grep -q "^pip install" $CALLS'; rm -rf $T
setup; mv $T/root/venv-vl $T/vvl; ln -s $T/vvl $T/root/venv-vl; bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1; check "a venv-vl that is a link is FAIL, untouched" "venv-vl with bin/python"; rm -rf $T
setup; NVRTC_PIN=nvidia-cuda-nvrtc-cu12==12.8.93 bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1; check "a pin outside 12.9 is FAIL" "is not nvidia-cuda-nvrtc-cu12==12.9.<x>"; rm -rf $T
# 4. [ops-lead's A1 conditions] (a) a link out of venv-vl under nvidia/cuda_nvrtc: FAIL naming it, nothing fetched or installed;
#    (b) venv-fa's libnvrtc changed during the install: FAIL and the event venv_contamination; the guard in the PASS report
setup; bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1
ok "[cond] PASS: guard.json has no outside path, venv / venv-fa NVRTC sha256 before = after, links recorded (none)" 'python3 -c "import json, sys; g = json.load(open(sys.argv[1]))[\"guard\"]; assert g[\"outside\"] == [] and g[\"others_unchanged\"] and len(g[\"sha_pre\"]) == 2 and g[\"links\"][\"venv-fa\"][\"cuda_nvrtc\"] is False and g[\"checked_paths\"] >= 5, g" $T/root/$D/report.json'
rm -rf $T
setup; SPV=$($T/root/venv-vl/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); rm -rf $SPV/nvidia/cuda_nvrtc; ln -s $T/root/venv/lib/python3.x/site-packages/nvidia/cuda_nvrtc $SPV/nvidia/cuda_nvrtc
H0=$(cat $T/root/venv/lib/python3.x/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12); bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1
check "[cond a] venv-vl's nvidia/cuda_nvrtc a link into venv: FAIL naming it" "paths outside venv-vl, nothing installed: "; ok "[cond a] the path named, nothing fetched or installed, venv's file as it was" '[[ "$(res reason)" == *"nvidia/cuda_nvrtc -> "*"/venv/lib/python3.x/site-packages/nvidia/cuda_nvrtc"* ]] && ! grep -q "^wheel_fetch\|^pip install" $CALLS && [ "$(cat $T/root/venv/lib/python3.x/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12)" = "$H0" ]'
rm -rf $T
setup; SPV=$($T/root/venv-vl/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); rm $SPV/nvidia/cuda_nvrtc/lib/libnvrtc.so.12; ln -s $T/root/venv-fa/lib/python3.x/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 $SPV/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1
check "[cond a] one file under cuda_nvrtc a link into venv-fa: FAIL naming that file" "venv-fa/lib/python3.x/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"; ok "[cond a] nothing installed" '! grep -q "^pip install" $CALLS'
rm -rf $T
setup; FAKE_CONTAMINATE=$T/root/venv-fa/lib/python3.x/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12 bash $T/rb/scripts/venv_vl_nvrtc.sh > /dev/null 2>&1
check "[cond b] venv-fa's libnvrtc changed during the install: FAIL" "venv / venv-fa libnvrtc sha256 changed during the install (venv_contamination)"; ok "[cond b] the event venv_contamination" 'grep -q "^control event VENV-VL-NVRTC venv_contamination" $CALLS'
rm -rf $T
[ $fails = 0 ] && echo "VENV_VL_NVRTC_TEST PASS ($n checks)" || echo "VENV_VL_NVRTC_TEST FAIL ($fails of $n)"
