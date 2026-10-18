#!/usr/bin/env bash
# Stub tests of fa_build.sh (runbook v2.5.6m; no GPU, nvcc, network or HF): a fake venv-fa python and fake nvcc.
# The build path needs the real source archive (sha256-pinned): FA_SDIST_FILE=<flash_attn-2.8.3.post2.tar.gz>, else
# those checks are skipped. Usage: [FA_SDIST_FILE=...] bash scripts/test_fa_build.sh -> FA_BUILD_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/venv/bin $T/root/venv-fa/bin $T/cuda128/bin $T/cuda124/bin
  cp $SRC/scripts/fa_build.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  : > $T/rb/scripts/fa_probe.py
  printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.8, V12.8.93"\n' > $T/cuda128/bin/nvcc
  printf '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.4, V12.4.131"\n' > $T/cuda124/bin/nvcc
  cat > $T/root/venv-fa/bin/python <<'SH'
#!/usr/bin/env bash
echo "py $*" >> $CALLS
case "$1" in
  */fa_probe.py) echo "{\"python\": \"3.10.22\", \"torch\": \"${FAKE_TORCH:-2.8.0+cu128}\", \"vllm\": \"0.10.2\"}";;
  -m) [ "$3" = --version ] && exit ${NOPIP:-0}; exit 0;;
  setup.py) echo "build FORCE=$FLASH_ATTENTION_FORCE_BUILD ARCHS=$FLASH_ATTN_CUDA_ARCHS JOBS=$MAX_JOBS THREADS=$NVCC_THREADS CUDA=$CUDA_HOME" >> $CALLS
    mkdir -p "$4"; : > "$4/flash_attn-2.8.3.post2-cp310-cp310-linux_x86_64.whl"; exit ${BUILD_RC:-0};;
  -c) case "$2" in *find_spec*) echo "${MISSING:-}";; esac; exit 0;;
esac
SH
  chmod +x $T/cuda128/bin/nvcc $T/cuda124/bin/nvcc $T/root/venv-fa/bin/python
  export CALLS=$T/calls; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/tmp/fa_build/result.json "$1"; }
# 1. no nvcc >= 12.8: FAIL with what was seen, an event, the report to logs/<node>/fa_build
setup
FA_CUDA_CANDIDATES="$T/none $T/cuda124" bash $T/rb/scripts/fa_build.sh > /dev/null 2>&1
ok "old nvcc refused" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"no nvcc >= 12.8"* && "$(res reason)" == *"cuda124=12.4"* ]]'
ok "fail event + report" 'grep -q "^control event FA-BUILD fa_build_fail" $CALLS && grep -q "^remote put $T/root/tmp/fa_build/report logs/abcd1234/fa_build" $CALLS'
ok "no build tried" '! grep -q "^build " $CALLS'
rm -rf $T
# 2. a stack other than torch 2.8 is refused before anything is installed
setup
FAKE_TORCH=2.7.1 FA_CUDA_CANDIDATES=$T/cuda128 bash $T/rb/scripts/fa_build.sh > /dev/null 2>&1
ok "stack refused" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"not torch 2.8"* ]] && ! grep -q "pip install" $CALLS'
rm -rf $T
# 3. a source archive with the wrong sha256 is deleted and nothing is built
setup
mkdir -p $T/root/tmp/fa_build/src; echo junk > $T/root/tmp/fa_build/src/flash_attn-2.8.3.post2.tar.gz
FA_CUDA_CANDIDATES=$T/cuda128 bash $T/rb/scripts/fa_build.sh > /dev/null 2>&1
ok "sha256 mismatch" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"sha256 mismatch"* ]] && [ ! -f $T/root/tmp/fa_build/src/flash_attn-2.8.3.post2.tar.gz ] && ! grep -q "^build " $CALLS'
ok "venv in use untouched" '! grep -q "$T/root/venv/" $CALLS'
rm -rf $T
# 4. the build path (real archive): forced source build for sm_100 only, missing tools with --no-deps, wheel reinstalled
if [ -f "${FA_SDIST_FILE:-}" ]; then
  setup
  mkdir -p $T/root/tmp/fa_build/src; cp "$FA_SDIST_FILE" $T/root/tmp/fa_build/src/flash_attn-2.8.3.post2.tar.gz
  MISSING="ninja psutil" FA_MAX_JOBS=6 FA_CUDA_CANDIDATES="$T/cuda124 $T/cuda128" bash $T/rb/scripts/fa_build.sh > /dev/null 2>&1
  ok "build PASS" '[ "$(res result)" = PASS ] && [ "$(res cuda_home)" = $T/cuda128 ] && grep -q "^control event FA-BUILD fa_build_pass" $CALLS'
  ok "forced sm_100 build" 'grep -q "^build FORCE=TRUE ARCHS=100 JOBS=6 THREADS=2 CUDA=$T/cuda128" $CALLS'
  ok "tools --no-deps" 'grep -q "^py -m pip install --no-deps ninja psutil" $CALLS'
  ok "wheel reinstalled offline" 'grep -q "^py -m pip install --no-deps --no-index --force-reinstall $T/root/tmp/fa_build/dist/flash_attn-2.8.3.post2-cp310-cp310-linux_x86_64.whl" $CALLS'
  rm -rf $T
else
  echo "  (build path skipped: FA_SDIST_FILE not set)"
fi
[ $fails = 0 ] && echo "FA_BUILD_TEST PASS ($n checks)" || echo "FA_BUILD_TEST FAIL ($fails of $n)"
