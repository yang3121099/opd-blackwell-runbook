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
  -m) [ "$3" = --version ] && exit ${NOPIP:-0}
    for i in $(seq 1 $#); do [ "${!i}" = --target ] && { j=$((i + 1)); P=${!j}   # pip --target: a fake CUDA 12.8 nvcc wheel
      mkdir -p $P/nvidia/cuda_nvcc/bin $P/nvidia/cuda_nvcc/nvvm/bin $P/nvidia/cuda_nvcc/include/crt $P/nvidia/cuda_cccl/include/cuda/std
      printf '#!/usr/bin/env bash\necho "nvcc $*" >> $CALLS\n[ "$1" = --version ] && { echo "Cuda compilation tools, release ${FAKE_PIPNVCC:-12.8}, V12.8.93"; exit 0; }\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && : > "$2"; shift; done; exit ${NVCC_RC:-0}\n' > $P/nvidia/cuda_nvcc/bin/nvcc
      [ "${NOEXEC:-0}" = 1 ] || chmod +x $P/nvidia/cuda_nvcc/bin/nvcc; : > $P/nvidia/cuda_cccl/include/cuda/std/cstdint; : > $P/nvidia/cuda_nvcc/include/crt/host_config.h; }; done; exit 0;;
  setup.py) echo "build FORCE=$FLASH_ATTENTION_FORCE_BUILD ARCHS=$FLASH_ATTN_CUDA_ARCHS JOBS=$MAX_JOBS THREADS=$NVCC_THREADS CUDA=$CUDA_HOME" >> $CALLS
    mkdir -p "$4"; : > "$4/flash_attn-2.8.3.post2-cp310-cp310-linux_x86_64.whl"; exit ${BUILD_RC:-0};;
  -c) case "$2" in *find_spec*) echo "${MISSING:-}";; *sysconfig*) echo "$SITEP";; esac; exit 0;;
esac
SH
  chmod +x $T/cuda128/bin/nvcc $T/cuda124/bin/nvcc $T/root/venv-fa/bin/python
  export SITEP=$T/root/venv-fa/lib/site; mkdir -p $SITEP/nvidia/cuda_runtime/include $SITEP/nvidia/cuda_runtime/lib $SITEP/nvidia/cublas/include
  : > $SITEP/nvidia/cuda_runtime/include/cuda_runtime.h; : > $SITEP/nvidia/cuda_runtime/lib/libcudart.so.12; : > $SITEP/nvidia/cublas/include/cublas_v2.h
  export CALLS=$T/calls; : > $CALLS
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/tmp/fa_build/result.json "$1"; }
# 1. no nvcc >= 12.8: FAIL with what was seen, an event, the report to logs/<node>/fa_build
setup
FA_PIP_CUDA=0 FA_CUDA_CANDIDATES="$T/none $T/cuda124" bash $T/rb/scripts/fa_build.sh > /dev/null 2>&1
ok "old nvcc refused" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"no nvcc >= 12.8"* && "$(res reason)" == *"cuda124=12.4"* ]]'
ok "fail event + report" 'grep -q "^control event FA-BUILD fa_build_fail" $CALLS && grep -q "^remote put $T/root/tmp/fa_build/report logs/abcd1234/fa_build" $CALLS'
ok "no build tried" '! grep -q "^build " $CALLS'
rm -rf $T
# 1b. v2.5.6n: only an old system nvcc -> CUDA 12.8 from pip wheels into $D/cuda_pkgs, laid out as $D/cuda; a test kernel first
setup
D=$T/root/tmp/fa_build
NVCC_RC=1 FA_CUDA_CANDIDATES="$T/cuda124" bash $T/rb/scripts/fa_build.sh > /dev/null 2>&1
ok "pip CUDA wheels pinned, --no-deps, --target" 'grep -q "^py -m pip install --no-deps --target $D/cuda_pkgs nvidia-cuda-nvcc-cu12==12.8.93 nvidia-cuda-cccl-cu12==12.8.90" $CALLS'
ok "CUDA_HOME layout" '[ -x $D/cuda/bin/nvcc ] && [ -L $D/cuda/nvvm ] && [ -L $D/cuda/include/cuda_runtime.h ] && [ -L $D/cuda/include/cublas_v2.h ] && [ -L $D/cuda/include/cuda/std/cstdint ] && [ -L $D/cuda/lib64/libcudart.so ]'
ok "test kernel for sm_100 first" 'grep -q "^nvcc -arch=sm_100 -I$D/cuda/include -L$D/cuda/lib64 -lcudart -o $D/t $D/t.cu" $CALLS'
ok "a failing test kernel stops before the build" '[ "$(res result)" = FAIL ] && [[ "$(res reason)" == *"cannot compile a test kernel"* ]] && ! grep -q "^build " $CALLS'
rm -rf $T
setup
D=$T/root/tmp/fa_build; mkdir -p $D/src; echo junk > $D/src/flash_attn-2.8.3.post2.tar.gz
NOEXEC=1 FA_CUDA_CANDIDATES="$T/cuda124" bash $T/rb/scripts/fa_build.sh > /dev/null 2>&1   # v2.5.6p: a wheel unpacked without +x
ok "pip CUDA becomes CUDA_HOME" '[ "$(res cuda_home)" = $D/cuda ] && [ "$(res cuda_from)" = pip ] && [[ "$(res reason)" == *"sha256 mismatch"* ]]'
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
