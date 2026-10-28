#!/usr/bin/env bash
# FA-BUILD (runbook v2.5.6m, rules 2.5.10): flash-attn 2.8.3.post2 built from its PyPI source on this node, into
# $OPD_ROOT/venv-fa only. FA-PROBE (2026-10-17 20:23 UTC) installed the pinned release wheel, whose
# flash_attn_2_cuda .so needs GLIBC_2.32, newer than this node's C library; a build here links against the node's own.
# Started by the verb `fa-build`; no GPU (nvcc compiles on the CPU), nice 19, MAX_JOBS (default 8) x NVCC_THREADS
# (default 2) compiler threads; never touches the venv in use, torch or vLLM. Steps, each in $OPD_ROOT/tmp/fa_build/build.log:
#   1. venv-fa exists (FA-PROBE made it) and still has the stack the probe read (torch 2.8 / Python 3.10)
#   2. an nvcc >= 12.8 (sm_100 needs it): $CUDA_HOME, /usr/local/cuda-12.9, -12.8, /usr/local/cuda, then PATH
#      (v2.5.6q: none there -> NVIDIA's CUDA 12.8.1 redistributable archives, sha256-pinned, under $D/cuda; step 2b)
#   3. build tools inside venv-fa (ninja, packaging, wheel, setuptools, psutil, einops) with --no-deps when missing
#   4. the PyPI source archive, stopped if its sha256 differs; built with FLASH_ATTENTION_FORCE_BUILD=TRUE (setup.py
#      would otherwise fetch the same release wheel) and FLASH_ATTN_CUDA_ARCHS=100 (sm_100; B300 is sm_103, same major)
#   5. the old flash-attn replaced by the built wheel (--no-deps, no index); torch / vLLM unchanged; the import that
#      verl's remove-padding path makes must work (no GPU call: that is FA-PROBE's varlen step)
#   6. result.json + build.log tail to logs/<node>/fa_build/ in the private repository, an event fa_build_pass / _fail
# Afterwards `rerun FA-PROBE` runs the probe again: its install step is skipped once the import works, so only the
# varlen and 3-step smoke steps run (on a GPU, in queue order).
# [v2.5.9z104] (Fable 10-28, an H800 / Hopper node) FA_CUDA_ARCHS=<n>[;<n>...] (default 100): the architectures built and the test
# kernel's (the first); a Hopper node sets 90 (the B300's sm_100 build has no sm_90 code)
# Usage: bash fa_build.sh   (FA_BUILD_DIR, FA_MAX_JOBS, FA_NVCC_THREADS, FA_CUDA_CANDIDATES for tests)
set -uo pipefail
source "$(dirname "$0")/env.sh"
D=${FA_BUILD_DIR:-$OPD_ROOT/tmp/fa_build}; VF=$OPD_ROOT/venv-fa; LOG=$D/build.log; mkdir -p $D
ARCHS=${FA_CUDA_ARCHS:-100}; [[ $ARCHS =~ ^[0-9]+(;[0-9]+)*$ ]] || { echo "FA_BUILD FA_CUDA_ARCHS '$ARCHS' is not <n>[;<n>...]"; exit 2; }; ARCH1=${ARCHS%%;*}
SDIST=flash_attn-2.8.3.post2.tar.gz
URL=https://files.pythonhosted.org/packages/6c/b1/b77625e3d11260d2bd6a4fc249a4be9a3da97ff848e2222e810c4f0e0211/$SDIST
SHA=d0f9f0b08a986cdeadbffcf5ec8b3a8f514366c68ad1853d3ab061bc83cc340b
RJ=$D/result.json; echo '{}' > $RJ
step() { echo "FA_BUILD $(date -u +%T) $*" | tee -a $LOG; }
put() { python3 -c "import json, sys; p, k, v = sys.argv[1:4]; d = json.load(open(p)); d[k] = v; json.dump(d, open(p, 'w'), indent=1)" "$RJ" "$1" "$2"; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  put result "$1"; put reason "$2"; put finished_utc "$(date -u +%FT%TZ)"
  tail -c 20000 $LOG > $D/build_tail.log
  mkdir -p $D/report; cp $RJ $D/build_tail.log $D/report/
  python3 $RUNBOOK/scripts/remote.py put $D/report logs/$(python3 $RUNBOOK/scripts/control.py node)/fa_build >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event FA-BUILD "fa_build_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; exit 0
}
step "start"; put started_utc "$(date -u +%FT%TZ)"

# 1. venv-fa and its stack
[ -x $VF/bin/python ] || finish FAIL "no venv-fa (FA-PROBE builds it first)"
[ "$(readlink -f $VF)" != "$(readlink -f $OPD_ROOT/venv)" ] || finish FAIL "venv-fa resolves to the venv in use"
BEFORE=$($VF/bin/python $RUNBOOK/scripts/fa_probe.py versions 2>/dev/null); put before "$BEFORE"
echo "$BEFORE" | python3 -c "import json, sys; d = json.load(sys.stdin); sys.exit(0 if str(d.get('torch', '')).startswith('2.8.') and str(d.get('python', '')).startswith('3.10') else 1)" 2>/dev/null \
  || finish FAIL "venv-fa stack is not torch 2.8 / Python 3.10: $BEFORE"
step "venv-fa: $BEFORE"

# 2. nvcc >= 12.8
CH=; SEEN=
for c in ${FA_CUDA_CANDIDATES:-${CUDA_HOME:-} /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda $(dirname "$(dirname "$(command -v nvcc 2> /dev/null || echo /x/x)")")}; do
  [ -x "$c/bin/nvcc" ] || continue
  v=$("$c/bin/nvcc" --version 2> /dev/null | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1); SEEN="$SEEN $c=$v"
  case $v in 12.8|12.9|12.1[0-9]) CH=$c; break;; esac
done
put cuda_seen "${SEEN# }"
inst() {   # pip if present, else uv (the uv-built venv has no pip; FA-PROBE used uv), always --no-deps
  if $VF/bin/python -m pip --version > /dev/null 2>&1; then $VF/bin/python -m pip install --no-deps "$@" >> $LOG 2>&1
  elif command -v uv > /dev/null 2>&1; then uv pip install --python $VF/bin/python --no-deps "$@" >> $LOG 2>&1
  else return 1; fi; }
# 2b. no system nvcc >= 12.8 (B300 2026-10-18 05:15 UTC: /usr/local/cuda is 12.1, fa-build seq 96). v2.5.6q: NVIDIA's CUDA
# 12.8.1 redistributable archives (developer.download.nvidia.com/compute/cuda/redist, sha256 of redistrib_12.8.1.json
# pinned here): cuda_nvcc (nvcc, cicc, ptxas, crt), cuda_cudart, cuda_cccl, cuda_nvtx, unpacked under $D/redist and laid
# out as a CUDA_HOME in $D/cuda, with the cuBLAS / cuSPARSE / cuSOLVER / cuRAND headers of the nvidia-* packages torch
# installed in venv-fa (the same 12.8.1 versions); nothing outside $D changes. (v2.5.6n-p used the PyPI wheel
# nvidia-cuda-nvcc-cu12, which has ptxas but no nvcc: seq 99 "cuda/bin/nvcc: No such file or directory".) A test kernel
# (bf16, cuda::std) is compiled for sm_100 before the long build, so a toolchain that cannot work fails in seconds.
if [ -z "$CH" ] && [ "${FA_REDIST_CUDA:-1}" = 1 ]; then
  RU=${FA_REDIST_URL:-https://developer.download.nvidia.com/compute/cuda/redist}; RD=$D/redist; C=$D/cuda; mkdir -p $RD
  step "no system nvcc >= 12.8:${SEEN:- none}; CUDA 12.8.1 redistributable archives into $RD"
  MANF=${FA_REDIST_MANIFEST:-$D/redist.manifest}   # tests point this at their own archives
  [ -n "${FA_REDIST_MANIFEST:-}" ] || cat > $MANF <<'MAN'
cuda_nvcc cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz 9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3
cuda_cudart cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz 8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68
cuda_cccl cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz 0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b
cuda_nvtx cuda_nvtx/linux-x86_64/cuda_nvtx-linux-x86_64-12.8.90-archive.tar.xz 7755524026d30e2c211d793392166517e379724d19eb9e8e352693f7d5fa0c95
MAN
  while read -r comp rel sha; do
    f=$RD/$(basename $rel)
    if [ ! -f $f ] || [ "$(sha256sum $f | cut -c1-64)" != "$sha" ]; then
      curl -fsSL --retry 3 -o $f.part "$RU/$rel" && mv $f.part $f || finish FAIL "download failed: $comp"
    fi
    [ "$(sha256sum $f | cut -c1-64)" = "$sha" ] || { rm -f $f; finish FAIL "sha256 mismatch for $comp (redistrib_12.8.1.json $sha)"; }
    rm -rf $RD/$comp; mkdir -p $RD/$comp; tar -xJf $f -C $RD/$comp --strip-components=1 || finish FAIL "unpack failed: $comp"
  done < $MANF
  put redist_ok true
  SP=$($VF/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia
  rm -rf $C; mkdir -p $C/lib64 $C/targets/x86_64-linux
  cp -r $RD/cuda_nvcc/bin $C/bin || finish FAIL "nvcc copy failed"   # a copy, so nvcc.profile's TOP is $C
  chmod -R u+x $C/bin $RD/cuda_nvcc/nvvm/bin 2> /dev/null
  ln -s $RD/cuda_nvcc/nvvm $C/nvvm
  python3 - $C/include $RD/cuda_nvcc/include $RD/cuda_cudart/include $RD/cuda_cccl/include $RD/cuda_nvtx/include $SP/*/include <<'PY' || finish FAIL "CUDA_HOME include layout failed"
import os, sys
dst, srcs = sys.argv[1], sys.argv[2:]
for s in srcs:                       # one include tree of symlinks; the first archive / package that has a file wins
    for r, ds, fs in os.walk(s):
        rel = os.path.relpath(r, s)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for f in fs:
            t = os.path.join(dst, rel, f)
            if not os.path.lexists(t):
                os.symlink(os.path.join(r, f), t)
PY
  for l in $RD/cuda_nvcc/lib/* $RD/cuda_cudart/lib/*; do [ -e "$l" ] && ln -sf $l $C/lib64/; done
  ln -s $C/include $C/targets/x86_64-linux/include; ln -s $C/lib64 $C/targets/x86_64-linux/lib   # nvcc.profile's target dir
  VOUT=$($C/bin/nvcc --version 2>&1); echo "$VOUT" >> $LOG
  v=$(echo "$VOUT" | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)
  [ "$v" = 12.8 ] || finish FAIL "redistributable nvcc does not run or is not 12.8 (got '${v}'; mode $(stat -c %A $C/bin/nvcc 2> /dev/null); $(echo "$VOUT" | tail -c 300 | tr '\n' ' '))"
  printf '#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda/std/cstdint>\n__global__ void k(__nv_bfloat16 *x, cuda::std::int32_t n) { if (threadIdx.x < n) x[threadIdx.x] = __float2bfloat16(1.0f); }\nint main() { return 0; }\n' > $D/t.cu
  $C/bin/nvcc -arch=sm_$ARCH1 -I$C/include -L$C/lib64 -lcudart -o $D/t $D/t.cu >> $LOG 2>&1 || finish FAIL "CUDA 12.8 cannot compile a test kernel for sm_$ARCH1: $(tail -c 300 $LOG | tr '\n' ' ')"
  CH=$C; put cuda_from redist
fi
[ -n "$CH" ] || finish FAIL "no nvcc >= 12.8 found (CUDA_HOME, /usr/local/cuda-12.9, -12.8, /usr/local/cuda, PATH):${SEEN:- none}"
put cuda_home "$CH"; step "nvcc: $CH (${v:-})"

# 3. build tools in venv-fa
MISS=$($VF/bin/python -c "import importlib.util as u; print(' '.join(m for m in ('ninja', 'packaging', 'wheel', 'setuptools', 'psutil', 'einops') if u.find_spec(m) is None))")
if [ -n "$MISS" ]; then
  step "installing build tools into venv-fa: $MISS"; inst $MISS || finish FAIL "build tools not installed: $MISS"
fi
$VF/bin/python -c "import ninja, packaging, wheel, setuptools, psutil, einops" 2>> $LOG || finish FAIL "build tools not importable in venv-fa"

# 4. source + build
S=$D/src; mkdir -p $S
if [ ! -f $S/$SDIST ]; then
  curl -fsSL --retry 3 -o $S/$SDIST.part "$URL" && mv $S/$SDIST.part $S/$SDIST || finish FAIL "download failed: $SDIST"
fi
GOT=$(sha256sum $S/$SDIST | cut -c1-64)
[ "$GOT" = "$SHA" ] || { rm -f $S/$SDIST; finish FAIL "sha256 mismatch for $SDIST (PyPI $SHA, got $GOT)"; }
put sha256_ok true
rm -rf $S/flash_attn-2.8.3.post2 $D/dist; tar -xzf $S/$SDIST -C $S || finish FAIL "unpack failed"
step "building (MAX_JOBS=${FA_MAX_JOBS:-8}, NVCC_THREADS=${FA_NVCC_THREADS:-2}, sm_${ARCHS//;/ sm_})"; T0=$(date +%s)
( cd $S/flash_attn-2.8.3.post2 && env CUDA_HOME=$CH PATH=$CH/bin:$VF/bin:$PATH FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS="$ARCHS" \
    TORCH_CUDA_ARCH_LIST=10.0 MAX_JOBS=${FA_MAX_JOBS:-8} NVCC_THREADS=${FA_NVCC_THREADS:-2} \
    nice -n 19 $VF/bin/python setup.py bdist_wheel -d $D/dist ) >> $LOG 2>&1 || finish FAIL "build failed after $(( $(date +%s) - T0 )) s: $(tail -c 300 $LOG | tr '\n' ' ')"
W=$(ls $D/dist/flash_attn-*.whl 2> /dev/null | head -1)
[ -n "$W" ] || finish FAIL "build produced no wheel"
put build_seconds "$(( $(date +%s) - T0 ))"; put wheel "$(basename $W)"; step "built $(basename $W) in $(( $(date +%s) - T0 )) s"

# 5. replace the release wheel in venv-fa
if $VF/bin/python -m pip --version > /dev/null 2>&1; then
  $VF/bin/python -m pip install --no-deps --no-index --force-reinstall "$W" >> $LOG 2>&1
else
  uv pip install --python $VF/bin/python --no-deps --offline --reinstall "$W" >> $LOG 2>&1
fi || finish FAIL "installing the built wheel failed"
AFTER=$($VF/bin/python $RUNBOOK/scripts/fa_probe.py versions 2>/dev/null); put after "$AFTER"
t0=$(echo "$BEFORE" | python3 -c "import json, sys; d = json.load(sys.stdin); print(d.get('torch'), d.get('vllm'))")
t1=$(echo "$AFTER" | python3 -c "import json, sys; d = json.load(sys.stdin); print(d.get('torch'), d.get('vllm'))")
[ "$t0" = "$t1" ] || finish FAIL "torch/vLLM changed by the install: $t0 -> $t1"
FAIMP="from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis; from flash_attn import flash_attn_varlen_func"
$VF/bin/python -c "$FAIMP" 2> $D/import.err || finish FAIL "import failed after the build: $(tail -c 300 $D/import.err | tr '\n' ' ')"
finish PASS "built from source and importable in venv-fa (torch/vLLM unchanged: $t1); next: rerun FA-PROBE for varlen + smoke"
