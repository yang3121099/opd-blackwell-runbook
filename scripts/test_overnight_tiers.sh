#!/usr/bin/env bash
# [v2.5.9z104] overnight.sh's vLLM tier block (step 6) and its move path (step 0) on stubs: OVERNIGHT_TIER_A_ONLY=1 (a Hopper node) fails at a failed tier A without
# trying B / C (no ptxas wheel installed, no CUDA 13 build); unset, a failed A goes on to B as before; a passing A is A either way.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d)
python3 - "$SRC/scripts/overnight.sh" "$T/tiers.sh" <<'PY'
import sys
s = open(sys.argv[1]).read()
i = s.index("rm -f $OPD_ROOT/triton.env; for v in $TIER_VARS")
j = s.index("\n", s.index('step 6 "vLLM OK via tier', i)) + 1
open(sys.argv[2], "w").write(s[i:j])
PY
run() {   # <A passes 0|1> [env]: the block with every outside call stubbed; CALLS lists what it ran
  ( export OPD_ROOT=$T/opd CALLS=$T/calls A_OK=$1; shift; export "$@" 2> /dev/null; mkdir -p $OPD_ROOT/venv; : > $CALLS
    TIER_VARS=TRITON_PTXAS_PATH; LOG=$T/log
    before_old() { return 1; }
    # [v3.13b] no gpu_cap stub: the block must read the capability by itself (overnight.sh does not source env.sh); a fake nvidia-smi
    mkdir -p $T/bin; printf '#!/bin/bash\ncase "$*" in *compute_cap*) [ -n "$CAP" ] && echo "$CAP";; -L) [ -n "$CAP" ] && echo "GPU 0: fake";; esac\nexit 0\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi; PATH=$T/bin:$PATH
    try_tier() { echo "try $1" >> $CALLS; [ $1 = A ] && [ $A_OK = 1 ]; }
    uv() { echo "uv $*" >> $CALLS; }; pip() { echo "pip $*" >> $CALLS; }; nvcc() { return 1; }
    step() { echo "STEP $1 $2" >> $CALLS; }; fail() { echo "FAIL $1 $2" >> $CALLS; exit 1; }
    source $T/tiers.sh ) > /dev/null 2>&1; }
run 0 OVERNIGHT_TIER_A_ONLY=1; rc=$?
ok "A fails, A only: step 6 fails, B / C never tried, nothing installed" '[ $rc = 1 ] && grep -q "^FAIL 6 tier A vLLM probe FAILED and OVERNIGHT_TIER_A_ONLY=1" $T/calls && ! grep -qE "^try B|^uv |^pip " $T/calls'
run 0; rc=$?
ok "A fails, flag unset: the ptxas wheel and tier B as before" 'grep -q "^uv pip install -q nvidia-cuda-nvcc-cu12==12.9.86" $T/calls'
run 1 OVERNIGHT_TIER_A_ONLY=1; rc=$?
ok "A passes: tier A, nothing installed" '[ $rc = 0 ] && grep -q "^STEP 6 vLLM OK via tier A: cu128 as is" $T/calls && ! grep -qE "^uv |^pip " $T/calls'
# [v3.12] FAST_START on a Hopper card: tier A without a probe; on a Blackwell card the probe runs as before
run 0 FAST=1 CAP=9.0; rc=$?
ok "fast start, Hopper: tier A is taken without a probe" '[ $rc = 0 ] && grep -q "^STEP 6 vLLM OK via tier A: cu128 as is (not probed" $T/calls && ! grep -q "^try " $T/calls'
run 1 FAST=1 CAP=10.3; rc=$?
ok "fast start, Blackwell: the probe runs" '[ $rc = 0 ] && grep -q "^try A" $T/calls'
run 0 FAST=1; rc=$?
ok "fast start, no GPU (a preparation machine): tier A without a probe" '[ $rc = 0 ] && grep -q "^STEP 6 vLLM OK via tier A: cu128 as is (not probed: no GPU" $T/calls && ! grep -q "^try " $T/calls'
run 0 CAP=; rc=$?
ok "no GPU without fast start: the probe runs as before" 'grep -q "^try A" $T/calls'
# the move path (step 0, FROM_OPD_ROOT + LINK_VENV): the old tier read unless A only; the venv linked, parquet copied, the old root unchanged
python3 - "$SRC/scripts/overnight.sh" "$T/move.sh" <<'PY'
import sys
s = open(sys.argv[1]).read()
i = s.index('OLD_TIER=""\nif [ -n "${FROM_OPD_ROOT:-}" ]')
j = s.index("\ntier_rank(){", i) + 1
open(sys.argv[2], "w").write(s[i:j])
PY
F=$T/old; mkdir -p $F/venv/bin $F/G-OPD $F/data; : > $F/venv/bin/activate; : > $F/data/train.parquet
echo "# written by overnight.sh, tier B, 2026-10-20T00:00:00Z; sourced by env.sh. Delete it to undo." > $F/triton.env
before=$(cd $F && find . | sort | md5sum 2> /dev/null || find . | sort | md5)
domove() { ( export OPD_ROOT=$T/new FROM_OPD_ROOT=$F LINK_VENV=1 "$@" 2> /dev/null; step() { echo "STEP $1 $2" >> $T/mv; }; source $T/move.sh; echo "OLD_TIER=$OLD_TIER" >> $T/mv ); }
rm -rf $T/new; : > $T/mv; domove OVERNIGHT_TIER_A_ONLY=1
ok "move, A only: the old tier B not read, the probe starts at A" 'grep -q "^OLD_TIER=$" $T/mv && grep -q "probe starts at tier A$" $T/mv'
ok "move: venv and G-OPD linked, parquet copied, triton.env not copied" '[ -L $T/new/venv ] && [ -L $T/new/G-OPD ] && [ -f $T/new/data/train.parquet ] && [ ! -e $T/new/triton.env ]'
ok "move: the old OPD_ROOT unchanged" '[ "$(cd $F && find . | sort | md5sum 2> /dev/null || find . | sort | md5)" = "$before" ]'
rm -rf $T/new; : > $T/mv; domove OVERNIGHT_TIER_A_ONLY=0
ok "move, flag unset: the old tier B read as before" 'grep -q "^OLD_TIER=B$" $T/mv'
rm -rf $T
[ $fails = 0 ] && echo "OVERNIGHT_TIERS_TEST PASS ($n checks)" || echo "OVERNIGHT_TIERS_TEST FAIL ($fails of $n)"
