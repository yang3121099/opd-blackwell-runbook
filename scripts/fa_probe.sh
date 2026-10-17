#!/usr/bin/env bash
# FA-PROBE (runbook v2.5.3, reviewer A1 2026-10-17): can flash-attn 2 varlen (remove-padding) run on this node? A task
# like any other (`run FA-PROBE`): it gets a GPU only after the queue's own tasks, and it never touches the venv in use
# or any E10b directory. Steps, each written to runs/FA-PROBE/probe.log:
#   1. read torch / vLLM / Python and the C++ ABI from the venv in use; refuse a stack the pinned wheel does not cover
#   2. copy the real directory behind $OPD_ROOT/venv to $OPD_ROOT/venv-fa (once) and relocate it: the old path in the
#      text files of bin/ (activate, entry-point shebangs) becomes the new one, so `source venv-fa/bin/activate` really
#      selects venv-fa; checked (sys.prefix, the python that activate puts first on PATH)
#   3. download the pinned flash-attn 2.8.3.post2 wheel for that ABI from the release page, stop if its sha256 differs
#      (no other source is tried), pip install --no-deps --no-index into venv-fa; torch and vLLM must be unchanged;
#      sys.prefix and flash_attn.__file__ are recorded to show where it landed
#   4. flash_attn_varlen_func forward + backward on this GPU against PyTorch SDPA (sm_100 code on an sm_103 GPU runs
#      through same-major compatibility: this is its acceptance test; the error text is kept if it fails)
#   5. 3-step smoke of the E10b-P2 configuration with venv-fa (remove-padding on), as runs/FA-PROBE-smoke: no checkpoint,
#      no upload, no validation; nvidia-smi samples this GPU every 5 s (memory peak); step timings from train.log
#   6. runs/FA-PROBE (report.json + logs, no wheel) to logs/<node>/fa_probe/ in the private repository, an event, .done
# A negative result is a result: the task ends with .done either way (no retries), and the event says pass or fail.
# Usage: bash fa_probe.sh <gpu>
set -uo pipefail
G=$1; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/FA-PROBE; VF=$OPD_ROOT/venv-fa; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE FA-PROBE"; exit 0; }
PJ=$D/probe.json; echo '{}' > $PJ
step() { echo "FA_PROBE $(date -u +%T) $*" | tee -a $LOG; }
put() {   # put <key> <json value>; puts <key> <string>
  python3 - "$PJ" "$1" "$2" "${3:-json}" <<'PY'
import json, sys
p, k, v, mode = sys.argv[1:5]; d = json.load(open(p))
if mode == "str":
    d[k] = v
else:
    try:
        d[k] = json.loads(v)
    except ValueError:
        d[k] = v
json.dump(d, open(p, "w"), indent=1)
PY
}
puts() { put "$1" "$2" str; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  puts result "$1"; puts reason "$2"
  python3 $RUNBOOK/scripts/fa_probe.py report $D > /dev/null 2>&1
  python3 $RUNBOOK/scripts/remote.py put $D logs/$(python3 $RUNBOOK/scripts/control.py node)/fa_probe >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  python3 $RUNBOOK/scripts/control.py event FA-PROBE "fa_probe_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
for _w in $(seq 19); do   # the previous task's vLLM can take tens of seconds to release its memory: wait up to 180 s
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s (${used} MiB used)"; sleep ${FA_WAIT_STEP:-10}   # (tests: 0)
done
step "start gpu=$G"

# 1. the stack of the venv in use, the wheel for it
SRC=$(readlink -f $OPD_ROOT/venv)
BEFORE=$($SRC/bin/python $RUNBOOK/scripts/fa_probe.py versions 2>/dev/null); put before "$BEFORE"
step "venv in use: $BEFORE"
ABI=$(echo "$BEFORE" | python3 -c "import json,sys; d=json.load(sys.stdin); ok=str(d.get('torch','')).startswith('2.8.') and str(d.get('python','')).startswith('3.10'); print(d.get('cxx11abi','') if ok else 'UNCOVERED')" 2>/dev/null)
[ "$ABI" = TRUE ] || [ "$ABI" = FALSE ] || finish FAIL "stack not covered by the pinned wheel (torch 2.8 / Python 3.10 / known ABI): $BEFORE"
read URL SHA NAME <<< "$(python3 $RUNBOOK/scripts/fa_probe.py wheel "$ABI")"
[ -n "${NAME:-}" ] || finish FAIL "no pinned wheel for ABI $ABI"
puts wheel "$NAME"

# 2. copy + relocate the venv
if [ ! -x $VF/bin/python ]; then
  step "copying the venv in use to venv-fa"
  rm -rf "$VF.tmp"; cp -a "$SRC" "$VF.tmp" || finish FAIL "venv copy failed"
  python3 - "$VF.tmp/bin" "$SRC" "$VF" <<'PY' || finish FAIL "venv relocation failed"
import os, sys
d, old, new = sys.argv[1:4]
for n in os.listdir(d):   # text files only (activate scripts, entry-point shebangs); binaries and symlinks stay
    p = os.path.join(d, n)
    if os.path.islink(p) or not os.path.isfile(p):
        continue
    b = open(p, "rb").read()
    if b"\0" in b[:4096] or old.encode() not in b:
        continue
    open(p, "wb").write(b.replace(old.encode(), new.encode()))
PY
  mv "$VF.tmp" "$VF" || finish FAIL "venv move failed"
fi
[ "$(readlink -f $VF)" != "$SRC" ] || finish FAIL "venv-fa resolves to the venv in use"
PFX=$($VF/bin/python -c "import sys; print(sys.prefix)" 2>/dev/null)
ACT=$(bash -c "source $VF/bin/activate && command -v python" 2>/dev/null)
puts sys_prefix "$PFX"; puts activate_python "$ACT"
{ [ "$(readlink -f "$PFX")" = "$(readlink -f $VF)" ] && [ "$ACT" = "$VF/bin/python" ]; } || finish FAIL "venv-fa not relocated (sys.prefix $PFX, activate puts $ACT first)"
step "venv-fa relocated: sys.prefix and activate both point to venv-fa"

# 3. pinned wheel, sha256, --no-deps
FAIMP="from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis; from flash_attn import flash_attn_varlen_func"   # what verl's remove-padding path imports
if ! $VF/bin/python -c "$FAIMP" 2>/dev/null; then
  WD=$OPD_ROOT/tmp/fa_wheel; mkdir -p $WD; W=$WD/$NAME   # outside runs/FA-PROBE, which is uploaded
  if [ ! -f "$W" ]; then
    curl -fsSL --retry 3 -o "$W.part" "$URL" && mv "$W.part" "$W" || finish FAIL "download failed: $NAME"
  fi
  GOT=$(sha256sum "$W" | cut -c1-64)
  [ "$GOT" = "$SHA" ] || { rm -f "$W"; finish FAIL "sha256 mismatch for $NAME (release page $SHA, got $GOT)"; }
  put sha256_ok true
  $VF/bin/python -m pip install --no-deps --no-index "$W" >> $LOG 2>&1 || finish FAIL "pip install --no-deps failed"
fi
AFTER=$($VF/bin/python $RUNBOOK/scripts/fa_probe.py versions 2>/dev/null); put after "$AFTER"
t0=$(echo "$BEFORE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('torch'), d.get('vllm'))")
t1=$(echo "$AFTER" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('torch'), d.get('vllm'))")
[ "$t0" = "$t1" ] || finish FAIL "torch/vLLM changed by the install: $t0 -> $t1"
$VF/bin/python -c "$FAIMP" 2> $D/import.err || finish FAIL "flash_attn.bert_padding / varlen import failed (einops?): $(tail -c 300 $D/import.err | tr '\n' ' ')"
FAF=$($VF/bin/python -c "import flash_attn; print(flash_attn.__file__)" 2>/dev/null); puts flash_attn_file "$FAF"
case "$FAF" in "$VF"/*|"$(readlink -f $VF)"/*) ;; *) finish FAIL "flash_attn not inside venv-fa: ${FAF:-not importable}";; esac
step "installed $NAME into venv-fa ($FAF); torch/vLLM unchanged ($t1)"

# 4. varlen forward/backward on this GPU
VL=$(CUDA_VISIBLE_DEVICES=$G $VF/bin/python $RUNBOOK/scripts/fa_probe.py varlen 2> $D/varlen.err); rc=$?
put varlen "${VL:-null}"; put varlen_pass $([ $rc = 0 ] && echo true || echo false)
[ $rc = 0 ] || finish FAIL "flash_attn_varlen_func failed on this GPU: $(tail -c 300 $D/varlen.err | tr '\n' ' ')"
step "varlen PASS $VL"

# 5. 3-step smoke, own task name and directory, no checkpoint / upload / validation
( while true; do echo "$(date -u +%s),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G | tr -d ' ')" >> $D/mem.csv; sleep 5; done ) &
MP=$!
rm -rf $OPD_ROOT/runs/FA-PROBE-smoke
VENV=$VF SMOKE=1 PROBE_TASK=FA-PROBE-smoke bash $RUNBOOK/scripts/run_arm.sh $G E10b-P2 > $D/smoke.out 2>&1; rc=$?
kill $MP 2> /dev/null
mkdir -p $D/smoke; cp $OPD_ROOT/runs/FA-PROBE-smoke/train.log $OPD_ROOT/runs/FA-PROBE-smoke/config.yaml $D/smoke/ 2> /dev/null
grep -q "use_remove_padding=True" $D/smoke/config.yaml 2> /dev/null && put rmpad true || put rmpad false
[ $rc = 0 ] || finish FAIL "smoke exited $rc: $(tail -c 300 $D/smoke.out | tr '\n' ' ')"
grep -q "use_remove_padding=True" $D/smoke/config.yaml 2> /dev/null || finish FAIL "smoke ran without remove-padding (flash_attn not seen by run_arm.sh)"
finish PASS "varlen ok, 3-step smoke ok with remove-padding"
