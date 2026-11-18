#!/usr/bin/env bash
# [v3.12] seed_copy.sh: a venv that uses an earlier install's packages where they are (install_uv.sh seed_venv: one .pth line)
# becomes a venv of its own, in the background (Runming 11-18: "use it directly first, and copy it as the base the formal
# environment is installed on"). The earlier install's site-packages are copied into a temporary directory on the local disk -
# no time limit, low priority - and then moved into $OPD_ROOT/venv's site-packages entry by entry; the .pth line goes last, so at
# every moment each package can be imported from one place or the other, and they are the same files. Running programs keep the
# path they started with. venv-fa (a copy of venv made while it was linked) gets the same packages from the now local venv.
# After it: venv/.seeded_from_history, no venv/.linked_to_history; store_sync.sh can pack the environment for the next machine.
# One at a time (flock). Log: logs/seed_copy.log. Usage: bash seed_copy.sh   (install_uv.sh starts it; SEED_COPY_SYNC=1 waits)
set -uo pipefail
source "$(dirname "$0")/env.sh"
L=$OPD_ROOT/logs/seed_copy.log; mkdir -p $OPD_ROOT/logs
say() { echo "SEED_COPY $(date -u +%T) $*" | tee -a $L; }
exec 6> $OPD_ROOT/venv/.seed_copy.lock; if command -v flock > /dev/null; then flock -n 6 || { say "another copy runs"; exit 0; }; fi
[ -s $OPD_ROOT/venv/.linked_to_history ] || { say "the venv is not linked: nothing to do"; exit 0; }
SP=$(head -1 $OPD_ROOT/venv/.linked_to_history); [ -d "$SP/torch" ] || { say "the linked packages are not there any more"; exit 1; }
LOW="nice -n 19"; command -v ionice > /dev/null 2>&1 && LOW="ionice -c3 $LOW"
swap() {   # <venv dir> <source site-packages>: copy through a temporary directory, move in, drop the link
  local V=$1 SRC=$2 dst tmp e t0; t0=$(date +%s)
  dst=$($V/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || return 1
  tmp=$OPD_ROOT/tmp/seed_copy_$(basename $V); rm -rf $tmp; mkdir -p $tmp   # beside the venv, not inside it: fa_probe.sh copies the venv directory while this runs
  ( cd "$SRC" && $LOW tar -cf - --exclude='__editable__*verl*' --exclude='verl-*.dist-info' --exclude='opd_shared_packages.pth' . ) | tar -xf - -C $tmp || { rm -rf $tmp; return 1; }
  for e in $tmp/* $tmp/.[!.]*; do [ -e "$e" ] || continue
    [ -e "$dst/$(basename "$e")" ] && continue   # what this venv has of its own (added packages, flash-attn, this node's verl) stays
    mv "$e" "$dst/"; done
  rm -rf $tmp $dst/opd_shared_packages.pth
  $V/bin/python -c 'import torch, vllm' 2>> $L || { echo "$SRC" > $dst/opd_shared_packages.pth; say "$(basename $V): the copy does not import torch / vllm: linked again"; return 1; }
  say "$(basename $V): $(ls $dst | wc -l | tr -d ' ') entries are local now ($(( $(date +%s) - t0 )) s)"; }
say "copying the earlier install's packages to the local disk"
swap $OPD_ROOT/venv "$SP" || { say "FAILED: the venv stays linked"; exit 1; }
rm -f $OPD_ROOT/venv/.linked_to_history; : > $OPD_ROOT/venv/.seeded_from_history
# the earlier install's console scripts, with this venv's interpreter on their first line
OB=$(dirname "$(dirname "$(dirname "$SP")")")/bin; n=0
for f in $OB/*; do case $(basename $f) in python*|activate*|pip*|uv*|Activate*|deactivate*) continue;; esac
  [ -f $f ] && [ ! -e $OPD_ROOT/venv/bin/$(basename $f) ] && head -1 $f 2> /dev/null | grep -q '^#!.*python' || continue
  { echo "#!$OPD_ROOT/venv/bin/python"; tail -n +2 $f; } > $OPD_ROOT/venv/bin/$(basename $f) && chmod +x $OPD_ROOT/venv/bin/$(basename $f) && n=$((n + 1)); done
say "$n console scripts"
LSP=$($OPD_ROOT/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
if [ -x $OPD_ROOT/venv-fa/bin/python ] && [ -f "$($OPD_ROOT/venv-fa/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2> /dev/null)/opd_shared_packages.pth" ]; then
  swap $OPD_ROOT/venv-fa "$LSP" || say "venv-fa stays linked"; fi
python3 $RUNBOOK/scripts/control.py event node seed_copy "the environment's packages are on the local disk now (were used from an earlier install)" > /dev/null 2>&1 || true
say "done"
