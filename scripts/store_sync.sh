#!/usr/bin/env bash
# [v3.10] store_sync.sh: this node's reusable files -> the long-term store on a shared disk (Fable 11-18; Runming: "the install is
# this slow - put the environment on the long-term disk and reuse it"). A borrowed machine's local disk is gone with the
# container; whatever the next machine can copy from the shared disk it does not download through the proxy (11-17/18: 22 GB of
# base models and 20 GB of our weights at 14 MB/s, three failed installs of the stack).
# The store ($OPD_STORE: history_scan.py chooses it, node.env may name it) looks like an install of ours, so the next machine's
# history scan finds it and the existing readers use it as they use any earlier install:
#   models/<name>/                         complete base models                      -> overnight.sh find_model copies them
#   adhoc_models/result/runs/<arm>/...     our fetched weights (with their .fetch_ok) -> fetch_model.py's local roots (z134)
#   envpack/site-packages-<py>-torch2.8.0-vllm0.10.2.tar   the verified venv's packages -> install_uv.sh seed_venv
#   wheels/<flash-attn wheel>              the pinned wheel fa_probe.sh downloaded    -> fa_probe.sh
#   paths.env, control/                    markers (history_scan.py: an install of ours)
# Only complete things are copied, each to a .tmp name first; what the store has at the same size is skipped; nothing in the
# store is deleted. CPU only, nice / ionice. Result: logs/store.json (result PASS / FAIL, the counts).
set -uo pipefail
source "$(dirname "$0")/env.sh"
[ -f $OPD_ROOT/node.env ] && { set -a; source $OPD_ROOT/node.env; set +a; }
L=$OPD_ROOT/logs; RJ=$L/store.json; LOG=$L/store.log; mkdir -p $L
S=${OPD_STORE:-}
fin() { python3 - "$RJ" "$1" "$2" "${NM:-0}" "${NW:-0}" "${NE:-0}" "${NWH:-0}" "$SECONDS" <<'PY'
import json, sys
p, res, why, nm, nw, ne, nwh, sec = sys.argv[1:9]
json.dump({"result": res, "reason": why, "models_copied": int(nm), "weights_copied": int(nw), "envpacks_written": int(ne), "wheels_copied": int(nwh), "seconds_total": int(sec)}, open(p, "w"))
PY
  echo "STORE $(date -u +%T) $1 $2" >> $LOG; [ "$1" = PASS ]; exit $?; }
[ -n "$S" ] || fin FAIL "no long-term store on this machine (history_scan.py found no writable shared directory with room; node.env OPD_STORE=<dir> names one)"
LOW="nice -n 19"; command -v ionice > /dev/null 2>&1 && LOW="ionice -c3 $LOW"
mkdir -p $S/control $S/models $S/adhoc_models/result/runs $S/envpack $S/wheels 2>> $LOG || fin FAIL "the store cannot be written"
[ -f $S/paths.env ] || echo "# opd-store: reusable files of earlier nodes (models, fetched weights, environment pack). Written by scripts/store_sync.sh; read by history_scan.py." > $S/paths.env
same() { [ -e "$2" ] && python3 - "$1" "$2" <<'PY'
import os, sys
def size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p, followlinks=True) for f in fs)
sys.exit(0 if size(sys.argv[1]) == size(sys.argv[2]) else 1)
PY
}
complete_model() { python3 - "$1" <<'PY'
import json, os, sys
d = sys.argv[1]; n = set(os.listdir(d)) if os.path.isdir(d) else set()
idx = os.path.join(d, "model.safetensors.index.json")
need = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
sys.exit(0 if "config.json" in n and all(f in n and os.path.getsize(os.path.join(d, f)) > 1000000 for f in need) and not any(x.endswith(".aria2") for x in n) else 1)
PY
}
NM=0; NW=0; NE=0; NWH=0
# 1. base models
for d in $OPD_ROOT/models/*/; do m=$(basename $d); complete_model $d || continue; same $d $S/models/$m && continue
  rm -rf $S/models/$m.tmp; $LOW cp -rL $d $S/models/$m.tmp 2>> $LOG && { rm -rf $S/models/$m.old; [ -e $S/models/$m ] && mv $S/models/$m $S/models/$m.old; mv $S/models/$m.tmp $S/models/$m && rm -rf $S/models/$m.old && NM=$((NM + 1)) && echo "STORE $(date -u +%T) model $m" >> $LOG; }
done
# 2. our fetched weights (a complete fetch has its .fetch_ok)
for W in ${CKPT_ROOT:+$CKPT_ROOT/adhoc_models/result} $OPD_ROOT/adhoc_models/result; do [ -d $W/runs ] || continue
  for ok in $(find $W/runs -maxdepth 6 -name .fetch_ok 2> /dev/null); do d=$(dirname $ok); rel=${d#$W/}
    same $d $S/adhoc_models/result/$rel && continue
    mkdir -p $(dirname $S/adhoc_models/result/$rel); rm -rf $S/adhoc_models/result/$rel.tmp
    $LOW cp -rL $d $S/adhoc_models/result/$rel.tmp 2>> $LOG && { rm -rf $S/adhoc_models/result/$rel; mv $S/adhoc_models/result/$rel.tmp $S/adhoc_models/result/$rel && NW=$((NW + 1)) && echo "STORE $(date -u +%T) weight $rel" >> $LOG; }
  done; done
# 3. the environment pack: only a venv whose install ended well, with the pinned stack
if [ -f $OPD_ROOT/venv/.install_ok ] && SP=$($OPD_ROOT/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2> /dev/null) \
   && ls -d $SP/torch-2.8.0*.dist-info $SP/vllm-0.10.2*.dist-info > /dev/null 2>&1; then
  P=$S/envpack/site-packages-$(basename $(dirname $SP))-torch2.8.0-vllm0.10.2.tar
  if [ ! -f $P ]; then
    ( cd $SP && $LOW tar -cf - --exclude='__editable__*verl*' --exclude='verl-*.dist-info' . ) | $LOW dd of=$P.tmp bs=64M iflag=fullblock status=none 2>> $LOG \
      && ( cd $(dirname $P) && sha256sum $(basename $P).tmp | cut -c1-64 > $(basename $P).sha256 ) && mv $P.tmp $P && NE=$((NE + 1)) && echo "STORE $(date -u +%T) envpack $(basename $P) $(du -h $P | cut -f1)" >> $LOG
  fi
fi
# 4. the pinned flash-attn wheel
for w in $OPD_ROOT/tmp/fa_wheel/*.whl; do [ -f "$w" ] || continue; same $w $S/wheels/$(basename $w) && continue
  cp $w $S/wheels/$(basename $w).tmp 2>> $LOG && mv $S/wheels/$(basename $w).tmp $S/wheels/$(basename $w) && NWH=$((NWH + 1)); done
fin PASS "store: $(ls $S/models 2> /dev/null | grep -vc '\.tmp$') model(s), $(find $S/adhoc_models/result/runs -maxdepth 6 -name .fetch_ok 2> /dev/null | wc -l | tr -d ' ') weight(s), $(ls $S/envpack/*.tar 2> /dev/null | wc -l | tr -d ' ') environment pack(s), $(ls $S/wheels/*.whl 2> /dev/null | wc -l | tr -d ' ') wheel(s); copied now: $NM / $NW / $NE / $NWH"
