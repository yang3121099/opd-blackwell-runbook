#!/usr/bin/env bash
# test_store_sync.sh ([v3.13a]): scripts/store_sync.sh on a fake node. A store on another disk gets the node's fetched weights and
# base models (here forced with STORE_SAME_DISK_COPY=1: a test has one disk); a store on the install's own disk - a preparation
# node whose base is on the shared disk - gets neither: they are read where they are. No store: FAIL, nothing written.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf $T' EXIT; fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
export OPD_ROOT=$T/n/run RUNBOOK=$SRC; unset OPD_STORE CKPT_ROOT
mkdir -p $OPD_ROOT/logs $OPD_ROOT/control $OPD_ROOT/models/Qwen3-0.6B $OPD_ROOT/adhoc_models/result/runs/E1/global_step_5/actor
echo '{}' > $OPD_ROOT/models/Qwen3-0.6B/config.json; python3 -c "open('$OPD_ROOT/models/Qwen3-0.6B/model.safetensors','wb').write(b'w' * 1000001)"
echo w > $OPD_ROOT/adhoc_models/result/runs/E1/global_step_5/actor/x.bin; : > $OPD_ROOT/adhoc_models/result/runs/E1/global_step_5/.fetch_ok
run() { ( export "$@" 2> /dev/null; bash $SRC/scripts/store_sync.sh > /dev/null 2>&1 ); }
run X=1; rc=$?
ok "no store: FAIL, the reason in logs/store.json" '[ $rc != 0 ] && grep -q "\"result\": \"FAIL\"" $OPD_ROOT/logs/store.json && grep -q "no long-term store" $OPD_ROOT/logs/store.json'
S=$T/store
run OPD_STORE=$S; rc=$?
ok "the store on the install's own disk: PASS, markers written, models and weights not copied" '[ $rc = 0 ] && [ -f $S/paths.env ] && [ ! -e $S/models/Qwen3-0.6B ] && [ ! -e $S/adhoc_models/result/runs/E1 ] && grep -q "read where they are" $OPD_ROOT/logs/store.json'
run OPD_STORE=$S STORE_SAME_DISK_COPY=1 STORE_RESERVE_GB=0; rc=$?
ok "another disk (forced): the fetched weight and the base model are copied, complete" '[ $rc = 0 ] && [ -f $S/adhoc_models/result/runs/E1/global_step_5/.fetch_ok ] && [ -f $S/models/Qwen3-0.6B/model.safetensors ] && grep -q "\"weights_copied\": 1" $OPD_ROOT/logs/store.json && grep -q "\"models_copied\": 1" $OPD_ROOT/logs/store.json'
run OPD_STORE=$S STORE_SAME_DISK_COPY=1 STORE_RESERVE_GB=0; rc=$?
ok "a second run copies nothing again" '[ $rc = 0 ] && grep -q "\"weights_copied\": 0" $OPD_ROOT/logs/store.json && grep -q "\"models_copied\": 0" $OPD_ROOT/logs/store.json'
[ $fails = 0 ] && echo "STORE_SYNC_TEST PASS ($n checks)" || { echo "STORE_SYNC_TEST FAIL ($fails of $n)"; exit 1; }
