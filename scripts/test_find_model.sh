#!/usr/bin/env bash
# test_find_model.sh ([v3.11]): overnight.sh's find_model on a fake machine - the store / an earlier install is copied to the local
# disk, a copy that is too slow leaves the model used where it is, a copy from anywhere else is used only with the published sha256.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf $T' EXIT; fails=0
ok() { if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
mkdir -p $T/bin; command -v timeout > /dev/null || cat > $T/bin/timeout <<'STUB'
#!/usr/bin/env python3
import subprocess, sys
a = sys.argv[1:]
try: sys.exit(subprocess.run(a[1:], timeout=float(a[0])).returncode)
except subprocess.TimeoutExpired: sys.exit(124)
STUB
chmod +x $T/bin/* 2> /dev/null; export PATH=$T/bin:$PATH
mkmodel() { mkdir -p $1; echo '{}' > $1/config.json; python3 -c "open('$1/model.safetensors','wb').write(b'${2:-w}' * 1000001)"; }
fresh() { rm -rf $T/n; mkdir -p $T/n/run/models $T/n/run/logs $T/n/run/control; export OPD_ROOT=$T/n/run LOG=$T/n/run/logs/overnight.log RB=$SRC; unset FROM_OPD_ROOT OPD_STORE OLD_BASE_CANDS MODEL_COPY_MAX_S MODEL_VERIFY_LIST; : > $LOG; }
eval "$(sed -n '/^complete_dir(){/,/^}/p' $SRC/scripts/overnight.sh)"
eval "$(sed -n '/^copy_model(){/,/^find_model(){/p' $SRC/scripts/overnight.sh | sed '$d')"
eval "$(sed -n '/^find_model(){/,/^}/p' $SRC/scripts/overnight.sh)"
mkmodel $T/store/models/Qwen3-8B; mkmodel $T/old/models/Qwen3-1.7B; mkmodel $T/wild/pretrained/Qwen3-0.6B g; mkdir -p $T/old/models/Qwen3-0.6B; echo '{}' > $T/old/models/Qwen3-0.6B/config.json

fresh; export OPD_STORE=$T/store OLD_BASE_CANDS=$T/none:$T/old
ok "the store's model is copied to the local disk" '[ "$(find_model Qwen3-8B)" = $OPD_ROOT/models/Qwen3-8B ] && complete_dir $OPD_ROOT/models/Qwen3-8B && grep -q "Qwen3-8B copied from" $LOG'
ok "an earlier install's model too; an incomplete one is not offered" '[ "$(find_model Qwen3-1.7B)" = $OPD_ROOT/models/Qwen3-1.7B ] && ! find_model Qwen3-0.6B > /dev/null'
ok "a second call finds the local copy" '[ "$(find_model Qwen3-8B)" = $OPD_ROOT/models/Qwen3-8B ]'

fresh; export OPD_STORE=$T/store MODEL_COPY_MAX_S=1; printf '#!/bin/bash\nsleep 5\n' > $T/bin/cp; chmod +x $T/bin/cp
ok "a copy that is too slow: the model is used where it is, nothing half-copied stays" '[ "$(find_model Qwen3-8B)" = $T/store/models/Qwen3-8B ] && [ -z "$(ls $OPD_ROOT/models)" ] && grep -q "used where it is" $LOG'
rm -f $T/bin/cp

fresh; python3 - $T <<'PY'
import hashlib, json, sys
t = sys.argv[1]
b = open(f"{t}/wild/pretrained/Qwen3-0.6B/model.safetensors", "rb").read()
json.dump({"siblings": [{"rfilename": "model.safetensors", "size": len(b), "lfs": {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}}, {"rfilename": "config.json", "size": 3}]}, open(f"{t}/list_ok.json", "w"))
json.dump({"siblings": [{"rfilename": "model.safetensors", "size": len(b), "lfs": {"sha256": "0" * 64, "size": len(b)}}]}, open(f"{t}/list_bad.json", "w"))
json.dump({"model_dirs": {"Qwen3-0.6B": [f"{t}/wild/pretrained/Qwen3-0.6B"]}}, open(f"{t}/n/run/control/history_secret.json", "w"))
PY
export MODEL_VERIFY_LIST=$T/list_ok.json
ok "a copy found elsewhere on the disks is copied and used once its sha256 matches the published one" '[ "$(find_model Qwen3-0.6B)" = $OPD_ROOT/models/Qwen3-0.6B ] && grep -q "published sha256 matched" $LOG'
rm -rf $OPD_ROOT/models/Qwen3-0.6B; export MODEL_VERIFY_LIST=$T/list_bad.json
ok "other weights under the right name are not used and not left behind" '! find_model Qwen3-0.6B > /dev/null && [ ! -e $OPD_ROOT/models/Qwen3-0.6B ] && grep -q "not verified" $LOG'
export MODEL_VERIFY_LIST=$T/none.json
ok "without the published list a copy from elsewhere is not used" '! find_model Qwen3-0.6B > /dev/null && [ ! -e $OPD_ROOT/models/Qwen3-0.6B ]'
[ $fails = 0 ] && echo "FIND_MODEL_TEST PASS (7 checks)" || { echo "FIND_MODEL_TEST FAIL ($fails of 7)"; exit 1; }
