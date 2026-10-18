#!/usr/bin/env bash
# Stub tests of v2.5.7a persist_env.sh and rebuild_drill.sh (no GPU, network or HF): a fake container disk (uv python,
# HF cache), a fake old base (venv in use, G-OPD, teacher) and a share holding $OPD_ROOT. Usage: bash scripts/test_persist.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(cd "$(mktemp -d)" && pwd -P); C=$T/c; O=$T/old/opd-run; S=$T/share; R=$S/opd-run; mkdir -p $T/rb/scripts $R/logs $R/data $S/opd-dr/state
  cp $SRC/scripts/persist_env.sh $SRC/scripts/rebuild_drill.sh $SRC/scripts/dr.sh $T/rb/scripts/
  printf 'export OPD_ROOT=%s RUNBOOK=%s\n' "$R" "$T/rb" > $T/rb/scripts/env.sh
  # uv's python on the container disk (python-build-standalone layout; the venvs name it through a version-less link)
  P=$C/uvpy/cpython-3.10.22-linux-x86_64-gnu; mkdir -p $P/bin; ln -s $P $C/uvpy/cpython-3.10-linux-x86_64-gnu
  cat > $P/bin/python3.10 <<'SH'
#!/usr/bin/env bash
# fake interpreter: -c prints what the import check reads (prefix = the venv it was started from, base = its real home)
[ "${1:-}" = -c ] || exit 0
case "$2" in *AutoTokenizer*) exit 0;; esac
pre=$(cd "$(dirname "$0")/.." && pwd); base=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd -P)
echo "{\"prefix\": \"${FAKE_PREFIX:-$pre}\", \"base\": \"$base\", \"torch\": \"2.8.0\", \"vllm\": \"0.10.2\"}"
SH
  chmod +x $P/bin/python3.10; ln -s python3.10 $P/bin/python3
  H=$C/uvpy/cpython-3.10-linux-x86_64-gnu/bin
  mkvenv() {   # <dir> <G-OPD dir>
    mkdir -p $1/bin $1/lib/python3.10/site-packages; printf 'home = %s\nversion_info = 3.10\n' "$H" > $1/pyvenv.cfg
    ln -s $H/python3.10 $1/bin/python; ln -s python $1/bin/python3; printf 'VIRTUAL_ENV=%s\n' "$1" > $1/bin/activate
    echo "$2/verl" > $1/lib/python3.10/site-packages/__editable__verl.pth; }
  mkdir -p $O/G-OPD/verl; echo x > $O/G-OPD/verl/x.py; mkvenv $O/venv $O/G-OPD; ln -s $O/venv $R/venv; ln -s $O/G-OPD $R/G-OPD
  mkvenv $R/venv-fa $O/G-OPD; printf 'VIRTUAL_ENV=%s\n' "$R/venv-fa" > $R/venv-fa/bin/activate
  mkmodel() { mkdir -p $1 $1/../blobs; echo '{}' > $1/config.json; head -c 4000 /dev/zero > $1/../blobs/w; ln -s ../blobs/w $1/model.safetensors; }
  mkmodel $C/hf/s06/snap; mkmodel $C/hf/s17/snap; mkdir -p $O/models/Qwen3-8B; echo '{}' > $O/models/Qwen3-8B/config.json; head -c 4000 /dev/zero > $O/models/Qwen3-8B/model.safetensors
  printf 'STUDENT_06B=%s\nSTUDENT_17B=%s\nTEACHER=%s\n' $C/hf/s06/snap $C/hf/s17/snap $O/models/Qwen3-8B > $R/models.env
  E=$S/opd-env; J=$R/logs/persist_env.json
}
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" "$1" "$2"; }
# 1. persist-env: python, venv, G-OPD and models on the share; links and models.env switched; originals untouched
setup
bash $T/rb/scripts/persist_env.sh > /dev/null 2>&1
ok "PASS" '[ "$(res $J result)" = PASS ]'
ok "python copied" '[ -x $E/python/cpython-3.10.22-linux-x86_64-gnu/bin/python3 ]'
ok "venv relocated" 'grep -q "^home = $E/python/cpython-3.10.22-linux-x86_64-gnu/bin$" $E/venv/pyvenv.cfg && [ "$(readlink -f $E/venv/bin/python)" = $E/python/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10 ] && grep -q "VIRTUAL_ENV=$E/venv" $E/venv/bin/activate && grep -q "^$E/G-OPD/verl$" $E/venv/lib/python3.10/site-packages/__editable__verl.pth'
ok "links switched" '[ "$(readlink -f $R/venv)" = $E/venv ] && [ "$(readlink -f $R/G-OPD)" = $E/G-OPD ] && [ -f $E/G-OPD/verl/x.py ]'
ok "venv-fa re-pointed in place" 'grep -q "^home = $E/python/" $R/venv-fa/pyvenv.cfg && grep -q "^$E/G-OPD/verl$" $R/venv-fa/lib/python3.10/site-packages/__editable__verl.pth'
ok "models copied, dereferenced, models.env rewritten" '[ -f $R/models/Qwen3-0.6B/model.safetensors ] && [ ! -L $R/models/Qwen3-0.6B/model.safetensors ] && grep -q "^STUDENT_17B=$R/models/Qwen3-1.7B$" $R/models.env && grep -q "^TEACHER=$R/models/Qwen3-8B$" $R/models.env && [ -f $R/models.env.prev ]'
ok "originals untouched" 'grep -q "^home = $H$" $O/venv/pyvenv.cfg && [ -d $O/models/Qwen3-8B ] && [ -d $E/hf ]'
bash $T/rb/scripts/persist_env.sh > /dev/null 2>&1
ok "idempotent" '[ "$(grep -c "copying the venv" $R/logs/persist_env.log)" = 1 ] && [ "$(res $J result)" = PASS ]'
# 2. rebuild drill on top of it (a local clone source; the state files as the heartbeat writes them)
G=$T/repo; mkdir -p $G/scripts $G/tasks; cp $SRC/scripts/dr.sh $G/scripts/; echo "v" > $G/tasks/queue.txt
printf 'export OPD_ROOT=${OPD_ROOT} RUNBOOK=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)\nexport CKPT_ROOT=$(sed -n "s/^CKPT_ROOT=//p" $OPD_ROOT/paths.env)\n' > $G/scripts/env.sh
(cd $G && git init -q && git add -A && git -c user.email=t@t -c user.name=t commit -qm init)
for f in models.env; do cp $R/$f $S/opd-dr/state/; done; echo "public=me/pub" > $S/opd-dr/state/hf_repos.txt; echo "CKPT_ROOT=$HOME/opd-ckpt" > $S/opd-dr/state/paths.env
mkdir -p $S/opd-dr/dr/E10b-X/global_step_25/actor; echo pt > $S/opd-dr/dr/E10b-X/global_step_25/actor/m.pt; echo 25 > $S/opd-dr/dr/E10b-X/latest
DRILL_REPO=$G bash $T/rb/scripts/rebuild_drill.sh > $T/drill.out 2>&1
D=$R/logs/rebuild_drill.json
ok "drill PASS, every step timed" 'python3 -c "import json, sys; d = json.load(open(sys.argv[1])); s = d[\"steps\"]; sys.exit(0 if d[\"pass\"] == \"true\" and sorted(s) == [\"clone\", \"env\", \"models\", \"opd_root\", \"restore\", \"venv\"] and all(v[\"ok\"] for v in s.values()) else 1)" $D'
ok "drill leaves nothing behind" '! ls -d $S/opd-drill-* > /dev/null 2>&1 && [ -d $S/opd-dr/dr/E10b-X/global_step_25 ]'
rm -rf $T
# 3. a copy that does not run from the share switches nothing
setup
FAKE_PREFIX=/somewhere/else bash $T/rb/scripts/persist_env.sh > /dev/null 2>&1
ok "failed check: nothing switched" '[ "$(res $J result)" = FAIL ] && [ "$(readlink -f $R/venv)" = $O/venv ] && grep -q "^TEACHER=$O/models/Qwen3-8B$" $R/models.env'
rm -rf $T
[ $fails = 0 ] && echo "PERSIST_TEST PASS ($n checks)" || echo "PERSIST_TEST FAIL ($fails of $n)"
