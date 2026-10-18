#!/usr/bin/env bash
# v2.5.7a persist-env (Runming 2026-10-18 via the reviewer: a new container must come up fast). On the B300 the venvs'
# base python is uv's under $HOME/.local/share/uv and two of the three models are in $HOME/.cache/huggingface (both on the
# container disk, gone after a rebuild), and the venv in use and G-OPD are in the old base on a full disk. This copies
# them to the network share next to $OPD_ROOT (<base> = the directory above $OPD_ROOT), CPU only, nice 19 / ionice idle:
#   <base>/opd-env/python/<name>   uv's CPython (the `home` of the venvs' pyvenv.cfg)
#   <base>/opd-env/venv            the venv in use, relocated: bin/ scripts, pyvenv.cfg home, python links, .pth paths
#   <base>/opd-env/G-OPD           G-OPD (the code verl runs from) when $OPD_ROOT/G-OPD links outside $OPD_ROOT
#   venv-fa                        in place: pyvenv.cfg home, python links and .pth paths re-pointed the same way
#   $OPD_ROOT/models/Qwen3-{0.6B,1.7B,8B}   the three models of models.env (dereferenced copies; overnight.sh's
#                                  find_model looks there first, so a rebuilt container finds them without a download)
#   <base>/opd-env/hf              HF_HOME for later downloads (env.sh uses it once it exists)
# Each venv is checked (python runs from the copy: sys.prefix, base_prefix, torch / vLLM import) before anything is
# switched; then $OPD_ROOT/venv and $OPD_ROOT/G-OPD are re-pointed (atomic symlink swap) and models.env rewritten
# (models.env.prev kept). Tasks started afterwards use the copies; running tasks keep the files they have open (the
# originals are not touched). Idempotent: a finished copy is not made again. Result: logs/persist_env.json.
# Usage: bash persist_env.sh      (PE_PYCHECK: the import check, for tests)
set -uo pipefail
source "$(dirname "$0")/env.sh"
B=$(dirname "$OPD_ROOT"); E=$B/opd-env; L=$OPD_ROOT/logs; LOG=$L/persist_env.log; RJ=$L/persist_env.json; mkdir -p $E/python $L
echo '{}' > $RJ
put() { python3 -c "import json, sys; p, k, v = sys.argv[1:4]; d = json.load(open(p)); d[k] = v; json.dump(d, open(p, 'w'), indent=1)" "$RJ" "$1" "$2"; }
step() { echo "PERSIST $(date -u +%T) $*" | tee -a $LOG; }
finish() { put result "$1"; put reason "$2"; put finished_utc "$(date -u +%FT%TZ)"; step "$1 $2"; exit 0; }
LOW="nice -n 19"; command -v ionice > /dev/null 2>&1 && LOW="ionice -c3 $LOW"
T0=$(date +%s); put started_utc "$(date -u +%FT%TZ)"
PYCHECK=${PE_PYCHECK:-'import json, sys, torch, vllm; print(json.dumps({"prefix": sys.prefix, "base": sys.base_prefix, "torch": torch.__version__, "vllm": vllm.__version__}))'}
relocate() {   # <venv dir> <new python bin dir> <old>=<new> ...: pyvenv.cfg home, bin/ python links, text paths
  local V=$1 NH=$2; shift 2
  sed -i.bak "s|^home = .*|home = $NH|" $V/pyvenv.cfg && rm -f $V/pyvenv.cfg.bak
  for f in $V/bin/*; do   # python, python3, python3.10: links into the old home -> the same name in the new one
    [ -L "$f" ] || continue
    case "$(readlink "$f")" in */python*|python*) case "$(basename "$f")" in python|python3|python3.*) ln -sfn $NH/$(basename "$(readlink -f "$f" 2> /dev/null || readlink "$f")") "$f" 2> /dev/null; [ -e "$f" ] || ln -sfn $NH/python3 "$f";; esac;; esac
  done
  python3 - "$V" "$@" <<'PY'
import glob, os, sys
v, pairs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
files = [p for p in glob.glob(os.path.join(v, "bin", "*")) if os.path.isfile(p) and not os.path.islink(p)]
for pat in ("*.pth", "*.egg-link", "__editable__*.py", "*.dist-info/direct_url.json"):
    files += glob.glob(os.path.join(v, "lib", "python*", "site-packages", pat))
n = 0
for p in files:
    b = open(p, "rb").read()
    if b"\0" in b[:4096]:
        continue
    c = b
    for old, new in pairs:
        c = c.replace(old.encode(), new.encode())
    if c != b:
        open(p, "wb").write(c); n += 1
print(n)
PY
}
check() {   # <venv dir> -> 0 when its python runs from the copy and imports torch / vLLM
  local V=$1 out; out=$(cd / && env -u PYTHONPATH -u VIRTUAL_ENV $V/bin/python -c "$PYCHECK" 2>> $LOG | tail -1)
  echo "$out" >> $LOG
  python3 -c "import json, os, sys; d = json.loads(sys.argv[1]); v, e = sys.argv[2], sys.argv[3]
sys.exit(0 if os.path.realpath(d['prefix']) == os.path.realpath(v) and os.path.realpath(d['base']).startswith(os.path.realpath(e)) else 1)" "$out" "$V" "$E/python" 2>/dev/null; }

# 1. the venv in use and its python
V0=$(readlink -f $OPD_ROOT/venv); put venv_before "$V0"
[ -f $V0/pyvenv.cfg ] || finish FAIL "no pyvenv.cfg in the venv in use"
H=$(sed -n 's/^home = //p' $V0/pyvenv.cfg | head -1); PYR=$(readlink -f "$(dirname "$H")"); NAME=$(basename $PYR)
OLD=$(dirname $V0)   # the old base (venv and G-OPD lived there)
case "$PYR" in "$E"/*) step "python already on the share";; *)
  if [ ! -x $E/python/$NAME/bin/python3 ]; then
    step "copying python $NAME"; rm -rf $E/python/$NAME.tmp
    $LOW cp -a $PYR $E/python/$NAME.tmp && mv $E/python/$NAME.tmp $E/python/$NAME || finish FAIL "python copy failed"
  fi;; esac
NH=$E/python/$NAME/bin; [ -x $NH/python3 ] || finish FAIL "copied python has no bin/python3"
put python "$NAME"
# 2. G-OPD next to it when it is a link to the old base
GL=$(readlink -f $OPD_ROOT/G-OPD 2> /dev/null || true); G2=$GL
if [ -n "$GL" ] && [ -L $OPD_ROOT/G-OPD ] && [ "${GL#$E/}" = "$GL" ]; then
  if [ ! -d $E/G-OPD/verl ]; then rm -rf $E/G-OPD.tmp; $LOW cp -a $GL $E/G-OPD.tmp && mv $E/G-OPD.tmp $E/G-OPD || finish FAIL "G-OPD copy failed"; fi
  G2=$E/G-OPD
fi
# 3. the venv itself
if [ "$V0" != "$E/venv" ]; then
  if [ ! -f $E/venv/.persist_ok ]; then
    step "copying the venv ($(du -sh $V0 2> /dev/null | cut -f1))"; rm -rf $E/venv.tmp $E/venv
    $LOW cp -a $V0 $E/venv.tmp && mv $E/venv.tmp $E/venv || finish FAIL "venv copy failed"
    n=$(relocate $E/venv $NH "$V0=$E/venv" "$OPD_ROOT/venv=$E/venv" "${GL:-/nonexistent}=$G2"); put venv_files_relocated "$n"
    check $E/venv || finish FAIL "the copied venv does not run from the share (see logs/persist_env.log); nothing switched"
    touch $E/venv/.persist_ok
  fi
fi
# 4. venv-fa in place
if [ -f $OPD_ROOT/venv-fa/pyvenv.cfg ] && ! grep -q "^home = $NH\$" $OPD_ROOT/venv-fa/pyvenv.cfg; then
  n=$(relocate $OPD_ROOT/venv-fa $NH "${GL:-/nonexistent}=$G2"); put venv_fa_files_relocated "$n"
  check $OPD_ROOT/venv-fa || put venv_fa_check FAIL
fi
# 5. switch (only now: every copy above is complete and checked)
if [ "$V0" != "$E/venv" ]; then ln -sfn $E/venv $OPD_ROOT/venv.new && mv -T $OPD_ROOT/venv.new $OPD_ROOT/venv 2> /dev/null || { rm -f $OPD_ROOT/venv; mv $OPD_ROOT/venv.new $OPD_ROOT/venv; }; fi
if [ "$G2" != "$GL" ]; then ln -sfn $G2 $OPD_ROOT/G-OPD.new && mv -T $OPD_ROOT/G-OPD.new $OPD_ROOT/G-OPD 2> /dev/null || { rm -f $OPD_ROOT/G-OPD; mv $OPD_ROOT/G-OPD.new $OPD_ROOT/G-OPD; }; fi
put venv_after "$(readlink -f $OPD_ROOT/venv)"
# 6. the three models
cp -p $OPD_ROOT/models.env $OPD_ROOT/models.env.prev 2> /dev/null
for kv in STUDENT_06B:Qwen3-0.6B STUDENT_17B:Qwen3-1.7B TEACHER:Qwen3-8B; do
  k=${kv%%:*}; m=${kv#*:}; src=$(sed -n "s/^$k=//p" $OPD_ROOT/models.env | tail -1); dst=$OPD_ROOT/models/$m
  [ -n "$src" ] && [ -d "$src" ] || { put model_$m "not in models.env"; continue; }
  case "$(readlink -f $src)" in "$(readlink -f $OPD_ROOT)"/*) put model_$m "already under OPD_ROOT"; continue;; esac
  if [ ! -f $dst/.persist_ok ]; then
    step "copying $m"; rm -rf $dst.tmp; mkdir -p $OPD_ROOT/models
    $LOW cp -rL $src $dst.tmp && touch $dst.tmp/.persist_ok && rm -rf $dst && mv $dst.tmp $dst || { put model_$m "copy failed"; continue; }
  fi
  python3 -c "import json, os, sys; d = sys.argv[1]; i = os.path.join(d, 'model.safetensors.index.json')
need = set(json.load(open(i))['weight_map'].values()) if os.path.exists(i) else {'model.safetensors'}
sys.exit(0 if os.path.exists(os.path.join(d, 'config.json')) and all(os.path.getsize(os.path.join(d, f)) > 1000 for f in need) else 1)" $dst \
    || { put model_$m "copy incomplete"; continue; }
  sed -i.bak "s|^$k=.*|$k=$dst|" $OPD_ROOT/models.env && rm -f $OPD_ROOT/models.env.bak; put model_$m "copied"
done
mkdir -p $E/hf
put seconds "$(( $(date +%s) - T0 ))"
finish PASS "python, venv, G-OPD and models on the share; \$OPD_ROOT/venv and models.env point there for tasks started from now on"
