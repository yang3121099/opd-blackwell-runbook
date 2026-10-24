#!/usr/bin/env bash
# BUILD-VENV-IFEVAL (runbook v2.5.9z16; the reviewer's general-ability checks; Fable / ops-lead 10-22: a grading-only venv, the training
# venv unchanged): $OPD_ROOT/venv-ifeval for Google's IFEval checker (code/eval/ifeval/). run_queue_eval.sh's ifeval rows grade through
# code/eval/ifeval_grade.py in it. No torch or vLLM inside. The CPU lane, niced, idle I/O; no GPU, nothing that runs is touched.
#   1. guards (as BUILD-VENV-VL):
#      - the target is $OPD_ROOT/venv-ifeval (absolute, resolved);
#      - an existing venv-ifeval is left alone (exit 0, "exists") unless VENV_IFEVAL_REBUILD=1, which first renames it to
#        venv-ifeval.tmp;
#      - every rm is of $OPD_ROOT/venv-ifeval.tmp only (a leftover of a failed run is removed at the start);
#      - venv, venv-fa and venv-vl are only read (venv's base python).
#   2. a fresh venv on venv's own base python (3.10), without pip; then ensurepip's bundled pip, else uv. Nothing is copied in.
#   3. the pins from PyPI by wheel_fetch.py, each file checked against PyPI's sha256:
#      - IFEVAL_PINS: the checker's four packages and every dependency, so nothing is resolved online;
#      - IFEVAL_BUILD_PINS: setuptools / wheel, to build langdetect, which publishes only an sdist (WHEEL_SDIST=langdetect).
#      Installed offline: the build pins first, then the rest --no-index --no-deps --no-build-isolation; pip check must pass.
#   4. nltk's punkt_tab: nltk 3.9 splits sentences with it (the checker's count_sentences and word_tokenize).
#      - fetched from nltk_data at a fixed commit over the node's proxy (par_fetch, 1 MB segments): raw.githubusercontent.com,
#        else jsDelivr's copy of the same commit;
#      - checked against PUNKT_SHA256 before anything is unpacked;
#      - unzipped into venv-ifeval.tmp/nltk_data/tokenizers/: sys.prefix/nltk_data is on nltk's own search path, and the grader
#        reads only this one;
#      - how it came (host, seconds, MB/s, sha256): runs/BUILD-VENV-IFEVAL/nltk_data.json.
#   5. checks in venv-ifeval.tmp, no GPU visible: `ifeval_grade.py --selftest` (Google's own unit tests, punkt_tab loaded, the known
#      responses graded strict / loose as expected) and the four packages at their pinned versions. Any miss = FAIL.
#   6. the venv's own scripts rewritten from venv-ifeval.tmp to venv-ifeval, then renamed into place (one rename); its python starts.
# Report: runs/BUILD-VENV-IFEVAL/{report.json, requirements.lock (pip freeze), wheels/manifest.json (sha256 of each file),
# nltk_data.json, selftest.json, build.log} to logs/<node>/venv_ifeval/ (private). Events venv_ifeval_built / venv_ifeval_exists /
# venv_ifeval_failed; .done after a build. Usage: bash build_venv_ifeval.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=BUILD-VENV-IFEVAL; D=$OPD_ROOT/runs/$T; LOG=$D/build.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "BUILD_VENV_IFEVAL $(date -u +%T) $*" | tee -a $LOG; }
ROOT=$(readlink -f "$OPD_ROOT"); DST=$ROOT/venv-ifeval; TMP=$ROOT/venv-ifeval.tmp; SRCV=${VENV_IFEVAL_FROM:-$ROOT/venv}
PINS=${IFEVAL_PINS:-nltk==3.9.1 langdetect==1.0.9 absl-py==2.1.0 immutabledict==4.2.1 click==8.1.7 joblib==1.4.2 regex==2024.11.6 tqdm==4.67.1 six==1.17.0}
BUILD_PINS=${IFEVAL_BUILD_PINS:-setuptools==75.3.0 wheel==0.45.1}
PUNKT_COMMIT=4f15a3d89eefe9748ec1c05be495d91289197155   # nltk_data, 2025-02-17 "Add Malayalam to punkt_tab" (punkt_tab.zip's latest change)
PUNKT_SIZE=${PUNKT_SIZE:-4319076}; PUNKT_SHA256=${PUNKT_SHA256:-e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106}
PUNKT_URLS=${PUNKT_URLS:-https://raw.githubusercontent.com/nltk/nltk_data/$PUNKT_COMMIT/packages/tokenizers/punkt_tab.zip https://cdn.jsdelivr.net/gh/nltk/nltk_data@$PUNKT_COMMIT/packages/tokenizers/punkt_tab.zip}
NI="nice -n 19"; command -v ionice > /dev/null && NI="$NI ionice -c3"
finish() {   # $1 = PASS / FAIL / EXISTS, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
for k in ("wheels/manifest", "nltk_data", "selftest"):
    p = os.path.join(d, k + ".json")
    if os.path.exists(p):
        try:
            r[k.split("/")[-1]] = json.load(open(p))
        except ValueError:
            r[k.split("/")[-1]] = "unreadable"
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $LOG $D/up/ 2> /dev/null; cp $D/requirements.lock $D/wheels/manifest.json $D/nltk_data.json $D/selftest.json $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/venv_ifeval >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  case $1 in PASS) ev=venv_ifeval_built;; EXISTS) ev=venv_ifeval_exists;; *) ev=venv_ifeval_failed;; esac
  python3 $RUNBOOK/scripts/control.py event $T $ev "$2" > /dev/null 2>&1
  step "$1 $2"; [ $1 = FAIL ] || touch $D/.done; exit 0
}
rm_tmp() {   # the only rm of this script: $OPD_ROOT/venv-ifeval.tmp, resolved, and nothing else
  [ -e "$TMP" ] || [ -L "$TMP" ] || return 0
  [ "$(readlink -f "$TMP")" = "$ROOT/venv-ifeval.tmp" ] && [ ! -L "$TMP" ] || { step "REFUSED: $TMP is not a plain directory at \$OPD_ROOT/venv-ifeval.tmp"; finish FAIL "venv-ifeval.tmp is a link or elsewhere"; }
  rm -rf -- "$ROOT/venv-ifeval.tmp"
}
# 1. guards
[ "$DST" = "$ROOT/venv-ifeval" ] && [ "$TMP" = "$ROOT/venv-ifeval.tmp" ] || finish FAIL "target paths not under \$OPD_ROOT"
if [ -e "$DST" ] || [ -L "$DST" ]; then
  [ "${VENV_IFEVAL_REBUILD:-0}" = 1 ] || finish EXISTS "venv-ifeval exists (set VENV_IFEVAL_REBUILD=1 to build it again)"
  [ -L "$DST" ] && finish FAIL "venv-ifeval is a link: not renamed"
  rm_tmp; mv -- "$DST" "$TMP" && step "rebuild: the old venv-ifeval moved to venv-ifeval.tmp"
fi
rm_tmp
[ -x "$SRCV/bin/python" ] || finish FAIL "no $(basename $SRCV) to take the base python from"
step "start base=$(basename $SRCV) pins=[$PINS] build=[$BUILD_PINS]"
# 2. a fresh venv on the base python, nothing copied in; pip from ensurepip, else uv (IFEVAL_PIP: the stub tests' pip)
BASEPY=$($SRCV/bin/python -c 'import sys; print(sys._base_executable)' 2>> $LOG)
[ -x "$BASEPY" ] || finish FAIL "base python of $(basename $SRCV) not found"
$BASEPY -m venv --without-pip --prompt venv-ifeval "$TMP" >> $LOG 2>&1 || finish FAIL "python -m venv failed"
PY=$TMP/bin/python; PYV=$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>> $LOG)
NOCACHE=--no-cache-dir
if [ -n "${IFEVAL_PIP:-}" ]; then PIP=$IFEVAL_PIP
elif $PY -m ensurepip --default-pip >> $LOG 2>&1 && $PY -m pip --version > /dev/null 2>&1; then PIP="$PY -m pip"
elif command -v uv > /dev/null; then PIP="uv pip --python $PY"; NOCACHE=--no-cache
else finish FAIL "no pip for venv-ifeval.tmp (no ensurepip, no uv)"; fi
step "venv on python $PYV, pip: ${PIP%% *}"
# 3. the pins, sha256-checked, installed offline: setuptools / wheel first (langdetect's sdist is built with them), then the rest
WHEEL_SDIST=langdetect $NI python3 $RUNBOOK/scripts/wheel_fetch.py $D/wheels $BUILD_PINS $PINS >> $LOG 2>&1 || finish FAIL "wheels: $(grep -h '^WHEEL.*Error\|^WHEEL.*sha256 ' $LOG | tail -2 | tr '\n' ' ')"
step "wheels in: $(grep '^PROBE_MBPS' $LOG | tail -1); $(grep -c '^WHEEL' $LOG) lines"
$NI $PIP install --no-index $NOCACHE --find-links $D/wheels --no-deps $BUILD_PINS >> $LOG 2>&1 || finish FAIL "offline install of the build pins failed"
$NI $PIP install --no-index $NOCACHE --find-links $D/wheels --no-deps --no-build-isolation $PINS >> $LOG 2>&1 || finish FAIL "offline install of the pins failed"
$PIP check >> $LOG 2>&1 || finish FAIL "pip check: $(tail -2 $LOG | tr '\n' ' ')"
$PIP freeze > $D/requirements.lock 2>> $LOG
# 4. punkt_tab, sha256-checked before it is unpacked, into the venv's own nltk_data
$NI python3 - "$D/nltk" "$TMP/nltk_data/tokenizers" "$PUNKT_SIZE" "$PUNKT_SHA256" "$RUNBOOK/scripts" $PUNKT_URLS > $D/nltk_data.json 2>> $LOG <<'PY'
import hashlib, json, os, re, sys, time, zipfile
out, dst, size, want, scripts = sys.argv[1:6]
sys.path.insert(0, scripts)
import par_fetch
os.makedirs(out, exist_ok=True)
z, size = os.path.join(out, "punkt_tab.zip"), int(size)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
r = {"file": "punkt_tab.zip", "size": size, "sha256": want, "tries": []}
for url in sys.argv[6:]:
    host = re.sub(r"^https?://([^/]+)/.*$", r"\1", url)
    if os.path.exists(z) and sha(z) == want:
        r["kept"] = True; break
    t0 = time.time()
    try:
        par_fetch.download(url, z, size, int(os.environ.get("PUNKT_CONNS", "4")), seg=1 << 20)
        got = sha(z)
        if got != want:
            os.remove(z); raise ValueError(f"sha256 {got[:16]} != pinned {want[:16]}")
        dt = max(time.time() - t0, 1e-6)
        r.update(host=host, seconds=round(dt, 2), MBps=round(size / 1e6 / dt, 3)); r["tries"].append({"host": host, "ok": True}); break
    except Exception as e:
        r["tries"].append({"host": host, "error": f"{type(e).__name__}: {e}"[:300]})
if not (os.path.exists(z) and sha(z) == want):
    r["ok"] = False; print(json.dumps(r)); sys.exit(1)
with zipfile.ZipFile(z) as f:
    names = f.namelist()
    if not names or any(not n.startswith("punkt_tab/") or n.startswith("/") or ".." in n.split("/") for n in names):
        r.update(ok=False, error="unexpected member paths"); print(json.dumps(r)); sys.exit(1)
    os.makedirs(dst, exist_ok=True); f.extractall(dst)
r.update(ok=True, members=len(names)); print(json.dumps(r))
PY
[ $? = 0 ] || finish FAIL "punkt_tab: $(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('tries'))" $D/nltk_data.json 2> /dev/null | cut -c1-300)"
step "punkt_tab in: $(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(d.get('host', 'kept'), d.get('MBps'), 'MB/s', d['members'], 'files')" $D/nltk_data.json 2>> $LOG)"
# 5. checks, no GPU visible: the selftest (Google's tests, punkt_tab, the known responses) and the pinned versions
CUDA_VISIBLE_DEVICES= $NI $PY -I $RUNBOOK/code/eval/ifeval_grade.py --selftest > $D/selftest.out 2>> $LOG
sed -n 's/^IFEVAL_SELFTEST //p' $D/selftest.out > $D/selftest.json
python3 - "$D/selftest.json" $PINS > $D/check_verdict.txt 2>> $LOG <<'PY'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except ValueError:
    print("the selftest did not run"); sys.exit()
pins = dict(p.split("==", 1) for p in sys.argv[2:])
bad = [] if c.get("ok") is True else ["selftest"]
bad += [f"{k} version" for k in ("nltk", "langdetect", "absl-py", "immutabledict") if c.get("versions", {}).get(k) != pins.get(k)]
print(" ".join(bad))
PY
BAD=$(cat $D/check_verdict.txt 2> /dev/null); [ -z "$BAD" ] || finish FAIL "checks: $BAD"
# 6. the venv's own scripts pointed at venv-ifeval, one rename into place
python3 - "$TMP" "$DST" <<'PY' >> $LOG 2>&1
import os, sys
old, new = sys.argv[1].encode(), sys.argv[2].encode()
paths = [os.path.join(sys.argv[1], "pyvenv.cfg")] + [os.path.join(sys.argv[1], "bin", f) for f in os.listdir(os.path.join(sys.argv[1], "bin"))]
for p in paths:   # text files only (activate scripts, console scripts' first lines, pyvenv.cfg); links and binaries left alone
    if os.path.islink(p) or not os.path.isfile(p):
        continue
    b = open(p, "rb").read()
    if b"\0" in b[:4096] or old not in b:
        continue
    open(p, "wb").write(b.replace(old, new)); print("REWRITTEN", os.path.basename(p))
PY
[ -z "$(grep -rIl -- "$TMP" "$TMP/bin" "$TMP/pyvenv.cfg" 2> /dev/null)" ] || finish FAIL "venv-ifeval.tmp still named in its scripts"
mv -- "$TMP" "$DST" || finish FAIL "rename to venv-ifeval failed"
V=$(CUDA_VISIBLE_DEVICES= $DST/bin/python -I -c 'import nltk, langdetect, absl, immutabledict; print(nltk.__version__)' 2>> $LOG) || finish FAIL "venv-ifeval does not start after the rename"
finish PASS "venv-ifeval built: nltk $V, punkt_tab from $(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(d.get('host', 'kept'), d.get('MBps'), 'MB/s')" $D/nltk_data.json), selftest $(python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(' / '.join(v['summary'] for v in d['google_tests'].values()), '/', sum(k['ok'] for k in d['known']), 'known')" $D/selftest.json 2> /dev/null), $(grep '^PROBE_MBPS' $LOG | tail -1)"
