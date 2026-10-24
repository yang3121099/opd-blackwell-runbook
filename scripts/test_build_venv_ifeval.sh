#!/usr/bin/env bash
# [v2.5.9z16] Stub tests of build_venv_ifeval.sh (no network, GPU or HF).
# Real: a throwaway source venv (its base python is used); the venv module.
# Fake: wheel_fetch.py, par_fetch.py (copies a fixture punkt_tab.zip, or garbage, or fails for one host), pip (IFEVAL_PIP: writes
# fake nltk / langdetect / absl / immutabledict into the venv it is given) and code/eval/ifeval_grade.py --selftest.
# Checked:
# - a PASS builds venv-ifeval with punkt_tab in its own nltk_data, the build pins installed before the rest, offline, langdetect as
#   an sdist;
# - venv / venv-fa / venv-vl untouched; an existing venv-ifeval is left alone; VENV_IFEVAL_REBUILD=1 rebuilds;
# - jsDelivr when raw.githubusercontent.com fails;
# - FAIL (no venv-ifeval, no .done) on a punkt_tab sha256 mismatch (nothing unpacked), a zip member outside punkt_tab/, a failed
#   selftest, a version off its pin, pip check, a wheel's sha256;
# - the only path removed is venv-ifeval.tmp (a link there is refused).
# Usage: bash scripts/test_build_venv_ifeval.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
mkzip() {   # $1 = zip path, $2 = extra member name or "": a punkt_tab-shaped zip; prints "<size> <sha256>"
  python3 - "$1" "$2" <<'PY'
import hashlib, os, sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w") as z:
    z.writestr("punkt_tab/README", "punkt_tab\n"); z.writestr("punkt_tab/english/collocations.tab", "a\tb\n")
    z.writestr("punkt_tab/english/sent_starters.txt", "however\n")
    if sys.argv[2]:
        z.writestr(sys.argv[2], "x\n")
b = open(sys.argv[1], "rb").read(); print(len(b), hashlib.sha256(b).hexdigest())
PY
}
setup() {
  T=$(mktemp -d); T=$(cd $T && pwd -P); mkdir -p $T/rb/scripts $T/rb/code/eval $T/root/runs $T/root/venv-fa $T/root/venv-vl
  cp $SRC/scripts/build_venv_ifeval.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  printf 'import json, os, sys\nopen(os.environ["CALLS"], "a").write("wheel_fetch WHEEL_SDIST=" + os.environ.get("WHEEL_SDIST", "") + " " + " ".join(sys.argv[1:]) + "\\n")\nos.makedirs(sys.argv[1], exist_ok=True)\nif os.environ.get("FAKE_WHEEL_FAIL"):\n    print("WHEEL nltk-3.9.1-py3-none-any.whl 1 ab sha256 00 != PyPI 11"); sys.exit(1)\njson.dump({"wheels": [{"name": p.split("==")[0], "sha256": "ab" * 32} for p in sys.argv[2:]], "probe_MBps": 0.3, "ok": True}, open(os.path.join(sys.argv[1], "manifest.json"), "w"))\nprint("PROBE_MBPS 0.3")\n' > $T/rb/scripts/wheel_fetch.py
  cat > $T/rb/scripts/par_fetch.py <<'PY'
import os, shutil
def download(url, dst, size, conns, seg=16 << 20, **kw):
    open(os.environ["CALLS"], "a").write(f"par_fetch {url} conns={conns} seg={seg}\n")
    if os.environ.get("FAKE_PUNKT_DOWN_HOST") and os.environ["FAKE_PUNKT_DOWN_HOST"] in url:
        raise OSError("Tunnel connection failed: 403 Forbidden")
    if os.environ.get("FAKE_PUNKT_BAD"):
        open(dst, "wb").write(b"\0" * size); return
    shutil.copy(os.environ["FIXTURE_ZIP"], dst)
PY
  cat > $T/fakepip.py <<'PY'
import os, sys, sysconfig
venv, a = sys.argv[1], sys.argv[2:]
open(os.environ["CALLS"], "a").write("pip " + " ".join(a) + "\n")
if a[:1] == ["check"]:
    sys.exit(1 if os.environ.get("FAKE_PIP_CHECK_FAIL") else 0)
if a[:1] == ["freeze"]:
    print("nltk==3.9.1\nlangdetect==1.0.9"); sys.exit(0)
if a[:1] == ["install"] and "--no-build-isolation" in a:
    sp = [os.path.join(venv, "lib", d, "site-packages") for d in os.listdir(os.path.join(venv, "lib"))][0]
    for p in [x for x in a if "==" in x]:
        name, ver = p.split("==")
        if name == "nltk" and os.environ.get("FAKE_PIP_VERSION_OFF"):
            ver = "3.8.1"
        mod = {"absl-py": "absl"}.get(name, name)
        os.makedirs(os.path.join(sp, mod), exist_ok=True)
        open(os.path.join(sp, mod, "__init__.py"), "w").write(f"__version__ = {ver!r}\n")
        di = os.path.join(sp, f"{name.replace('-', '_')}-{ver}.dist-info"); os.makedirs(di, exist_ok=True)
        open(os.path.join(di, "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
    open(os.path.join(venv, "bin", "nltk"), "w").write("#!" + os.path.join(venv, "bin", "python") + "\nprint(1)\n")
sys.exit(0)
PY
  cat > $T/rb/code/eval/ifeval_grade.py <<'PY'
import importlib.metadata as md, json, os, sys
open(os.environ["CALLS"], "a").write("selftest " + sys.prefix + "\n")
punkt = os.path.isdir(os.path.join(sys.prefix, "nltk_data", "tokenizers", "punkt_tab", "english"))
v = {p: md.version(p) for p in ("nltk", "langdetect", "absl-py", "immutabledict")}
print("IFEVAL_SELFTEST " + json.dumps({"ok": punkt and not os.environ.get("FAKE_SELFTEST_FAIL"), "versions": v, "punkt_tab_in_venv": punkt,
      "google_tests": {"instructions_util_test": {"rc": 0, "summary": "Ran 7 tests OK"}, "instructions_test": {"rc": 0, "summary": "Ran 41 tests OK"}},
      "known": [{"ok": True}] * 5}))
PY
  python3 -m venv --without-pip $T/root/venv; echo keep > $T/root/venv/marker; echo fa > $T/root/venv-fa/marker; echo vl > $T/root/venv-vl/marker
  export CALLS=$T/calls IFEVAL_PIP="python3 $T/fakepip.py $T/root/venv-ifeval.tmp" FIXTURE_ZIP=$T/punkt_tab.zip; : > $CALLS
  read -r PUNKT_SIZE PUNKT_SHA256 <<< "$(mkzip $T/punkt_tab.zip "")"; export PUNKT_SIZE PUNKT_SHA256
  SUMS=$(cd $T/root && find venv venv-fa venv-vl -type f | sort | xargs cksum)
}
D=runs/BUILD-VENV-IFEVAL; VI=venv-ifeval
res() { python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ''))" $T/root/$D/report.json "$1"; }
untouched() { [ "$(cd $T/root && find venv venv-fa venv-vl -type f | sort | xargs cksum)" = "$SUMS" ]; }
failed_clean() { [ "$(res result)" = FAIL ] && [ ! -e $T/root/$VI ] && [ ! -f $T/root/$D/.done ] && grep -q "^control event BUILD-VENV-IFEVAL venv_ifeval_failed" $CALLS && untouched; }
# 1. PASS
setup
bash $T/rb/scripts/build_venv_ifeval.sh > $T/out 2>&1
ok "PASS: venv-ifeval built, no venv-ifeval.tmp left, .done, event, report to logs/<node>/venv_ifeval" '[ "$(res result)" = PASS ] && [ -x $T/root/$VI/bin/python ] && [ ! -e $T/root/$VI.tmp ] && [ -f $T/root/$D/.done ] && grep -q "^control event BUILD-VENV-IFEVAL venv_ifeval_built" $CALLS && grep -q "^remote put $T/root/$D/up logs/abcd1234/venv_ifeval$" $CALLS && [[ "$(res reason)" == *"nltk 3.9.1, punkt_tab from raw.githubusercontent.com"*"Ran 41 tests OK / 5 known"* ]]'
ok "wheels: langdetect as an sdist (WHEEL_SDIST), the build pins and the pins in one fetch" 'grep -q "^wheel_fetch WHEEL_SDIST=langdetect $T/root/$D/wheels setuptools==75.3.0 wheel==0.45.1 nltk==3.9.1 langdetect==1.0.9 absl-py==2.1.0 immutabledict==4.2.1 click==8.1.7 joblib==1.4.2 regex==2024.11.6 tqdm==4.67.1 six==1.17.0$" $CALLS'
ok "offline installs: setuptools / wheel first, then the pins --no-build-isolation; pip check; freeze" 'a=$(grep -n "^pip install --no-index --no-cache-dir --find-links $T/root/$D/wheels --no-deps setuptools==75.3.0 wheel==0.45.1$" $CALLS | cut -d: -f1); b=$(grep -n "^pip install --no-index --no-cache-dir --find-links $T/root/$D/wheels --no-deps --no-build-isolation nltk==3.9.1 .* six==1.17.0$" $CALLS | cut -d: -f1); [ -n "$a" ] && [ -n "$b" ] && [ $a -lt $b ] && grep -q "^pip check$" $CALLS && [ -s $T/root/$D/requirements.lock ]'
ok "punkt_tab fetched (1 MB segments, raw.githubusercontent.com at the pinned commit), sha256 in nltk_data.json, unzipped into the venv" 'grep -q "^par_fetch https://raw.githubusercontent.com/nltk/nltk_data/4f15a3d89eefe9748ec1c05be495d91289197155/packages/tokenizers/punkt_tab.zip conns=4 seg=1048576$" $CALLS && [ -f $T/root/$VI/nltk_data/tokenizers/punkt_tab/english/collocations.tab ] && python3 -c "import json, sys; d = json.load(open(sys.argv[1])); assert d[\"ok\"] and d[\"host\"] == \"raw.githubusercontent.com\" and d[\"sha256\"] == sys.argv[2] and d[\"members\"] == 3" $T/root/$D/nltk_data.json $PUNKT_SHA256'
ok "the selftest ran in venv-ifeval.tmp; venv-ifeval's scripts point at venv-ifeval, none at .tmp; venv / venv-fa / venv-vl untouched" 'grep -q "^selftest $T/root/$VI.tmp$" $CALLS && ! grep -rIl "$VI.tmp" $T/root/$VI/bin $T/root/$VI/pyvenv.cfg 2> /dev/null | grep -q . && grep -q "$T/root/$VI\"*$" $T/root/$VI/bin/activate && head -1 $T/root/$VI/bin/nltk | grep -q "^#!$T/root/$VI/bin/python$" && untouched && [ "$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))[\"selftest\"][\"versions\"][\"nltk\"])" $T/root/$D/report.json)" = 3.9.1 ]'
# 2. an existing venv-ifeval is left alone (EXISTS, .done, nothing pulled)
rm -f $T/root/$D/.done; : > $CALLS; echo keep > $T/root/$VI/marker
bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "an existing venv-ifeval: EXISTS, untouched, no pull, .done (the #after reruns fire)" '[ "$(res result)" = EXISTS ] && [ -f $T/root/$VI/marker ] && ! grep -q "^wheel_fetch\|^par_fetch" $CALLS && [ -f $T/root/$D/.done ] && grep -q "venv_ifeval_exists" $CALLS'
# 3. VENV_IFEVAL_REBUILD=1: the old one moved to venv-ifeval.tmp, removed there, built again
rm -f $T/root/$D/.done; : > $CALLS
VENV_IFEVAL_REBUILD=1 bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "VENV_IFEVAL_REBUILD=1 builds again (the old marker gone)" '[ "$(res result)" = PASS ] && [ ! -f $T/root/$VI/marker ] && [ -x $T/root/$VI/bin/python ] && grep -q "^wheel_fetch" $CALLS && untouched'
cd $SRC; rm -rf $T
# 4. raw.githubusercontent.com refused by the proxy: jsDelivr's copy of the same commit
setup; FAKE_PUNKT_DOWN_HOST=raw.githubusercontent.com bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "first host down: punkt_tab from cdn.jsdelivr.net, the refusal kept in nltk_data.json" '[ "$(res result)" = PASS ] && grep -q "^par_fetch https://cdn.jsdelivr.net/gh/nltk/nltk_data@4f15a3d89eefe9748ec1c05be495d91289197155/packages/tokenizers/punkt_tab.zip " $CALLS && python3 -c "import json, sys; d = json.load(open(sys.argv[1])); assert d[\"host\"] == \"cdn.jsdelivr.net\" and \"403\" in d[\"tries\"][0][\"error\"]" $T/root/$D/nltk_data.json'
cd $SRC; rm -rf $T
# 5. punkt_tab's sha256 off on both hosts: FAIL, nothing unpacked, no install skipped silently
setup; FAKE_PUNKT_BAD=1 bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "punkt_tab sha256 mismatch: FAIL (punkt_tab), nothing unpacked, the bad file removed" 'failed_clean && [[ "$(res reason)" == punkt_tab:*"sha256"* ]] && [ ! -e $T/root/$VI.tmp/nltk_data ] && [ ! -e $T/root/$D/nltk/punkt_tab.zip ]'
cd $SRC; rm -rf $T
# 6. a zip member outside punkt_tab/ (its sha256 pinned): FAIL, nothing extracted
setup; read -r PUNKT_SIZE PUNKT_SHA256 <<< "$(mkzip $T/punkt_tab.zip "../evil.txt")"
bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "a member outside punkt_tab/: FAIL, nothing extracted anywhere" 'failed_clean && [ ! -e $T/root/$VI.tmp/nltk_data ] && [ ! -e $T/root/$VI.tmp/evil.txt ] && ! find $T -name evil.txt | grep -q .'
cd $SRC; rm -rf $T
# 7. the selftest fails / a version off its pin / pip check: FAIL on the checks, no venv-ifeval
setup; FAKE_SELFTEST_FAIL=1 bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "the selftest fails: FAIL (checks: selftest)" 'failed_clean && [ "$(res reason)" = "checks: selftest" ]'
: > $CALLS; FAKE_PIP_VERSION_OFF=1 bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "nltk 3.8.1 installed: FAIL (nltk version); the leftover venv-ifeval.tmp of the run before was removed and rebuilt" 'failed_clean && [ "$(res reason)" = "checks: nltk version" ]'
: > $CALLS; FAKE_PIP_CHECK_FAIL=1 bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "pip check fails: FAIL (pip check), no punkt_tab fetched" 'failed_clean && [[ "$(res reason)" == "pip check"* ]] && ! grep -q "^par_fetch" $CALLS'
: > $CALLS; FAKE_WHEEL_FAIL=1 bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "a wheel failing its sha256: FAIL before any install" 'failed_clean && [[ "$(res reason)" == wheels:* ]] && ! grep -q "^pip install" $CALLS'
cd $SRC; rm -rf $T
# 8. venv-ifeval.tmp is a link to elsewhere: refused, the link target kept
setup; mkdir -p $T/elsewhere; echo precious > $T/elsewhere/f; ln -s $T/elsewhere $T/root/$VI.tmp
bash $T/rb/scripts/build_venv_ifeval.sh > /dev/null 2>&1
ok "venv-ifeval.tmp a link: FAIL, the target untouched" '[ "$(res result)" = FAIL ] && [ -f $T/elsewhere/f ] && [[ "$(res reason)" == *"link"* ]] && untouched'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "BUILD_VENV_IFEVAL_TEST PASS ($n checks)" || echo "BUILD_VENV_IFEVAL_TEST FAIL ($fails of $n)"
