#!/usr/bin/env bash
# [v2.5.9c] (reviewer 10-21; Runming: "is ToolRL passed over too?") One public GitHub repository at a fixed commit onto the
# node, without a GPU and without the Mac's proxy: a shallow fetch of exactly that commit into <dest>.tmp, the commit
# checked, the .git directory dropped, every file's sha256 in .manifest.sha256, the data files described by
# code/agent/inspect_data.py in .summary.json (sizes, sha256, rows, the first record's fields, the licence's first lines),
# then moved to <dest> with .fetch_ok = the commit; the summary to the private repository (logs/<node>/git/<owner>__<repo>.json)
# and an event git_done / git_failed. run_tasks.sh starts it for the queue's "#git <owner>/<repo> <commit>" lines.
# [v2.5.9e] <dest> must lie under $OPD_ROOT/data/git/ (checked before anything is removed); mode `code` describes a code
# repository by its file list only (no data parsing); a repository with hf/<owner>__<name>/ folders (opd-lab-assets) gets
# each linked as $OPD_ROOT/data/hf/<owner>__<name> with a .fetch_ok (files, sizes, sha256, the commit), the placement
# layout the tool-call converters read; a real directory already there is left alone.
# Usage: bash git_fetch.sh <owner>/<repo> <40-hex commit> <dest> [code]      GIT_BASE (tests): another base URL than github.com
set -uo pipefail
source "$(dirname "$0")/env.sh"; R=$1; C=$2; D=$3; MODE=${4:-data}
fail() { echo "GIT_FETCH_FAILED $R $C: $1"; touch $D.git_fail; rm -rf $D.tmp; python3 $RUNBOOK/scripts/control.py event git_fetch git_failed "$R $C: $1" > /dev/null 2>&1; exit 1; }
[[ $R =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ && $C =~ ^[0-9a-f]{40}$ ]] || fail "bad repository or commit"
ab() { python3 -c 'import os, sys; print(os.path.abspath(sys.argv[1]))' "$1"; }
case "$(ab $D)/" in "$(ab $OPD_ROOT)/data/git/"?*/) ;; *) echo "GIT_FETCH_FAILED $R $C: destination outside data/git"; exit 1;; esac   # [v2.5.9e] before any rm
link_hf() {   # [v2.5.9e] hf/<owner>__<name>/ folders -> $OPD_ROOT/data/hf/<owner>__<name> (link) + .fetch_ok in the folder
  if [ -d $D/hf ]; then
    mkdir -p $OPD_ROOT/data/hf
    for x in $D/hf/*/; do
      n=$(basename $x); [[ $n =~ ^[A-Za-z0-9._-]+__[A-Za-z0-9._-]+$ ]] || continue; t=$OPD_ROOT/data/hf/$n
      if [ -e $t ] && [ ! -L $t ]; then echo "GIT_FETCH_LINK_SKIPPED $n: a placed copy is already there"; continue; fi
      python3 - "$D/hf/$n" "$R" "$C" <<'PY' || continue
import hashlib, json, os, sys, time
d, repo, commit = sys.argv[1:4]
files = []
for dp, dn, fn in os.walk(d):
    for f in sorted(fn):
        if f == ".fetch_ok":
            continue
        p = os.path.join(dp, f)
        files.append((os.path.relpath(p, d), os.path.getsize(p), hashlib.sha256(open(p, "rb").read()).hexdigest()))
src = open(os.path.join(d, "SOURCE.md")).read() if os.path.exists(os.path.join(d, "SOURCE.md")) else ""
hf = next((l.split("datasets/", 1)[1].strip() for l in src.splitlines() if "huggingface.co/datasets/" in l), None)
json.dump({"repo": hf, "kind": "dataset", "via": f"github:{repo}@{commit}", "files": sorted(files),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(os.path.join(d, ".fetch_ok"), "w"))
PY
      ln -sfn $D/hf/$n $t && echo "GIT_FETCH_LINKED $n"
    done
  fi
}
if [ "$MODE" = link ]; then   # [v2.5.9e] the copy is at the commit already: only the links (git_pass, when one is missing)
  [ "$(cat $D/.fetch_ok 2> /dev/null)" = $C ] || { echo "GIT_FETCH_FAILED $R $C: link mode needs the copy at the commit"; exit 1; }
  link_hf
  for x in $D/hf/*/; do [ -e $OPD_ROOT/data/hf/$(basename $x) ] || { touch $D.link_fail; exit 1; }; done   # retried after 30 min
  rm -f $D.link_fail; exit 0
fi
TO=""; command -v timeout > /dev/null && TO="timeout ${GIT_TIMEOUT:-1800}"
rm -rf $D.tmp; mkdir -p $D.tmp
( cd $D.tmp && git init -q && git remote add origin ${GIT_BASE:-https://github.com}/$R.git && $TO git fetch -q --depth 1 origin $C && git checkout -q FETCH_HEAD ) \
  || fail "fetch failed"
[ "$(git -C $D.tmp rev-parse HEAD)" = $C ] || fail "checked out $(git -C $D.tmp rev-parse HEAD), not $C"
rm -rf $D.tmp/.git
SHA="sha256sum"; command -v sha256sum > /dev/null || SHA="shasum -a 256"
( cd $D.tmp && find . -type f ! -name .manifest.sha256 -print0 | sort -z | xargs -0 $SHA > .manifest.sha256 ) || fail "manifest"
python3 $RUNBOOK/code/agent/inspect_data.py $D.tmp --repo $R --commit $C $([ "$MODE" = code ] && echo --list-only) > $D.tmp/.summary.json || fail "inspect_data"
rm -rf $D; mv $D.tmp $D; echo $C > $D/.fetch_ok; rm -f $D.git_fail
link_hf
python3 $RUNBOOK/scripts/remote.py put $D/.summary.json logs/$(python3 $RUNBOOK/scripts/control.py node)/git/${R/\//__}.json > /dev/null 2>&1 || echo "summary upload failed (kept in $D)"
n=$(grep -c . $D/.manifest.sha256); mb=$(du -sm $D | cut -f1)
python3 $RUNBOOK/scripts/control.py event git_fetch git_done "$R $C: $n files, $mb MB" > /dev/null 2>&1
echo "GIT_FETCH_OK $R $C: $n files, $mb MB"
