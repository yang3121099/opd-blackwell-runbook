#!/usr/bin/env bash
# [v2.5.9c] (reviewer 10-21; Runming: "is ToolRL passed over too?") One public GitHub repository at a fixed commit onto the
# node, without a GPU and without the Mac's proxy: a shallow fetch of exactly that commit into <dest>.tmp, the commit
# checked, the .git directory dropped, every file's sha256 in .manifest.sha256, the data files described by
# code/agent/inspect_data.py in .summary.json (sizes, sha256, rows, the first record's fields, the licence's first lines),
# then moved to <dest> with .fetch_ok = the commit; the summary to the private repository (logs/<node>/git/<owner>__<repo>.json)
# and an event git_done / git_failed. run_tasks.sh starts it for the queue's "#git <owner>/<repo> <commit>" lines.
# Usage: bash git_fetch.sh <owner>/<repo> <40-hex commit> <dest>      GIT_BASE (tests): another base URL than github.com
set -uo pipefail
source "$(dirname "$0")/env.sh"; R=$1; C=$2; D=$3
fail() { echo "GIT_FETCH_FAILED $R $C: $1"; touch $D.git_fail; rm -rf $D.tmp; python3 $RUNBOOK/scripts/control.py event git_fetch git_failed "$R $C: $1" > /dev/null 2>&1; exit 1; }
[[ $R =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ && $C =~ ^[0-9a-f]{40}$ ]] || fail "bad repository or commit"
TO=""; command -v timeout > /dev/null && TO="timeout ${GIT_TIMEOUT:-1800}"
rm -rf $D.tmp; mkdir -p $D.tmp
( cd $D.tmp && git init -q && git remote add origin ${GIT_BASE:-https://github.com}/$R.git && $TO git fetch -q --depth 1 origin $C && git checkout -q FETCH_HEAD ) \
  || fail "fetch failed"
[ "$(git -C $D.tmp rev-parse HEAD)" = $C ] || fail "checked out $(git -C $D.tmp rev-parse HEAD), not $C"
rm -rf $D.tmp/.git
SHA="sha256sum"; command -v sha256sum > /dev/null || SHA="shasum -a 256"
( cd $D.tmp && find . -type f ! -name .manifest.sha256 -print0 | sort -z | xargs -0 $SHA > .manifest.sha256 ) || fail "manifest"
python3 $RUNBOOK/code/agent/inspect_data.py $D.tmp --repo $R --commit $C > $D.tmp/.summary.json || fail "inspect_data"
rm -rf $D; mv $D.tmp $D; echo $C > $D/.fetch_ok; rm -f $D.git_fail
python3 $RUNBOOK/scripts/remote.py put $D/.summary.json logs/$(python3 $RUNBOOK/scripts/control.py node)/git/${R/\//__}.json > /dev/null 2>&1 || echo "summary upload failed (kept in $D)"
n=$(grep -c . $D/.manifest.sha256); mb=$(du -sm $D | cut -f1)
python3 $RUNBOOK/scripts/control.py event git_fetch git_done "$R $C: $n files, $mb MB" > /dev/null 2>&1
echo "GIT_FETCH_OK $R $C: $n files, $mb MB"
