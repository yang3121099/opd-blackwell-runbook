#!/usr/bin/env bash
# [v2.5.9c] Tests of git_fetch.sh, inspect_data.py and run_tasks.sh's git_pass against a local repository (no network).
# Usage: bash scripts/test_git_fetch.sh -> GIT_FETCH_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); cd $T
mkdir -p $T/rb/scripts $T/rb/code/agent $T/root/logs $T/bin $T/work
cp $SRC/scripts/git_fetch.sh $T/rb/scripts/; cp $SRC/code/agent/inspect_data.py $T/rb/code/agent/
echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
export CALLS=$T/calls; : > $CALLS
# a source repository with a licence and three data files, fetchable by commit id over file://
( cd $T/work && git init -q && git config user.email t@example.com && git config user.name t
  printf '                                 Apache License\n                           Version 2.0, January 2004\n' > LICENSE
  mkdir -p dataset/a && printf '{"tools": [{"name": "f"}], "query": "q1", "answer": "a1"}\n{"query": "q2"}\n' > dataset/a/train.jsonl
  printf '[{"id": 1, "text": "x"}, {"id": 2}]' > dataset/raw.json; printf 'a,b\n1,2\n3,4\n' > dataset/t.csv; echo "# readme" > README.md
  git add -A && git commit -q -m one && echo second >> README.md && git commit -qam two )
C1=$(git -C $T/work rev-parse HEAD~1); C2=$(git -C $T/work rev-parse HEAD)
mkdir -p $T/remote/own; git clone -q --bare $T/work $T/remote/own/repo.git; git -C $T/remote/own/repo.git config uploadpack.allowAnySHA1InWant true
export GIT_BASE=file://$T/remote
D=$T/root/data/git/own__repo
# 1. the older commit, exactly: manifest, summary, .fetch_ok, upload, event
bash $T/rb/scripts/git_fetch.sh own/repo $C1 $D > $T/out1 2>&1
ok "fetched at the commit asked for" '[ "$(cat $D/.fetch_ok)" = $C1 ] && ! grep -q second $D/README.md && [ ! -d $D/.git ] && [ ! -d $D.tmp ] && grep -q "^GIT_FETCH_OK own/repo $C1" $T/out1'
ok "manifest holds every file with sha256" 'grep -q "  ./dataset/a/train.jsonl$" $D/.manifest.sha256 && grep -q "  ./LICENSE$" $D/.manifest.sha256 && [ "$(grep -c . $D/.manifest.sha256)" = 5 ]'
summ=$(python3 - $D/.summary.json "$T" <<'PY'
import json, sys
v = json.load(open(sys.argv[1])); f = v["data_files"]
ok = (v["licence"]["spdx_guess"] == "Apache-2.0" and f["dataset/a/train.jsonl"]["rows"] == 2
      and sorted(f["dataset/a/train.jsonl"]["first"]) == ["answer", "query", "tools"] and f["dataset/raw.json"]["rows"] == 2
      and f["dataset/t.csv"]["rows"] == 2 and sys.argv[2] not in open(sys.argv[1]).read() and v["commit"] and v["repo"] == "own/repo")
print("SUMMARY_OK" if ok else json.dumps(v)[:400])
PY
)
ok "summary: licence, rows, first record fields, no absolute path" '[ "$summ" = SUMMARY_OK ]'
ok "summary to the private repository, event git_done" 'grep -q "^remote put $D/.summary.json logs/abcd1234/git/own__repo.json$" $CALLS && grep -q "^control event git_fetch git_done own/repo $C1: 5 files" $CALLS'
# 2. a commit that does not exist: failure marker, event, the earlier copy kept
bash $T/rb/scripts/git_fetch.sh own/repo 0123456789abcdef0123456789abcdef01234567 $D > $T/out2 2>&1; r2=$?
ok "a missing commit fails, keeps the earlier copy" '[ $r2 = 1 ] && [ -f $D.git_fail ] && [ ! -d $D.tmp ] && [ "$(cat $D/.fetch_ok)" = $C1 ] && grep -q "control event git_fetch git_failed own/repo 0123" $CALLS'
# 3. names and commits outside the pattern are refused before any network
bash $T/rb/scripts/git_fetch.sh "own/repo;rm" $C2 $T/root/x > $T/out3 2>&1; r3=$?; bash $T/rb/scripts/git_fetch.sh own/repo main $T/root/y > $T/out4 2>&1; r4=$?
ok "bad repository or commit refused" '[ $r3 = 1 ] && [ $r4 = 1 ] && grep -q "bad repository or commit" $T/out3 && grep -q "bad repository or commit" $T/out4'
# 4. the newer commit replaces the copy
rm -f $D.git_fail; bash $T/rb/scripts/git_fetch.sh own/repo $C2 $D > /dev/null 2>&1
ok "a new commit replaces the copy" '[ "$(cat $D/.fetch_ok)" = $C2 ] && grep -q second $D/README.md && [ ! -f $D.git_fail ]'
# 5. git_pass: starts git_fetch.sh for a #git line whose copy is missing or at another commit; not for one already there,
#    not within 30 min of a failure, not for a malformed line
printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; chmod +x $T/bin/setsid
printf 'echo "git_fetch $*" >> %s/starts\n' $T > $T/rb/scripts/git_fetch.sh.stub
gp() { ( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb Q=$T/q; cp $T/rb/scripts/git_fetch.sh.stub $T/rb/scripts/git_fetch.sh
         eval "$(sed -n '/^git_pass() {/,/; }$/p' $SRC/scripts/run_tasks.sh)"; git_pass
         for _ in $(seq 30); do [ -s $T/starts ] && break; sleep 0.1; done; sleep 0.2 ); }   # the start is in the background
printf '#git own/repo %s\n#git bad;name %s\n#git own/other main\n' $C2 $C2 > $T/q; : > $T/starts; gp
ok "git_pass: nothing to do when the copy is at the commit" '[ ! -s $T/starts ]'
printf '#git own/repo %s\n' $C1 > $T/q; gp
ok "git_pass: another commit starts git_fetch.sh" 'grep -q "^git_fetch own/repo $C1 $D$" $T/starts'
: > $T/starts; touch $D.git_fail; gp
ok "git_pass: not within 30 min of a failure" '[ ! -s $T/starts ]'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "GIT_FETCH_TEST PASS ($n checks)" || echo "GIT_FETCH_TEST FAIL ($fails of $n)"
