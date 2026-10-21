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
  git add -A && git commit -q -m one && echo second >> README.md
  mkdir -p hf/own__ds hf/own__placed && printf -- '- Source: https://huggingface.co/datasets/own/ds\n' > hf/own__ds/SOURCE.md && echo '[1]' > hf/own__ds/d.json && echo x > hf/own__placed/f.txt
  git add -A && git commit -qm two )
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
# [v2.5.9e] hf/<owner>__<name>/ linked into data/hf with a .fetch_ok; a real placed directory left alone
ok "[v2.5.9e] hf folder linked as data/hf/own__ds with .fetch_ok (repo from SOURCE.md, the commit)" '[ -L $T/root/data/hf/own__ds ] && [ -f $T/root/data/hf/own__ds/d.json ] && python3 -c "import json, sys; d = json.load(open(sys.argv[1])); assert d[\"repo\"] == \"own/ds\" and d[\"via\"].endswith(sys.argv[2]) and [f[0] for f in d[\"files\"]] == [\"SOURCE.md\", \"d.json\"]" $T/root/data/hf/own__ds/.fetch_ok $C2'
mkdir -p $T/root/data/hf/own__keep; rm -f $D.git_fail; mkdir -p $T/work/hf/own__keep && echo y > $T/work/hf/own__keep/g.txt && (cd $T/work && git add -A && git commit -qm three) && C3=$(git -C $T/work rev-parse HEAD) && git -C $T/work push -q $T/remote/own/repo.git HEAD:refs/heads/c3
bash $T/rb/scripts/git_fetch.sh own/repo $C3 $D > $T/out5 2>&1
ok "[v2.5.9e] a real directory at data/hf/<name> is not replaced by a link" '[ ! -L $T/root/data/hf/own__keep ] && grep -q "GIT_FETCH_LINK_SKIPPED own__keep" $T/out5 && [ -L $T/root/data/hf/own__ds ]'
# [v2.5.9e] a destination outside data/git is refused before anything is removed; code mode lists files only
mkdir -p $T/root/elsewhere && echo keep > $T/root/elsewhere/k
bash $T/rb/scripts/git_fetch.sh own/repo $C3 $T/root/elsewhere > $T/out6 2>&1; r6=$?
bash $T/rb/scripts/git_fetch.sh own/repo $C3 $T/root/data/git > $T/out7 2>&1; r7=$?
ok "[v2.5.9e] destinations outside data/git/<name> refused, nothing removed" '[ $r6 = 1 ] && [ $r7 = 1 ] && grep -q "destination outside data/git" $T/out6 && [ -f $T/root/elsewhere/k ] && [ -d $T/root/data/git/own__repo ]'
bash $T/rb/scripts/git_fetch.sh own/repo $C3 $T/root/data/git/own__code code > $T/out8 2>&1
ok "[v2.5.9e] code mode: the summary lists files only" 'python3 -c "import json, sys; d = json.load(open(sys.argv[1])); assert d[\"list_only\"] is True and d[\"files\"] > 0 and \"data_files\" not in d" $T/root/data/git/own__code/.summary.json'
# [v2.5.9e] link mode (real script): restores a removed link without fetching; refused when the copy is not at the commit
rm -f $T/root/data/hf/own__ds; bash $T/rb/scripts/git_fetch.sh own/repo $C3 $D link > $T/out9 2>&1; r9=$?
bash $T/rb/scripts/git_fetch.sh own/repo $C1 $D link > $T/out10 2>&1; r10=$?
ok "[v2.5.9e] link mode restores the link; another commit refused" '[ $r9 = 0 ] && [ -L $T/root/data/hf/own__ds ] && grep -q "GIT_FETCH_LINKED own__ds" $T/out9 && [ $r10 = 1 ] && grep -q "needs the copy at the commit" $T/out10'
# 5. git_pass: starts git_fetch.sh for a #git line whose copy is missing or at another commit; not for one already there,
#    not within 30 min of a failure, not for a malformed line
printf '#!/usr/bin/env bash\nexec "$@"\n' > $T/bin/setsid; chmod +x $T/bin/setsid
printf 'echo "git_fetch $*" >> %s/starts\n' $T > $T/rb/scripts/git_fetch.sh.stub
gp() { ( PATH=$T/bin:$PATH OPD_ROOT=$T/root RUNBOOK=$T/rb Q=$T/q; cp $T/rb/scripts/git_fetch.sh.stub $T/rb/scripts/git_fetch.sh
         eval "$(sed -n '/^git_pass() {/,/; }$/p' $SRC/scripts/run_tasks.sh)"; git_pass
         for _ in $(seq 30); do [ -s $T/starts ] && break; sleep 0.1; done; sleep 0.2 ); }   # the start is in the background
printf '#git own/repo %s\n#git bad;name %s\n#git own/other main\n' $C3 $C3 > $T/q; : > $T/starts; gp
ok "git_pass: nothing to do when the copy is at the commit" '[ ! -s $T/starts ]'
printf '#git own/repo %s\n' $C1 > $T/q; gp
ok "git_pass: another commit starts git_fetch.sh" 'grep -q "^git_fetch own/repo $C1 $D$" $T/starts'
: > $T/starts; printf '#git own/other %s code\n' $C1 > $T/q; gp
ok "[v2.5.9e] git_pass passes code mode on" 'grep -q "^git_fetch own/other $C1 $T/root/data/git/own__other code$" $T/starts'
# [v2.5.9e] a copy at its commit with a missing data/hf link: git_pass runs link mode (not again within 30 min of a link failure)
: > $T/starts; rm -f $T/root/data/hf/own__ds; printf '#git own/repo %s\n' $C3 > $T/q; gp
ok "[v2.5.9e] a missing link: git_pass starts link mode" 'grep -q "^git_fetch own/repo $C3 $D link$" $T/starts'
: > $T/starts; touch $D.link_fail; gp; rm -f $D.link_fail
ok "[v2.5.9e] not within 30 min of a link failure" '[ ! -s $T/starts ]'
printf '#git own/repo %s\n' $C1 > $T/q; : > $T/starts; touch $D.git_fail; gp
ok "git_pass: not within 30 min of a failure" '[ ! -s $T/starts ]'
cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "GIT_FETCH_TEST PASS ($n checks)" || echo "GIT_FETCH_TEST FAIL ($fails of $n)"
