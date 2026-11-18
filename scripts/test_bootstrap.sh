#!/usr/bin/env bash
# [v2.5.9z106] bootstrap.sh --dry on stubs (no network: curl answers from a stub; the runbook cloned from this checkout): direct and
# proxied networks, no network at all, the files it writes (node.env 0600, coldstart_args, bootstrap.json, dispatch frozen, heartbeat
# 120 s, STATUS_PUT_S=120), and that no token or proxy value reaches its output. Run: bash scripts/test_bootstrap.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]: ${out:0:600}"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); mkdir -p $T/bin $T/base
TOKV="hf_$(printf 'Zq9%.0s' $(seq 12))"; printf '%s\n' "$TOKV" > $T/tok; chmod 600 $T/tok
# curl: 200 when NET=direct, or when NET=proxy and a proxy is set to http://good:1; [z135] NET=hfonly: huggingface.co only,
# NET=githubonly: github.com only; else 000
cat > $T/bin/curl <<'EOF'
#!/bin/bash
u=${@: -1}
case ${NET:-direct} in
  direct) c=200;; proxy) [ "${https_proxy:-}" = "http://good:1" ] && c=200 || c=000;;
  hfonly) [[ $u == https://huggingface.co/* ]] && c=200 || c=000;; githubonly) [[ $u == https://github.com/* ]] && c=200 || c=000;; *) c=000;;
esac
printf $c
EOF
printf '#!/bin/bash\n[ "$1" = -L ] && [ -z "$NOGPU" ] && echo "GPU 0: stub"\nexit 1\n' > $T/bin/nvidia-smi   # [v3.13] one GPU listed; NOGPU=1: none (a CPU machine)
REALDF=$(command -v df)
cat > $T/bin/df <<DFEOF
#!/bin/bash
# [v3.13] DF_FAKE=<dir>: the mount table from <dir>/table, the room of a directory from <dir>/free_<path with _ for />
if [ -n "\$DF_FAKE" ]; then
  case "\$*" in
    "-P -T") cat \$DF_FAKE/table;;
    "-Pk "*) d=\${@: -1}; echo "Filesystem 1024-blocks Used Available Capacity Mounted on"; echo "x 0 0 \$(cat \$DF_FAKE/free_\$(echo \$d | tr / _) 2> /dev/null || echo 0) 0% \$d";;
  esac; exit 0; fi
exec $REALDF "\$@"
DFEOF
printf '#!/bin/bash\nexit 1\n' > $T/bin/pgrep   # [z126] the retire check sees only this test (no stray host process stops it)
chmod +x $T/bin/*; export PATH=$T/bin:$PATH RB_URL=$SRC
B() { bash $SRC/scripts/bootstrap.sh --base $T/base --token-file $T/tok --dry "$@" 2>&1; }

out=$(NET=direct B); rc=$?; O=$T/base/opd/run
ok "direct: runs through, loops only printed" '[ $rc = 0 ] && [[ "$out" == *"network: direct"* ]] && [[ "$out" == *"would run loops.sh start; node "* ]]'
ok "node.env 0600, no proxy file, the token by reference" '[ "$(stat -c %a $O/node.env 2> /dev/null || stat -f %Lp $O/node.env)" = 600 ] && [ ! -e $O/proxy.env ] && grep -q "head -1 $T/tok" $O/node.env && ! grep -q "$TOKV" $O/node.env'
ok "coldstart_args, bootstrap.json, dispatch frozen, heartbeat 120, STATUS_PUT_S=120" 'grep -q -- "--token-file $T/tok --opd-root $O " $O/control/coldstart_args && grep -q -- "--no-loops" $O/control/coldstart_args && [ -f $O/control/bootstrap.json ] && [ -f $O/control/dispatch_frozen ] && [ "$(cat $O/control/heartbeat_sec)" = 120 ] && grep -qx "export STATUS_PUT_S=120" $O/control.env'
ok "HF_REPO is the public repository alone" 'grep -q "HF_REPO=yang31210999/opd-lab-b300-pub$" $O/node.env'
ok "no token in the output" '[[ "$out" != *"$TOKV"* ]]'
out=$(NET=direct B); rc=$?
ok "a second run is safe (STATUS_PUT_S once, step 1 not again)" '[ $rc = 0 ] && [ "$(grep -c STATUS_PUT_S $O/control.env)" = 1 ] && [[ "$out" != *"would run cold_start.sh --step 1"* ]]'
rm -rf $T/base/opd
out=$(NET=proxy https_proxy=http://good:1 B); rc=$?
ok "proxy: found among the candidates, value not printed, proxy.env 0600 with no_proxy" '[ $rc = 0 ] && [[ "$out" == *"network: proxy candidate"* ]] && [[ "$out" != *"good:1"* ]] && [ "$(stat -c %a $O/proxy.env 2> /dev/null || stat -f %Lp $O/proxy.env)" = 600 ] && grep -q "^export no_proxy=localhost,127.0.0.1,::1," $O/proxy.env'
rm -rf $T/base/opd
out=$(NET=none B); rc=$?
ok "no network: STOP, nothing started" '[ $rc = 1 ] && [[ "$out" == *"STOP: neither direct nor any of"* ]] && [ ! -e $O/node.env ]'
out=$(B --role boss 2>&1); rc=$?; ok "bad --role: usage, exit 2" '[ $rc = 2 ]'
touch $T/old_loops.sh; out=$(NET=direct B --retire-file $T/old_loops.sh); rc=$?
ok "--retire-file in --dry: only printed" '[ $rc = 0 ] && [[ "$out" == *"would rename $T/old_loops.sh"* ]] && [ -f $T/old_loops.sh ]'
# [v2.5.9z126] the GitHub heartbeat token (B0b / B1b) and --profile / --role rehearsal (B5)
export GH_HB_CRED=$T/ghc/gh_hb_credentials
GHV="github_pat_$(printf 'Xy7%.0s' $(seq 10))"; printf '%s\n' "$GHV" > $T/ghtok; chmod 600 $T/ghtok
rm -rf $T/base/opd; out=$(NET=direct B); rc=$?
ok "z126: no token file in --dry: only says it would ask, nothing stored" '[ $rc = 0 ] && [[ "$out" == *"would ask for the GitHub heartbeat token"* ]] && [ ! -e $GH_HB_CRED ] && grep -q "\"gh_token\": false" $O/control/bootstrap.json'
rm -rf $T/base/opd; out=$(NET=direct B --gh-token-file $T/ghtok); rc=$?
ok "z126: --gh-token-file: git's credential line, 0600, never printed; checked after B1" '[ $rc = 0 ] && [ "$(cat $GH_HB_CRED)" = "https://yang3121099:$GHV@github.com" ] && [ "$(stat -c %a $GH_HB_CRED 2> /dev/null || stat -f %Lp $GH_HB_CRED)" = 600 ] && [[ "$out" != *"$GHV"* ]] && [[ "$out" == *"token stored for git"* ]] && [[ "$out" == *"would check the GitHub heartbeat repository"* ]]'
ok "z126: bootstrap.json says a token is there; node.env carries the moved credential path, not the token" 'grep -q "\"gh_token\": true" $O/control/bootstrap.json && grep -q "^export GH_HB_CRED=$GH_HB_CRED$" $O/node.env && ! grep -rq "$GHV" $O'
out=$(NET=direct B); rc=$?
ok "z126: a rerun finds the stored token and does not ask" '[ $rc = 0 ] && [[ "$out" == *"its token is stored already"* ]] && [[ "$out" != *"would ask for the GitHub"* ]]'
rm -f $GH_HB_CRED; out=$(NET=direct B --no-gh); rc=$?
ok "z126: --no-gh does not ask or check" '[ $rc = 0 ] && [[ "$out" == *"not asked (--no-gh)"* ]] && [[ "$out" != *"would check the GitHub"* ]] && [ ! -e $GH_HB_CRED ]'
printf 'not a token!\n' > $T/badtok; chmod 600 $T/badtok; out=$(NET=direct B --gh-token-file $T/badtok); rc=$?
ok "z126: not a GitHub token: not stored, bootstrap goes on" '[ $rc = 0 ] && [[ "$out" == *"not a GitHub token"* ]] && [ ! -e $GH_HB_CRED ] && [[ "$out" != *"not a token!"* ]]'
chmod 644 $T/ghtok; out=$(NET=direct B --gh-token-file $T/ghtok); rc=$?; chmod 600 $T/ghtok
ok "z126: a token file readable by others: STOP" '[ $rc = 1 ] && [[ "$out" == *"--gh-token-file missing or not 0600"* ]]'
rm -rf $T/base/opd; out=$(NET=direct B --no-gh --profile 80g); rc=$?
ok "z126: --profile 80g writes its seven keys, STATUS_PUT_S once" '[ $rc = 0 ] && for kv in VENV=fa FIXED_MICRO_BSZ=4 LOGP_MB=1 VLLM_UTIL=0.35 GC=1 STATUS_PUT_S=120 PUBLIC_UPLOADS=1; do grep -qx "export $kv" $O/control.env || exit 1; done && [ "$(grep -c STATUS_PUT_S $O/control.env)" = 1 ] && [ "$(cat $O/control/profile)" = 80g ] && grep -q "\"profile\": \"80g\"" $O/control/bootstrap.json'
sed -i.bak 's/^export VLLM_UTIL=.*/export VLLM_UTIL=0.3/' $O/control.env; out=$(NET=direct B --no-gh --profile 80g); rc=$?
ok "z126: a rerun keeps what setenv changed" '[ $rc = 0 ] && grep -qx "export VLLM_UTIL=0.3" $O/control.env && [ "$(grep -c VLLM_UTIL $O/control.env)" = 1 ] && [[ "$out" == *"already set, kept: VENV FIXED_MICRO_BSZ LOGP_MB VLLM_UTIL GC STATUS_PUT_S PUBLIC_UPLOADS"* ]]'
out=$(NET=direct B --no-gh --profile nosuch); rc=$?
ok "z126: an unknown profile: STOP naming the profiles there are" '[ $rc = 1 ] && [[ "$out" == *"no profile nosuch (there are: "*"80g"* ]]'
out=$(B --profile ../x 2>&1); rc=$?; ok "z126: a profile name with a path: usage, exit 2" '[ $rc = 2 ]'
rm -rf $T/base/opd; out=$(NET=direct B --no-gh --profile 80g --role rehearsal); rc=$?
ok "z126: --role rehearsal writes its limits over the profile's (one line each)" '[ $rc = 0 ] && grep -qx "export STATUS_PUT_S=300" $O/control.env && grep -qx "export CONTROL_BATCH=1" $O/control.env && grep -qx "export HF_COMMIT_BUDGET=30" $O/control.env && [ "$(grep -c STATUS_PUT_S $O/control.env)" = 1 ] && grep -qx "export VENV=fa" $O/control.env'
# [v2.5.9z135] the runbook's git mirror on Hugging Face (RB_URL on another host): the reach without GitHub (four cases), node.env
# RB_URL + OPD_CONTROL_ORIGIN, the codes line, B1b's check skipped when GitHub does not answer. The mirror URL clones this
# checkout (git's insteadOf), nothing goes to the network.
HFRB=https://huggingface.co/x/opd-blackwell-runbook
HB() { env RB_URL=$HFRB GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.$SRC.insteadOf GIT_CONFIG_VALUE_0=$HFRB bash $SRC/scripts/bootstrap.sh --base $T/base --token-file $T/tok --dry "$@" 2>&1; }
rm -rf $T/base/opd; out=$(NET=hfonly HB --gh-token-file $T/ghtok); rc=$?
ok "z135: GitHub blocked, RB_URL = the HF mirror: runs through; node.env RB_URL + OPD_CONTROL_ORIGIN; B1b skipped" '[ $rc = 0 ] && [[ "$out" == *"network: direct"* ]] && grep -qx "export RB_URL=$HFRB OPD_CONTROL_ORIGIN=huggingface.co/x/opd-blackwell-runbook" $O/node.env && [ -f $T/base/opd/runbook/scripts/loops.sh ] && [[ "$out" == *"GitHub does not answer here, its check skipped"* ]] && [[ "$out" != *"would check the GitHub"* ]]'
ok "z135: the codes line names GitHub, the runbook host, the HF API, PyPI; PyPI blocked is said" '[[ "$out" == *"network codes: github 000, runbook (huggingface.co) 200, hf api 200, pypi 000"* ]] && [[ "$out" == *"PyPI does not answer: setup step 2 can only use an old base"* ]]'
rm -rf $T/base/opd; rm -f $GH_HB_CRED; out=$(NET=hfonly B --no-gh); rc=$?
ok "z135: GitHub blocked with the GitHub RB_URL: STOP with the direct codes and the mirror hint" '[ $rc = 1 ] && [[ "$out" == *"(direct: github 000, hf api 200, pypi 000); GitHub blocked, Hugging Face answers: RB_URL=https://huggingface.co/yang31210999/opd-blackwell-runbook"* ]] && [ ! -e $O/node.env ]'
rm -rf $T/base/opd; out=$(NET=direct HB --no-gh); rc=$?
ok "z135: everything open, RB_URL = the mirror: runs through, the mirror in node.env" '[ $rc = 0 ] && grep -q "^export RB_URL=$HFRB " $O/node.env'
rm -rf $T/base/opd; out=$(NET=githubonly HB --no-gh); rc=$?
ok "z135: Hugging Face blocked, RB_URL = the mirror: STOP before any clone" '[ $rc = 1 ] && [[ "$out" == *"reaches the runbook (huggingface.co) and Hugging Face"* ]] && [ ! -d $T/base/opd/runbook ]'
rm -rf $T/base/opd; out=$(NET=direct B --no-gh); rc=$?
ok "z135: the GitHub RB_URL: node.env without RB_URL / OPD_CONTROL_ORIGIN (control.py's default)" '[ $rc = 0 ] && ! grep -q "OPD_CONTROL_ORIGIN" $O/node.env'
# [v3.0] (Fable 11-17) --name: the node id from the word; a base another node owns is never shared (two machines on one disk)
H() { python3 -c "import hashlib, sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:8])" $1; }
rm -rf $T/base/opd $T/base/opd-*; NA=$(H h2a); NB=$(H h2b)
out=$(NET=direct B --no-gh --name h2a); rc=$?
ok "v3.0 --name: the node id is the word's hash, in node.env and control/owner" '[ $rc = 0 ] && grep -q " OPD_NODE=$NA " $T/base/opd/run/node.env && [ "$(cat $T/base/opd/run/control/owner)" = $NA ] && [[ "$out" == *"node $NA"* ]]'
out=$(NET=direct B --no-gh --name h2b); rc=$?
ok "v3.0 a second node on the same base takes opd-<its id>; the first node's files stay" '[ $rc = 0 ] && [[ "$out" == *"belongs to another node"* ]] && grep -q " OPD_NODE=$NB " $T/base/opd-$NB/run/node.env && grep -q " OPD_NODE=$NA " $T/base/opd/run/node.env && [ "$(cat $T/base/opd/run/control/owner)" = $NA ]'
ok "v3.2 the second node reuses the first one's install: FROM_OPD_ROOT and LINK_VENV in its node.env, none in the first node's" 'grep -q "^export FROM_OPD_ROOT=$T/base/opd/run LINK_VENV=1" $T/base/opd-$NB/run/node.env && ! grep -q FROM_OPD_ROOT $T/base/opd/run/node.env && [[ "$out" == *"will be reused"* ]]'
out=$(NET=direct B --no-gh --name h2b); rc=$?
ok "v3.0 the second node's rerun finds its own directory (and still its peer)" '[ $rc = 0 ] && [[ "$out" != *"belongs to another node"* ]] && [[ "$out" == *"base: $T/base/opd-$NB "* ]] && grep -q "^export FROM_OPD_ROOT=$T/base/opd/run LINK_VENV=1" $T/base/opd-$NB/run/node.env'
out=$(NET=direct B --no-gh --name h2a); rc=$?
ok "v3.0 the first node's rerun keeps <base>/opd" '[ $rc = 0 ] && [[ "$out" == *"base: $T/base/opd "* ]]'
out=$(NET=direct B --no-gh --name "a/b"); rc=$?
ok "v3.0 --name takes one word" '[ $rc = 2 ]'
# [v3.1] --fill-cmd: the machine owner's filler command stays on the machine (0600), with the pasting shell's PATH; never printed
rm -rf $T/base/opd $T/base/opd-*
FC="python /some/owner/dir/train.py --duration_hours 888"
out=$(NET=direct B --no-gh --name h2a --fill-cmd "$FC"); rc=$?; O=$T/base/opd/run
ok "v3.1 --fill-cmd: stored 0600 with the PATH, not in the output" '[ $rc = 0 ] && [ "$(cat $O/control/gpu_fill_secret.cmd)" = "$FC" ] && [ "$(stat -c %a $O/control/gpu_fill_secret.cmd 2> /dev/null || stat -f %Lp $O/control/gpu_fill_secret.cmd)" = 600 ] && grep -q "^export PATH=" $O/control/gpu_fill_secret.env && [[ "$out" != *"/some/owner"* ]] && [[ "$out" == *"filler: the owner"* ]]'
out=$(NET=direct B --no-gh --name h2a); rc=$?
ok "v3.1 a rerun without --fill-cmd keeps the stored command" '[ $rc = 0 ] && [ "$(cat $O/control/gpu_fill_secret.cmd)" = "$FC" ]'
rm -rf $T/base/opd $T/base/opd-*
# [v3.13] a CPU development machine (11-18: "STOP: no writable local disk found"): the base by room among the WRITABLE directories;
# no GPU -> a preparation node whose id comes from <name>-cpu; the status is built without nvidia-smi
mkdir -p $T/df $T/ro $T/small $T/home $T/big; : > $T/hostsfile; chmod 555 $T/ro
fr() { echo $2 > $T/df/free_$(echo $1 | tr / _); }
{ echo "Filesystem Type 1024-blocks Used Available Capacity Mounted on"
  echo "/dev/a ext4 1 1 9000000000 1% $T/hostsfile"; echo "/dev/b xfs 1 1 8000000000 1% $T/ro"; echo "overlay overlay 1 1 10485760 1% $T/small"
  echo "ceph ceph 1 1 99000000000 1% $T/net"; } > $T/df/table
fr $T/hostsfile 9000000000; fr $T/ro 8000000000; fr $T/small 10485760; fr $T/home 314572800
B2() { HOME=$T/home DF_FAKE=$T/df bash $SRC/scripts/bootstrap.sh --token-file $T/tok --dry --no-gh "$@" 2>&1; }
out=$(NET=direct B2 --name h2a); rc=$?
if [ "$(id -u)" != 0 ]; then
ok "v3.13 base: a file mount and a read-only disk are passed over; nothing local has 50 GB -> \$HOME" '[ $rc = 0 ] && [[ "$out" == *"base: $T/home (300 GB free"* ]] && [ -f $T/home/opd/run/node.env ]'
else n=$((n + 1)); fi
echo "/dev/c xfs 1 1 943718400 1% $T/big" >> $T/df/table; fr $T/big 943718400; rm -rf $T/home/opd
out=$(NET=direct B2 --name h2a); rc=$?
ok "v3.13 base: the writable local directory with the most room" '[ $rc = 0 ] && [[ "$out" == *"base: $T/big (900 GB free"* ]] && [ -f $T/big/opd/run/node.env ] && [ ! -e $T/home/opd ]'
rm -rf $T/big/opd; NC=$(H h2a-cpu)
out=$(NOGPU=1 NET=direct B2 --name h2a); rc=$?
ok "v3.13 no GPU: a preparation node, id from <name>-cpu (not the GPU machine's id)" '[ $rc = 0 ] && [[ "$out" == *"a preparation node"* ]] && grep -q " OPD_NODE=$NC " $T/big/opd/run/node.env && [ $NC != $NA ]'
out=$(NOGPU=1 NET=direct B2 --name h2a); rc=$?
ok "v3.13 no GPU: a rerun keeps the same id and directory" '[ $rc = 0 ] && grep -q " OPD_NODE=$NC " $T/big/opd/run/node.env && [ ! -e $T/big/opd-$NC ]'
chmod 755 $T/ro; rm -rf $T/big/opd
unset GH_HB_CRED
[ -n "${KEEP_T:-}" ] && echo "kept $T" || rm -rf $T
[ $fails = 0 ] && echo "BOOTSTRAP_TEST PASS ($n checks)" || { echo "BOOTSTRAP_TEST FAIL ($fails of $n)"; exit 1; }
