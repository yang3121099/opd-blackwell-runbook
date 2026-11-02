#!/usr/bin/env bash
# [v2.5.9z106] bootstrap.sh --dry on stubs (no network: curl answers from a stub; the runbook cloned from this checkout): direct and
# proxied networks, no network at all, the files it writes (node.env 0600, coldstart_args, bootstrap.json, dispatch frozen, heartbeat
# 120 s, STATUS_PUT_S=120), and that no token or proxy value reaches its output. Run: bash scripts/test_bootstrap.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]: ${out:0:600}"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); mkdir -p $T/bin $T/base
TOKV="hf_$(printf 'Zq9%.0s' $(seq 12))"; printf '%s\n' "$TOKV" > $T/tok; chmod 600 $T/tok
# curl: 200 when NET=direct, or when NET=proxy and a proxy is set to http://good:1; else 000
cat > $T/bin/curl <<'EOF'
#!/bin/bash
if [ "${NET:-direct}" = direct ] || { [ "$NET" = proxy ] && [ "${https_proxy:-}" = "http://good:1" ]; }; then printf 200; else printf 000; fi
EOF
printf '#!/bin/bash\nexit 1\n' > $T/bin/nvidia-smi
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
unset GH_HB_CRED
[ -n "${KEEP_T:-}" ] && echo "kept $T" || rm -rf $T
[ $fails = 0 ] && echo "BOOTSTRAP_TEST PASS ($n checks)" || { echo "BOOTSTRAP_TEST FAIL ($fails of $n)"; exit 1; }
