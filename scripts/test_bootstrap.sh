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
[ -n "${KEEP_T:-}" ] && echo "kept $T" || rm -rf $T
[ $fails = 0 ] && echo "BOOTSTRAP_TEST PASS ($n checks)" || { echo "BOOTSTRAP_TEST FAIL ($fails of $n)"; exit 1; }
