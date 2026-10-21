#!/usr/bin/env bash
# [v2.5.9c] Stub tests of our_du.sh and heartbeat.sh's our_du_due (no real du). Usage: bash scripts/test_our_du.sh -> OUR_DU_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); cd $T; mkdir -p $T/rb/scripts $T/root/control $T/ckpt $T/bin
cp $SRC/scripts/our_du.sh $T/rb/scripts/
echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb CKPT_ROOT=$T/ckpt" > $T/rb/scripts/env.sh
# du -sk: 600 GB of ours on the work disk, 1100 GB on the checkpoint disk; FAKE_DU_FAIL: the work-disk du times out
printf '#!/usr/bin/env bash\ncase "$*" in *ckpt*) echo -e "1074316406\\t$2";; *) [ "${FAKE_DU_FAIL:-0}" = 1 ] && exit 124; echo -e "585937500\\t$2";; esac\n' > $T/bin/du; chmod +x $T/bin/du
v() { python3 -c "import json, sys; d = json.load(open(sys.argv[1])); print(d.get(sys.argv[2]))" $T/root/control/our_du.json "$1"; }
PATH=$T/bin:$PATH bash $T/rb/scripts/our_du.sh
ok "work and checkpoint usage in GB, a UTC time" '[ "$(v work_gb)" = 600.0 ] && [ "$(v ckpt_gb)" = 1100.1 ] && [ "$(v timed_out)" = False ] && [[ "$(v utc)" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}T ]]'
FAKE_DU_FAIL=1 PATH=$T/bin:$PATH bash $T/rb/scripts/our_du.sh
ok "a du that fails: work_gb null, timed_out" '[ "$(v work_gb)" = None ] && [ "$(v timed_out)" = True ] && [ "$(v ckpt_gb)" = 1100.1 ]'
# our_du_due: due without the file or with an old one, not with a fresh one
due() { ( OPD_ROOT=$T/root; eval "$(grep '^our_du_due()' $SRC/scripts/heartbeat.sh)"; our_du_due && echo due || echo fresh ); }
r1=$(due); touch -t 202001010000 $T/root/control/our_du.json; r2=$(due); touch $T/root/control/our_du.json; r3=$(due)
ok "our_du_due: fresh file not due, an old one due" '[ "$r1" = fresh ] && [ "$r2" = due ] && [ "$r3" = fresh ]'
rm -f $T/root/control/our_du.json; ok "our_du_due: no file is due" '[ "$(due)" = due ]'
cd $SRC
[ $fails = 0 ] && echo "OUR_DU_TEST PASS ($n checks)" || echo "OUR_DU_TEST FAIL ($fails of $n)"
