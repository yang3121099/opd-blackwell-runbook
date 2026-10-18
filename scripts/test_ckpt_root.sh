# env.sh v2.4 CKPT_ROOT choice and ckpt_link (bash scripts/test_ckpt_root.sh) (temp HOME and OPD_ROOT; touches nothing else)
set -u; T=$(mktemp -d); mkdir -p $T/home $T/root; cp ${1:-$(dirname "$0")}/env.sh $T/
r=$(HOME=$T/home OPD_ROOT=$T/root bash -c "source $T/env.sh; echo \$CKPT_ROOT; mkdir -p \$OPD_ROOT/runs/E10b-PX; ckpt_link \$OPD_ROOT/runs/E10b-PX; readlink \$OPD_ROOT/runs/E10b-PX/ckpt; cat \$OPD_ROOT/paths.env")
echo "$r"; c=$(echo "$r" | head -1)
second=$(HOME=/nonexistent OPD_ROOT=$T/root bash -c "source $T/env.sh; echo \$CKPT_ROOT")   # paths.env wins on later sources
[ -d "$c" ] && [ "$(echo "$r" | sed -n 2p)" = "$c/E10b-PX/ckpt" ] && [ "$second" = "$c" ] && echo "CKPT_ROOT_TEST PASS" || echo "CKPT_ROOT_TEST FAIL second=$second"
# v2.5.7i: PROBE_ENV wins over control.env (a well-formed KEY=VAL only)
printf 'export VENV=fa\nexport FIXED_MICRO_BSZ=4\n' > $T/root/control.env
p1=$(HOME=$T/home OPD_ROOT=$T/root bash -c "source $T/env.sh; echo \"\$VENV|\$FIXED_MICRO_BSZ|\${GC:-}\"")
p2=$(HOME=$T/home OPD_ROOT=$T/root PROBE_ENV='VENV= FIXED_MICRO_BSZ= GC=1 bad;x=1' bash -c "source $T/env.sh; echo \"\$VENV|\$FIXED_MICRO_BSZ|\${GC:-}\"")
[ "$p1" = "fa|4|" ] && [ "$p2" = "||1" ] && echo "PROBE_ENV_TEST PASS" || echo "PROBE_ENV_TEST FAIL p1=$p1 p2=$p2"
rm -rf $T
