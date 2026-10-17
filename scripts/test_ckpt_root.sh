# env.sh v2.4 CKPT_ROOT choice and ckpt_link (bash scripts/test_ckpt_root.sh) (temp HOME and OPD_ROOT; touches nothing else)
set -u; T=$(mktemp -d); mkdir -p $T/home $T/root; cp ${1:-$(dirname "$0")}/env.sh $T/
r=$(HOME=$T/home OPD_ROOT=$T/root bash -c "source $T/env.sh; echo \$CKPT_ROOT; mkdir -p \$OPD_ROOT/runs/E10b-PX; ckpt_link \$OPD_ROOT/runs/E10b-PX; readlink \$OPD_ROOT/runs/E10b-PX/ckpt; cat \$OPD_ROOT/paths.env")
echo "$r"; c=$(echo "$r" | head -1)
second=$(HOME=/nonexistent OPD_ROOT=$T/root bash -c "source $T/env.sh; echo \$CKPT_ROOT")   # paths.env wins on later sources
[ -d "$c" ] && [ "$(echo "$r" | sed -n 2p)" = "$c/E10b-PX/ckpt" ] && [ "$second" = "$c" ] && echo "CKPT_ROOT_TEST PASS" || echo "CKPT_ROOT_TEST FAIL second=$second"
rm -rf $T
