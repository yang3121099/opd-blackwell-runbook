#!/usr/bin/env bash
# test_seed_venv.sh ([v3.11]): install_uv.sh's seed_venv on a fake machine - the store's environment pack or an earlier install's
# site-packages are copied through a temporary directory; a copy that is too slow becomes a link (.pth); nothing half-copied stays.
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d); trap 'rm -rf $T' EXIT; fails=0
ok() { if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
mkdir -p $T/bin; command -v timeout > /dev/null || cat > $T/bin/timeout <<'STUB'
#!/usr/bin/env python3
import subprocess, sys
a = sys.argv[1:]
try: sys.exit(subprocess.run(a[1:], timeout=float(a[0])).returncode)
except subprocess.TimeoutExpired: sys.exit(124)
STUB
command -v sha256sum > /dev/null || printf '#!/bin/bash\nshasum -a 256 "$@"\n' > $T/bin/sha256sum
chmod +x $T/bin/*; export PATH=$T/bin:$PATH
OLD=$T/old/venv/lib/python3.10/site-packages; mkdir -p $OLD/torch-2.8.0+cu128.dist-info $OLD/vllm-0.10.2.dist-info $OLD/torch $OLD/vllm $OLD/verl-0.6.1.dist-info $T/old/venv/bin
echo x > $OLD/torch/a.py; : > "$OLD/__editable__.verl-0.6.1.pth"; printf '#!/gone/python\nprint(1)\n' > $T/old/venv/bin/ray
fresh() { rm -rf $T/n; mkdir -p $T/n/venv/lib/python3.10/site-packages $T/n/venv/bin; cd $T/n; export OPD_ROOT=$T/n; unset OPD_STORE OLD_BASE_CANDS SEED_COPY_MAX_S SEED_RELINK; DST=$T/n/venv/lib/python3.10/site-packages; }
python() { if [ "${2:-}" != "${2#*sysconfig}" ]; then echo $DST; elif [ "${2:-}" = "import torch, vllm" ]; then [ -d $DST/torch ] || [ -f $DST/opd_shared_packages.pth ]; else echo "2.8.0+cu128"; fi; }
eval "$(sed -n '/^seed_venv() {/,/^  done; done; return 1; }/p' $SRC/scripts/install_uv.sh)"

fresh; export OLD_BASE_CANDS=$T/none:$T/old; out=$(seed_venv)
ok "an earlier install's packages are copied; its G-OPD link is not; console scripts get this venv's interpreter" '[[ "$out" == *"copied:"* ]] && [ -f $DST/torch/a.py ] && [ ! -e "$DST/__editable__.verl-0.6.1.pth" ] && [ ! -e $DST/verl-0.6.1.dist-info ] && [ "$(head -1 venv/bin/ray)" = "#!$T/n/venv/bin/python" ] && [ -f venv/.seeded_from_history ] && [ ! -e venv/.seed_tmp ]'
ok "a second call has nothing to do" '! seed_venv > /dev/null'

fresh; export OLD_BASE_CANDS=$T/old SEED_COPY_MAX_S=1; printf '#!/bin/bash\nsleep 5\n' > $T/bin/tar; chmod +x $T/bin/tar; out=$(seed_venv); rm -f $T/bin/tar
ok "a copy that is too slow: the packages are linked where they are, nothing half-copied in the venv" '[[ "$out" == *"used where they are"* ]] && [ "$(cat $DST/opd_shared_packages.pth)" = "$OLD" ] && [ -f venv/.linked_to_history ] && [ ! -e $DST/torch ] && [ ! -e venv/.seed_tmp ]'
ok "a linked venv that works is left alone" '! seed_venv > /dev/null'
export SEED_RELINK=1 SEED_COPY_MAX_S=60; out=$(seed_venv)
ok "a later run with more time turns the link into a copy" '[[ "$out" == *"copied:"* ]] && [ -f $DST/torch/a.py ] && [ ! -e $DST/opd_shared_packages.pth ] && [ ! -e venv/.linked_to_history ]'

fresh; mkdir -p $T/store/envpack; ( cd $OLD && tar -cf $T/store/envpack/site-packages-python3.10-torch2.8.0-vllm0.10.2.tar --exclude='__editable__*verl*' --exclude='verl-*.dist-info' . )
( cd $T/store/envpack && sha256sum site-packages-python3.10-torch2.8.0-vllm0.10.2.tar | cut -c1-64 > site-packages-python3.10-torch2.8.0-vllm0.10.2.tar.sha256 )
export OPD_STORE=$T/store OLD_BASE_CANDS=$T/old; out=$(seed_venv)
ok "the store's environment pack comes first and is checked by its sha256" '[[ "$out" == *"environment pack"*"copied:"* ]] && [ -f $DST/torch/a.py ] && [ -f venv/.seeded_from_store ]'
fresh; echo 0000 > $T/store/envpack/site-packages-python3.10-torch2.8.0-vllm0.10.2.tar.sha256; export OPD_STORE=$T/store; out=$(seed_venv); rc=$?
ok "a pack whose sha256 differs is not used" '[ $rc = 1 ] && [[ "$out" == *"sha256 differs"* ]] && [ ! -e $DST/torch ]'
cd /; [ $fails = 0 ] && echo "SEED_VENV_TEST PASS (7 checks)" || { echo "SEED_VENV_TEST FAIL ($fails of 7)"; exit 1; }
