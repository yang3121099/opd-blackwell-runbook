#!/usr/bin/env bash
# test_pypi_index.sh: pypi_index.sh with a fake curl (no network): fastest answering candidate wins, a dead one is skipped,
# the choice is cached, `force` measures again, nothing answering leaves no file, env.sh exports the stored index.
set -u; cd "$(dirname "$0")/.."; T=$(mktemp -d); trap 'rm -rf $T' EXIT; fail=0
ok() { if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got '$2', want '$3'"; fail=$((fail + 1)); fi; }
mkdir -p $T/bin $T/opd
cat > $T/bin/curl <<'C'
#!/usr/bin/env bash
u="${@: -1}"; t=$(grep -m1 -F "$(echo "$u" | sed -E 's#(/vllm/|/interegular/)$##') " "$FAKE" | cut -d' ' -f2-)
[ -n "$t" ] && echo -n "$t" || echo -n "000 10.0"
C
chmod +x $T/bin/curl; export FAKE=$T/fake PATH=$T/bin:$PATH OPD_ROOT=$T/opd
printf '%s\n' 'https://mirrors.aliyun.com/pypi/simple 200 0.9' 'https://pypi.tuna.tsinghua.edu.cn/simple 200 0.2' 'https://mirrors.ustc.edu.cn/pypi/simple 403 0.1' 'https://pypi.org/simple 200 4.0' > $FAKE
ok "fastest 200 wins" "$(bash scripts/pypi_index.sh 2> /dev/null)" https://pypi.tuna.tsinghua.edu.cn/simple
ok "stored" "$(cat $T/opd/control/pypi_index)" https://pypi.tuna.tsinghua.edu.cn/simple
printf '%s\n' 'https://pypi.org/simple 200 4.0' > $FAKE
ok "cached choice kept" "$(bash scripts/pypi_index.sh 2> /dev/null)" https://pypi.tuna.tsinghua.edu.cn/simple
ok "force measures again" "$(bash scripts/pypi_index.sh force 2> /dev/null)" https://pypi.org/simple
ok "explicit index first" "$(OPD_PYPI_INDEX=https://pypi.org/simple/ bash scripts/pypi_index.sh force 2> /dev/null)" https://pypi.org/simple
: > $FAKE; rm -f $T/opd/control/pypi_index
bash scripts/pypi_index.sh > /dev/null 2>&1; ok "nothing answers: rc 1" $? 1
ok "nothing answers: no file" "$([ -e $T/opd/control/pypi_index ] && echo yes || echo no)" no
echo https://mirrors.aliyun.com/pypi/simple > $T/opd/control/pypi_index
ok "sourced under set -eu" "$(bash -c 'set -euo pipefail; source scripts/pypi_index.sh; pypi_index_export 2> /dev/null; echo $UV_DEFAULT_INDEX,$PIP_INDEX_URL,$UV_HTTP_TIMEOUT')" https://mirrors.aliyun.com/pypi/simple,https://mirrors.aliyun.com/pypi/simple,180
ok "env.sh line" "$(bash -c "$(grep -A1 'pypi_index ]; then' scripts/env.sh | head -1); echo \$UV_INDEX_URL")" https://mirrors.aliyun.com/pypi/simple
[ $fail = 0 ] && echo "PASS test_pypi_index (9)" || { echo "FAILED $fail"; exit 1; }
