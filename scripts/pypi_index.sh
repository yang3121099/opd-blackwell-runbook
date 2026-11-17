#!/usr/bin/env bash
# [v3.5] pypi_index.sh (Fable 11-17): the package index of this node. A borrowed mainland machine reached pypi.org only through a
# slow proxy and setup step 3 ended with "Failed to fetch https://pypi.org/simple/interegular/ ... operation timed out" (uv, 3 tries,
# 128 s). The candidates - OPD_PYPI_INDEX if set, the machine's own pip configuration, three mainland mirrors, PyPI - are timed on two
# index pages; the fastest that answers 200 on both is written to $OPD_ROOT/control/pypi_index (one line) and exported for uv and pip.
# env.sh reads that file, so every script of the node installs from the same index. A choice younger than PYPI_INDEX_TTL_S (86400) is
# kept; `pypi_index.sh force` measures again. Sourced by the install scripts (functions + one call), or run as a command (prints it).
pypi_index_pick() {
  local f=$OPD_ROOT/control/pypi_index u c t1 t2 best="" bt="" own
  if [ "${1:-}" != force ] && [ -s "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f" 2> /dev/null || stat -f %m "$f") )) -lt ${PYPI_INDEX_TTL_S:-86400} ]; then
    head -1 "$f"; return 0; fi
  own=$( { python3 -m pip config get global.index-url || pip config get global.index-url; } 2> /dev/null | grep -m1 -E '^https?://[A-Za-z0-9._:/@-]+$')
  for u in ${OPD_PYPI_INDEX:-} $own https://mirrors.aliyun.com/pypi/simple https://pypi.tuna.tsinghua.edu.cn/simple https://mirrors.ustc.edu.cn/pypi/simple https://pypi.org/simple; do
    u=${u%/}
    c=$(curl -sL -o /dev/null -m ${PYPI_INDEX_PROBE_S:-10} -w '%{http_code} %{time_total}' "$u/vllm/" 2> /dev/null); [ "${c%% *}" = 200 ] || continue; t1=${c##* }
    c=$(curl -sL -o /dev/null -m ${PYPI_INDEX_PROBE_S:-10} -w '%{http_code} %{time_total}' "$u/interegular/" 2> /dev/null); [ "${c%% *}" = 200 ] || continue; t2=${c##* }
    t1=$(awk -v a="$t1" -v b="$t2" 'BEGIN { printf "%.3f", a + b }')
    echo "PYPI_INDEX candidate $u: ${t1} s" >&2
    if [ -z "$best" ] || awk -v a="$t1" -v b="$bt" 'BEGIN { exit !(a < b) }'; then best=$u; bt=$t1; fi
    [ -n "${OPD_PYPI_INDEX:-}" ] && [ "$u" = "${OPD_PYPI_INDEX%/}" ] && break   # an explicit choice that answers is taken as it is
  done
  [ -n "$best" ] || { echo "PYPI_INDEX none of the candidates answered: the tools' own default stays" >&2; return 1; }
  { mkdir -p "$OPD_ROOT/control" && echo "$best" > "$f.tmp" && mv "$f.tmp" "$f"; } 2> /dev/null
  echo "$best"; }
pypi_index_export() { local i; i=$(pypi_index_pick "${1:-}") && [ -n "$i" ] && export UV_DEFAULT_INDEX=$i UV_INDEX_URL=$i PIP_INDEX_URL=$i
  export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-180}; echo "PYPI_INDEX ${i:-default}" >&2; }
if [ "${BASH_SOURCE[0]}" = "$0" ]; then : ${OPD_ROOT:?}; pypi_index_pick "${1:-}"; fi
