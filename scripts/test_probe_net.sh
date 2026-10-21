#!/usr/bin/env bash
# [v2.5.9c] Stub tests of probe_net.sh (no network). Usage: bash scripts/test_probe_net.sh -> PROBE_NET_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/root/venv/bin $T/bin; cd $T
  cp $SRC/scripts/probe_net.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb" > $T/rb/scripts/env.sh
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  # curl: ModelScope answers 206 at 20 MB/s through the proxy and 000 directly; huggingface 000; hf-mirror 206 at 0.1 MB/s
  cat > $T/bin/curl <<'C'
#!/usr/bin/env bash
url="${@: -1}"; direct=0; [ -z "${https_proxy:-}" ] && direct=1; echo "curl direct=$direct $url" >> $CALLS
case "$url" in
  */api/v1/datasets/*) case "$url" in *AI-ModelScope/geometry3k|*Team-ACE/ToolACE) echo -n 200;; *) echo -n 404;; esac;;
  *modelscope.cn*) [ $direct = 1 ] && { echo -n "http=000 ip= bytes=0 secs=15 Bps=0"; echo "connect timeout" >&2; exit 28; }
                   echo -n "http=206 ip=fake-ms bytes=67108864 secs=3.2 Bps=${FAKE_MS_BPS:-20000000}";;
  *huggingface.co*) echo -n "http=000 ip= bytes=0 secs=120 Bps=0"; exit 28;;
  *hf-mirror.com*) echo -n "http=206 ip=fake-mirror bytes=12000000 secs=120 Bps=100000";;
esac
C
  chmod +x $T/bin/curl
  printf '#!/usr/bin/env bash\nshift; exec "$@"\n' > $T/bin/timeout; chmod +x $T/bin/timeout
  printf '#!/usr/bin/env bash\nd=$(echo "$*" | sed -n "s/.* -d \\([^ ]*\\).*/\\1/p"); [ "${FAKE_PIP_FAIL:-0}" = 1 ] && { echo "ERROR: Could not find a version"; exit 1; }; mkdir -p $d; : > $d/modelscope-1.20.0-py3-none-any.whl\n' > $T/root/venv/bin/python; chmod +x $T/root/venv/bin/python
  export CALLS=$T/calls PATH=$T/bin:$PATH https_proxy=http://proxy.invalid:3128; : > $CALLS
}
r() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-NET/report.json "$1"; }
setup
bash $T/rb/scripts/probe_net.sh cpu > /dev/null 2>&1
ok "eight route measurements, proxy and direct" '[ "$(r "len(v[\"routes\"])")" = 8 ] && [ "$(grep -c "^curl direct=1 .*\(modelscope\|huggingface\|hf-mirror\)" $CALLS | tr -d " ")" -ge 4 ]'
ok "ModelScope usable through the proxy only" '[ "$(r "\" \".join(v[\"modelscope_usable\"])")" = "proxy:ms_small proxy:ms_shard" ]'
ok "per-route fields: http, MB/s, the error text" '[ "$(r "[x for x in v[\"routes\"] if x[\"mode\"] == \"proxy\" and x[\"route\"] == \"ms_shard\"][0][\"MBps\"]")" = 20.0 ] && [ "$(r "[x for x in v[\"routes\"] if x[\"mode\"] == \"direct\" and x[\"route\"] == \"ms_small\"][0][\"error\"]")" = "connect timeout" ]'
ok "pip can fetch modelscope; datasets found on ModelScope" '[ "$(r "v[\"pip_modelscope\"]")" = True ] && [ "$(r "\" \".join(sorted(d[\"id\"] for d in v[\"modelscope_datasets\"] if d[\"http\"] == \"200\"))")" = "AI-ModelScope/geometry3k Team-ACE/ToolACE" ] && [ "$(r "len(v[\"modelscope_datasets\"])")" = 12 ]'
ok "report to the private repository, event, .done" 'grep -q "^remote put $T/root/runs/PROBE-NET/up logs/abcd1234/probe_net/PROBE-NET" $CALLS && grep -q "^control event PROBE-NET probe_net_done modelscope usable: proxy:ms_small proxy:ms_shard" $CALLS && [ -f $T/root/runs/PROBE-NET/.done ]'
cd $SRC
T2=$T; setup
FAKE_MS_BPS=1000000 FAKE_PIP_FAIL=1 bash $T/rb/scripts/probe_net.sh cpu > /dev/null 2>&1
ok "a slow shard is not usable (< 5 MB/s); a pip failure is reported" '[ "$(r "\" \".join(v[\"modelscope_usable\"])")" = "proxy:ms_small" ] && [ "$(r "v[\"pip_modelscope\"]")" = False ] && [[ "$(r "v[\"pip_error\"]")" == *"Could not find"* ]]'
cd $SRC
[ $fails = 0 ] && echo "PROBE_NET_TEST PASS ($n checks)" || echo "PROBE_NET_TEST FAIL ($fails of $n)"
