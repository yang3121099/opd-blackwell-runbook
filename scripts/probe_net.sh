#!/usr/bin/env bash
# PROBE-NET (runbook [v2.5.9c]; Runming via the reviewer 08:3x UTC 10-21: "no way to download Qwen3-32B / Qwen3-VL-8B-Thinking?
# please find one, the data too"): the node's routes to the model hubs, no GPU (the CPU lane), read-only network, fixed URLs.
#  1. four routes, each with the node's proxy and without it (direct): ModelScope's API for a small file (Qwen3-VL-8B-Thinking
#     config.json) and for the first shard of Qwen3-32B, huggingface.co and hf-mirror.com for that shard: a range request
#     of 64 MB (120 s cap): http code, remote IP, bytes, seconds, bytes/s
#  2. whether pip can fetch the modelscope package (no install)
#  3. ModelScope's dataset API for the image / math / tool sets waiting for placement, under their own owner and under
#     AI-ModelScope: the http code of each
# report.json (private: logs/<node>/probe_net/<task>), an event, .done; a rerun measures again.
# Usage: bash probe_net.sh <gpu|cpu>
set -uo pipefail
T=PROBE-NET; source "$(dirname "$0")/env.sh"
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D; : > $D/routes.tsv; : > $D/datasets.tsv
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_NET $(date -u +%T) $*" | tee -a $LOG; }
MS=https://modelscope.cn/api/v1
declare -a ROUTES=(
  "ms_small|$MS/models/Qwen/Qwen3-VL-8B-Thinking/repo?Revision=master&FilePath=config.json"
  "ms_shard|$MS/models/Qwen/Qwen3-32B/repo?Revision=master&FilePath=model-00001-of-00017.safetensors"
  "hf|https://huggingface.co/Qwen/Qwen3-32B/resolve/main/model-00001-of-00017.safetensors"
  "hf_mirror|https://hf-mirror.com/Qwen/Qwen3-32B/resolve/main/model-00001-of-00017.safetensors")
for mode in proxy direct; do
  P=env; [ $mode = direct ] && P="env -u https_proxy -u HTTPS_PROXY -u http_proxy -u HTTP_PROXY -u all_proxy -u ALL_PROXY"
  for r in "${ROUTES[@]}"; do
    name=${r%%|*}; url=${r#*|}
    w=$($P curl -sS -L --max-time ${PN_MAX_TIME:-120} --connect-timeout 15 -r 0-67108863 -o /dev/null \
          -w 'http=%{http_code} ip=%{remote_ip} bytes=%{size_download} secs=%{time_total} Bps=%{speed_download}' "$url" 2> $D/err.txt)
    printf '%s\t%s\t%s\t%s\n' "$mode" "$name" "$w" "$(head -c 200 $D/err.txt | tr '\n\t' '  ')" >> $D/routes.tsv
    step "$mode $name $w"
  done
done
pip_out=$(timeout 300 $OPD_ROOT/venv/bin/python -m pip download modelscope --no-deps -d $D/pip_probe -q 2>&1 | tail -1)
pip_ok=$(ls $D/pip_probe 2> /dev/null | grep -c modelscope)
rm -rf $D/pip_probe
step "pip download modelscope: $([ ${pip_ok:-0} -gt 0 ] && echo ok || echo "failed: $pip_out")"
for ds in hiyouga/geometry3k MathLLMs/MathVision AI4Math/MathVista FanqingM/MMK12 BytedTsinghua-SIA/DAPO-Math-17k Team-ACE/ToolACE; do
  for id in $ds AI-ModelScope/${ds#*/}; do
    c=$(curl -sS -o /dev/null --max-time 30 --connect-timeout 15 -w '%{http_code}' "$MS/datasets/$id" 2> /dev/null || echo 000)
    printf '%s\t%s\n' "$id" "$c" >> $D/datasets.tsv
  done
done
step "datasets: $(awk '$2 == 200 {print $1}' $D/datasets.tsv | tr '\n' ' ')"
python3 - "$D" "${pip_ok:-0}" "$pip_out" <<'PY' > $D/report.json
import json, re, sys
d, pip_ok, pip_out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
routes = []
for line in open(f"{d}/routes.tsv"):
    mode, name, w, err = (line.rstrip("\n").split("\t") + ["", "", "", ""])[:4]
    kv = dict(re.findall(r"(\w+)=(\S*)", w))
    bps = float(kv.get("Bps") or 0)
    routes.append({"mode": mode, "route": name, "http": kv.get("http"), "remote_ip": kv.get("ip"), "bytes": int(float(kv.get("bytes") or 0)),
                   "secs": float(kv.get("secs") or 0), "MBps": round(bps / 1e6, 2), "error": err.strip() or None})
ok = [r for r in routes if r["route"].startswith("ms") and r["http"] in ("200", "206") and (r["route"] == "ms_small" or r["MBps"] >= 5)]
ds = [dict(zip(("id", "http"), l.split())) for l in open(f"{d}/datasets.tsv") if l.strip()]
json.dump({"routes": routes, "modelscope_usable": [f"{r['mode']}:{r['route']}" for r in ok],
           "pip_modelscope": bool(pip_ok), "pip_error": None if pip_ok else pip_out[:300], "modelscope_datasets": ds}, sys.stdout, indent=1)
PY
mkdir -p $D/up; cp $D/report.json $D/probe.log $D/up/
python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_net/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
u=$(python3 -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['modelscope_usable']) or 'none')" $D/report.json)
python3 $RUNBOOK/scripts/control.py event $T probe_net_done "modelscope usable: $u" > /dev/null 2>&1
step "done; modelscope usable: $u"; touch $D/.done
