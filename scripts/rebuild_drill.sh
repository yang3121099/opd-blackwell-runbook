#!/usr/bin/env bash
# v2.5.7a rebuild drill (Runming 2026-10-18 via the reviewer: a new container from zero in <= 10 min). REBUILD.md's steps,
# timed, without a GPU and without touching anything in use: in <base>/opd-drill-<utc> on the share it
#   1. clones the public runbook (the clone a new container would make)
#   2. builds a fresh $OPD_ROOT: venv / G-OPD links to <base>/opd-env, data linked, the small control files from
#      $(dr_root)/state (hf_repos.txt, paths.env with CKPT_ROOT moved into the drill, models.env, control.env, queue)
#   3. runs the venv's python from there (torch / vLLM import, sys.prefix in the share copy)
#   4. checks the three models of models.env are complete (config + every weight shard) and loads the 1.7B tokenizer
#   5. copies the smallest mirrored resume state (opd-dr/dr/<task>) into the drill's CKPT_ROOT, as dr_restore would
#   6. sources env.sh / dr.sh of the clone (paths, candidates, dr_root) and reads the DR report
# then deletes the drill directory. Never started: overnight.sh, run_tasks, the heartbeat, any upload except this
# report. Result logs/rebuild_drill.json (seconds per step, total, pass). Usage: bash rebuild_drill.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/dr.sh"
B=$(dirname "$OPD_ROOT"); E=$B/opd-env; DRR=$(dr_root); L=$OPD_ROOT/logs; RJ=$L/rebuild_drill.json; mkdir -p $L
X=$B/opd-drill-$(date -u +%Y%m%dT%H%M%S); NR=$X/opd-run; mkdir -p $NR/logs $X/ckpt
echo '{"steps": {}}' > $RJ
put() { python3 -c "import json, sys; p, k, v = sys.argv[1:4]; d = json.load(open(p)); d[k] = v; json.dump(d, open(p, 'w'), indent=1)" "$RJ" "$1" "$2"; }
sec() { python3 -c "import json, sys; p, k, v, ok = sys.argv[1:5]; d = json.load(open(p)); d['steps'][k] = {'seconds': int(v), 'ok': ok == '0'}; json.dump(d, open(p, 'w'), indent=1)" "$RJ" "$1" "$2" "$3"; }
T0=$(date +%s); put started_utc "$(date -u +%FT%TZ)"; fails=0
run() {   # <name> <command...>: one timed step
  local step_name=$1 step_t0=$(date +%s) rc; shift; "$@" >> $L/rebuild_drill.log 2>&1; rc=$?
  sec $step_name $(( $(date +%s) - step_t0 )) $rc; [ $rc = 0 ] || fails=$((fails + 1)); echo "DRILL $step_name rc=$rc $(( $(date +%s) - step_t0 )) s" | tee -a $L/rebuild_drill.log; }
s1() { git clone -q --depth 1 ${DRILL_REPO:-https://github.com/yang3121099/opd-blackwell-runbook} $X/runbook; }
s2() {
  ln -s $(readlink -f $OPD_ROOT/venv) $NR/venv && ln -s $(readlink -f $OPD_ROOT/G-OPD) $NR/G-OPD && ln -s $(readlink -f $OPD_ROOT/data) $NR/data || return 1
  [ -d $OPD_ROOT/venv-fa ] && ln -s $(readlink -f $OPD_ROOT/venv-fa) $NR/venv-fa
  for f in hf_repos.txt models.env control.env; do [ -f $DRR/state/$f ] && cp $DRR/state/$f $NR/; done
  [ -f $DRR/state/paths.env ] && sed "s|^CKPT_ROOT=.*|CKPT_ROOT=$X/ckpt|" $DRR/state/paths.env > $NR/paths.env
  [ -f $DRR/state/queue.txt ] && cp $DRR/state/queue.txt $X/runbook/tasks/queue.txt
  [ -f $NR/models.env ] && [ -f $NR/hf_repos.txt ]; }
s3() { (cd / && env -u PYTHONPATH -u VIRTUAL_ENV $NR/venv/bin/python -c "import json, sys, torch, vllm; print(json.dumps({'prefix': sys.prefix, 'torch': torch.__version__, 'vllm': vllm.__version__}))") && \
       case "$(readlink -f $NR/venv)" in "$E"/*) true;; *) echo "venv not on the share yet (persist-env first)"; false;; esac; }
s4() { python3 - $NR/models.env <<'PY' && (cd / && $NR/venv/bin/python -c "import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])" "$(sed -n 's/^STUDENT_17B=//p' $NR/models.env)")
import json, os, sys
env = dict(l.strip().split("=", 1) for l in open(sys.argv[1]) if "=" in l)
bad = []
for k in ("STUDENT_06B", "STUDENT_17B", "TEACHER"):
    d = env.get(k, ""); i = os.path.join(d, "model.safetensors.index.json")
    need = set(json.load(open(i))["weight_map"].values()) if os.path.exists(i) else {"model.safetensors"}
    if not (os.path.isfile(os.path.join(d, "config.json")) and all(os.path.isfile(os.path.join(d, f)) for f in need)):
        bad.append(k)
    elif d.startswith(os.path.expanduser("~") + "/"):   # still on the container disk ($HOME)
        bad.append(k + " (on the container disk)")
print(json.dumps({"incomplete_or_local": bad})); sys.exit(1 if bad else 0)
PY
}
s5() { local t st; t=$(du -sk $DRR/dr/*/ 2> /dev/null | sort -n | head -1 | cut -f2); [ -n "$t" ] || { echo "no mirrored resume state (none to restore)"; return 0; }
       t=${t%/}; st=$(cat $t/latest); local n=$st; mkdir -p $X/ckpt/$(basename $t)/ckpt && DR_BWLIMIT= dr_copy $t/global_step_$n $X/ckpt/$(basename $t)/ckpt/global_step_$n && echo "restored $(basename $t) step $n: $(du -sh $X/ckpt | cut -f1)"; }
s6() { OPD_ROOT=$NR bash -c "source $X/runbook/scripts/env.sh > /dev/null && source $X/runbook/scripts/dr.sh && echo CKPT_ROOT=\$CKPT_ROOT dr_root=\$(dr_root) && [ \"\$CKPT_ROOT\" = $X/ckpt ]"; }
run clone s1; run opd_root s2; run venv s3; run models s4; run restore s5; run env s6
rm -rf $X
put seconds_total "$(( $(date +%s) - T0 ))"; put pass "$([ $fails = 0 ] && echo true || echo false)"; put finished_utc "$(date -u +%FT%TZ)"
echo "DRILL_DONE $(( $(date +%s) - T0 )) s, $fails failed step(s)"
