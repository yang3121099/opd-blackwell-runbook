#!/usr/bin/env bash
# [v2.5.9z100] cold_start.sh in DRYRUN on stubs (requirements v26 §11's Mac acceptance): the arguments and the token file, the control
# bootstrap (applied_seq = the history's largest seq, no old command run, control.env only proposed), the done list from the last
# status, the labels (checked / missing), the weights list, the health checks (all PASS -> the loops; one FAIL -> nothing started).
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
T=$(mktemp -d); RB=$T/rb; mkdir -p $RB/scripts $RB/control $RB/tasks $T/bin
cp $SRC/scripts/cold_start.sh $SRC/scripts/coldstart_hf.py $SRC/scripts/coldstart_check.py $RB/scripts/
export CALLS=$T/calls; : > $CALLS
printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\n' > $RB/scripts/control.py
cat > $RB/control/abcd1234.yaml <<'Y'
entries:
  - {"seq": 1, "cmd": "setenv DR=0", "reason": "r", "utc": "2026-10-20T00:00:00Z"}
  - {"seq": 2, "cmd": "kill E10b-P5", "reason": "r", "utc": "2026-10-20T00:00:00Z"}
  - {"seq": 3, "cmd": "setenv GPUS_HOLD_MIN=1440", "reason": "r", "utc": "2026-10-20T00:00:00Z"}
  - {"seq": 4, "cmd": "rerun EVAL-x", "reason": "r", "utc": "2026-10-20T00:00:00Z"}
  - {"seq": 5, "cmd": "setenv DR=1", "reason": "r", "utc": "2026-10-20T00:00:00Z"}
Y
cat > $RB/tasks/coldstart.json <<'J'
{"schema": "coldstart/1", "models": [{"name": "Qwen3-4B", "hf": "Qwen/Qwen3-4B", "always": true, "tasks": []},
 {"name": "Qwen3-8B", "hf": "Qwen/Qwen3-8B", "always": true, "by": "overnight.sh", "tasks": []},
 {"name": "Qwen3-VL-2B-Instruct", "hf": "Qwen/Qwen3-VL-2B-Instruct", "tasks": ["E15-VLM-P2lo"]},
 {"name": "Qwen3-32B", "hf": "Qwen/Qwen3-32B", "tasks": ["E10b-P5-t32"]}],
 "artifacts": [{"name": "labels_17.jsonl", "repo_path": "runs/PROBE-DIFF17/labels_17.jsonl", "node_path": "runs/PROBE-DIFF17/labels_17.jsonl", "rerun": "PROBE-DIFF17"},
  {"name": "labels_probe_17.jsonl", "repo_path": "runs/PROBE-SELFROUTE/labels_probe_17.jsonl", "node_path": "runs/PROBE-SELFROUTE/labels_probe_17.jsonl", "rerun": "PROBE-SELFROUTE"}],
 "weights": [{"path": "runs/E10e-P5tail/global_step_50/actor/huggingface", "tasks": ["E12e-P5tail-8k"]},
  {"path": "runs/E10c-P5tail-c50/global_step_30/actor/huggingface", "tasks": ["E12c-P5tail-8k"]},
  {"path": "runs/E10b-P5/global_step_150/actor/huggingface", "tasks": ["EVAL-done-one"]}]}
J
fakehub() { python3 - "$@" <<'PY'
import hashlib, json, sys
lab = '{"split": "math500", "pos": 0, "r": 0.5}\n'
blob = hashlib.sha1(b"blob %d\0" % len(lab.encode()) + lab.encode()).hexdigest()
st = {"utc": sys.argv[2], "tasks": {"EVAL-done-one": {"state": "done"}, "E10b-P5-t32": {"state": "done"}, "E12e-P5tail-8k": {"state": "running"}}}
json.dump({"_writable": {"me/ops": sys.argv[3] == "1"}, "me/pub": {"runs/E10e-P5tail/global_step_50/actor/huggingface/config.json": {"size": 5}},
           "me/ops": {"status/abcd1234.json": {"text": json.dumps(st)}, "runs/PROBE-DIFF17/labels_17.jsonl": {"blob_id": blob, "text": lab}}},
          open(sys.argv[1], "w"))
PY
}
fakehub "$T/fake.json" 2026-10-25T00:00:00Z 1
export COLDSTART_FAKE=$T/fake.json COLDSTART_DRYRUN=1 COLDSTART_MOUNTS=$T/mounts
echo "x / ext4 rw 0 0" > $T/mounts
printf '#!/bin/bash\nfor i in $(seq ${NGPU_STUB:-8}); do echo "GPU $i"; done\n' > $T/bin/nvidia-smi
printf '#!/bin/bash\necho "Filesystem 1024-blocks Used Available Capacity Mounted"; echo "x 9999999999 1 3000000000 1%% /"\n' > $T/bin/df
printf '#!/bin/bash\nexit 0\n' > $T/bin/git
printf '#!/bin/bash\nshift; exec "$@"\n' > $T/bin/timeout   # GNU timeout is not on the Mac
chmod +x $T/bin/*; export PATH=$T/bin:$PATH
O=$T/opd; mkdir -p $O/logs
for v in venv venv-fa venv-vl venv-ifeval; do mkdir -p $O/$v/bin; printf '#!/bin/bash\n[ "$1" = -c ] && exit 0\nexec python3 "$@"\n' > $O/$v/bin/python; chmod +x $O/$v/bin/python; done
printf 'G2 PASS\nVLLM PASS\nSMOKE PASS\n' > $O/logs/verify.log
tok=$T/token; printf 'x%.0s' $(seq 30) > $tok; chmod 600 $tok
ARGS="--token-file $tok --opd-root $O --ckpt-root $T/ck --repos public=me/pub,private=me/ops --node abcd1234"

out=$(bash $RB/scripts/cold_start.sh --opd-root $O 2>&1); rc=$?
ok "missing arguments: usage, exit 2" '[ $rc = 2 ] && [[ "$out" == "COLDSTART usage: "* ]]'
chmod 644 $tok; out=$(bash $RB/scripts/cold_start.sh $ARGS 2>&1); rc=$?; chmod 600 $tok
ok "a token file not 0600 stops at step 0" '[ $rc = 1 ] && [[ "$out" == *"STEP 0 FAILED: token file mode 644"* ]] && [ ! -f $O/control/applied_seq ]'
out=$(bash $RB/scripts/cold_start.sh $ARGS 2>&1); rc=$?
ok "the whole DRYRUN passes: health all PASS, loops only printed" '[ $rc = 0 ] && grep -q "STEP 7 health checks all PASS" $O/logs/coldstart.log && grep -q "DRYRUN would run: setsid nohup bash scripts/run_tasks.sh" $O/logs/coldstart.log && ! grep -q "FAIL" $O/logs/coldstart_health.txt'
ok "token never printed or written" '! grep -rq "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" $O $T/calls && [[ "$out" != *xxxxxxxxxxxxxxxxxxxx* ]]'
ok "applied_seq = 5, none of the old commands run, no reply written" '[ "$(cat $O/control/applied_seq)" = 5 ] && [ ! -d $O/control/replies ] && ! grep -q "kill\|rerun" $CALLS'
ok "control.env only proposed: last values, nothing applied" '[ ! -f $O/control.env ] && grep -q "^DR=1$" $O/control/control.env.proposed && ! grep -q "^DR=0$" $O/control/control.env.proposed && grep -q "^GPUS_HOLD_MIN=1440$" $O/control/control.env.proposed'
ok "paths.env carries CKPT_ROOT and OPD_NODE; hf_repos.txt the two repositories" 'grep -q "^CKPT_ROOT=$T/ck$" $O/paths.env && grep -q "^export OPD_NODE=abcd1234$" $O/paths.env && grep -q "^private=me/ops$" $O/hf_repos.txt'
ok "done list from the last status" '[ "$(sort $O/control/coldstart_done.txt | tr "\n" " ")" = "E10b-P5-t32 EVAL-done-one " ]'
ok "models: 4B always, the VL one for an unfinished task, the 32B skipped (its task done), overnight'\''s left to it" 'grep -q "MODEL Qwen3-4B fetched (fake)" $O/logs/coldstart.log && grep -q "MODEL Qwen3-VL-2B-Instruct fetched (fake)" $O/logs/coldstart.log && ! grep -q "MODEL Qwen3-32B" $O/logs/coldstart.log && grep -q "MODEL Qwen3-8B by overnight.sh" $O/logs/coldstart.log'
ok "labels: one checked, one missing (rerun as a new version)" 'grep -q "LABELS labels_17.jsonl PASS" $O/logs/coldstart.log && grep -q "LABELS labels_probe_17.jsonl MISSING .* rerun PROBE-SELFROUTE as a NEW label version" $O/logs/coldstart.log && [ -f $O/runs/PROBE-DIFF17/labels_17.jsonl ]'
ok "weights: the done task'\''s not needed; one on the public repository, one missing" '[ "$(python3 -c "import json; print(sorted((w[\"path\"].split(\"/\")[1], w[\"where\"]) for w in json.load(open(\"$O/control/coldstart_weights.json\"))[\"weights\"]))")" = "[('"'"'E10c-P5tail-c50'"'"', '"'"'missing'"'"'), ('"'"'E10e-P5tail'"'"', '"'"'public'"'"')]" ]'
echo 7 > $O/control/applied_seq; out=$(bash $RB/scripts/cold_start.sh $ARGS 2>&1); rc=$?
ok "a larger applied_seq is kept (rerun), the check compares with the history" '[ "$(cat $O/control/applied_seq)" = 7 ] && grep -q "FAIL control: applied_seq == 5" $O/logs/coldstart_health.txt && [ $rc = 1 ]'
echo 5 > $O/control/applied_seq; NGPU_STUB=4 bash $RB/scripts/cold_start.sh $ARGS > $T/o2 2>&1; rc=$?
ok "a FAIL (4 GPUs of 8): nothing started" '[ $rc = 1 ] && grep -q "FAIL gpus: 8 visible" $O/logs/coldstart_health.txt && grep -q "STEP 7 FAILED: health checks" $T/o2 && ! grep -q "STEP 8" $T/o2'
mkdir -p $O/control/outbox; touch $O/control/outbox/x.json; bash $RB/scripts/cold_start.sh $ARGS > $T/o3 2>&1; rc=$?; rm -rf $O/control/outbox
ok "a non-empty outbox fails the health check" '[ $rc = 1 ] && grep -q "FAIL control: outbox empty" $O/logs/coldstart_health.txt'
bash $RB/scripts/cold_start.sh $ARGS > /dev/null 2>&1   # back to a passing state
fakehub "$T/fake.json" 2026-10-25T00:00:00Z 0; bash $RB/scripts/cold_start.sh $ARGS > /dev/null 2>&1; rc=$?
ok "a token that cannot write: FAIL, nothing started" '[ $rc = 1 ] && grep -q "FAIL Hugging Face: the token writes" $O/logs/coldstart_health.txt'
fakehub "$T/fake.json" 2026-10-25T00:00:00Z 1; echo "x $(python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" $T) nfs4 rw 0 0" >> $T/mounts; bash $RB/scripts/cold_start.sh $ARGS > /dev/null 2>&1; rc=$?
ok "the control plane on a network filesystem: FAIL" '[ $rc = 1 ] && grep -q "FAIL control plane on one local disk" $O/logs/coldstart_health.txt'
echo "x / ext4 rw 0 0" > $T/mounts; rm -f $O/control/coldstart_done.txt*; fakehub "$T/fake.json" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" 1
bash $RB/scripts/cold_start.sh $ARGS > /dev/null 2>&1; rc=$?
ok "a reused node id whose last status is fresh (< 2 h): FAIL" '[ $rc = 1 ] && grep -q "FAIL node abcd1234 reused: its last status is older than 2 h" $O/logs/coldstart_health.txt'
# [v2.5.9z104] --only / --no-loops / --ckpt-min-gb (a temporary node to adapt: one canary task, the queue never started)
O2=$T/opd2; mkdir -p $O2/logs; for v in venv venv-fa; do mkdir -p $O2/$v/bin; cp $O/$v/bin/python $O2/$v/bin/python; done; cp $O/logs/verify.log $O2/logs/
A2="--token-file $tok --opd-root $O2 --ckpt-root $T/ck2 --repos public=me/pub,private=me/ops --node ef567890"
out=$(bash $RB/scripts/cold_start.sh $A2 --only PROBE-TAIL --no-loops 2>&1); rc=$?
ok "--only PROBE-TAIL --no-loops: health PASS, no loop or event, step 8 says so; a new node id: applied_seq 0" '[ $rc = 0 ] && [ "$(cat $O2/control/applied_seq)" = 0 ] && grep -q "STEP 7 health checks all PASS" $O2/logs/coldstart.log && grep -q "STEP 8 loops NOT started (--no-loops)" $O2/logs/coldstart.log && ! grep -q "setsid nohup" $O2/logs/coldstart.log && ! grep -q "event node coldstart" $CALLS'
ok "--only: the always-models only (no VL / 32B), venv-vl / venv-ifeval not built, their checks SKIP" 'grep -q "MODEL Qwen3-4B fetched (fake)" $O2/logs/coldstart.log && ! grep -q "MODEL Qwen3-VL" $O2/logs/coldstart.log && ! grep -q "MODEL Qwen3-32B" $O2/logs/coldstart.log && grep -q "venv-vl, venv-ifeval not built (--only PROBE-TAIL)" $O2/logs/coldstart.log && grep -q "^SKIP venv-vl: vllm, transformers (--only PROBE-TAIL)$" $O2/logs/coldstart_health.txt && grep -q "^SKIP venv-ifeval: langdetect (--only PROBE-TAIL)$" $O2/logs/coldstart_health.txt'
ok "--only: no labels, no weights for a task that uses none" '! grep -q "^LABELS" $O2/logs/coldstart.log && grep -q "WEIGHTS 0 needed" $O2/logs/coldstart.log'
out=$(bash $RB/scripts/cold_start.sh $A2 --only PROBE-TAIL --no-loops --ckpt-min-gb 4000 2>&1); rc=$?
ok "--ckpt-min-gb 4000 with 3000 GB free: FAIL, nothing started" '[ $rc = 1 ] && grep -q "FAIL CKPT_ROOT >= 4000 GB free" $O2/logs/coldstart_health.txt'
out=$(bash $RB/scripts/cold_start.sh $A2 --only "PROBE-TAIL;x" 2>&1); rc=$?
ok "--only with a bad name: usage, exit 2" '[ $rc = 2 ] && [[ "$out" == "COLDSTART usage: "* ]]'
printf '{"models": [{"name": "A", "always": true, "tasks": []}, {"name": "B", "tasks": ["E10b-P5route-x"]}], "artifacts": [{"name": "l1", "used_by": "E10b-P5route*, READINGS-EXTRA splits"}, {"name": "l2", "used_by": "DIAG-PROBE-GATE"}], "weights": [{"path": "w1", "tasks": ["E10b-P5route-x"]}, {"path": "w2", "tasks": ["E12b-P5"]}]}' > $T/m.json
python3 $RB/scripts/coldstart_hf.py only $T/m.json E10b-P5route-x $T/m2.json > /dev/null
ok "coldstart_hf.py only: always + used models, labels by a used_by glob, the task's weights" '[ "$(python3 -c "import json; m = json.load(open(\"$T/m2.json\")); print(\" \".join([\",\".join(x[\"name\"] for x in m[\"models\"]), \",\".join(a[\"name\"] for a in m[\"artifacts\"]), \",\".join(w[\"path\"] for w in m[\"weights\"]), \",\".join(m[\"only\"])]))")" = "A,B l1 w1 E10b-P5route-x" ]'
[ -n "${KEEP_T:-}" ] && echo "kept $T" || rm -rf $T
[ $fails = 0 ] && echo "COLD_START_TEST PASS ($n checks)" || echo "COLD_START_TEST FAIL ($fails of $n)"
