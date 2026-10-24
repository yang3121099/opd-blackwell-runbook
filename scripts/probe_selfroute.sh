#!/usr/bin/env bash
# PROBE-SELFROUTE (runbook [v2.5.9z82]; Q-072, Fable 10-24 18:55 Beijing, research_program 3c): the gate of E10b-P5selfroute, Q-062's
# stage two. On ONE GPU code/analysis/diff_probe.py --emit: the 1.7B base's last-prompt-token hidden states at H1's layer (PROBE-DIFFPROBE's
# diff_probe.json: its layer and L2 strength) for PROBE-DIFF17's 6,400 train prompts and MATH-500's 500; the train prompts labelled out
# of fold (5 folds, each prompt by the probe fit without its fold), MATH-500 by the fit on all of train; labels_probe_17.jsonl in
# labels_17's format (r = the probe's p_easy: the route arm's rule r >= 0.5 -> easy -> the non-thinking teacher unchanged).
# Inputs: runs/PROBE-DIFF17/labels_17.jsonl, runs/PROBE-DIFFPROBE/out/diff_probe.json (WAIT_INPUTS when absent), data/dapo_train.parquet.
# PASS = exit 0, >= 6,000 train prompts and all 500 of MATH-500, and the pre-registered check on the train labels: the probe's easy
# share within 5 pp of the true one and agreement >= 0.75. report.json + selfroute.json + labels_probe_17.jsonl (keys and numbers,
# path-free) to the result repository as runs/PROBE-SELFROUTE; probe.log and the output tail to logs/<node>/probe_selfroute/
# (private); an event; .done. ~10 min on one GPU. Usage: bash probe_selfroute.sh <gpu>
set -uo pipefail
G=$1; T=PROBE-SELFROUTE; source "$(dirname "$0")/env.sh"; source ${VENV_EVAL:-$OPD_ROOT/venv}/bin/activate
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_SELFROUTE $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    r["selfroute"] = json.load(open(os.path.join(d, "out", "selfroute.json")))
except (OSError, ValueError):
    pass
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/up/; [ -f $D/out/selfroute.json ] && cp $D/out/selfroute.json $D/up/
  [ -f $D/labels_probe_17.jsonl ] && cp $D/labels_probe_17.jsonl $D/up/
  python3 $RUNBOOK/scripts/remote.py put $D/up runs/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  mkdir -p $D/logs; cp $LOG $D/logs/ 2> /dev/null; tail -c 200000 $D/gen.out > $D/logs/out_tail.log 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/logs logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_selfroute/$T >> $LOG 2>&1 || true
  python3 $RUNBOOK/scripts/control.py event $T "probe_selfroute_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
LAB=$OPD_ROOT/runs/PROBE-DIFF17/labels_17.jsonl; REP=$OPD_ROOT/runs/PROBE-DIFFPROBE/out/diff_probe.json
[ -f $LAB ] || { echo "WAIT_INPUTS $T: runs/PROBE-DIFF17/labels_17.jsonl not on this node"; exit 1; }
[ -f $REP ] || { echo "WAIT_INPUTS $T: runs/PROBE-DIFFPROBE/out/diff_probe.json not on this node"; exit 1; }
[ -f $OPD_ROOT/data/dapo_train.parquet ] || finish FAIL "no data/dapo_train.parquet"
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PSR_WAIT_STEP:-10}
done
step "start gpu=$G model=$(basename $STUDENT_17B)"
rm -rf $D/out $D/labels_probe_17.jsonl; mkdir -p $D/out
CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/analysis/diff_probe.py --model $STUDENT_17B --labels $LAB --train $OPD_ROOT/data/dapo_train.parquet \
  --out $D/out --emit $D/labels_probe_17.jsonl --probe-report $REP $([ -n "${PSR_LIMIT:-}" ] && echo --limit $PSR_LIMIT) > $D/gen.out 2>&1; rc=$?
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
[ $rc = 0 ] || finish FAIL "diff_probe --emit exited $rc: $(tail -c 300 $D/gen.out | tr '\n' ' ')"
why=$(python3 - $D/out/selfroute.json $D/labels_probe_17.jsonl "${PSR_LIMIT:-0}" <<'PY'
import json, sys
r = json.load(open(sys.argv[1])); lim = int(sys.argv[3]); why = []
n = sum(1 for x in open(sys.argv[2]) if x.strip())
if r.get("n_train", 0) < (6000 if not lim else 1):   # PSR_LIMIT (a smoke): any
    why.append(f"{r.get('n_train')} train prompts labelled, expected >= 6000")
if not lim and r.get("n_math500") != 500:
    why.append(f"{r.get('n_math500')} MATH-500 problems, expected 500")
if n != r.get("n_train", 0) + r.get("n_math500", 0):
    why.append(f"labels_probe_17.jsonl has {n} rows, expected {r.get('n_train', 0) + r.get('n_math500', 0)}")
t = r.get("train", {})
if not t.get("pass"):
    why.append(f"train labels: easy share {t.get('probe_easy_share')} vs true {t.get('true_easy_share')} ({t.get('share_diff_pp')} pp, line < 5), agreement {t.get('agreement')} (line >= 0.75)")
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "$(python3 -c "import json, sys; r = json.load(open(sys.argv[1])); t, m = r['train'], r['math500']; print(f\"layer {r['layer']}: train out of fold easy {t['probe_easy_share']} vs true {t['true_easy_share']} ({t['share_diff_pp']} pp), agreement {t['agreement']}, AUC {r['auc_oof_train']}; MATH-500 easy {m['probe_easy_share']} vs {m['true_easy_share']}, agreement {m['agreement']}, AUC {r['auc_math500']}\")" $D/out/selfroute.json)"
