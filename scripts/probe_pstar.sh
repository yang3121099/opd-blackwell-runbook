#!/usr/bin/env bash
# PROBE-PSTAR (runbook [v2.5.9z75]; Q-064, Runming 10-22 19:1x Beijing, pre-registered research_program 10-22): the critical thinking
# share p* = Delta_long / (Delta_long + Delta_short) of a mixed teacher, from the 8B teacher's sequence log-probs. On ONE GPU
# code/analysis/pstar.py: 256 MATH-500 problems (seed 0) with one short answer from P2 s150's MATH-500 16K T0.6 reading and one long
# answer from P5 s150's (sample 0 each), each scored on the teacher's thinking template (P5's reading, '<think>\n' opened) and its
# non-thinking one (P2's); p* from sequence sums and from per-token means, bootstrap CIs; TERM's thinking prompt as a sensitivity row.
# Inputs: runs/E11b-P2/e11b_P2_math500_T0.6_math500_no_think_t0.6.jsonl and runs/E11b-P5/e11b_P5_math500_T0.6_math500_no_think_t0.6.jsonl
# (WAIT_INPUTS when absent). PASS = exit 0 and 256 problems scored with finite log-probs. The hypotheses are readings, not the gate.
# pstar.json + rows.jsonl + report.json to the public result repository as runs/PROBE-PSTAR (numbers only); probe.log and the
# output tail to logs/<node>/probe_pstar/ (private); an event; .done. ~10 min on one GPU. Usage: bash probe_pstar.sh <gpu>
set -uo pipefail
G=$1; T=PROBE-PSTAR; source "$(dirname "$0")/env.sh"; source ${VENV_EVAL:-$OPD_ROOT/venv}/bin/activate
D=$OPD_ROOT/runs/$T; LOG=$D/probe.log; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
step() { echo "PROBE_PSTAR $(date -u +%T) $*" | tee -a $LOG; }
finish() {   # $1 = PASS / FAIL, $2 = reason
  python3 - "$D" "$1" "$2" <<'PY'
import json, os, sys
d, res, why = sys.argv[1:4]
r = {"result": res, "reason": why}
try:
    r["pstar"] = json.load(open(os.path.join(d, "out", "pstar.json")))
except (OSError, ValueError):
    pass
json.dump(r, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
  mkdir -p $D/up; cp $D/report.json $D/up/; cp $D/out/pstar.json $D/out/rows.jsonl $D/up/ 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/up runs/$T >> $LOG 2>&1 || step "report upload failed (kept in $D)"
  mkdir -p $D/logs; cp $LOG $D/logs/ 2> /dev/null; tail -c 200000 $D/gen.out > $D/logs/out_tail.log 2> /dev/null
  python3 $RUNBOOK/scripts/remote.py put $D/logs logs/$(python3 $RUNBOOK/scripts/control.py node)/probe_pstar/$T >> $LOG 2>&1 || true
  python3 $RUNBOOK/scripts/control.py event $T "probe_pstar_$(echo $1 | tr A-Z a-z)" "$2" > /dev/null 2>&1
  step "$1 $2"; touch $D/.done; exit 0
}
SHORT=$OPD_ROOT/runs/E11b-P2/e11b_P2_math500_T0.6_math500_no_think_t0.6.jsonl
LONG=$OPD_ROOT/runs/E11b-P5/e11b_P5_math500_T0.6_math500_no_think_t0.6.jsonl
for f in $SHORT $LONG; do [ -f $f ] || { echo "WAIT_INPUTS $T: runs/${f#$OPD_ROOT/runs/} not on this node"; exit 1; }; done
for _w in $(seq 19); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G 2>/dev/null | tr -d ' ')
  [ "${used:-1}" -le 1024 ] && break; [ $_w = 19 ] && finish FAIL "gpu $G not empty after 180 s"; sleep ${PPS_WAIT_STEP:-10}
done
step "start gpu=$G teacher=$(basename $TEACHER) n=${PPS_N:-256}"
rm -rf $D/out; mkdir -p $D/out
CUDA_VISIBLE_DEVICES=$G python $RUNBOOK/code/analysis/pstar.py --teacher $TEACHER --short $SHORT --long $LONG --out $D/out --n ${PPS_N:-256} --seed 0 > $D/gen.out 2>&1; rc=$?
python3 $RUNBOOK/scripts/control.py reap $G $T > /dev/null 2>&1 || true
[ $rc = 0 ] || finish FAIL "pstar exited $rc: $(tail -c 300 $D/gen.out | tr '\n' ' ')"
why=$(python3 - $D/out "${PPS_N:-256}" <<'PY'
import json, math, os, sys
d, n = sys.argv[1], int(sys.argv[2]); why = []
r = json.load(open(os.path.join(d, "pstar.json")))
rows = [json.loads(x) for x in open(os.path.join(d, "rows.jsonl")) if x.strip()]
if r.get("problems") != n or len(rows) != n:
    why.append(f"{r.get('problems')} problems scored ({len(rows)} rows), expected {n}")
if any(not math.isfinite(x[k]) for x in rows for k in ("lt_long", "lp_long", "lt_short", "lp_short")):
    why.append("a log-prob that is not finite")
if r.get("sums", {}).get("p_star") is None:
    why.append("no p* (Delta_long + Delta_short <= 0)")
print("; ".join(why))
PY
)
[ -z "$why" ] || finish FAIL "$why"
finish PASS "$(python3 -c "import json, sys; r = json.load(open(sys.argv[1])); s, t = r['sums'], r['per_token']; print(f\"p* {s['p_star']} {s['p_star_ci95']} (Delta_long {s['delta_long']}, Delta_short {s['delta_short']}; H1 {r['H1']}, in (0.5, 0.6) {r['in_measured_interval']}); per token {t['p_star']} {t['p_star_ci95']} (H2 {r['H2']}); lengths {r['mean_len_long']} / {r['mean_len_short']}\")" $D/out/pstar.json)"
