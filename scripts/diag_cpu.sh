#!/usr/bin/env bash
# The zero-GPU diagnostics of 10-27 (runbook [v2.5.9z99]; Fable's research_program 3i / 3j / 3l, order t32 -> Q-078 -> Q-085-A ->
# DIAG-VERIFY-ONSET -> DIAG-VLM-LOOP2): CPU-lane tasks (no GPU, no process touched), at nice 19 / ionice idle, reading readings and
# rollout dumps that exist (code/analysis/zero_gpu_diag.py). report.json (result PASS = the analysis ran; the hypotheses' verdicts are
# in it), to logs/<node>/<task in lower case>/ (private), an event, .done. Waits (no attempt) while an input is not on the node.
# Usage: bash diag_cpu.sh <DIAG-T32-TAILS | DIAG-PROBE-GATE | DIAG-WALL-BIMODAL | DIAG-VERIFY-ONSET | DIAG-VLM-LOOP2>
set -uo pipefail
source "$(dirname "$0")/env.sh"
T=$1; D=$OPD_ROOT/runs/$T; R=$OPD_ROOT/runs; PY=${DIAG_PY:-$OPD_ROOT/venv/bin/python}; mkdir -p $D
[ -f $D/.done ] && { echo "TASK_ALREADY_DONE $T"; exit 0; }
need() { for f in "$@"; do [ -e $R/$f ] || { echo "WAIT_INPUTS $T: runs/$f not on this node"; exit 1; }; done; }
LOW=""; command -v ionice > /dev/null && LOW="ionice -c3"
Z="nice -n 19 $LOW $PY $RUNBOOK/code/analysis/zero_gpu_diag.py"
P5T32=EVAL-e10b-P5t32/e10b_P5t32_s50_math500_T0.6_math500_no_think_t0.6.jsonl
PROBE=PROBE-SELFROUTE/labels_probe_17.jsonl; LAB=PROBE-DIFF17/labels_17.jsonl
case $T in
  DIAG-T32-TAILS)   # Q-082: t32's truncated 16K answers - loop or still writing; 20 drawn to read
    need $P5T32; $Z t32tails --out $D/out.json --input $R/$P5T32 --n 20 > $D/diag.log 2>&1;;
  DIAG-PROBE-GATE)   # Q-078: the probe-gated think / no-think choice of the 1.7B base at 16K (off the 32K readings)
    NT=EVAL-budA17a/budA_base17_math500_T0.6_32k_math500_no_think_t0.6.jsonl; TH=EVAL-budB/budB_P0think_math500_T0.6_32k_math500_think_t0.6.jsonl
    need $NT $TH $PROBE $LAB
    $Z probegate --out $D/out.json --input nothink=$R/$NT --input think=$R/$TH --probe $R/$PROBE --labels $R/$LAB --budget 16384 --tokenizer $STUDENT_17B > $D/diag.log 2>&1;;
  DIAG-WALL-BIMODAL)   # Q-085-A: every E10b arm with >= 10 rollout dumps at the 4K cap
    need $PROBE
    mapfile -t RUNS < <(python3 - "$R" <<'PY'
import os, re, sys
r = sys.argv[1]
for d in sorted(os.listdir(r)):
    rd = os.path.join(r, d, "rollouts")
    if not d.startswith("E10b-") or not os.path.isdir(rd) or sum(1 for f in os.listdir(rd) if re.fullmatch(r"\d+\.jsonl", f)) < 10:
        continue
    try:
        cfg = open(os.path.join(r, d, "config.yaml")).read()
    except OSError:
        cfg = ""
    m = re.search(r"max_response_length[:=]\s*(\d+)", cfg)
    if (int(m.group(1)) if m else 4096) == 4096:
        print(os.path.join(r, d))
PY
)
    if [ ${#RUNS[@]} -gt 0 ]; then
      $Z wallbimodal --out $D/out.json --probe $R/$PROBE --tokenizer $STUDENT_17B --thresh 3686 --runs "${RUNS[@]}" > $D/diag.log 2>&1
    else   # no arm with dumps is itself the answer: an empty report, not a wait
      echo '{"thresh": 3686, "arms": {}}' > $D/out.json; echo "no E10b run at the 4K cap with >= 10 rollout dumps" > $D/diag.log
    fi;;
  DIAG-VERIFY-ONSET)   # the gate of Q-086: where the first verification marker falls (P5 s150, tail s50; 100 samples each)
    A=E11b-P5/e11b_P5_math500_T0.6_math500_no_think_t0.6.jsonl; B=EVAL-tail/e10b_P5tail_s50_math500_T0.6_math500_no_think_t0.6.jsonl
    need $A $B; $Z verifyonset --out $D/out.json --input p5s150=$R/$A --input tail_s50=$R/$B --n 100 > $D/diag.log 2>&1;;
  DIAG-VLM-LOOP2)   # Q-080's mechanism: the VLM loops, and per problem whether both decodings loop
    args=(); for b in mathvista mathvision mathverse; do
      t=T0; m=no_think_t0.0; [ $b = mathvista ] && { t=T0.6; m=no_think_t0.6; }; e=EVAL-e15-base2; [ $b = mathvista ] && e=EVAL-e15-base
      fb=$e/e15_base2B_${b}_${t}_${b}_$m.jsonl; fs=EVAL-e15-vlm-s50/e15_P5_s50_${b}_${t}_${b}_$m.jsonl
      gb=EVAL-vlm32-base-$b/vlm32_base2B_${b}_T0.6_32k_${b}_no_think_t0.6.jsonl; gs=EVAL-vlm32-s50-$b/vlm32_P5_s50_${b}_T0.6_32k_${b}_no_think_t0.6.jsonl
      need $fb $fs $gb $gs
      args+=(--pair "base_$b=base16k_$b:$R/$fb,base32k_$b:$R/$gb" --pair "s50_$b=s50_16k_$b:$R/$fs,s50_32k_$b:$R/$gs")
    done
    $Z vlmloop2 --out $D/out.json --tokenizer $OPD_ROOT/models/Qwen3-VL-2B-Instruct "${args[@]}" > $D/diag.log 2>&1;;
  *) echo "unknown diagnostic $T"; exit 2;;
esac
rc=$?; [ $rc = 0 ] || { tail -5 $D/diag.log; echo "DIAG_FAILED $T"; exit 1; }
python3 - "$D" "$T" <<'PY'
import json, os, sys
d, t = sys.argv[1:3]; o = json.load(open(os.path.join(d, "out.json")))
def pct(x): return f"{100 * x:.1f}%"
if t == "DIAG-T32-TAILS":
    s = o["shares"]; why = f"truncated {o['truncated']} of {o['samples']}: loop {pct(s['loop'])} unfinished {pct(s['unfinished'])} box_more {pct(s['box_more'])}; drawn 20: loop {o['drawn_loop']}, still writing {o['drawn_continuation']}"
elif t == "DIAG-PROBE-GATE":
    f = lambda k: f"{k} {o[k]['acc']} / {o[k]['mean_len']:.0f} tok"
    why = "; ".join(f(k) for k in ("probe_gate", "always_think", "never_think", "oracle")) + f"; gate-think CI {o['gate_minus_think_ci']}; H1 {o['H1']} H2 {o['H2']}"
elif t == "DIAG-WALL-BIMODAL":
    a = o["arms"]; why = f"{len(a)} arms; H-A1 true {sum(1 for v in a.values() if v.get('H_A1'))}, H-A2 true {sum(1 for v in a.values() if v.get('H_A2'))}; " + ", ".join(f"{k} wall s{v.get('wall_step')}" for k, v in a.items())
elif t == "DIAG-VERIFY-ONSET":
    why = f"gate {'PASS' if o['gate'] else 'FAIL'}; " + "; ".join(f"{k} first-before-box {v['first_before_box_share']} median {v['median_rel_len']}" for k, v in o["readings"].items())
else:
    why = "; ".join(f"{k}: both {v['both']} neither {v['neither']} of {v['problems']}" for k, v in o["pairs"].items())
json.dump({"result": "PASS", "reason": why[:1500], **o}, open(os.path.join(d, "report.json"), "w"), indent=1)
PY
mkdir -p $D/up; cp $D/report.json $D/diag.log $D/up/
python3 $RUNBOOK/scripts/remote.py put $D/up logs/$(python3 $RUNBOOK/scripts/control.py node)/$(echo $T | tr A-Z a-z) > /dev/null 2>&1 || echo "report upload failed (kept in $D)"
python3 $RUNBOOK/scripts/control.py event $T diag_done "$(python3 -c "import json; print(json.load(open('$D/report.json'))['reason'][:400])")" > /dev/null 2>&1
cat $D/diag.log | tail -3; touch $D/.done; exit 0
