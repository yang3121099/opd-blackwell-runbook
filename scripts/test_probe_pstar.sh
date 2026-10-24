#!/usr/bin/env bash
# [v2.5.9z75] (Q-064) Stub tests of probe_pstar.sh (no GPU, network or HF; pstar.py faked). Usage: bash scripts/test_probe_pstar.sh
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/analysis $T/root/runs/E11b-P2 $T/root/runs/E11b-P5 $T/root/venv/bin $T/bin; cd $T
  cp $SRC/scripts/probe_pstar.sh $T/rb/scripts/
  echo "export OPD_ROOT=$T/root RUNBOOK=$T/rb TEACHER=$T/m8" > $T/rb/scripts/env.sh; : > $T/root/venv/bin/activate
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["node"] and print("abcd1234")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/remote.py
  cat > $T/rb/code/analysis/pstar.py <<'PY'
import json, os, sys
a = sys.argv; open(os.environ["CALLS"], "a").write("pstar " + " ".join(a[1:]) + "\n")
out = a[a.index("--out") + 1]; n = int(a[a.index("--n") + 1]); rows = int(os.environ.get("FAKE_ROWS", n))
lt = float("nan") if os.environ.get("FAKE_NAN") else -100.0
with open(os.path.join(out, "rows.jsonl"), "w") as f:
    for i in range(rows):
        f.write(json.dumps({"pid": f"math500-{i}", "lt_long": lt, "lp_long": -160.0, "lt_short": -30.0, "lp_short": -10.0, "n_long": 3000, "n_short": 600}) + "\n")
json.dump({"problems": rows, "sums": {"p_star": None if os.environ.get("FAKE_NOP") else 0.55, "p_star_ci95": [0.52, 0.58], "delta_long": 60.0, "delta_short": 49.0},
           "per_token": {"p_star": 0.2, "p_star_ci95": [0.18, 0.22]}, "H1": True, "in_measured_interval": True, "H2": True,
           "mean_len_long": 3000.0, "mean_len_short": 600.0}, open(os.path.join(out, "pstar.json"), "w"))
sys.exit(int(os.environ.get("FAKE_RC", "0")))
PY
  printf '#!/usr/bin/env bash\necho 0\n' > $T/bin/nvidia-smi; chmod +x $T/bin/nvidia-smi
  : > $T/root/runs/E11b-P2/e11b_P2_math500_T0.6_math500_no_think_t0.6.jsonl; : > $T/root/runs/E11b-P5/e11b_P5_math500_T0.6_math500_no_think_t0.6.jsonl
  export CALLS=$T/calls PATH=$T/bin:$PATH PPS_WAIT_STEP=0; : > $CALLS
}
res() { python3 -c "import json, sys; v = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))" $T/root/runs/PROBE-PSTAR/report.json "$1"; }
setup; bash $T/rb/scripts/probe_pstar.sh 4 > /dev/null 2>&1
ok "PASS: the two readings and the 8B, 256 problems seed 0; report, public upload, private log, event" '[ "$(res "v[\"result\"]")" = PASS ] && [ -f $T/root/runs/PROBE-PSTAR/.done ] && grep -q "^pstar --teacher $T/m8 --short $T/root/runs/E11b-P2/e11b_P2_math500_T0.6_math500_no_think_t0.6.jsonl --long $T/root/runs/E11b-P5/e11b_P5_math500_T0.6_math500_no_think_t0.6.jsonl --out $T/root/runs/PROBE-PSTAR/out --n 256 --seed 0$" $CALLS && grep -q "^remote put $T/root/runs/PROBE-PSTAR/up runs/PROBE-PSTAR$" $CALLS && grep -q "^control event PROBE-PSTAR probe_pstar_pass" $CALLS && [[ "$(res "v[\"reason\"]")" == "p* 0.55 [0.52, 0.58] (Delta_long 60.0, Delta_short 49.0; H1 True, in (0.5, 0.6) True); per token 0.2 [0.18, 0.22] (H2 True)"* ]] && [ -f $T/root/runs/PROBE-PSTAR/up/rows.jsonl ]'
cd $SRC; rm -rf $T
setup; rm $T/root/runs/E11b-P5/*.jsonl; out=$(bash $T/rb/scripts/probe_pstar.sh 4 2>&1); rc=$?
ok "a reading missing: WAIT_INPUTS, exit 1, no report" '[ $rc = 1 ] && [[ "$out" == *"WAIT_INPUTS PROBE-PSTAR: runs/E11b-P5/e11b_P5_math500_T0.6_math500_no_think_t0.6.jsonl not on this node"* ]] && [ ! -f $T/root/runs/PROBE-PSTAR/report.json ]'
cd $SRC; rm -rf $T
check() { ok "$1" '[ "$(res "v[\"result\"]")" = FAIL ] && [[ "$(res "v[\"reason\"]")" == *"'"$2"'"* ]]'; }
setup; FAKE_ROWS=200 bash $T/rb/scripts/probe_pstar.sh 4 > /dev/null 2>&1; check "fewer problems scored is FAIL" "expected 256"; cd $SRC; rm -rf $T
setup; FAKE_NAN=1 bash $T/rb/scripts/probe_pstar.sh 4 > /dev/null 2>&1; check "a non-finite log-prob is FAIL" "not finite"; cd $SRC; rm -rf $T
setup; FAKE_NOP=1 bash $T/rb/scripts/probe_pstar.sh 4 > /dev/null 2>&1; check "no p* is FAIL" "no p*"; cd $SRC; rm -rf $T
setup; FAKE_RC=2 bash $T/rb/scripts/probe_pstar.sh 4 > /dev/null 2>&1; check "pstar failing is FAIL" "pstar exited 2"; cd $SRC; rm -rf $T
[ $fails = 0 ] && echo "PROBE_PSTAR_TEST PASS ($n checks)" || echo "PROBE_PSTAR_TEST FAIL ($fails of $n)"
