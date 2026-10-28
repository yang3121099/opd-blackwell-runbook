#!/usr/bin/env bash
# Cold start (runbook [v2.5.9z100]; Fable's requirements v26 §11, 10-27, after the B300 was lost): an empty machine back to running
# from git (this runbook) and Hugging Face only, its disks assumed gone. One command, the person at the node types it once:
#   bash scripts/cold_start.sh --token-file <0600 file> --opd-root <dir> --ckpt-root <dir> --repos public=<owner/name>,private=<owner/name>
#        [--node <id>] [--gpus 8] [--only <task>[,<task>...]] [--no-loops] [--ckpt-min-gb 1000]
# [v2.5.9z104] (Fable 10-28: a temporary H800 to adapt; the queue must not start there) --only: the manifest cut to those tasks
# (coldstart_hf.py only: the always-models, the models / labels / weights they use), venv-vl and venv-ifeval not built and their
# checks SKIP; --no-loops: stop after the health checks (step 8 not run: no heartbeat, no run_tasks, no event; a canary is started
# by hand); --ckpt-min-gb: the CKPT_ROOT free-space check's floor in GB (default 1000).
# Steps (STEP n lines in $OPD_ROOT/logs/coldstart.log; a rerun skips what is done):
#   0  arguments; the token from its 0600 file into HF_TOKEN (never printed, never written); paths.env (CKPT_ROOT, OPD_NODE), hf_repos.txt
#   1  control bootstrap BEFORE any loop: control/applied_seq = the largest seq of control/<node>.yaml, so none of the old commands
#      runs; control.env.proposed = each setenv key's last value - printed, NOT applied (ops-lead's A1, then written by hand)
#   2  overnight.sh with OVERNIGHT_NO_LOOPS=1: pull, base models (0.6B / 1.7B / 8B), the cu128 venv, data, vLLM tier probe, verify.sh
#   3  the node's last status from the private repository: which tasks are done (one read)
#   4  in parallel: the other models the unfinished tasks need (tasks/coldstart.json), venv-fa (FA-PROBE; the source build when the
#      wheel does not import), venv-vl, venv-ifeval
#   5  the probe labels from the private repository, checked against the Hub's hash (missing: rerun the probe as a NEW label version)
#   6  which start / reading weights the unfinished tasks need are on the Hub (missing ones listed; those tasks wait, they never start)
#   7  health checks, every one PASS before anything starts (logs/coldstart_health.txt)
#   8  the loops (heartbeat, run_tasks; run_tasks keeps the others), then one event "coldstart" with the skipped seqs
# COLDSTART_DRYRUN=1: nothing is installed, downloaded or started; every such step prints what it would run (the Mac tests).
set -uo pipefail
RB=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TOKF=""; OPD=""; CK=""; REPOS=""; NODE=""; NGPU=8; ONLY=""; NOLOOPS=0; CKMIN=1000
while [ $# -gt 0 ]; do case $1 in
  --token-file) TOKF=$2; shift 2;; --opd-root) OPD=$2; shift 2;; --ckpt-root) CK=$2; shift 2;; --repos) REPOS=$2; shift 2;;
  --node) NODE=$2; shift 2;; --gpus) NGPU=$2; shift 2;; --only) ONLY=$2; shift 2;; --no-loops) NOLOOPS=1; shift;;
  --ckpt-min-gb) CKMIN=$2; shift 2;; *) echo "COLDSTART usage: unknown argument $1"; exit 2;; esac; done
[[ $NGPU =~ ^[0-9]+$ ]] && [[ $CKMIN =~ ^[0-9]+$ ]] && [[ -z $ONLY || $ONLY =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]] || { echo "COLDSTART usage: --gpus / --ckpt-min-gb take a number, --only task names (comma separated)"; exit 2; }
[ -n "$TOKF" ] && [ -n "$OPD" ] && [ -n "$CK" ] && [ -n "$REPOS" ] || { echo "COLDSTART usage: --token-file --opd-root --ckpt-root --repos are required"; exit 2; }
PUB=$(echo "$REPOS" | tr ',' '\n' | sed -n 's/^public=//p'); PRIV=$(echo "$REPOS" | tr ',' '\n' | sed -n 's/^private=//p')
[ -n "$PUB" ] && [ -n "$PRIV" ] || { echo "COLDSTART usage: --repos public=<owner/name>,private=<owner/name>"; exit 2; }
DRY=${COLDSTART_DRYRUN:-0}; export OPD_ROOT=$OPD
mkdir -p $OPD/logs $OPD/control $OPD/runs $OPD/data $OPD/models $CK || { echo "COLDSTART cannot create $OPD / $CK"; exit 1; }
LOG=$OPD/logs/coldstart.log
step() { echo "[$(date -u +%FT%TZ)] STEP $1 $2" | tee -a $LOG; }
fail() { step $1 "FAILED: $2"; exit 1; }
run() { if [ $DRY = 1 ]; then echo "DRYRUN would run: $*" | tee -a $LOG; else "$@"; fi; }
PY=python3; [ -x $OPD/venv/bin/python ] && PY=$OPD/venv/bin/python

# ---------- 0. token, node, paths ----------
[ -f "$TOKF" ] && [ -s "$TOKF" ] || fail 0 "token file missing or empty"
perm=$(stat -c %a "$TOKF" 2>/dev/null || stat -f %Lp "$TOKF"); [ "$perm" = 600 ] || [ "$perm" = 400 ] || fail 0 "token file mode $perm (needs 0600)"
HF_TOKEN=$(head -1 "$TOKF" | tr -d '[:space:]'); export HF_TOKEN; export HF_REPO=$PUB
[ -n "$NODE" ] || NODE=$(python3 -c "import hashlib, socket; print(hashlib.sha256(socket.gethostname().split('.')[0].encode()).hexdigest()[:8])")
export OPD_NODE=$NODE
{ echo "CKPT_ROOT=$CK"; echo "export OPD_NODE=$NODE"; } > $OPD/paths.env.new && mv $OPD/paths.env.new $OPD/paths.env
printf 'public=%s\nprivate=%s\n' "$PUB" "$PRIV" > $OPD/hf_repos.txt
step 0 "node $NODE; paths.env and hf_repos.txt written; token from its file (not printed)"

# ---------- 1. control bootstrap: no old command may run on this machine ----------
Y=$RB/control/$NODE.yaml
read -r MAXSEQ PROPOSED_N < <(python3 - "$Y" "$OPD/control/control.env.proposed" <<'PY'
import json, re, sys
y, out = sys.argv[1:3]
try:
    lines = open(y, encoding="utf-8").read().splitlines()
except OSError:
    lines = []
seqs, env = [0], {}
for l in lines:
    m = re.search(r"\{.*\}", l)
    if not m:
        continue
    try:
        e = json.loads(m.group(0))
    except ValueError:
        continue
    seqs.append(int(e.get("seq", 0)))
    c = str(e.get("cmd", "")).split()
    if len(c) == 2 and c[0] in ("setenv", "set") and "=" in c[1]:
        k, v = c[1].split("=", 1); env[k] = (v, e.get("seq"))
with open(out, "w") as f:
    f.write("# control.env proposed by cold_start.sh: each setenv key's last value in the old control history (seq). NOT applied:\n"
            "# review (ops-lead A1), edit for the new machine, then copy the lines you keep to $OPD_ROOT/control.env by hand.\n")
    for k, (v, s) in sorted(env.items()):
        f.write(f"# seq {s}\n{k}={v}\n")
print(max(seqs), len(env))
PY
)
CUR=$(cat $OPD/control/applied_seq 2>/dev/null || echo 0)
# [v2.5.9z104] also written when missing: a new node id (no control/<node>.yaml, MAXSEQ 0) had none, and its own check failed
if [ ! -f $OPD/control/applied_seq ] || [ "${CUR:-0}" -lt "$MAXSEQ" ]; then echo $MAXSEQ > $OPD/control/applied_seq.tmp && mv $OPD/control/applied_seq.tmp $OPD/control/applied_seq; fi
REPLIES0=$(find $OPD/control/replies -type f 2>/dev/null | wc -l | tr -d ' ')
echo "${CUR:-0} $MAXSEQ" > $OPD/control/coldstart_skipped
step 1 "applied_seq $(cat $OPD/control/applied_seq) (old commands 1..$MAXSEQ skipped, none run); control.env.proposed: $PROPOSED_N keys, NOT applied - review it:"
cat $OPD/control/control.env.proposed | tee -a $LOG

# ---------- 2. overnight.sh without its loops ----------
if [ -f $OPD/logs/verify.log ] && grep -q "SMOKE PASS" $OPD/logs/verify.log; then step 2 "overnight.sh already passed (verify.log SMOKE PASS)"
else run env OVERNIGHT_NO_LOOPS=1 OPD_ROOT=$OPD bash $RB/scripts/overnight.sh || fail 2 "overnight.sh (see logs/overnight.log)"; step 2 "overnight.sh done"; fi
[ -x $OPD/venv/bin/python ] && PY=$OPD/venv/bin/python

# ---------- 3. which tasks are done (the last status) ----------
DONE=$OPD/control/coldstart_done.txt
[ -f $DONE ] || $PY $RB/scripts/coldstart_hf.py status "$PRIV" "$NODE" $DONE | tee -a $LOG
M=$RB/tasks/coldstart.json   # the committed first version; regenerated from the queue this machine pulled, when it is there
if [ -f $RB/tasks/queue.txt ]; then python3 $RB/scripts/coldstart_manifest.py $RB/tasks/queue.txt $OPD/control/coldstart.json | tee -a $LOG && M=$OPD/control/coldstart.json; fi
if [ -n "$ONLY" ]; then python3 $RB/scripts/coldstart_hf.py only $M "$ONLY" $OPD/control/coldstart_only.json | tee -a $LOG || fail 3 "--only $ONLY"; M=$OPD/control/coldstart_only.json; fi

# ---------- 4. models and the other venvs, in parallel ----------
fa() { [ -x $OPD/venv-fa/bin/python ] && $OPD/venv-fa/bin/python -c "import flash_attn" 2> /dev/null && return 0
       bash $RB/scripts/fa_probe.sh 0 > $OPD/logs/coldstart_fa.log 2>&1
       $OPD/venv-fa/bin/python -c "import flash_attn" 2> /dev/null || { bash $RB/scripts/fa_build.sh >> $OPD/logs/coldstart_fa.log 2>&1 && rm -f $OPD/runs/FA-PROBE/.done && bash $RB/scripts/fa_probe.sh 0 >> $OPD/logs/coldstart_fa.log 2>&1; }; }
if [ $DRY = 1 ]; then
  echo "DRYRUN would run: coldstart_hf.py models, fa_probe.sh / fa_build.sh$([ -z "$ONLY" ] && echo ', build_venv_vl.sh, build_venv_ifeval.sh') (in parallel)" | tee -a $LOG
  [ -n "$ONLY" ] && echo "venv-vl, venv-ifeval not built (--only $ONLY)" | tee -a $LOG
  COLDSTART_FAKE=${COLDSTART_FAKE:-} $PY $RB/scripts/coldstart_hf.py models $M $DONE $OPD/models | tee -a $LOG
else
  ( $PY $RB/scripts/coldstart_hf.py models $M $DONE $OPD/models > $OPD/logs/coldstart_models.log 2>&1 ) & P1=$!
  ( fa ) & P2=$!
  P3=; P4=
  if [ -z "$ONLY" ]; then
    ( bash $RB/scripts/build_venv_vl.sh > $OPD/logs/coldstart_vl.log 2>&1 ) & P3=$!
    ( bash $RB/scripts/build_venv_ifeval.sh > $OPD/logs/coldstart_ifeval.log 2>&1 ) & P4=$!
  else echo "venv-vl, venv-ifeval not built (--only $ONLY)" | tee -a $LOG; fi
  wait $P1 $P2 $P3 $P4; cat $OPD/logs/coldstart_models.log | tee -a $LOG
fi
step 4 "models and venvs finished (health checks below say whether each works)"

# ---------- 5. labels ----------
$PY $RB/scripts/coldstart_hf.py labels $M "$PRIV" $OPD | tee -a $LOG; step 5 "labels checked"

# ---------- 6. weights the unfinished tasks need ----------
$PY $RB/scripts/coldstart_hf.py weights $M $DONE "$PUB" "$PRIV" $OPD/control/coldstart_weights.json | tee -a $LOG; step 6 "weights listed (control/coldstart_weights.json)"

# ---------- 7. health checks ----------
H=$OPD/logs/coldstart_health.txt; : > $H; bad=0
chk() { if eval "$2" > /dev/null 2>&1; then echo "PASS $1" >> $H; else echo "FAIL $1" >> $H; bad=1; fi; }
skp() { echo "SKIP $1 (--only $ONLY)" >> $H; }
chk "gpus: $NGPU visible" "[ \$(nvidia-smi -L | wc -l) -eq $NGPU ]"
chk "CKPT_ROOT >= $([ $CKMIN = 1000 ] && echo '1 TB' || echo "$CKMIN GB") free" "[ \$(df -Pk $CK | awk 'NR==2{print \$4}') -ge $((CKMIN * 1000000)) ]"   # the default's wording as before
chk "Hugging Face: both repositories answer" "$PY $RB/scripts/coldstart_hf.py reach $PUB $PRIV"
chk "Hugging Face: the token writes to the private repository (a probe under status/$NODE/)" "$PY $RB/scripts/coldstart_hf.py write $PRIV $NODE"
chk "control plane on one local disk (runbook, control/, logs/; not a network filesystem)" "python3 $RB/scripts/coldstart_check.py localdisk $RB $OPD/control $OPD/logs"
[ -f $Y ] && chk "node $NODE reused: its last status is older than 2 h" "python3 $RB/scripts/coldstart_check.py silent $DONE.utc 2"
chk "GitHub: the runbook remote answers" "timeout 60 git -C $RB ls-remote -q origin HEAD"
chk "venv: torch, vllm, verl" "$OPD/venv/bin/python -c 'import torch, vllm, verl'"
chk "venv-fa: flash_attn" "$OPD/venv-fa/bin/python -c 'import flash_attn'"
if [ -z "$ONLY" ]; then chk "venv-vl: vllm, transformers" "$OPD/venv-vl/bin/python -c 'import vllm, transformers'"
  chk "venv-ifeval: langdetect" "$OPD/venv-ifeval/bin/python -c 'import langdetect'"
else skp "venv-vl: vllm, transformers"; skp "venv-ifeval: langdetect"; fi
chk "verify.sh: G2, VLLM, SMOKE PASS" "grep -q 'G2 PASS' $OPD/logs/verify.log && grep -q 'VLLM PASS' $OPD/logs/verify.log && grep -q 'SMOKE PASS' $OPD/logs/verify.log"
chk "control: applied_seq == $MAXSEQ (the old history's largest)" "[ \$(cat $OPD/control/applied_seq) -eq $MAXSEQ ]"
chk "control: no command ran during the bootstrap" "[ \$(find $OPD/control/replies -type f 2>/dev/null | wc -l) -eq $REPLIES0 ]"
chk "control: outbox empty" "[ \$(find $OPD/control/outbox -type f 2>/dev/null | wc -l) -eq 0 ]"
chk "no loop running yet" "! pgrep -f 'scripts/(run_tasks|heartbeat)[.]sh'"
cat $H | tee -a $LOG
[ $bad = 0 ] || fail 7 "health checks (logs/coldstart_health.txt); nothing started"
step 7 "health checks all PASS"

# ---------- 8. the loops ----------
[ $NOLOOPS = 1 ] && { step 8 "loops NOT started (--no-loops): nothing runs until a task is started by hand (bash scripts/task.sh <gpu> <task>)"; exit 0; }
cd $RB
run setsid nohup bash scripts/heartbeat.sh > $OPD/logs/heartbeat.log 2>&1 9>&- &
run setsid nohup bash scripts/run_tasks.sh > $OPD/logs/run_tasks.log 2>&1 9>&- &
run python3 scripts/control.py event node coldstart "skipped seq $(cut -d' ' -f1 $OPD/control/coldstart_skipped)+1..$MAXSEQ (none run); control.env proposed, not applied; health all PASS"
step 8 "loops started (heartbeat, run_tasks; run_tasks keeps uploader / watchdog / gh_heartbeat / weights_replay); event coldstart sent"
