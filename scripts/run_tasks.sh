#!/usr/bin/env bash
# Remote-node loop (queue version 2): every POLL seconds (default 60; v2.3 used 5 minutes), start the next runnable
# tasks of the open stages, strictly in queue order, one task per free GPU (no compute process and <= 2 GB used), 90 s
# apart.
#  - an "auto" stage opens once every task of every earlier stage is finished (.done locally or in the result
#    repository, or failed 3 times); a "locked" stage never opens and holds back everything after it
#  - a task starts only when its deps are .done; a task whose dep failed for good gets runs/<task>/.skipped and counts
#    as finished, so later stages still open (v2.3: before, it held its stage open forever and stalled the ladder)
#  - fail-fast: a task that exits without .done gets runs/<task>/.failed (attempt count + the log tail, reported by the
#    heartbeat) and is retried at most twice (3 attempts in all)
#  - v2.4 (ops-1): `git pull` every 60 s (PULL_EVERY; a pull of an unchanged repository is one small fetch), then
#    scripts/control.py applies the new commands of
#    control/<node>.yaml (pause/resume/kill/prio/restore/rerun_verify/setenv/snapshot; once each, in seq order);
#    paused tasks are not started, a stage's tasks are tried in prio order (runs/<task>/.prio, default 100, then queue
#    order), nothing starts while a remote verify runs, and a task stopped by `kill` is not counted as a failure.
#  - v2.4: when a pull changes run_tasks.sh or env.sh, this loop restarts itself (exec) after `bash -n` passes;
#    running tasks are separate sessions and keep running.
# DRY=1 prints what would start, without starting anything.
set -uo pipefail
source "$(dirname "$0")/env.sh"; source $OPD_ROOT/venv/bin/activate
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
Q=$RUNBOOK/tasks/queue.txt; last_pull=0; CDIR=$OPD_ROOT/control; mkdir -p $CDIR
SELF_SHA=$(cat "$RUNBOOK/scripts/run_tasks.sh" "$RUNBOOK/scripts/env.sh" | sha1sum | cut -c1-12)
echo "RUN_TASKS_START $(date -u +%T) sha $SELF_SHA"; echo $SELF_SHA > $CDIR/run_tasks.sha
paused() { [ -f $OPD_ROOT/runs/$1/.paused ]; }
prio() { cat $OPD_ROOT/runs/$1/.prio 2>/dev/null | tr -cd '0-9' | grep . || echo 100; }
verify_busy() { [ -f $CDIR/verify.running ] && kill -0 "$(cat $CDIR/verify.running 2>/dev/null)" 2>/dev/null; }
# v2.5.7zb: the loop's network calls are bounded. On 2026-10-19 one of them hung 12:17-13:50 UTC with no timeout and
# nothing was dispatched for 94 min. bounded <seconds> <command...>: the command under `timeout` (when present); a
# timeout is logged and sent as a (bounded) event, and the loop goes on.
bounded() {
  local s=$1; shift
  local to=0; command -v timeout > /dev/null && to=1
  if [ $to = 1 ]; then timeout -k 30 $s "$@"; else "$@"; fi   # [v2.5.8] -k 30: KILL 30 s after a TERM that is ignored
  local rc=$?
  if [ $to = 1 ] && { [ $rc = 124 ] || [ $rc = 137 ]; }; then
    echo "LOOP_TIMEOUT $(date -u +%T) ${s}s rc=$rc $*" >> $OPD_ROOT/logs/control.log
    timeout -k 10 60 python $RUNBOOK/scripts/control.py event node loop_timeout "$(basename $1) $2" > /dev/null 2>&1 &
  fi
  return $rc; }
# v2.5.7zb: the watchdog (watchdog.sh) restarts this loop or the heartbeat when their liveness files go stale; it is
# started here when it is not running (pgrep + its own flock against a second copy)
nap() {   # v2.5.7zb: sleep in 10-s steps, refreshing the liveness file (90 s between launches must not look stale)
  local t=$1
  while [ $t -gt 0 ]; do sleep $(( t < 10 ? t : 10 )); t=$(( t - 10 )); date -u +%s > $CDIR/loop_run_tasks; done; }
# [v2.5.8] ensure_loop <name>: the watchdog / the uploader runs, on the current script: one whose file changed since it
# started (control/<name>.sha, like the heartbeat's) is stopped and started again, so a pull upgrades them too
ensure_loop() {
  [ "${DRY:-0}" = 1 ] && return
  local sha p; sha=$(sha1sum $RUNBOOK/scripts/$1.sh | cut -c1-12)
  if pgrep -f "scripts/$1.s[h]" > /dev/null; then
    [ "$(cat $CDIR/$1.sha 2> /dev/null)" = "$sha" ] && return
    for p in $(pgrep -f "scripts/$1.s[h]"); do kill $p 2> /dev/null; done; sleep 2
    python $RUNBOOK/scripts/control.py event node $1_upgraded "$sha" > /dev/null 2>&1 &
  fi
  echo $sha > $CDIR/$1.sha
  (cd $RUNBOOK && setsid nohup bash scripts/$1.sh >> $OPD_ROOT/logs/$1.log 2>&1 9>&- < /dev/null &); }
ensure_watchdog() { ensure_loop watchdog; }
ensure_uploader() { ensure_loop uploader; }   # [v2.5.8] the outbox's shipper
pull_and_control() {   # git pull, record it, apply new control commands, restart this loop if it changed
  local before=$(git -C "$RUNBOOK" rev-parse --short HEAD 2>/dev/null) ok=false
  bounded ${PULL_TIMEOUT:-120} git -C "$RUNBOOK" pull -q --ff-only 2>/dev/null && ok=true   # v2.5.7zb
  local after=$(git -C "$RUNBOOK" rev-parse --short HEAD 2>/dev/null)   # v2.5: changed control rules need a sign-off line
  [ -n "$before" ] && [ "$before" != "$after" ] && { python $RUNBOOK/scripts/control.py check_pull $before $after >> $OPD_ROOT/logs/control.log 2>&1 || ok=false; }
  printf '{"utc": "%s", "ok": %s, "head_before": "%s", "head": "%s"}\n' "$(date -u +%FT%TZ)" $ok "$before" \
    "$(git -C "$RUNBOOK" rev-parse --short HEAD 2>/dev/null)" > $CDIR/last_pull.json
  bounded ${APPLY_TIMEOUT:-300} python $RUNBOOK/scripts/control.py apply >> $OPD_ROOT/logs/control.log 2>&1   # v2.5.7zb
  local now=$(cat "$RUNBOOK/scripts/run_tasks.sh" "$RUNBOOK/scripts/env.sh" | sha1sum | cut -c1-12)
  if [ "$now" != "$SELF_SHA" ]; then
    if bash -n "$RUNBOOK/scripts/run_tasks.sh" && bash -n "$RUNBOOK/scripts/env.sh"; then
      echo "SELF_RESTART $(date -u +%T) $SELF_SHA->$now"; rm -f $CDIR/self_restart_refused; exec bash "$RUNBOOK/scripts/run_tasks.sh"
    else echo "$(date -u +%FT%TZ) new version $now fails bash -n; staying on $SELF_SHA" > $CDIR/self_restart_refused; fi
  fi
}
# done locally, or in the result repository; a remote hit is cached as a local .done so it is asked only once (v2.3)
# the remote question is asked at most every REMOTE_EVERY s per task (600; with 60 s passes it would otherwise cost one
# HF API call per unfinished task per minute); the marker lives in control/, so runs/ only holds tasks that exist
done_() { [ -f $OPD_ROOT/runs/$1/.done ] && return 0
  local m=$CDIR/remote_checked/$1; mkdir -p $CDIR/remote_checked
  [ -f $m ] && [ $(( $(date +%s) - $(stat -c %Y $m) )) -lt ${REMOTE_EVERY:-600} ] && return 1
  touch $m
  python $RUNBOOK/scripts/remote.py done $1 2>/dev/null && { mkdir -p $OPD_ROOT/runs/$1; touch $OPD_ROOT/runs/$1/.done; }; }
skipped() { [ -f $OPD_ROOT/runs/$1/.skipped ]; }
attempts() { sed -n 's/^attempts: //p' $OPD_ROOT/runs/$1/.failed 2>/dev/null || echo 0; }
dead() { [ "$(attempts $1)" -ge 3 ]; }
running() { pgrep -f "task.sh ([0-9]+(,[0-9]+)*|cpu) $1\$" >/dev/null; }   # [v2.5.8] cpu: a CPU-lane task; [v2.5.9a] a,b: a #gpus task
# [v2.5.8] (reviewer 08:5x UTC 10-19: CPU phases without a GPU slot) CPU-only tasks run in a lane of their own, "gpu" cpu,
# at most CPU_MAX (default 1) at once; STRICT-C1 held a GPU slot for its rejudge (and three on its failed attempts)
cpu_task() { case $1 in STRICT-C1|PROBE-NET|PROBE-NET2) return 0;; esac; return 1; }   # [v2.5.9c] PROBE-NET; [v2.5.9h] PROBE-NET2
cpu_busy() { [ "$(pgrep -f "task.sh cpu [A-Za-z0-9._-]+\$" | wc -l | tr -d ' ')" -ge "${CPU_MAX:-1}" ]; }
free_gpus() { for i in $(seq 0 $((NGPU-1))); do
    # v2.5.4d: a GPU given to a task stays taken while its task.sh / ad-hoc evaluation runs, also before the task touches it
    # (2026-10-17: FA-PROBE's CPU-only venv copy left GPU6 empty, and EVAL-bridge and EVAL-p5s43-s20 were started on it too)
    pgrep -f "scripts/(task|run_adhoc_eval)\.sh ([0-9]+,)*$i(,[0-9]+)*( |\$)" > /dev/null && continue   # [v2.5.9a] also in a GPU list
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $i)" ] || continue
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i | tr -d ' ')" -le 2048 ] && echo $i; done; }
# [v2.5.9a] (Runming via the reviewer 06:4x UTC 10-21: "a task heavy for one GPU gets two") "#gpus <task> <n>" lines of the
# queue (n 1-4, default 1): the task gets n GPUs, passed as a,b (task.sh accepts a list for the trainings and PROBE-2GPU;
# run_e12.sh / run_arm.sh: CUDA_VISIBLE_DEVICES=a,b, trainer.n_gpus_per_node=n, one vLLM engine per rank). A verl FSDP
# checkpoint is sharded by world size: a #gpus task resumes only with the same n.
gpus_of() { local n; n=$(awk -v t="$1" '$1 == "#gpus" && $2 == t {print $3}' "$Q" | tail -1); [[ $n =~ ^[1-4]$ ]] && echo $n || echo 1; }
# take_gpus <task>: G = its first n free GPUs (a,b), dropped from FREE. With too few free GPUs a multi-GPU task holds the
# free ones (FREE emptied: the later tasks of this pass get none) in turns of GPUS_HOLD_MIN (default 30) minutes from its
# first wait: held for one turn, released to the later tasks for the next, and so on, so neither side waits for ever.
# [v2.5.9c] (reviewer 08:4x UTC 10-21: GPU7 sat held 25 min while a training waited) it holds only while another GPU is
# about to free: a running task's ETA (the latest status) within GPUS_SOON_MIN (30) minutes; the held GPUs and the task go
# to control/held.json for the pulse (held_for), cleared at the start of every pass
soon_free() { python3 - "$OPD_ROOT/logs" "${GPUS_SOON_MIN:-30}" <<'PY' 2> /dev/null
import glob, json, sys
fs = sorted(glob.glob(sys.argv[1] + "/status_*.json"))
t = json.load(open(fs[-1])).get("tasks", {}) if fs else {}
sys.exit(0 if any(isinstance(v, dict) and v.get("state") in ("running", "retrying") and isinstance(v.get("eta_min"), (int, float))
                  and v["eta_min"] <= float(sys.argv[2]) for v in t.values()) else 1)
PY
}
take_gpus() {
  local n w; n=$(gpus_of $1); G=""
  if [ ${#FREE[@]} -ge $n ]; then G=$(IFS=,; echo "${FREE[*]:0:$n}"); FREE=("${FREE[@]:$n}"); rm -f $OPD_ROOT/runs/$1/.gpus_wait; return 0; fi
  if [ $n -gt 1 ] && [ ${#FREE[@]} -gt 0 ]; then
    w=$OPD_ROOT/runs/$1/.gpus_wait; mkdir -p $OPD_ROOT/runs/$1
    if [ ! -f $w ]; then date +%s > $w; echo "HOLD $(date -u +%T) gpu=$(IFS=,; echo "${FREE[*]}") for $1 (needs $n)"; fi
    if [ $(( ($(date +%s) - $(cat $w)) / (${GPUS_HOLD_MIN:-30} * 60 + 1) % 2 )) = 0 ] && soon_free; then
      printf '{"task": "%s", "gpus": [%s], "since_utc": "%s"}\n' $1 "$(IFS=,; echo "${FREE[*]}")" "$(date -u -d @$(cat $w) +%FT%TZ 2> /dev/null || date -u +%FT%TZ)" > $OPD_ROOT/control/held.json
      FREE=()
    fi
  fi
  return 1; }
# v2.5.4e: EVAL tasks get a GPU only once their models are local and checked (.inputs_ok = the hash of their #eval
# lines, written by `run_queue_eval.sh prefetch`); CPU-side prefetches for every EVAL task of the queue
# (also in locked stages), a failed one again after 30 min
eval_hash() { awk -v t="$1" '$1 == "#eval" && $2 == t' "$Q" | cksum | cut -d' ' -f1; }
inputs_ok() { [ "$(cat $OPD_ROOT/runs/$1/.inputs_ok 2>/dev/null)" = "$(eval_hash $1)" ]; }
# v2.5.6h: an EVAL task whose models are all result:runs/... (read back from this node's public repository after the
# training's upload) is ready once its deps are done; it waits for the upload itself, so it takes the first free GPU
result_only() { local m; m=$(awk -v t="$1" '$1 == "#eval" && $2 == t {print $4}' "$Q"); [ -n "$m" ] && ! echo "$m" | grep -qv '^result:'; }
# [v2.5.8] an EVAL task with nothing to download (only Qwen/ base models and local: saves) is ready at once: on 10-19
# the two readings switched to local: waited for a prefetch while trainings took the GPUs freed for them (15:33 UTC)
no_fetch() { local m; m=$(awk -v t="$1" '$1 == "#eval" && $2 == t {print $4}' "$Q"); [ -n "$m" ] && ! echo "$m" | grep -qv -E '^(Qwen/|local:)'; }
# [v2.5.8] up to PREFETCH_PAR (default 3) prefetches at once (10-19: one stuck prefetch held back every reading for an
# hour), each ended after PREFETCH_TIMEOUT (default 5400 s); fetch_model.py locks each path, so two prefetches needing
# one model download it once
prefetch_pass() {
  local n t PTO=""   # n = distinct tasks being prefetched (the wrapper, timeout and bash of one prefetch all match)
  n=$(ps -eo args= | grep -o "run_queue_eval\.sh prefetch [A-Za-z0-9._-]*" | sort -u | wc -l | tr -d ' ')
  command -v timeout > /dev/null && PTO="timeout -k 60 ${PREFETCH_TIMEOUT:-5400}"
  for t in $(awk '$1 == "#eval" {print $2}' "$Q" | sort -u); do
    [ "${n:-0}" -ge "${PREFETCH_PAR:-3}" ] && return
    pgrep -f "run_queue_eval\.sh prefetch $t( |\$)" > /dev/null && continue
    [ -f $OPD_ROOT/runs/$t/.done ] && continue; inputs_ok $t && continue
    [ -n "$(find $OPD_ROOT/runs/$t/.prefetch_fail -mmin -10 2>/dev/null)" ] && continue   # v2.5.6h: retry after 10 min
    _wait=0; for d in $(deps_of $t); do [ -f $OPD_ROOT/runs/$d/.done ] || _wait=1; done; [ $_wait = 1 ] && continue   # v2.5.6h: only once its inputs exist
    mkdir -p $OPD_ROOT/runs/$t
    setsid nohup bash -c "$PTO bash $RUNBOOK/scripts/run_queue_eval.sh prefetch $t >> $OPD_ROOT/logs/prefetch_$t.log 2>&1 || touch $OPD_ROOT/runs/$t/.prefetch_fail" < /dev/null > /dev/null 2>&1 &
    n=$((n + 1))
  done; }
net2_unfreeze() {   # [v2.5.9i] (ops-lead's A1 of 9h) the slow pull a PROBE-NET2 stopped and could not resume (SIGKILL: no trap) runs again
  local f=$OPD_ROOT/runs/PROBE-NET2/paused_pids
  [ -s $f ] || return 0; pgrep -f "scripts/probe_net2\.sh" > /dev/null && return 0
  kill -CONT $(cat $f) 2> /dev/null; python3 $RUNBOOK/scripts/control.py event PROBE-NET2 pulls_resumed_by_run_tasks "$(cat $f)" > /dev/null 2>&1 &
  : > $f; }
# v2.5.7w (reviewer 08:5x UTC): "#fetch Qwen/Qwen3-<n>B" lines of the queue = a model fetched to $OPD_ROOT/models/<name> in the
# background, without a GPU (fetch_model.py: every file by size and sha256, .fetch_ok, events fetch_done / fetch_failed);
# one model at a time, a failure retried after 30 min
fetch_pass() {
  pgrep -f "fetch_model\.py .* --task FETCH-" > /dev/null && { fetch_progress; return; }
  # [v2.5.9e] no new pull while result uploads are timing out (the node's proxy is shared; results go first)
  [ -n "$(find $OPD_ROOT/logs/heartbeat.err -mmin -60 2> /dev/null)" ] && tail -20 $OPD_ROOT/logs/heartbeat.err | grep -q UPLOAD_TIMEOUT && return
  local m d r ds nm io
  for m in $(awk '$1 == "#fetch" {print $2}' "$Q"); do
    case $m in   # [v2.5.9e] Qwen3 text and VL models, and datasets (dataset:<owner>/<name> -> data/hf/<owner>__<name>)
      Qwen/*) [[ $m =~ ^Qwen/Qwen3-(VL-)?[0-9.]+B(-Instruct|-Thinking)?(-[0-9]{4})?$ ]] || continue; r=$m; d=$OPD_ROOT/models/${m#Qwen/}; ds=""; nm=${m#Qwen/};;
      dataset:*) r=${m#dataset:}; [[ $r =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || continue; d=$OPD_ROOT/data/hf/${r/\//__}; ds=--dataset; nm=${r/\//__};;
      *) continue;;
    esac
    [ -f $d/.fetch_ok ] && continue
    [ -n "$(find $d.fetch_fail -mmin -${FETCH_RETRY_MIN:-10} 2>/dev/null)" ] && continue
    mkdir -p $(dirname $d) $OPD_ROOT/logs
    # [v2.5.9e] slow background pulls (0.03-0.05 MB/s through the proxy, PROBE-NET 10-21): niced, idle I/O class, each call
    # bounded by FETCH_TIMEOUT_BG (6 h; the partial files stay and the next call goes on), retried after FETCH_RETRY_MIN (10)
    io=""; command -v ionice > /dev/null && io="ionice -c3"
    setsid nohup bash -c "FETCH_TIMEOUT=${FETCH_TIMEOUT_BG:-21600} FETCH_PARALLEL=${FETCH_PARALLEL_BG:-1} nice -n 19 $io python $RUNBOOK/scripts/fetch_model.py $r '' $d $ds --task FETCH-$nm >> $OPD_ROOT/logs/fetch_$nm.log 2>&1 && rm -f $d.fetch_fail || touch $d.fetch_fail" < /dev/null > /dev/null 2>&1 &
    date +%s > ${CDIR:-$OPD_ROOT/control}/fetch_progress.t
    return
  done; }
# [v2.5.9e] while a pull runs: once an hour an event fetch_progress with the MB already on disk (its directory, partial
# downloads included)
fetch_progress() {
  local c=${CDIR:-$OPD_ROOT/control} last p a d; last=$(cat $c/fetch_progress.t 2> /dev/null || echo 0)
  [ $(( $(date +%s) - last )) -ge ${FETCH_PROGRESS_SEC:-3600} ] || return
  date +%s > $c/fetch_progress.t
  for p in $(pgrep -f "fetch_model\.py .* --task FETCH-"); do
    a=$(tr '\0' ' ' < /proc/$p/cmdline 2> /dev/null); d=$(echo "$a" | grep -oE "$OPD_ROOT/(models|data/hf)/[A-Za-z0-9._-]+" | head -1); [ -n "$d" ] || continue
    python $RUNBOOK/scripts/control.py event FETCH-$(basename $d) fetch_progress "$(du -sm $d 2> /dev/null | cut -f1) MB on disk" > /dev/null 2>&1 &
    break
  done; }
# [v2.5.9c] (reviewer 10-21; Runming: "is ToolRL passed over too?") "#git <owner>/<repo> <40-hex commit>" lines of the queue = a
# public GitHub repository at that commit onto $OPD_ROOT/data/git/<owner>__<repo> (git_fetch.sh: shallow, the commit
# checked, sha256 manifest, a summary of its data files to the private repository, .fetch_ok = the commit), without a
# GPU or the Mac; one at a time, a failure retried after 30 min, a changed commit fetched again
git_pass() {
  pgrep -f "scripts/git_fetch\.sh " > /dev/null && return
  local r c d k
  while read -r r c k; do
    [[ $r =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ && $c =~ ^[0-9a-f]{40}$ ]] || continue
    d=$OPD_ROOT/data/git/${r/\//__}
    if [ "$(cat $d/.fetch_ok 2>/dev/null)" = $c ]; then   # [v2.5.9e] at the commit; its hf/ folders linked into data/hf?
      [ -n "$(find $d.link_fail -mmin -30 2>/dev/null)" ] && continue
      for x in $d/hf/*/; do [ -d "$x" ] || continue; [[ $(basename $x) =~ ^[A-Za-z0-9._-]+__[A-Za-z0-9._-]+$ ]] || continue
        [ -e $OPD_ROOT/data/hf/$(basename $x) ] && continue
        setsid nohup bash $RUNBOOK/scripts/git_fetch.sh $r $c $d link >> $OPD_ROOT/logs/git_fetch.log 2>&1 < /dev/null &
        return; done
      continue
    fi
    [ -n "$(find $d.git_fail -mmin -30 2>/dev/null)" ] && continue
    mkdir -p $OPD_ROOT/data/git $OPD_ROOT/logs
    [ "${k:-}" = code ] || k=""   # [v2.5.9e] a 4th field `code`: file list only (link is internal)
    setsid nohup bash $RUNBOOK/scripts/git_fetch.sh $r $c $d $k >> $OPD_ROOT/logs/git_fetch.log 2>&1 < /dev/null &
    return
  done <<< "$(awk '$1 == "#git" {print $2, $3, $4}' "$Q")"; }
launch() { # gpu task   (v2.5: start / done / killed / failed also go to status/<node>/events/ as receipts; [v2.5.8] gpu cpu: no reap)
  mkdir -p $OPD_ROOT/runs/$2
  python $RUNBOOK/scripts/control.py event $2 start "gpu=$1 attempt=$(( $(attempts $2) + 1 ))" > /dev/null 2>&1 &
  setsid nohup bash -c "bash $RUNBOOK/scripts/task.sh $1 $2 > $OPD_ROOT/logs/$2.log 2>&1
    [ $1 = cpu ] || for _g in \$(echo $1 | tr , ' '); do python $RUNBOOK/scripts/control.py reap \$_g $2 > /dev/null 2>&1; done   # v2.5.4b: the task's orphaned vLLM / Ray processes off its GPU(s)
    if [ -f $OPD_ROOT/runs/$2/.done ]; then python $RUNBOOK/scripts/control.py event $2 done gpu=$1
    elif [ -f $OPD_ROOT/runs/$2/.killed ]; then python $RUNBOOK/scripts/control.py event $2 killed gpu=$1
    else a=\$(( \$(sed -n 's/^attempts: //p' $OPD_ROOT/runs/$2/.failed 2>/dev/null || echo 0) + 1 ))
      { echo \"attempts: \$a\"; echo \"last: \$(date -u +%FT%TZ)\"; tail -20 $OPD_ROOT/logs/$2.log; } > $OPD_ROOT/runs/$2/.failed
      python $RUNBOOK/scripts/control.py event $2 failed \"attempt=\$a\"
      if tail -c 200000 $OPD_ROOT/logs/$2.log $OPD_ROOT/runs/$2/train.log 2>/dev/null | grep -qE 'CUDA out of memory|OutOfMemoryError'; then touch $OPD_ROOT/runs/$2/.oom   # v2.5.4a: task.sh retries it with FIXED_MICRO_BSZ=4
        python $RUNBOOK/scripts/control.py event $2 oom_retry_fixed_micro \"next attempt with FIXED_MICRO_BSZ=4\"; fi; fi" < /dev/null > /dev/null 2>&1 &
}
# deps of a task id as written in the queue (task:dep+dep), for `run` requests (v2.5)
stage_lines() {   # [v2.5.9m] (Runming 10-21 21:2x Beijing via ops-lead: padding readings yield to formal work) the queue's
  # stage lines in dispatch order: every stage not named on a "#pad <stage> [<stage> ...]" line first, then the padding
  # stages, each group in file order; comments dropped (the loop skipped them). Without #pad lines: the file order.
  awk 'NR == FNR { if ($1 == "#pad") for (i = 2; i <= NF; i++) pad[$i] = 1; next }
       $1 == "" || $1 ~ /^#/ || $1 == "version:" { next }
       ($1 in pad) { tail = tail $0 "\n"; next }
       { print }
       END { printf "%s", tail }' "$1" "$1"; }
deps_of() { local s=$(grep -v '^#' "$Q" | tr ' ' '\n' | grep -E "^$1(:|\$)" | head -1); [ "$s" != "${s#*:}" ] && echo ${s#*:} | tr '+' ' '; }
while true; do
  date -u +%s > $CDIR/loop_run_tasks   # v2.5.7zb: liveness for the watchdog (local, every round)
  ensure_watchdog; ensure_uploader
  if [ "${PULL:-1}" = 1 ] && [ $(( $(date +%s) - last_pull )) -ge ${PULL_EVERY:-60} ]; then last_pull=$(date +%s); pull_and_control; fi
  # v2.5 self-heal: the heartbeat loop runs the current heartbeat.sh (restarted when it died or the file changed), and every
  # running E10b/E12b task has its upload_sync (restarted with the settings run_arm.sh / run_e12.sh use); one event each
  if [ "${DRY:-0}" != 1 ]; then
    hb_sha=$(sha1sum $RUNBOOK/scripts/heartbeat.sh | cut -c1-12); hb_pid=$(pgrep -f "scripts/heartbeat.s[h]" | head -1)
    if [ -n "$hb_pid" ] && [ "$(cat $CDIR/heartbeat.sha 2>/dev/null)" != "$hb_sha" ]; then
      for p in $(pgrep -f "scripts/heartbeat.s[h]"); do kill $p 2> /dev/null; done; sleep 2; hb_pid=""   # only our heartbeat loop
      python $RUNBOOK/scripts/control.py event node heartbeat_upgraded "$hb_sha" > /dev/null 2>&1 &
    fi
    if [ -z "$hb_pid" ]; then
      setsid nohup bash $RUNBOOK/scripts/heartbeat.sh >> $OPD_ROOT/logs/heartbeat.log 2>&1 < /dev/null &
      python $RUNBOOK/scripts/control.py event node heartbeat_restarted > /dev/null 2>&1 &
    fi
    for p in $(pgrep -f "task.sh [0-9]+(,[0-9]+)* E1(0[bcd]|2[bcd])-[A-Za-z0-9._-]+\$"); do
      t=$(tr '\0' ' ' < /proc/$p/cmdline | grep -oE "E1(0[bcd]|2[bcd])-[A-Za-z0-9._-]+" | tail -1); [ -n "$t" ] || continue
      pgrep -f "upload_syn[c].sh $OPD_ROOT/runs/$t\$" > /dev/null && continue
      case $t in
        E10b-P5tail|E10b-P5stopmerge|E10b-P5top10|E10b-P5rand10m|E10b-P5think|E10b-P5self|E10b-P5self-s43|E10d-P5self|E10b-P2self|E10b-P5len[SLR]) env UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-50} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$t >> $OPD_ROOT/runs/$t/upload.log 2>&1 < /dev/null &;;   # v2.5.7r: the 50-step arms
        E10d-X1|E10d-X2|E10d-X4) env UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-25,50} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$t >> $OPD_ROOT/runs/$t/upload.log 2>&1 < /dev/null &;;   # [v2.5.8i] the X arms: s25 and s50
        E10c-P5-polish) env UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-20} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$t >> $OPD_ROOT/runs/$t/upload.log 2>&1 < /dev/null &;;   # [v2.5.9u] its s20
        E10c-P5-fix4) env UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-100,150} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$t >> $OPD_ROOT/runs/$t/upload.log 2>&1 < /dev/null &;;   # [v2.5.8a] s100 too (the lab's 0.6B P5 reading step)
        E10b-*|E10c-*|E10d-*) env UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-150} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$t >> $OPD_ROOT/runs/$t/upload.log 2>&1 < /dev/null &;;
        E12b-*|E12c-*|E12d-*) case $t in E12b-P5-s44|E12d-P5) s=25,50,75,100;; E12b-P5-8k|E12b-P5-16k) s=25,50;; E12c-*|E12d-*|E12b-P0|E12b-P1|E12b-P2|E12b-P3|E12b-P5|E12b-P5-s43|E12b-P2-s43|E12b-P5self|E12b-P5self-s43|E12b-P5-kl|E12b-P5-stop|E12b-P5-stop2) s=100;; *) s=60;; esac   # v2.5.7u: s44's mid-RL steps
                env KEEP_BF16=0 UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-$s} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$t >> $OPD_ROOT/runs/$t/upload.log 2>&1 < /dev/null &;;
      esac
      python $RUNBOOK/scripts/control.py event $t upload_sync_restarted > /dev/null 2>&1 &
    done
  fi
  [ "${DRY:-0}" != 1 ] && prefetch_pass
  [ "${DRY:-0}" != 1 ] && net2_unfreeze   # [v2.5.9i]
  [ "${DRY:-0}" != 1 ] && fetch_pass   # v2.5.7w
  [ "${DRY:-0}" != 1 ] && git_pass   # [v2.5.9c]
  FREE=($(free_gpus)); earlier_finished=1; rm -f $CDIR/held.json   # [v2.5.9c] a hold is written again by take_gpus
  verify_busy && FREE=()   # a remote rerun_verify owns the GPUs
  while read -r stage status rest; do
    case $stage in ''|\#*|version:) continue;; esac
    opened=0; [ "$status" = open ] && opened=1; [ "$status" = auto ] && [ $earlier_finished = 1 ] && opened=1
    stage_finished=1
    for spec in $(for s in $rest; do echo "$(prio ${s%%:*}) $s"; done | sort -s -n -k1,1 | cut -d' ' -f2); do
      t=${spec%%:*}; deps=""; [ "$spec" != "$t" ] && deps=$(echo ${spec#*:} | tr '+' ' ')
      if done_ $t || dead $t || skipped $t; then continue; fi
      bad=""; for d in $deps; do dead $d || skipped $d && bad="$bad $d"; done
      if [ -n "$bad" ]; then   # a dependency failed for good: this task can never run; it no longer blocks its stage
        mkdir -p $OPD_ROOT/runs/$t; echo "skipped: dependency failed for good:$bad ($(date -u +%FT%TZ))" > $OPD_ROOT/runs/$t/.skipped
        echo "SKIP $t: dependency$bad failed for good"; continue
      fi
      stage_finished=0
      [ $opened = 1 ] || continue
      running $t && continue
      paused $t && continue
      ok=1; for d in $deps; do done_ $d || ok=0; done
      [ $ok = 1 ] || continue
      case $t in EVAL-*) inputs_ok $t || result_only $t || no_fetch $t || continue;; esac   # v2.5.4e: no GPU while its models are being fetched ([v2.5.8] no_fetch)
      if cpu_task $t; then   # [v2.5.8] the CPU lane: no GPU taken
        cpu_busy && continue
        if [ "${DRY:-0}" = 1 ]; then echo "WOULD_START gpu=cpu $t (stage $stage, attempt $(( $(attempts $t) + 1 )))"; continue; fi
        echo "START $(date -u +%T) gpu=cpu $t (stage $stage, attempt $(( $(attempts $t) + 1 )))"; launch cpu $t; nap 10; continue
      fi
      take_gpus $t || continue; g=$G   # [v2.5.9a] one GPU, or n for a #gpus task
      if [ "${DRY:-0}" = 1 ]; then echo "WOULD_START gpu=$g $t (stage $stage, attempt $(( $(attempts $t) + 1 )))"; continue; fi
      echo "START $(date -u +%T) gpu=$g $t (stage $stage, queue v$(sed -n 's/^version:[[:space:]]*//p' "$Q"), attempt $(( $(attempts $t) + 1 )))"
      launch $g $t; nap 90
    done
    [ "$status" = locked ] && stage_finished=0
    [ $stage_finished = 1 ] || earlier_finished=0
  done < <(stage_lines "$Q")   # [v2.5.9m] padding stages last
  # v2.5: ad-hoc `run` / `eval` requests (control.py) take only the GPUs this pass left free: the queue goes first
  if [ "${DRY:-0}" != 1 ] && ! verify_busy; then
    while [ ${#FREE[@]} -gt 0 ]; do
      r=$(python $RUNBOOK/scripts/control.py next_request 2>/dev/null); [ -n "$r" ] || break
      set -- $r; g=${FREE[0]}; FREE=("${FREE[@]:1}")
      if [ "$1" = run ]; then
        ok=1; for d in $(deps_of $2); do done_ $d || ok=0; done
        if [ "$(gpus_of $2)" != 1 ]; then   # [v2.5.9a] a #gpus task starts from the queue only (its n GPUs, its checkpoint's world size)
          python $RUNBOOK/scripts/control.py rcpt $3 rejected "$2 takes $(gpus_of $2) GPUs: it starts from the queue"; FREE=($g "${FREE[@]}")
        elif [ $ok = 1 ] && ! running $2 && ! done_ $2; then
          echo "START $(date -u +%T) gpu=$g $2 (control seq $3)"; launch $g $2; python $RUNBOOK/scripts/control.py rcpt $3 accepted "started on gpu $g"; nap 90
        else python $RUNBOOK/scripts/control.py rcpt $3 rejected "deps not done, already running or done"; FREE=($g "${FREE[@]}"); fi
      else
        echo "START $(date -u +%T) gpu=$g adhoc_eval_$2 (control seq $2)"
        setsid nohup bash $RUNBOOK/scripts/run_adhoc_eval.sh $g $2 > $OPD_ROOT/logs/adhoc_eval_$2.log 2>&1 < /dev/null &
        python $RUNBOOK/scripts/control.py rcpt $2 accepted "started on gpu $g"; nap 30
      fi
    done
  fi
  [ "${DRY:-0}" = 1 ] && break
  nap ${POLL:-60}
done
