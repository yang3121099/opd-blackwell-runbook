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
pull_and_control() {   # git pull, record it, apply new control commands, restart this loop if it changed
  local before=$(git -C "$RUNBOOK" rev-parse --short HEAD 2>/dev/null) ok=false
  git -C "$RUNBOOK" pull -q --ff-only 2>/dev/null && ok=true
  local after=$(git -C "$RUNBOOK" rev-parse --short HEAD 2>/dev/null)   # v2.5: changed control rules need a sign-off line
  [ -n "$before" ] && [ "$before" != "$after" ] && { python $RUNBOOK/scripts/control.py check_pull $before $after >> $OPD_ROOT/logs/control.log 2>&1 || ok=false; }
  printf '{"utc": "%s", "ok": %s, "head_before": "%s", "head": "%s"}\n' "$(date -u +%FT%TZ)" $ok "$before" \
    "$(git -C "$RUNBOOK" rev-parse --short HEAD 2>/dev/null)" > $CDIR/last_pull.json
  python $RUNBOOK/scripts/control.py apply >> $OPD_ROOT/logs/control.log 2>&1
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
running() { pgrep -f "task.sh [0-9]+ $1\$" >/dev/null; }
free_gpus() { for i in $(seq 0 $((NGPU-1))); do
    # v2.5.4d: a GPU given to a task stays taken while its task.sh / ad-hoc evaluation runs, also before the task touches it
    # (2026-10-17: FA-PROBE's CPU-only venv copy left GPU6 empty, and EVAL-bridge and EVAL-p5s43-s20 were started on it too)
    pgrep -f "scripts/(task|run_adhoc_eval)\.sh $i( |\$)" > /dev/null && continue
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $i)" ] || continue
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i | tr -d ' ')" -le 2048 ] && echo $i; done; }
# v2.5.4e: EVAL tasks get a GPU only once their models are local and checked (.inputs_ok = the hash of their #eval
# lines, written by `run_queue_eval.sh prefetch`); one CPU-side prefetch at a time, for every EVAL task of the queue
# (also in locked stages), a failed one again after 30 min
eval_hash() { awk -v t="$1" '$1 == "#eval" && $2 == t' "$Q" | cksum | cut -d' ' -f1; }
inputs_ok() { [ "$(cat $OPD_ROOT/runs/$1/.inputs_ok 2>/dev/null)" = "$(eval_hash $1)" ]; }
prefetch_pass() {
  pgrep -f "run_queue_eval\.sh prefetch " > /dev/null && return
  for t in $(awk '$1 == "#eval" {print $2}' "$Q" | sort -u); do
    [ -f $OPD_ROOT/runs/$t/.done ] && continue; inputs_ok $t && continue
    [ -n "$(find $OPD_ROOT/runs/$t/.prefetch_fail -mmin -10 2>/dev/null)" ] && continue   # v2.5.6h: retry after 10 min
    _wait=0; for d in $(deps_of $t); do [ -f $OPD_ROOT/runs/$d/.done ] || _wait=1; done; [ $_wait = 1 ] && continue   # v2.5.6h: only once its inputs exist
    mkdir -p $OPD_ROOT/runs/$t
    setsid nohup bash -c "bash $RUNBOOK/scripts/run_queue_eval.sh prefetch $t >> $OPD_ROOT/logs/prefetch_$t.log 2>&1 || touch $OPD_ROOT/runs/$t/.prefetch_fail" < /dev/null > /dev/null 2>&1 &
    return
  done; }
launch() { # gpu task   (v2.5: start / done / killed / failed also go to status/<node>/events/ as receipts)
  mkdir -p $OPD_ROOT/runs/$2
  python $RUNBOOK/scripts/control.py event $2 start "gpu=$1 attempt=$(( $(attempts $2) + 1 ))" > /dev/null 2>&1 &
  setsid nohup bash -c "bash $RUNBOOK/scripts/task.sh $1 $2 > $OPD_ROOT/logs/$2.log 2>&1
    python $RUNBOOK/scripts/control.py reap $1 $2 > /dev/null 2>&1   # v2.5.4b: the task's orphaned vLLM / Ray processes off its GPU
    if [ -f $OPD_ROOT/runs/$2/.done ]; then python $RUNBOOK/scripts/control.py event $2 done gpu=$1
    elif [ -f $OPD_ROOT/runs/$2/.killed ]; then python $RUNBOOK/scripts/control.py event $2 killed gpu=$1
    else a=\$(( \$(sed -n 's/^attempts: //p' $OPD_ROOT/runs/$2/.failed 2>/dev/null || echo 0) + 1 ))
      { echo \"attempts: \$a\"; echo \"last: \$(date -u +%FT%TZ)\"; tail -20 $OPD_ROOT/logs/$2.log; } > $OPD_ROOT/runs/$2/.failed
      python $RUNBOOK/scripts/control.py event $2 failed \"attempt=\$a\"
      if tail -c 200000 $OPD_ROOT/logs/$2.log $OPD_ROOT/runs/$2/train.log 2>/dev/null | grep -qE 'CUDA out of memory|OutOfMemoryError'; then touch $OPD_ROOT/runs/$2/.oom   # v2.5.4a: task.sh retries it with FIXED_MICRO_BSZ=4
        python $RUNBOOK/scripts/control.py event $2 oom_retry_fixed_micro \"next attempt with FIXED_MICRO_BSZ=4\"; fi; fi" < /dev/null > /dev/null 2>&1 &
}
# deps of a task id as written in the queue (task:dep+dep), for `run` requests (v2.5)
deps_of() { local s=$(grep -v '^#' "$Q" | tr ' ' '\n' | grep -E "^$1(:|\$)" | head -1); [ "$s" != "${s#*:}" ] && echo ${s#*:} | tr '+' ' '; }
while true; do
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
    for p in $(pgrep -f "task.sh [0-9]+ E1(0[bc]|2b)-[A-Za-z0-9._-]+\$"); do
      t=$(tr '\0' ' ' < /proc/$p/cmdline | grep -oE "E1(0[bc]|2b)-[A-Za-z0-9._-]+" | tail -1); [ -n "$t" ] || continue
      pgrep -f "upload_syn[c].sh $OPD_ROOT/runs/$t\$" > /dev/null && continue
      case $t in
        E10b-*|E10c-*) env UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-150} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$t >> $OPD_ROOT/runs/$t/upload.log 2>&1 < /dev/null &;;
        E12b-*) case ${t#E12b-} in P0|P1|P2|P3|P5) s=100;; *) s=60;; esac
                env KEEP_BF16=0 UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-$s} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OPD_ROOT/runs/$t >> $OPD_ROOT/runs/$t/upload.log 2>&1 < /dev/null &;;
      esac
      python $RUNBOOK/scripts/control.py event $t upload_sync_restarted > /dev/null 2>&1 &
    done
  fi
  [ "${DRY:-0}" != 1 ] && prefetch_pass
  FREE=($(free_gpus)); earlier_finished=1
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
      case $t in EVAL-*) inputs_ok $t || continue;; esac   # v2.5.4e: no GPU while its models are being fetched
      [ ${#FREE[@]} -gt 0 ] || continue
      g=${FREE[0]}; FREE=("${FREE[@]:1}")
      if [ "${DRY:-0}" = 1 ]; then echo "WOULD_START gpu=$g $t (stage $stage, attempt $(( $(attempts $t) + 1 )))"; continue; fi
      echo "START $(date -u +%T) gpu=$g $t (stage $stage, queue v$(sed -n 's/^version:[[:space:]]*//p' "$Q"), attempt $(( $(attempts $t) + 1 )))"
      launch $g $t; sleep 90
    done
    [ "$status" = locked ] && stage_finished=0
    [ $stage_finished = 1 ] || earlier_finished=0
  done < "$Q"
  # v2.5: ad-hoc `run` / `eval` requests (control.py) take only the GPUs this pass left free: the queue goes first
  if [ "${DRY:-0}" != 1 ] && ! verify_busy; then
    while [ ${#FREE[@]} -gt 0 ]; do
      r=$(python $RUNBOOK/scripts/control.py next_request 2>/dev/null); [ -n "$r" ] || break
      set -- $r; g=${FREE[0]}; FREE=("${FREE[@]:1}")
      if [ "$1" = run ]; then
        ok=1; for d in $(deps_of $2); do done_ $d || ok=0; done
        if [ $ok = 1 ] && ! running $2 && ! done_ $2; then
          echo "START $(date -u +%T) gpu=$g $2 (control seq $3)"; launch $g $2; python $RUNBOOK/scripts/control.py rcpt $3 accepted "started on gpu $g"; sleep 90
        else python $RUNBOOK/scripts/control.py rcpt $3 rejected "deps not done, already running or done"; FREE=($g "${FREE[@]}"); fi
      else
        echo "START $(date -u +%T) gpu=$g adhoc_eval_$2 (control seq $2)"
        setsid nohup bash $RUNBOOK/scripts/run_adhoc_eval.sh $g $2 > $OPD_ROOT/logs/adhoc_eval_$2.log 2>&1 < /dev/null &
        python $RUNBOOK/scripts/control.py rcpt $2 accepted "started on gpu $g"; sleep 30
      fi
    done
  fi
  [ "${DRY:-0}" = 1 ] && break
  sleep ${POLL:-60}
done
