#!/usr/bin/env bash
# [v3.2] gpu_fill.sh (Fable 11-17; Runming: the machine's rule is "GPU utilisation >= 85 %, about 90 %, else the job is ended after two
# hours", and its owner's instruction "when your program fails or ends, start <the owner's filler command>"; the owner 11-17: "my
# filler may already be on the cards - clear them before you start", "the disk fills up often: a disk-full exit must still start the
# filler"): an idle GPU of this node runs THE OWNER'S OWN filler, never a load of ours.
#   - The command is node-local: one line in $OPD_ROOT/control/gpu_fill_secret.cmd (bootstrap.sh --fill-cmd '<command>'; it names paths
#     of that machine, so it is never in this repository, in a status or in a log we upload; "secret" in the file's name keeps it out
#     of snapshot / upload, control.py SECRET_NAME). Without the file, or without GPU_FILL=1 in control.env (setenv;
#     profiles/h200.env), nothing is started and nothing is cleared: the node only measures (scripts/gpu_monitor.py).
#   - Its own loop (`gpu_fill.sh`, no argument; run_tasks.sh's ensure_loop keeps it alive): every FILL_POLL_S (20) s the GPUs that are
#     empty - no task of ours holds them, no process, <= 2 GB - are looked at, and one empty at two looks in a row gets a filler: one
#     instance per GPU, in its own session, CUDA_VISIBLE_DEVICES=<gpu>, the PATH the bootstrap line was pasted in
#     (control/gpu_fill_secret.env). The loop does not depend on run_tasks.sh, on the heartbeat or on the disk: its state is in
#     /dev/shm (memory), so a task that dies of a full disk - or a scheduler that cannot write any more - still gets its cards filled.
#     While setup's own checks run, their card (VERIFY_GPU, 0) is left empty.
#   - A GPU that only runs the filler is FREE for the queue (run_tasks.sh free_gpus -> fill_free): the filler we started, or one that
#     was on the card before us - a process whose command line names the filler's script, whoever started it.
#   - Before a task starts on its GPUs, fill_stop clears them of the filler: TERM to our sessions and to such processes, KILL -9 after
#     FILL_TERM_S (20) s, then it waits until the cards' memory is back under 2 GB (up to FILL_FREE_S 90 s). Only the filler, only
#     those cards: never `fuser -k /dev/nvidia*`, which would end our own tasks on the other cards too.
# Sourced by run_tasks.sh (functions only). As a command: gpu_fill.sh [loop] | status | stop [<gpu>[,<gpu>...]] (all without a list).
FILL_DIR=${FILL_DIR:-$( [ -d /dev/shm ] && [ -w /dev/shm ] && echo /dev/shm/opd_gpu_fill_$(id -u)_$(echo "$OPD_ROOT" | cksum | cut -d' ' -f1) || echo $OPD_ROOT/control/gpu_fill )}
FILL_CMDF=$OPD_ROOT/control/gpu_fill_secret.cmd
fill_enabled() { [ -s $FILL_CMDF ] && [ "$(sed -n 's/^export GPU_FILL=\([01]\)$/\1/p' $OPD_ROOT/control.env 2> /dev/null | tail -1)" = 1 ]; }
fill_sig() { head -1 $FILL_CMDF 2> /dev/null | tr ' ' '\n' | grep -m1 -E '\.py$'; }   # the filler's script: what its processes are known by
fill_pgid() { local p=""; [ -f $FILL_DIR/$1.pid ] && p=$(tr -cd 0-9 < $FILL_DIR/$1.pid); [ -n "$p" ] && kill -0 -- -$p 2> /dev/null && echo $p; }   # our filler's session on GPU $1, if alive
fill_on() { [ -n "$(fill_pgid $1)" ]; }
fill_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $1 2> /dev/null | tr -d ' ' | grep -E '^[0-9]+$'; }
fill_app_pids() {   # the processes on GPU $1 that are the filler (by its script's name on their command line), ours or not
  local s p; s=$(fill_sig); [ -n "$s" ] || return 0
  for p in $(fill_apps $1); do ps -o args= -p $p 2> /dev/null | grep -qF -- "$s" && echo $p; done; }
fill_free() {   # GPU $1 runs nothing but the filler: our session, or every process on it is the filler
  fill_enabled || return 1
  fill_on $1 && return 0
  local a f; a=$(fill_apps $1 | sort -u); [ -n "$a" ] || return 1
  f=$(fill_app_pids $1 | sort -u); [ "$a" = "$f" ]; }
fill_held() {   # GPU $1 is held by a task.sh / ad-hoc evaluation of ours that has not released it (run_tasks.sh gpu_held, without its switch)
  pgrep -f "scripts/(task|run_adhoc_eval)\.sh ([0-9]+,)*$1(,[0-9]+)*( |\$)" > /dev/null || return 1
  pgrep -f "scripts/run_adhoc_eval\.sh $1( |\$)" > /dev/null && return 0
  local t n=0
  for t in $(ps -eo args= | sed -nE "s#^([^ ]*/)?bash [^ ]*scripts/task\.sh ([0-9]+,)*$1(,[0-9]+)* ([A-Za-z0-9._-]+)\$#\4#p" | sort -u); do
    n=$((n + 1)); [ "$(cut -d' ' -f1 $OPD_ROOT/runs/$t/.gpu_released 2> /dev/null)" = "$1" ] || return 0
  done
  [ $n -gt 0 ] || return 0
  return 1; }
fill_empty_gpus() {   # the GPUs with nothing on them and no task of ours holding them
  local i n; n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null | wc -l)
  for i in $(seq 0 $((n - 1))); do
    fill_held $i && continue; fill_on $i && continue
    [ -z "$(fill_apps $i)" ] || continue
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2> /dev/null | tr -d ' ' | head -1)" -le 2048 ] 2> /dev/null && echo $i; done; }
fill_start() {   # $1 = a GPU
  mkdir -p $FILL_DIR; rm -f $FILL_DIR/$1.pid
  local log=$OPD_ROOT/logs/gpu_fill_$1.log; { mkdir -p $OPD_ROOT/logs && : >> $log; } 2> /dev/null || log=/dev/null   # a full disk does not stop the filler
  ( [ -f $OPD_ROOT/control/gpu_fill_secret.env ] && . $OPD_ROOT/control/gpu_fill_secret.env
    export CUDA_VISIBLE_DEVICES=$1 FILL_CMD_LINE="$(head -1 $FILL_CMDF)" FILL_PIDF=$FILL_DIR/$1.pid
    exec setsid bash -c 'echo $$ > $FILL_PIDF; unset FILL_PIDF; eval "$FILL_CMD_LINE"' >> $log 2>&1 8>&- 9>&- < /dev/null ) &
  local i; for i in $(seq 30); do [ -s $FILL_DIR/$1.pid ] && break; sleep 0.1; done   # its session id is on file before anyone asks for it
  echo "FILL_START $(date -u +%T) gpu=$1"; }
fill_stop() {   # $1 = a GPU list a,b,c: clear them of the filler (ours and one that was there before us) and wait for the cards' memory
  local g p i any=0 L; L=$(echo "$1" | tr , ' ')
  [ -d $FILL_DIR ] || [ -s $FILL_CMDF ] || return 0
  for g in $L; do
    p=$(fill_pgid $g); [ -n "$p" ] && { any=1; kill -TERM -- -$p 2> /dev/null; }
    fill_enabled && for p in $(fill_app_pids $g); do any=1; kill -TERM $p 2> /dev/null; done
  done
  [ $any = 1 ] || return 0
  for i in $(seq $(( ${FILL_TERM_S:-20} * 2 ))); do any=0
    for g in $L; do fill_on $g && any=1; fill_enabled && [ -n "$(fill_app_pids $g)" ] && any=1; done
    [ $any = 0 ] && break; sleep 0.5; done
  for g in $L; do
    p=$(fill_pgid $g); [ -n "$p" ] && kill -KILL -- -$p 2> /dev/null
    fill_enabled && for p in $(fill_app_pids $g); do kill -KILL $p 2> /dev/null; done
    rm -f $FILL_DIR/$g.pid $FILL_DIR/$g.empty; done
  command -v nvidia-smi > /dev/null || { echo "FILL_STOP $(date -u +%T) gpu=$1"; return 0; }
  for i in $(seq ${FILL_FREE_S:-90}); do any=0
    for g in $L; do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2> /dev/null | tr -d ' ' | head -1)" -le 2048 ] 2> /dev/null || any=1; done
    [ $any = 0 ] && { echo "FILL_STOP $(date -u +%T) gpu=$1 (cards empty after $i s)"; return 0; }; sleep 1; done
  echo "FILL_STOP $(date -u +%T) gpu=$1: memory not back under 2 GB after ${FILL_FREE_S:-90} s"; return 1; }
fill_round() {   # $1 = the GPUs that are empty right now (space separated): fill those that were empty at the look before too
  local g e=" $1 " sg=""
  if ! fill_enabled; then   # switched off: the fillers we started end; nothing else is touched
    [ -d $FILL_DIR ] && for g in $(ls $FILL_DIR 2> /dev/null | sed -n 's/\.pid$//p'); do
      g=$(fill_pgid $g) && kill -TERM -- -$g 2> /dev/null; done
    rm -f $FILL_DIR/*.pid $FILL_DIR/*.empty 2> /dev/null; return 0; fi
  mkdir -p $FILL_DIR
  pgrep -f "scripts/(cold_start|overnight|verify|fa_probe|fa_build)\.s[h]" > /dev/null 2>&1 && sg=${VERIFY_GPU:-0}   # setup's checks want their card empty
  for g in $(ls $FILL_DIR 2> /dev/null | sed -n 's/\.empty$//p'); do [[ $e == *" $g "* ]] || rm -f $FILL_DIR/$g.empty; done
  [ -n "$sg" ] && fill_free $sg && fill_stop $sg
  for g in $1; do
    [ "$g" = "$sg" ] && continue; fill_on $g && continue   # never a second filler on a card
    if [ -f $FILL_DIR/$g.empty ]; then rm -f $FILL_DIR/$g.empty; fill_start $g; else date +%s > $FILL_DIR/$g.empty; fi
  done; }
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  source "$(dirname "$0")/env.sh"
  [ -f $OPD_ROOT/node.env ] && { set -a; source $OPD_ROOT/node.env; set +a; }
  case ${1:-loop} in
    loop) mkdir -p $FILL_DIR; exec 8> $FILL_DIR/loop.lock
      if command -v flock > /dev/null; then flock -n 8 || { echo "GPU_FILL $(date -u +%T) another loop holds the lock"; exit 0; }; fi
      while true; do   # no file of ours on a disk is needed for a round: the state is in $FILL_DIR, the log line is dropped when it cannot be written
        out=$(fill_round "$(fill_empty_gpus | tr '\n' ' ')" 2>&1 8>&-); [ -n "$out" ] && { echo "$out" >> $OPD_ROOT/logs/gpu_fill.log; } 2> /dev/null
        sleep ${FILL_POLL_S:-20}; done;;
    status) n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null | wc -l)
      for g in $(seq 0 $((n - 1))); do echo "gpu $g: $(fill_on $g && echo "our filler (session $(fill_pgid $g))" || { fill_free $g && echo "a filler that was there before us" || { fill_held $g && echo "a task of ours" || echo "-"; }; })"; done
      fill_enabled && echo "filling: on" || echo "filling: off (GPU_FILL=1 and control/gpu_fill_secret.cmd are both needed)";;
    stop) n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null | wc -l)
      L=${2:-$(seq 0 $((n - 1)) | paste -sd, -)}; [ -n "$L" ] && fill_stop $L || echo "no GPU";;
    *) echo "usage: gpu_fill.sh [loop] | status | stop [<gpu>[,<gpu>...]]"; exit 2;;
  esac
fi
