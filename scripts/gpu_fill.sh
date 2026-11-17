#!/usr/bin/env bash
# [v3.1] gpu_fill.sh (Fable 11-17; Runming: the machine's rule is "GPU utilisation >= 85 %, about 90 %, else the job is ended after two
# hours", and its owner's instruction "when your program fails or ends, start <the owner's filler command>"): an idle GPU of this node
# runs THE OWNER'S OWN filler, never a load of ours.
#   - The command is node-local: one line in $OPD_ROOT/control/gpu_fill_secret.cmd (bootstrap.sh --fill-cmd '<command>'; it names paths of that
#     machine, so it is never in this repository, in a status or in a log we upload; "secret" in the file's name keeps it out of
#     snapshot / upload, control.py SECRET_NAME). Without the file, or without GPU_FILL=1 in
#     control.env (setenv; profiles/h200.env), nothing is started: the node only measures (scripts/gpu_monitor.py).
#   - One instance per idle GPU, in its own session, with CUDA_VISIBLE_DEVICES=<gpu> and the PATH the bootstrap line was pasted in
#     (control/gpu_fill_secret.env), output in logs/gpu_fill_<gpu>.log (local only).
#   - run_tasks.sh calls fill_round at the end of every round: a GPU that was empty at two rounds in a row (no task holds it, no
#     process, <= 2 GB) gets a filler; while setup's own checks run, their card (VERIFY_GPU, 0) is left empty.
#   - A GPU that only runs a filler is FREE for the queue (free_gpus); before a task starts on it, fill_stop ends the filler (TERM
#     to its session, KILL after FILL_TERM_S 20 s) and waits until the card's memory is back under 2 GB (up to FILL_FREE_S 90 s).
# Sourced by run_tasks.sh (functions only). As a command: gpu_fill.sh status | stop [<gpu>[,<gpu>...]] (all without a list).
FILL_DIR=$OPD_ROOT/control/gpu_fill
fill_enabled() { [ -s $OPD_ROOT/control/gpu_fill_secret.cmd ] && [ "$(sed -n 's/^export GPU_FILL=\([01]\)$/\1/p' $OPD_ROOT/control.env 2> /dev/null | tail -1)" = 1 ]; }
fill_pgid() { local p=""; [ -f $FILL_DIR/$1.pid ] && p=$(tr -cd 0-9 < $FILL_DIR/$1.pid); [ -n "$p" ] && kill -0 -- -$p 2> /dev/null && echo $p; }   # the filler's session on GPU $1, if alive
fill_on() { [ -n "$(fill_pgid $1)" ]; }
fill_start() {   # $1 = a GPU
  mkdir -p $FILL_DIR $OPD_ROOT/logs; rm -f $FILL_DIR/$1.pid
  ( [ -f $OPD_ROOT/control/gpu_fill_secret.env ] && . $OPD_ROOT/control/gpu_fill_secret.env
    export CUDA_VISIBLE_DEVICES=$1 FILL_CMD_LINE="$(head -1 $OPD_ROOT/control/gpu_fill_secret.cmd)" FILL_PIDF=$FILL_DIR/$1.pid
    exec setsid bash -c 'echo $$ > $FILL_PIDF; unset FILL_PIDF; eval "$FILL_CMD_LINE"' >> $OPD_ROOT/logs/gpu_fill_$1.log 2>&1 9>&- < /dev/null ) &
  local i; for i in $(seq 30); do [ -s $FILL_DIR/$1.pid ] && break; sleep 0.1; done   # its session id is on file before anyone asks for it
  echo "FILL_START $(date -u +%T) gpu=$1"; }
fill_stop() {   # $1 = a GPU list a,b,c: end their fillers and wait for the cards' memory
  local g p i any=0
  for g in $(echo "$1" | tr , ' '); do p=$(fill_pgid $g); [ -n "$p" ] || continue; any=1; kill -TERM -- -$p 2> /dev/null; done
  [ $any = 1 ] || return 0
  for i in $(seq $(( ${FILL_TERM_S:-20} * 2 ))); do any=0; for g in $(echo "$1" | tr , ' '); do fill_on $g && any=1; done; [ $any = 0 ] && break; sleep 0.5; done
  for g in $(echo "$1" | tr , ' '); do p=$(fill_pgid $g); [ -n "$p" ] && kill -KILL -- -$p 2> /dev/null; rm -f $FILL_DIR/$g.pid $FILL_DIR/$g.empty; done
  command -v nvidia-smi > /dev/null || { echo "FILL_STOP $(date -u +%T) gpu=$1"; return 0; }
  for i in $(seq ${FILL_FREE_S:-90}); do any=0
    for g in $(echo "$1" | tr , ' '); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2> /dev/null | tr -d ' ' | head -1)" -le 2048 ] 2> /dev/null || any=1; done
    [ $any = 0 ] && { echo "FILL_STOP $(date -u +%T) gpu=$1 (cards empty after $i s)"; return 0; }; sleep 1; done
  echo "FILL_STOP $(date -u +%T) gpu=$1: memory not back under 2 GB after ${FILL_FREE_S:-90} s"; return 1; }
fill_round() {   # $1 = the GPUs that are empty right now (space separated; run_tasks.sh empty_gpus): fill those empty for two rounds
  local g e=" $1 " sg=""
  fill_enabled || { [ -d $FILL_DIR ] && for g in $(ls $FILL_DIR 2> /dev/null | sed -n 's/\.pid$//p'); do fill_stop $g; done; return 0; }   # switched off: the fillers end
  mkdir -p $FILL_DIR
  pgrep -f "scripts/(cold_start|overnight|verify|fa_probe|fa_build)\.s[h]" > /dev/null 2>&1 && sg=${VERIFY_GPU:-0}   # setup's checks want their card empty
  for g in $(ls $FILL_DIR 2> /dev/null | sed -n 's/\.empty$//p'); do [[ $e == *" $g "* ]] || rm -f $FILL_DIR/$g.empty; done
  [ -n "$sg" ] && fill_on $sg && fill_stop $sg
  for g in $1; do
    [ "$g" = "$sg" ] && continue
    if [ -f $FILL_DIR/$g.empty ]; then rm -f $FILL_DIR/$g.empty; fill_start $g; else date +%s > $FILL_DIR/$g.empty; fi
  done; }
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  source "$(dirname "$0")/env.sh"; FILL_DIR=$OPD_ROOT/control/gpu_fill
  case ${1:-status} in
    status) for f in $FILL_DIR/*.pid; do [ -f "$f" ] || continue; g=$(basename $f .pid); echo "gpu $g: $(fill_on $g && echo "filler session $(fill_pgid $g)" || echo "no live filler")"; done
            fill_enabled && echo "filling: on" || echo "filling: off (GPU_FILL=1 and control/gpu_fill_secret.cmd are both needed)";;
    stop) L=${2:-$(ls $FILL_DIR 2> /dev/null | sed -n 's/\.pid$//p' | paste -sd, -)}; [ -n "$L" ] && fill_stop $L || echo "no filler";;
    *) echo "usage: gpu_fill.sh status | stop [<gpu>[,<gpu>...]]"; exit 2;;
  esac
fi
