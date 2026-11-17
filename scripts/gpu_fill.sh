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
#     [v3.3] Setup never clears a card by itself (11-17: the rule "setup's card stays empty" ended the owner's filler, one process
#     on all eight cards, for the whole install - Runming: "do not end the filler before a GPU program of ours runs"). A script
#     that needs an EMPTY card for a few minutes (fa_probe.sh) says so: fill_hold <gpu> <s>, fill_stop, and fill_unhold at its exit.
#   - A GPU that only runs the filler is FREE for the queue (run_tasks.sh free_gpus -> fill_free): the filler we started, or one that
#     was on the card before us - a process whose command line names the filler's script, whoever started it.
#   - Before a task starts on its GPUs, fill_stop clears them of the filler: TERM to our sessions and to such processes, KILL -9 after
#     FILL_TERM_S (20) s, then it waits until the cards' memory is back under 2 GB (up to FILL_FREE_S 90 s). Only the filler, only
#     those cards: never `fuser -k /dev/nvidia*`, which would end our own tasks on the other cards too.
# Sourced by run_tasks.sh (functions only). As a command: gpu_fill.sh [loop] | status | stop [<gpu>[,<gpu>...]] (all without a list).
FILL_DIR=${FILL_DIR:-$( [ -d /dev/shm ] && [ -w /dev/shm ] && echo /dev/shm/opd_gpu_fill_$(id -u)_$(echo "$OPD_ROOT" | cksum | cut -d' ' -f1) || echo $OPD_ROOT/control/gpu_fill )}
FILL_CMDF=$OPD_ROOT/control/gpu_fill_secret.cmd
# [v3.4] (11-17: the owner's filler runs under an interpreter of his own conda environment; the stored line said `python`, which is
# something else in our loops, so the fillers we started ended at once) the filler's exact command is LEARNED from a filler process
# that is on a card - its working directory and its argv, word for word - into control/gpu_fill_secret.seen (node-local, 0600), and
# that is what fill_start runs; the stored line is only the fallback and the name fillers are known by
FILL_SEEN=$OPD_ROOT/control/gpu_fill_secret.seen
fill_learn() {   # $1 = a filler's pid: its cwd and argv become the command fill_start runs
  local p=$1 cwd="" args=""
  local -a A=(); local e="" envs="" k v
  if [ -r /proc/$p/cmdline ]; then cwd=$(readlink /proc/$p/cwd 2> /dev/null)
    while IFS= read -r -d '' k; do A+=("$k"); done < /proc/$p/cmdline 2> /dev/null
    # [v3.6] the interpreter by its real path (argv[0] `python` is another program in our loops), and the environment variables that
    # choose its libraries, as the filler itself has them
    [ ${#A[@]} -gt 0 ] && [[ ${A[0]} != /* ]] && e=$(readlink /proc/$p/exe 2> /dev/null) && [ -x "$e" ] && A[0]=$e
    [ ${#A[@]} -gt 0 ] && args=$(printf '%q ' "${A[@]}")
    if [ -r /proc/$p/environ ]; then while IFS= read -r -d '' k; do v=${k#*=}; k=${k%%=*}
        case $k in PATH|LD_LIBRARY_PATH|PYTHONPATH|CONDA_PREFIX|CONDA_DEFAULT_ENV|VIRTUAL_ENV) envs="$envs$(printf 'export %s=%q; ' "$k" "$v")";; esac
      done < /proc/$p/environ 2> /dev/null; fi
  else args=$(ps -o args= -p $p 2> /dev/null); fi   # no /proc (macOS, the tests): the words as ps gives them
  [ -n "$args" ] || return 1
  ( umask 077; { printf '%s' "$envs"; [ -n "$cwd" ] && printf 'cd %q 2> /dev/null; ' "$cwd"; printf '%s\n' "$args"; } > $FILL_SEEN.tmp && mv $FILL_SEEN.tmp $FILL_SEEN ) 2> /dev/null; }
fill_seen_ok() { local s; s=$(fill_sig); [ -s $FILL_SEEN ] && [ -n "$s" ] && grep -qF -- "$s" $FILL_SEEN; }   # learned, and from the filler named now
fill_cmd_line() { local s; s=$(fill_sig)
  if fill_seen_ok; then head -1 $FILL_SEEN
  elif [ -s $FILL_CMDF ] && { [ -z "$(fill_name)" ] || grep -qF -- "$s" $FILL_CMDF; }; then head -1 $FILL_CMDF; fi; }   # else nothing: no command known yet
# [v3.4] `setenv GPU_FILL_NAME=<script file name>` (no path: control files are public) names the filler when the owner changes it
# (11-17: train.py -> a new script): its processes are then known by "/<name>" on their command line, and the command is learned
# from one of them; the stored line is not needed any more
fill_name() { sed -n 's/^export GPU_FILL_NAME=\([A-Za-z0-9_.-]*\)$/\1/p' $OPD_ROOT/control.env 2> /dev/null | tail -1; }
fill_enabled() { { [ -s $FILL_CMDF ] || [ -n "$(fill_name)" ]; } && [ "$(sed -n 's/^export GPU_FILL=\([01]\)$/\1/p' $OPD_ROOT/control.env 2> /dev/null | tail -1)" = 1 ]; }
fill_sig() { local n; n=$(fill_name); if [ -n "$n" ]; then echo "$n"; else head -1 $FILL_CMDF 2> /dev/null | tr ' ' '\n' | grep -m1 -E '\.py$'; fi; }   # what the filler's processes are known by
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
fill_hold() { local g; mkdir -p $FILL_DIR; for g in $(echo "$1" | tr , ' '); do echo $(( $(date +%s) + ${2:-1800} )) > $FILL_DIR/$g.hold; done; }   # keep the loop off these cards for $2 s
fill_unhold() { local g; for g in $(echo "$1" | tr , ' '); do rm -f $FILL_DIR/$g.hold; done; }
fill_holding() { local u=""; [ -f $FILL_DIR/$1.hold ] && u=$(tr -cd 0-9 < $FILL_DIR/$1.hold); [ -n "$u" ] && [ "$u" -gt "$(date +%s)" ]; }
fill_empty_gpus() {   # the GPUs with nothing on them and no task of ours holding them
  local i n; n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null | wc -l)
  for i in $(seq 0 $((n - 1))); do
    fill_held $i && continue; fill_on $i && continue; fill_holding $i && continue
    [ -z "$(fill_apps $i)" ] || continue
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i 2> /dev/null | tr -d ' ' | head -1)" -le 2048 ] 2> /dev/null && echo $i; done; }
# [v3.6] (11-17, a second borrowed machine: the owner's filler is ONE process on all eight cards, started with the container) the
# owner's filler is ended only once a RESTART of it has been proven on this node: the loop runs the known command once, beside
# whatever is on a card, for FILL_SELFTEST_S (30) s - alive at the end = ok (control/gpu_fill_selftest: "ok|fail <command id> <utc>").
# Until then fill_stop ends only the fillers we started and the task shares its card with the owner's filler (FILL_KEEP).
FILL_TEST=$OPD_ROOT/control/gpu_fill_selftest
fill_cmd_id() { fill_cmd_line | cksum | cut -d' ' -f1; }
fill_selftest_ok() { [ -s $FILL_TEST ] && [ "$(cut -d' ' -f1,2 $FILL_TEST 2> /dev/null)" = "ok $(fill_cmd_id)" ]; }
fill_selftest_due() {   # a command is known and it has no result yet (or a failure older than FILL_SELFTEST_RETRY_S 1800 s)
  [ -n "$(fill_cmd_line)" ] || return 1
  [ "$(cut -d' ' -f2 $FILL_TEST 2> /dev/null)" = "$(fill_cmd_id)" ] || return 0
  [ "$(cut -d' ' -f1 $FILL_TEST)" = fail ] && [ $(( $(date +%s) - $(stat -c %Y $FILL_TEST 2> /dev/null || stat -f %m $FILL_TEST) )) -ge ${FILL_SELFTEST_RETRY_S:-1800} ]; }
fill_selftest() {   # $1 = a GPU
  local rc id log=$OPD_ROOT/logs/gpu_fill_selftest.log n=${FILL_SELFTEST_S:-30}; id=$(fill_cmd_id); [ -n "$(fill_cmd_line)" ] || return 1
  { mkdir -p $OPD_ROOT/logs && : > $log; } 2> /dev/null || log=/dev/null
  ( [ -f $OPD_ROOT/control/gpu_fill_secret.env ] && . $OPD_ROOT/control/gpu_fill_secret.env
    export CUDA_VISIBLE_DEVICES=$1 FILL_CMD_LINE="$(fill_cmd_line)"
    exec timeout -k 5 $n bash -c 'eval "$FILL_CMD_LINE"' > $log 2>&1 8>&- 9>&- < /dev/null ); rc=$?
  if [ $rc = 124 ] || [ $rc = 137 ]; then ( umask 077; echo "ok $id $(date -u +%FT%TZ)" > $FILL_TEST ) 2> /dev/null
    echo "FILL_SELFTEST $(date -u +%T) gpu=$1 ok: the filler's command was alive after $n s"
  else ( umask 077; echo "fail $id $(date -u +%FT%TZ) rc=$rc" > $FILL_TEST ) 2> /dev/null
    echo "FILL_SELFTEST $(date -u +%T) gpu=$1 FAILED (exit $rc before $n s): the owner's filler will not be ended; see logs/gpu_fill_selftest.log"; fi; }
fill_kill_pids() {   # the filler processes on GPU $1 that fill_stop may end: all of them once a restart is proven, else only ours
  local p o; o=" $(cat $FILL_DIR/*.pid 2> /dev/null | tr '\n' ' ') "
  for p in $(fill_app_pids $1); do
    if fill_selftest_ok || [[ $o == *" $(ps -o pgid= -p $p 2> /dev/null | tr -d ' ') "* ]]; then echo $p; fi; done; }
fill_start() {   # $1 = a GPU
  [ -n "$(fill_cmd_line)" ] || { echo "FILL_NO_COMMAND $(date -u +%T) gpu=$1: the filler's command is not known yet (none learned, none stored for this name)"; return 1; }
  mkdir -p $FILL_DIR; rm -f $FILL_DIR/$1.pid
  local log=$OPD_ROOT/logs/gpu_fill_$1.log; { mkdir -p $OPD_ROOT/logs && : >> $log; } 2> /dev/null || log=/dev/null   # a full disk does not stop the filler
  ( [ -f $OPD_ROOT/control/gpu_fill_secret.env ] && . $OPD_ROOT/control/gpu_fill_secret.env
    export CUDA_VISIBLE_DEVICES=$1 FILL_CMD_LINE="$(fill_cmd_line)" FILL_PIDF=$FILL_DIR/$1.pid
    exec setsid bash -c 'echo $$ > $FILL_PIDF; unset FILL_PIDF; eval "$FILL_CMD_LINE"' >> $log 2>&1 8>&- 9>&- < /dev/null ) &
  local i; for i in $(seq 30); do [ -s $FILL_DIR/$1.pid ] && break; sleep 0.1; done   # its session id is on file before anyone asks for it
  echo "FILL_START $(date -u +%T) gpu=$1"; }
fill_stop() {   # $1 = a GPU list a,b,c: clear them of the filler (ours and one that was there before us) and wait for the cards' memory
  local g p i any=0 L; L=$(echo "$1" | tr , ' ')
  [ -d $FILL_DIR ] || [ -s $FILL_CMDF ] || [ -n "$(fill_name)" ] || return 0
  local theirs=0
  if fill_enabled && ! fill_seen_ok; then   # [v3.6] learn the command before deciding: the proof below is about the command a restart runs
    for g in $L; do p=$(fill_app_pids $g | head -1); [ -n "$p" ] && { fill_learn $p; break; }; done; fi
  for g in $L; do
    p=$(fill_pgid $g); [ -n "$p" ] && { any=1; kill -TERM -- -$p 2> /dev/null; }
    if fill_enabled; then
      fill_selftest_ok || [ -z "$(fill_app_pids $g)" ] || [ -n "$p" ] || echo "FILL_KEEP $(date -u +%T) gpu=$g: the owner's filler stays on the card (no proven restart yet: $(cut -d' ' -f1,4- $FILL_TEST 2> /dev/null || echo not tested)); the task shares it"
      for p in $(fill_kill_pids $g); do any=1; fill_on $g || theirs=1; kill -TERM $p 2> /dev/null; done; fi
  done
  [ $any = 1 ] || return 0
  for i in $(seq $(( ${FILL_TERM_S:-20} * 2 ))); do any=0
    for g in $L; do fill_on $g && any=1; fill_enabled && [ -n "$(fill_kill_pids $g)" ] && any=1; done
    [ $any = 0 ] && break; sleep 0.5; done
  for g in $L; do
    p=$(fill_pgid $g); [ -n "$p" ] && kill -KILL -- -$p 2> /dev/null
    fill_enabled && for p in $(fill_kill_pids $g); do kill -KILL $p 2> /dev/null; done
    rm -f $FILL_DIR/$g.pid $FILL_DIR/$g.empty; done
  # [v3.6] a filler that was one process on many cards is gone from all of them: the cards this call did not ask for get their own
  # filler at once (not after the loop's two looks)
  if [ $theirs = 1 ] && fill_enabled; then sleep 1; for g in $(fill_empty_gpus); do [[ " $L " == *" $g "* ]] || fill_holding $g || fill_start $g; done; fi
  command -v nvidia-smi > /dev/null || { echo "FILL_STOP $(date -u +%T) gpu=$1"; return 0; }
  for i in $(seq ${FILL_FREE_S:-90}); do any=0
    for g in $L; do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2> /dev/null | tr -d ' ' | head -1)" -le 2048 ] 2> /dev/null || any=1; done
    [ $any = 0 ] && { echo "FILL_STOP $(date -u +%T) gpu=$1 (cards empty after $i s)"; return 0; }; sleep 1; done
  echo "FILL_STOP $(date -u +%T) gpu=$1: memory not back under 2 GB after ${FILL_FREE_S:-90} s"; return 1; }
fill_round() {   # $1 = the GPUs that are empty right now (space separated): fill those that were empty at the look before too
  local g e=" $1 "
  if ! fill_enabled; then   # switched off: the fillers we started end; nothing else is touched
    [ -d $FILL_DIR ] && for g in $(ls $FILL_DIR 2> /dev/null | sed -n 's/\.pid$//p'); do
      g=$(fill_pgid $g) && kill -TERM -- -$g 2> /dev/null; done
    rm -f $FILL_DIR/*.pid $FILL_DIR/*.empty 2> /dev/null; return 0; fi
  mkdir -p $FILL_DIR
  if ! fill_seen_ok; then   # [v3.4] a filler is on some card: learn its exact command while it is there
    for g in $(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null); do g=$(fill_app_pids $g | head -1); [ -n "$g" ] && { fill_learn $g; break; }; done; fi
  if fill_selftest_due; then   # [v3.6] prove once that the command can be started here, on a card no task of ours holds
    for g in $(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null); do fill_held $g || fill_holding $g || { fill_selftest $g; break; }; done; fi
  for g in $(ls $FILL_DIR 2> /dev/null | sed -n 's/\.empty$//p'); do [[ $e == *" $g "* ]] || rm -f $FILL_DIR/$g.empty; done
  for g in $1; do
    fill_on $g && continue; fill_holding $g && continue   # never a second filler on a card; not on a card a check asked to keep empty
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
      fill_enabled && echo "filling: on (command $(fill_seen_ok && echo learned || { [ -n "$(fill_cmd_line)" ] && echo stored || echo "not known yet"; }))" || echo "filling: off (GPU_FILL=1 and a stored command or GPU_FILL_NAME are needed)";;
    stop) n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null | wc -l)
      L=${2:-$(seq 0 $((n - 1)) | paste -sd, -)}; [ -n "$L" ] && fill_stop $L || echo "no GPU";;
    *) echo "usage: gpu_fill.sh [loop] | status | stop [<gpu>[,<gpu>...]]"; exit 2;;
  esac
fi
