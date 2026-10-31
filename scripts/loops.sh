#!/usr/bin/env bash
# [v2.5.9z106] loops.sh start|stop|restart|ensure|status [heartbeat|run_tasks|both]: the one place where the two main loops
# (heartbeat.sh, run_tasks.sh) are started and stopped (connect-first bootstrap, Fable 11-01; ops-lead's A1 16:11 UTC C2 / C3 / C6).
# - One lock (control/loops.lock, flock; `ensure` only tries it and skips the round when it is held).
# - A loop is ours when its command line runs THIS runbook's scripts/<loop>.sh (absolute, or relative with this runbook as its
#   working directory); of the matching processes only the roots count ($(...) subshells carry the same command line).
# - start / ensure: no root -> start one (setsid, as watchdog.sh and overnight.sh did); one root -> adopted (control/loops_<loop>.pid,
#   one event loops_adopted), never a second one (C2: the H800's loops, started by h800_loops.sh, are taken over this way);
#   two or more -> refused (exit 3, event loops_duplicate); `restart` is the way out.
# - stop / restart: the loop's tree as watchdog.sh's tree_of sees it (its session; task sessions, prefetches, upload_sync and the
#   other loops are never in it, C6): TERM, KILL after LOOPS_TERM_WAIT (30) s.
# - ensure heartbeat (run_tasks, every round): as start, and a heartbeat whose script changed since it started (control/heartbeat.sha)
#   is restarted (was run_tasks.sh's own pgrep / kill / setsid).
# - The loops run with $OPD_ROOT/node.env when it exists (bootstrap writes it; the first adoption writes it from the adopted
#   heartbeat's environment, selected keys only, the token by reference to its file when one matches).
# - status: one line, and control/loops.json for the heartbeat's status.
# Exit 0 ok; 2 usage; 3 duplicate loops; 4 a loop still there after KILL / did not start; 5 the lock is held (ensure: 0, skipped).
set -uo pipefail
source "$(dirname "$0")/env.sh"
CDIR=$OPD_ROOT/control; mkdir -p $CDIR $OPD_ROOT/logs
[ -f $OPD_ROOT/node.env ] && source $OPD_ROOT/node.env
VERB=${1:-status}; WHICH=${2:-both}
case $VERB in start|stop|restart|ensure|status) ;; *) echo "LOOPS usage: loops.sh start|stop|restart|ensure|status [heartbeat|run_tasks|both]"; exit 2;; esac
case $WHICH in both) LS="heartbeat run_tasks";; heartbeat|run_tasks) LS=$WHICH;; *) echo "LOOPS usage: loop heartbeat, run_tasks or both"; exit 2;; esac
RB_REAL=$(cd $RUNBOOK 2> /dev/null && pwd -P)   # macOS: /var is /private/var
ev() { if [ "${LOOPS_SYNC_EV:-0}" = 1 ]; then python3 $RUNBOOK/scripts/control.py event node "$1" "$2" > /dev/null 2>&1   # the tests
  else command -v timeout > /dev/null && (timeout -k 10 60 python3 $RUNBOOK/scripts/control.py event node "$1" "$2" > /dev/null 2>&1 &); fi; }
say() { echo "LOOPS $(date -u +%T) $*"; }

ours() {   # $1 = pid, $2 = loop: does it run THIS runbook's scripts/<loop>.sh?
  local args cwd
  args=$(ps -o args= -p $1 2> /dev/null) || return 1
  case "$args" in
    *" $RUNBOOK/scripts/$2.sh"|*" $RUNBOOK/scripts/$2.sh "*) return 0;;
    *" scripts/$2.sh"|*" scripts/$2.sh "*)
      cwd=$(readlink /proc/$1/cwd 2> /dev/null || lsof -a -p $1 -d cwd -Fn 2> /dev/null | sed -n 's/^n//p')   # no /proc: macOS, the tests
      [ -n "$cwd" ] || cwd=${LOOPS_TEST_CWD:-}
      [ "$cwd" = "$RUNBOOK" ] || [ "$cwd" = "$RB_REAL" ];;
    *) return 1;;
  esac; }
ours_all() { local p; pgrep -f "scripts/$1[.]sh" 2> /dev/null | while read -r p; do ours $p $1 && echo $p; done; }
roots_of() {   # the loops of kind $1 that are ours, one pid per line: one per session (its leader when that is ours, else its lowest pid;
  # a loop's own $(...) and ( ... ) & subshells share its session, some re-parented to init); without session ids (macOS, the tests)
  # the processes whose parent is not one of them
  local all p pp
  all=$(ours_all $1); [ -n "$all" ] || return 0
  if ps -o sid= -p $$ > /dev/null 2>&1; then
    for p in $all; do echo "$(ps -o sid= -p $p 2> /dev/null | tr -d ' ') $p"; done |
      awk 'NF == 2 {if ($1 == $2) lead[$1] = $2; if (!($1 in low) || $2 + 0 < low[$1] + 0) low[$1] = $2} END {for (s in low) print ((s in lead) ? lead[s] : low[s])}'
  else
    for p in $all; do pp=$(ps -o ppid= -p $p 2> /dev/null | tr -d ' '); grep -qx "$pp" <<< "$all" || echo $p; done
  fi; }
TREE_SKIP='scripts/(task|run_adhoc_eval|upload_sync|run_queue_eval|heartbeat|watchdog|run_tasks|uploader|dr|dr_sync|loops)[.]sh|fetch_model[.]py'
tree_of() {   # as watchdog.sh: $1 and its descendants in its own session, minus TREE_SKIP and what is below it
  local hassid=1; ps -o sid= -p $$ > /dev/null 2>&1 || hassid=0
  { if [ $hassid = 1 ]; then ps -A -o pid=,ppid=,sid=,args=; else ps -A -o pid=,ppid=,args= | awk '{$2 = $2 " -"; print}'; fi; } |
  awk -v r=$1 -v skip="$TREE_SKIP" '{pid = $1; par[pid] = $2; s[pid] = $3; $1 = $2 = $3 = ""; a[pid] = $0}
    END {inn[r] = 1; ch = 1
         while (ch) { ch = 0; for (p in par) if (!(p in inn) && (par[p] in inn) && s[p] == s[r] && a[p] !~ skip) { inn[p] = 1; ch = 1 } }
         for (p in inn) print p}'; }
alive_of() { local p; for p in "$@"; do kill -0 $p 2> /dev/null && echo $p; done; }
stop_one() {   # TERM the trees of every process of loop $1, KILL what is left after LOOPS_TERM_WAIT (30) s; 0 when none is left.
  # The set only grows: what the first snapshot saw (an upload in its own process group that ignores TERM stays in it after its
  # loop died) and, each second, any process of the loop seen since (a $(...) subshell forked after the snapshot)
  local p members="" left t=0
  while :; do
    for p in $(ours_all $1); do members="$members $(tree_of $p | tr '\n' ' ')"; done
    left=$(alive_of $members | sort -u | tr '\n' ' ')
    [ -z "${left// /}" ] && break
    [ $t = 0 ] && say "stop $1: $left"
    if [ $t -ge ${LOOPS_TERM_WAIT:-30} ]; then say "KILL $1: $left"; kill -KILL $left 2> /dev/null; sleep 2; break; fi
    kill -TERM $left 2> /dev/null; sleep 1; t=$((t + 1))
  done
  rm -f $CDIR/loops_$1.pid
  [ -z "$(alive_of $members)" ] && [ -z "$(roots_of $1)" ]; }
start_one() {   # no root -> start; one -> adopt; more -> refuse
  local r n
  r=$(roots_of $1); n=$(echo "$r" | grep -c .)
  if [ $n -gt 1 ]; then say "DUPLICATE $1: $(echo $r)"; ev loops_duplicate "$1: $n roots"; return 3; fi
  if [ $n = 1 ]; then
    if [ "$(cat $CDIR/loops_$1.pid 2> /dev/null)" != "$r" ]; then echo $r > $CDIR/loops_$1.pid; say "adopted $1 $r"; ev loops_adopted "$1"; fi
    return 0; fi
  [ "${LOOPS_DRY:-0}" = 1 ] && { say "would start $1"; return 0; }
  date -u +%s > $CDIR/loop_$1   # the watchdog gives the new loop a full window
  # the subshell exits at once and holds none of our descriptors (a caller reading our output, $(...) or control.py, must not
  # wait for the loop); the loop runs from this runbook, relative, as h800_loops.sh / overnight.sh started it
  ( cd $RUNBOOK || exit 1; setsid nohup bash scripts/$1.sh >> $OPD_ROOT/logs/$1.log 2>&1 9>&- 8>&- < /dev/null & ) > /dev/null 2>&1 < /dev/null
  for _ in $(seq 20); do sleep 0.5; r=$(roots_of $1); [ -n "$r" ] && break; done
  [ -n "$r" ] || { say "$1 did not start"; ev loops_start_failed "$1"; return 4; }
  echo "$r" | head -1 > $CDIR/loops_$1.pid; say "started $1 $(echo $r)"; ev loops_started "$1"; return 0; }
node_env_from() {   # [C2] the first adoption: node.env from the adopted heartbeat's environment (selected keys; 0600)
  [ -f $OPD_ROOT/node.env ] || [ ! -r /proc/$1/environ ] && return 0
  python3 - "$1" "$OPD_ROOT" <<'PY'
import os, shlex, sys
pid, root = sys.argv[1:3]
env = dict(l.split("=", 1) for l in open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace").split("\0") if "=" in l)
keys = ("OPD_ROOT RUNBOOK HF_REPO OPD_NODE HF_HOME XDG_CACHE_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR PIP_CACHE_DIR TRITON_CACHE_DIR "
        "OVERNIGHT_TIER_A_ONLY FA_CUDA_ARCHS PYTHONDONTWRITEBYTECODE VLLM_NO_USAGE_STATS VERIFY_GPU "
        "https_proxy HTTPS_PROXY http_proxy HTTP_PROXY no_proxy NO_PROXY").split()
out = ["# [v2.5.9z106] written by loops.sh at the first adoption, from the adopted heartbeat's environment (selected keys)"]
out += [f"export {k}={shlex.quote(env[k])}" for k in keys if k in env]
tok = env.get("HF_TOKEN", "")
if tok:   # by reference: a token file whose first line is this token; else this node's own 0600 file
    cands = [os.path.join(root, ".hf_token"), "/root/.hf_token_opd", os.path.expanduser("~/.hf_token_opd")]
    ref = next((c for c in cands if os.path.isfile(c) and open(c).readline().strip() == tok), None)
    if ref is None:
        ref = os.path.join(root, ".hf_token")
        fd = os.open(ref, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.write(fd, (tok + "\n").encode()); os.close(fd)
    out.append(f'[ -s {shlex.quote(ref)} ] && export HF_TOKEN="$(head -1 {shlex.quote(ref)} | tr -d \'[:space:]\')"')
fd = os.open(os.path.join(root, "node.env"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(fd, ("\n".join(out) + "\n").encode()); os.close(fd)
print(f"LOOPS node.env written ({len(out) - 1} lines, values not shown)")
PY
}
write_json() {
  local h r; h=$(roots_of heartbeat | tr '\n' ' '); r=$(roots_of run_tasks | tr '\n' ' ')
  printf '{"utc": "%s", "heartbeat": [%s], "run_tasks": [%s], "frozen": %s}\n' "$(date -u +%FT%TZ)" \
    "$(echo $h | tr ' ' ',')" "$(echo $r | tr ' ' ',')" "$([ -f $CDIR/dispatch_frozen ] && echo true || echo false)" > $CDIR/loops.json.tmp \
    && mv -f $CDIR/loops.json.tmp $CDIR/loops.json; }

if [ $VERB = status ]; then
  for L in heartbeat run_tasks; do r=$(roots_of $L); echo "LOOPS $L $(echo "$r" | grep -c .) ${r:+($(echo $r))}"; done
  write_json; exit 0
fi
if command -v flock > /dev/null; then
  exec 8> $CDIR/loops.lock
  if [ $VERB = ensure ]; then flock -n 8 || exit 0; else flock -w ${LOOPS_LOCK_WAIT:-120} 8 || { say "lock held"; exit 5; }; fi
else   # no flock (macOS, the tests): a lock directory, removed on exit
  got=0; for _ in $(seq $([ $VERB = ensure ] && echo 1 || echo ${LOOPS_LOCK_WAIT:-120})); do mkdir $CDIR/loops.lockdir 2> /dev/null && { got=1; break; }; sleep 1; done
  [ $got = 1 ] || { [ $VERB = ensure ] && exit 0; say "lock held"; exit 5; }
  trap 'rmdir $CDIR/loops.lockdir 2> /dev/null' EXIT
fi
rc=0
for L in $LS; do
  case $VERB in
    start) start_one $L || rc=$?;;
    stop) stop_one $L || { say "$L still running after KILL"; ev loops_stuck "$L"; rc=4; };;
    restart) if stop_one $L; then start_one $L || rc=$?; ev loops_restarted "$L"; else say "$L still running after KILL"; ev loops_stuck "$L"; rc=4; fi;;
    ensure)
      if [ $L = heartbeat ] && [ -n "$(roots_of heartbeat)" ] && [ "$(cat $CDIR/heartbeat.sha 2> /dev/null)" != "$(sha1sum $RUNBOOK/scripts/heartbeat.sh | cut -c1-12)" ]; then
        stop_one heartbeat && { start_one heartbeat || rc=$?; ev heartbeat_upgraded "$(sha1sum $RUNBOOK/scripts/heartbeat.sh | cut -c1-12)"; }
      else start_one $L || rc=$?; fi;;
  esac
done
r=$(roots_of heartbeat | head -1); [ -n "$r" ] && node_env_from $r
write_json; exit $rc
