#!/usr/bin/env bash
# [v2.5.9z106] bootstrap.sh: connect first, everything else remote (Runming 10-31 23:5x Beijing via Fable; ops-lead's A1 11-01 16:11 UTC,
# C1-C8). The one thing a person at a new machine runs. Needs bash, git, curl, python3. About 2-5 minutes, then
#   node <id> online; everything else is remote
# and from then on the project installs, checks and repairs this machine through control commands (env-probe, setup, verify,
# tail, restart-loops, dispatch, setenv, ...). What it does, in order:
#   B2 a base directory: --base, else the largest writable local disk; T=<base>/opd (runbook, run, ckpt, cache below it)
#   B0 the Hugging Face token: --token-file <0600 file>, or --hf-home <dir> (its token file, read in place, never copied), else typed
#      once with read -rs (not echoed, not in the shell history) into $T/run/.hf_token (0600)
#   B0b [z126] the GitHub heartbeat token (Fable 11-02: a second way out): --gh-token-file <0600 file>, else typed once with read -rs
#      (Enter skips; --no-gh does not ask), stored for git alone as ~/.config/opd/gh_hb_credentials (0600, git's credential store;
#      GH_HB_CRED moves it); after B1 one `git ls-remote` of GH_HB_URL (default the private heartbeat repository) says whether it is
#      reachable, and B6 waits up to 90 s for gh_heartbeat's first push of hb/<node>.json. No token is printed or uploaded.
#   B1 the network: direct when GitHub and Hugging Face answer 200, else the proxy candidates found in the environment, the profile
#      files and other processes' environments (proxy variables only), each tested; $T/run/proxy.env (0600) with no_proxy for this
#      host. No proxy value is printed or uploaded.
#   B3 the runbook cloned (or pulled) into $T/runbook; the node id (as cold_start.sh)
#   B4 $T/run/venv-ctl: requirements.ctl.txt only (huggingface_hub, pyyaml; no torch)
#   B5 cold_start.sh --step 1 (no old command may ever run here), then node.env, control/coldstart_args, control/bootstrap.json,
#      heartbeat every 120 s, STATUS_PUT_S=120, dispatch frozen (control/dispatch_frozen). [z126] --profile <name>[,<name>]: the
#      export lines of profiles/<name>.env into control.env, a key already there kept (setenv wins on a rerun); --role rehearsal
#      then writes profiles/rehearsal.env over the same keys (the rehearsal's commit limits)
#   B6 loops.sh start (heartbeat and run_tasks; a running set is adopted, never doubled), then waits for the first status upload
# Rerunning it is safe: every step looks at what is there first. --retire-tmux / --retire-file hand a machine over from earlier
# blocks: refused while any of our scripts runs; a tmux session is closed only when its panes hold nothing but sh / bash / tee /
# sleep (C8); a file is renamed *.retired. --dry (the Mac tests): no install, no cold_start step, no tmux change, no loops.
set -uo pipefail
RB_URL=${RB_URL:-https://github.com/yang3121099/opd-blackwell-runbook.git}
REPOS=public=yang31210999/opd-lab-b300-pub,private=yang31210999/opd-lab-b300-ops
BASE=""; TOKF=""; HFH=""; ONLY=""; CKMIN=200; ROLE=node; RETIRE_TMUX=""; RETIRE_FILES=""; DRY=0; OLDC=""; GHTOKF=""; NOGH=0; PROFILE=""
while [ $# -gt 0 ]; do case $1 in
  --base) BASE=$2; shift 2;; --token-file) TOKF=$2; shift 2;; --hf-home) HFH=$2; shift 2;; --repos) REPOS=$2; shift 2;;
  --only) ONLY=$2; shift 2;; --ckpt-min-gb) CKMIN=$2; shift 2;; --role) ROLE=$2; shift 2;; --old-base-cands) OLDC=$2; shift 2;;
  --retire-tmux) RETIRE_TMUX="$RETIRE_TMUX $2"; shift 2;; --retire-file) RETIRE_FILES="$RETIRE_FILES $2"; shift 2;; --dry) DRY=1; shift;;
  --gh-token-file) GHTOKF=$2; shift 2;; --no-gh) NOGH=1; shift;; --profile) PROFILE=$2; shift 2;;
  *) echo "BOOTSTRAP usage: unknown argument $1"; exit 2;; esac; done
say() { echo "[bootstrap $(date -u +%H:%MZ)] $*"; }
stop() { say "STOP: $*"; exit 1; }
[[ $ROLE =~ ^(node|rehearsal)$ ]] && [[ $CKMIN =~ ^[0-9]+$ ]] && [[ -z $ONLY || $ONLY =~ ^[A-Za-z0-9._-]+(,[A-Za-z0-9._-]+)*$ ]] \
  && [[ -z $PROFILE || $PROFILE =~ ^[A-Za-z0-9_-]+(,[A-Za-z0-9_-]+)*$ ]] || { echo "BOOTSTRAP usage: --role node|rehearsal, --ckpt-min-gb N, --only task[,task], --profile name[,name]"; exit 2; }
for c in bash git curl python3; do command -v $c > /dev/null || stop "$c is missing on this machine"; done
OURS='^bash /root/h800_(setup|loops|fix)\.sh|scripts/(verify|run_arm|overnight|cold_start|setup_runner)\.sh|scripts/task\.sh'

# ---------- B2. the base directory ----------
[ -n "$BASE" ] || BASE=$(df -P -T 2> /dev/null | awk 'NR > 1 && $2 ~ /^(ext[234]|xfs|btrfs|zfs|overlay)$/ {print $5, $7}' | sort -n | tail -1 | cut -d' ' -f2-)
[ -n "$BASE" ] && [ -d "$BASE" ] && [ -w "$BASE" ] || stop "no writable local disk found (run again with --base <dir>)"
T=${BASE%/}/opd; O=$T/run
mkdir -p $O/logs $O/control $T/cache $T/tools $T/ckpt || stop "cannot create $T"
say "base: $T ($(df -Ph $BASE | awk 'NR == 2 {print $4}') free)"

# ---------- B0. the token ----------
if [ -n "$HFH" ]; then TOKF=$HFH/token; [ -s "$TOKF" ] || stop "--hf-home: no token file in it"
elif [ -z "$TOKF" ]; then
  TOKF=$O/.hf_token
  if [ ! -s "$TOKF" ]; then
    [ $DRY = 1 ] && { say "would ask for the token (read -rs)"; } || {
      read -rs -p "Hugging Face token (typed, not shown): " tok < /dev/tty; echo
      [ -n "$tok" ] || stop "no token typed"
      ( umask 077; printf '%s\n' "$tok" > "$TOKF" ); unset tok; chmod 600 "$TOKF"; }
  fi
fi
[ $DRY = 1 ] || { [ -s "$TOKF" ] && [[ "$(stat -c %a "$TOKF" 2> /dev/null || stat -f %Lp "$TOKF")" =~ ^[46]00$ ]] || stop "token file missing or not 0600 / 0400"; }
say "token: from its file, not shown"

# ---------- B0b. the GitHub heartbeat token (z126) ----------
# git's credential store line, written with the shell's own printf (no process gets the token as an argument); a GitHub token is
# letters, digits and _ (ghp_..., github_pat_...), anything else is not stored
GHC=${GH_HB_CRED:-$HOME/.config/opd/gh_hb_credentials}; GHURL=${GH_HB_URL:-https://github.com/yang3121099/opd-b300-heartbeat.git}
if [ $NOGH = 1 ]; then say "GitHub heartbeat: not asked (--no-gh)"
elif [ -s "$GHC" ]; then say "GitHub heartbeat: its token is stored already (not shown)"
else
  ghtok=""
  if [ -n "$GHTOKF" ]; then
    [ -s "$GHTOKF" ] && [[ "$(stat -c %a "$GHTOKF" 2> /dev/null || stat -f %Lp "$GHTOKF")" =~ ^[46]00$ ]] || stop "--gh-token-file missing or not 0600 / 0400"
    IFS= read -r ghtok < "$GHTOKF" || true
  elif [ $DRY = 1 ]; then say "would ask for the GitHub heartbeat token (read -rs, Enter skips)"
  else read -rs -p "GitHub token for the heartbeat repository (typed, not shown; Enter skips): " ghtok < /dev/tty; echo; fi
  ghtok=${ghtok//[[:space:]]/}
  if [ -z "$ghtok" ]; then [ $DRY = 1 ] && [ -z "$GHTOKF" ] || say "GitHub heartbeat: no token, off (Hugging Face is the only way out)"
  elif [[ $ghtok =~ ^[A-Za-z0-9_]{20,255}$ ]]; then
    mkdir -p "$(dirname "$GHC")" && chmod 700 "$(dirname "$GHC")"
    ( umask 077; printf 'https://%s:%s@github.com\n' "${GH_HB_USER:-yang3121099}" "$ghtok" > "$GHC" ); chmod 600 "$GHC"
    say "GitHub heartbeat: token stored for git (0600, not shown)"
  else say "GitHub heartbeat: what was typed is not a GitHub token (letters, digits, _): not stored, off"; fi
  unset ghtok
fi

# ---------- hand-over from earlier blocks (C8) ----------
if [ -n "${RETIRE_TMUX// /}${RETIRE_FILES// /}" ]; then
  P=$(pgrep -fa "$OURS" | cut -c1-120); [ -z "$P" ] || stop "still running, nothing changed: $P"
  for s in $RETIRE_TMUX; do
    tmux has-session -t "$s" 2> /dev/null || continue
    bad=""
    for pp in $(tmux list-panes -s -t "$s" -F '#{pane_pid}'); do
      tree=$(ps -A -o pid=,ppid=,comm= | awk -v r=$pp '{par[$1] = $2; cm[$1] = $3} END {inn[r] = 1; ch = 1
        while (ch) { ch = 0; for (p in par) if (!(p in inn) && (par[p] in inn)) { inn[p] = 1; ch = 1 } } for (p in inn) print cm[p]}')
      for c in $tree; do case $(basename "$c") in sh|bash|dash|tee|sleep) ;; *) bad="$bad $c";; esac; done
    done
    [ -z "$bad" ] || stop "tmux session $s runs more than sh / tee / sleep ($bad): nothing changed"
    [ $DRY = 1 ] && say "would close tmux session $s (sh / tee / sleep only)" || { tmux kill-session -t "$s"; say "tmux session $s closed (it held sh / tee / sleep only)"; }
  done
  for f in $RETIRE_FILES; do [ -f "$f" ] || continue; [ $DRY = 1 ] && say "would rename $f to $f.retired" || { mv "$f" "$f.retired"; say "retired $(basename $f)"; }; done
fi

# ---------- B1. the network ----------
code() { curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$1" 2> /dev/null; }
reach() { [ "$(code https://github.com/)" = 200 ] && [ "$(code https://huggingface.co/api/models/Qwen/Qwen3-0.6B)" = 200 ]; }
PX=$O/proxy.env
if [ -s $PX ]; then . $PX; reach && say "network: the proxy found before (values not shown)" || { rm -f $PX; }; fi
if [ ! -s $PX ]; then
  ( unset https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY; reach ) && { say "network: direct"; NETOK=1; } || NETOK=0
  if [ $NETOK = 0 ]; then
    C=$(mktemp); chmod 600 $C   # proxy variables only, from the environment, the profile files and other processes' environments
    { env; for f in /etc/environment /etc/profile /etc/profile.d/*.sh /root/.bashrc /root/.profile /root/.bash_profile /home/*/.bashrc /home/*/.profile; do
        [ -f "$f" ] && cat "$f" 2> /dev/null; done; for p in /proc/[0-9]*; do tr '\0' '\n' < $p/environ 2> /dev/null; done; } |
      grep -hoE "^(export +)?(https?_proxy|HTTPS?_PROXY|all_proxy|ALL_PROXY)=['\"]?[^'\" ]+" | sed -E "s/^export +//; s/['\"]//g; s/^[A-Za-z_]+=//" | sort -u > $C
    n=$(grep -c . $C); k=0; GOOD=""
    while read -r u; do k=$((k + 1)); [ -n "$u" ] || continue
      ( export https_proxy=$u HTTPS_PROXY=$u http_proxy=$u HTTP_PROXY=$u; reach ) && { GOOD=$u; break; }; done < $C
    rm -f $C
    [ -n "$GOOD" ] || stop "neither direct nor any of $n proxy candidate(s) reaches GitHub and Hugging Face: ask the machine's owner how git / pip reach the internet here"
    NP="localhost,127.0.0.1,::1,$(hostname),$(hostname -I 2> /dev/null | tr -s ' ' ',' | sed 's/,$//')"
    ( umask 077; printf 'export https_proxy=%s\nexport HTTPS_PROXY=%s\nexport http_proxy=%s\nexport HTTP_PROXY=%s\nexport no_proxy=%s\nexport NO_PROXY=%s\n' \
      "$GOOD" "$GOOD" "$GOOD" "$GOOD" "$NP" "$NP" > $PX ); unset GOOD
    . $PX; say "network: proxy candidate $k of $n (value not shown), no_proxy for this host"
  fi
fi

# ---------- B1b. can the heartbeat repository be reached with that token (z126) ----------
if [ $NOGH = 0 ] && [ -s "$GHC" ]; then
  if [ $DRY = 1 ]; then say "would check the GitHub heartbeat repository (git ls-remote)"
  else TO=""; command -v timeout > /dev/null && TO="timeout -k 5 40"
    if GIT_TERMINAL_PROMPT=0 $TO git -c credential.helper= -c "credential.helper=store --file=$GHC" ls-remote "$GHURL" > /dev/null 2>&1
    then say "GitHub heartbeat: repository reachable with its token"
    else say "GitHub heartbeat: repository not reachable (missing, or the token has no access to it): the node goes on, gh_heartbeat retries every 1-15 min"; fi
  fi
fi

# ---------- B3. the runbook, the node id ----------
if [ -d $T/runbook/.git ]; then git -C $T/runbook pull -q --ff-only || say "git pull failed: the clone as it is"
else git clone -q ${RB_BRANCH:+-b $RB_BRANCH} $RB_URL $T/runbook || stop "git clone failed"; fi   # RB_BRANCH: a rehearsal's branch
RB=$T/runbook; [ -f $RB/scripts/loops.sh ] || stop "this runbook has no z106 (scripts/loops.sh) yet"
NODE=$(python3 -c "import hashlib, socket; print(hashlib.sha256(socket.gethostname().split('.')[0].encode()).hexdigest()[:8])")
NG=$(nvidia-smi -L 2> /dev/null | grep -c '^GPU'); CAPS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' ' | sort -u | paste -sd, -)
say "runbook $(git -C $RB log -1 --format=%h); node $NODE; $NG GPU(s)${CAPS:+, compute capability $CAPS}"

# ---------- B4. the control venv ----------
V=$O/venv-ctl
if ! [ -x $V/bin/python ] || ! $V/bin/python -c "import huggingface_hub, yaml" 2> /dev/null; then
  if [ $DRY = 1 ]; then say "would build $V from requirements.ctl.txt"
  else
    say "[installs] the control venv: huggingface_hub, pyyaml (no torch; about 1-2 min)"
    if python3 -m venv $V > $O/logs/bootstrap_venv.log 2>&1; then $V/bin/python -m pip install -q -r $RB/requirements.ctl.txt >> $O/logs/bootstrap_venv.log 2>&1
    else   # no ensurepip (python3-venv not installed): a venv without pip, filled by the system pip
      rm -rf $V; python3 -m venv --without-pip $V >> $O/logs/bootstrap_venv.log 2>&1 && \
        python3 -m pip install -q --target "$($V/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" -r $RB/requirements.ctl.txt >> $O/logs/bootstrap_venv.log 2>&1
    fi
    $V/bin/python -c "import huggingface_hub, yaml" 2> /dev/null || { tail -n 5 $O/logs/bootstrap_venv.log | sed -E 's#//[^/@ ]+@#//***@#g'; stop "the control venv did not build (logs/bootstrap_venv.log)"; }
  fi
fi

# ---------- B5. the control plane's own state ----------
PUB=$(echo $REPOS | tr ',' '\n' | sed -n 's/^public=//p'); [ -n "$PUB" ] || stop "--repos needs public=<owner/name>"
ARGS="--token-file $TOKF --opd-root $O --ckpt-root $T/ckpt --repos $REPOS --node $NODE --gpus ${NG:-0} --ckpt-min-gb $CKMIN --no-loops${ONLY:+ --only $ONLY}"
if [ ! -f $O/control/bootstrap.json ]; then   # step 1 before any loop (no old command may run), once
  if [ $DRY = 1 ]; then say "would run cold_start.sh --step 1"
  else COLDSTART_DRYRUN=0 bash $RB/scripts/cold_start.sh $ARGS --step 1 > $O/logs/bootstrap_step1.log 2>&1 || { tail -n 5 $O/logs/bootstrap_step1.log; stop "cold_start.sh --step 1"; }; fi
fi
( umask 077
  { echo "# [v2.5.9z106] written by bootstrap.sh: the environment of this node's loops (loops.sh sources it); values stay on this machine"
    echo "export OPD_ROOT=$O RUNBOOK=$RB OPD_NODE=$NODE HF_REPO=$PUB"
    echo "export XDG_CACHE_HOME=$T/cache UV_CACHE_DIR=$T/cache/uv UV_PYTHON_INSTALL_DIR=$T/tools/python PIP_CACHE_DIR=$T/cache/pip TRITON_CACHE_DIR=$T/cache/triton"
    echo "export PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1"
    echo "export HF_HOME=$T/cache/hf"   # also with --hf-home: its token is read from there, the caches stay below $T
    [ "$CAPS" = 9.0 ] && echo "export OVERNIGHT_TIER_A_ONLY=1 FA_CUDA_ARCHS=90   # Hopper: vLLM tier A, flash-attn for sm_90"
    [ -n "$OLDC" ] && echo "export OLD_BASE_CANDS=$OLDC"
    [ -n "${GH_HB_CRED:-}" ] && echo "export GH_HB_CRED=$GHC"   # [z126] a credential file moved from its default
    [ -n "${GH_HB_URL:-}" ] && echo "export GH_HB_URL=$GHURL"
    [ -s $PX ] && echo "[ -s $PX ] && . $PX"
    echo "[ -s $TOKF ] && export HF_TOKEN=\"\$(head -1 $TOKF | tr -d '[:space:]')\""
  } > $O/node.env.new && mv $O/node.env.new $O/node.env )
grep -q "^export OPD_ROOT=$O RUNBOOK=$RB OPD_NODE=$NODE HF_REPO=" $O/node.env || stop "node.env not written"
echo "$ARGS" > $O/control/coldstart_args
[ -f $O/control/heartbeat_sec ] || echo 120 > $O/control/heartbeat_sec
touch $O/control.env
if [ -n "$PROFILE" ]; then   # [z126] the node class's settings; a key already in control.env is kept (setenv wins on a rerun)
  for pn in ${PROFILE//,/ }; do
    PF=$RB/profiles/$pn.env; [ -f $PF ] || stop "no profile $pn (there are: $(ls $RB/profiles 2> /dev/null | sed -n 's/\.env$//p' | paste -sd' ' -))"
    added=""; kept=""
    while IFS= read -r line; do [[ $line =~ ^export\ ([A-Z][A-Z0-9_]*)=([A-Za-z0-9_.,:=+-]*)$ ]] || continue; k=${BASH_REMATCH[1]}
      if grep -q "^export $k=" $O/control.env; then kept="$kept $k"; else echo "$line" >> $O/control.env; added="$added $k"; fi
    done < $PF
    say "profile $pn: control.env +${added:- nothing}${kept:+ (already set, kept:$kept)}"
  done
  echo "$PROFILE" > $O/control/profile
fi
if [ $ROLE = rehearsal ]; then   # [z126] the rehearsal's limits over whatever is there
  while IFS= read -r line; do [[ $line =~ ^export\ ([A-Z][A-Z0-9_]*)=([A-Za-z0-9_.,:=+-]*)$ ]] || continue; k=${BASH_REMATCH[1]}
    grep -v "^export $k=" $O/control.env > $O/control.env.new; echo "$line" >> $O/control.env.new; mv $O/control.env.new $O/control.env
  done < $RB/profiles/rehearsal.env
  say "rehearsal limits: $(sed -n 's/^export //p' $RB/profiles/rehearsal.env | paste -sd' ' -)"
fi
grep -q '^export STATUS_PUT_S=' $O/control.env || echo 'export STATUS_PUT_S=120' >> $O/control.env
# the first bootstrap freezes dispatch (also over loops it adopts: their run_tasks honours the file once it runs z106)
[ -f $O/control/bootstrap.json ] || { date -u +%FT%TZ > $O/control/dispatch_frozen
  printf '{"utc": "%s", "node": "%s", "role": "%s", "runbook": "%s", "profile": "%s", "gh_token": %s}\n' "$(date -u +%FT%TZ)" $NODE $ROLE \
    "$(git -C $RB log -1 --format=%h)" "$PROFILE" "$([ $NOGH = 0 ] && [ -s "$GHC" ] && echo true || echo false)" > $O/control/bootstrap.json; }
[ -f $O/control/setup_state.json ] || printf '{"step": "B", "name": "bootstrap", "state": "done", "utc": "%s", "last_lines": []}\n' "$(date -u +%FT%TZ)" > $O/control/setup_state.json
say "control: applied_seq $(cat $O/control/applied_seq 2> /dev/null), heartbeat every $(cat $O/control/heartbeat_sec) s, dispatch $([ -f $O/control/dispatch_frozen ] && echo frozen || echo on)"

# ---------- B6. the loops ----------
[ $DRY = 1 ] && { say "would run loops.sh start; node $NODE"; exit 0; }
set -a; . $O/node.env; set +a
bash $RB/scripts/loops.sh start > $O/logs/bootstrap_loops.log 2>&1; rc=$?
[ $rc = 0 ] || { tail -n 5 $O/logs/bootstrap_loops.log; stop "loops.sh start exit $rc"; }
for i in $(seq 36); do grep -q "^HEARTBEAT " $O/logs/heartbeat.log 2> /dev/null && break; sleep 5; done
if grep -q "^HEARTBEAT " $O/logs/heartbeat.log 2> /dev/null; then echo "node $NODE online; everything else is remote"
else say "node $NODE started, first status not confirmed yet; heartbeat.err:"; tail -n 5 $O/logs/heartbeat.err 2> /dev/null | sed -E 's#//[^/@ ]+@#//***@#g; s/hf_[A-Za-z0-9]{8,}/hf_***/g'; fi
# [z126] the second way out: run_tasks (frozen or not) starts gh_heartbeat, whose first round pushes at once
if [ $NOGH = 0 ] && [ -s "$GHC" ]; then
  for i in $(seq 18); do grep -q '"last_ok_utc": "' $O/control/gh_hb.json 2> /dev/null && break; sleep 5; done
  if grep -q '"last_ok_utc": "' $O/control/gh_hb.json 2> /dev/null; then echo "GitHub heartbeat: hb/$NODE.json pushed"
  else say "GitHub heartbeat: no push yet ($(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('last_error') or 'no round yet')" $O/control/gh_hb.json 2> /dev/null \
    | sed -E 's#//[^/@ ]+@#//***@#g; s/(gh[pousr]_|github_pat_)[A-Za-z0-9_]+/\1***/g' || echo 'no round yet')); the HF path goes on"; fi
fi
