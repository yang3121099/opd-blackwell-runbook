# v2.5.7 disaster recovery (Runming 2026-10-18 via the reviewer, A1 06:4x UTC): helpers sourced by run_arm.sh and
# run_e12.sh after env.sh. A container rebuild loses the container disk ($HOME, CKPT_ROOT) but keeps the network shares
# where $OPD_ROOT lives; DR_ROOT (default: the directory above $OPD_ROOT, /opd-dr) is on that share, is never uploaded,
# and holds dr/<task>/global_step_<N>/ (the newest resume state of each running training, written by dr_sync.sh, one
# copy per task, deleted when the task is done), keep/<task>/ (dr-keep: a finished task's final model kept on the share,
# e.g. SFT-P1, the start of E12b-P1) and state/ (small control files, heartbeat.sh). DR=0 (setenv) turns it off;
# DR_EVERY (setenv, default 25) is the save interval; DR_CAP_GB (default 1500; [v2.5.8] was 900, Runming: the share up to 2 TB) the ceiling for all of DR_ROOT.
dr_root() { echo "${DR_ROOT:-$(dirname "$OPD_ROOT")/opd-dr}"; }
dr_on() { [ "${DR:-1}" = 1 ]; }
dr_copy() {   # <src dir> <dst dir>: resume state without HF weights (resume reads model / optimizer / extra / data.pt)
  if command -v rsync > /dev/null 2>&1; then rsync -a --delete ${DR_BWLIMIT:+--bwlimit=$DR_BWLIMIT} --exclude 'actor/huggingface*' --exclude 'huggingface*' "$1/" "$2/"
  else mkdir -p "$2" && cp -a "$1/." "$2/" && rm -rf "$2/actor/huggingface" "$2/actor/huggingface_bf16" "$2/huggingface"; fi; }
dr_restore() {   # <task dir>: no resume state on the container disk but one on the share -> copy it back; verl's
  # resume_mode=auto then continues from it. A local resume state (any global_step_*/actor/*.pt) always wins.
  dr_on || return 0
  local RUN=$1 T D n; T=$(basename "$1"); D=$(dr_root)/dr/$T
  ls $RUN/ckpt/global_step_*/actor/*.pt > /dev/null 2>&1 && return 0
  [ -f $D/latest ] || return 0
  n=$(cat $D/latest); [[ $n =~ ^[0-9]+$ ]] && ls $D/global_step_$n/actor/*.pt > /dev/null 2>&1 || return 0
  mkdir -p $RUN/ckpt/global_step_$n && DR_BWLIMIT= dr_copy $D/global_step_$n $RUN/ckpt/global_step_$n || return 1
  echo $n > $RUN/ckpt/latest_checkpointed_iteration.txt
  echo "DR_RESTORED $T step $n"
  python3 $RUNBOOK/scripts/control.py event $T dr_restore "resume state of step $n copied back from the share" > /dev/null 2>&1 || true; }
dr_start_sync() {   # <task dir>: the mirror loop, once per task (like upload_sync.sh)
  dr_on || return 0
  pgrep -f "dr_syn[c].sh $1\$" > /dev/null || setsid nohup bash $RUNBOOK/scripts/dr_sync.sh $1 >> $1/dr.log 2>&1 < /dev/null & }
