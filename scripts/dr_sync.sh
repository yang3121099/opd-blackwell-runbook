#!/usr/bin/env bash
# v2.5.7 disaster recovery: mirror one training's newest resume state to $(dr_root)/dr/<task>/ (see dr.sh). Every
# DR_SYNC_SEC (120 s): when ckpt/latest_checkpointed_iteration.txt names a step whose resume state (actor/*.pt) is
# complete and not yet mirrored, and DR_ROOT holds less than DR_CAP_GB (900), the step goes to global_step_<N>.tmp
# (rsync, nice 19, ionice idle, --bwlimit DR_MBPS = 300 MB/s; HF weights excluded), is renamed when complete, `latest`
# is written, and the older mirrored step is deleted (one copy per task). Over the cap: event dr_cap, nothing written.
# The task done (.done or finished:): the mirror and the local resume files (*.pt) are deleted (event dr_cleared),
# the final weights stay, and the loop ends; the training gone
# without finishing: the mirror stays for the retry (whose run_arm.sh / run_e12.sh starts this loop again).
# Usage: bash dr_sync.sh <task dir>
set -uo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/dr.sh"
RUN=$1; TASK=$(basename $RUN); D=$(dr_root)/dr/$TASK; CAP=${DR_CAP_GB:-900}
export DR_BWLIMIT=$(( ${DR_MBPS:-300} * 1024 ))   # rsync counts KiB/s
LOW="nice -n 19"; command -v ionice > /dev/null 2>&1 && LOW="ionice -c3 $LOW"
ev() { python3 $RUNBOOK/scripts/control.py event $TASK "$1" "$2" > /dev/null 2>&1 || true; }
capped=""; seen=0; t_start=$(date +%s)
while true; do
  if [ -f $RUN/.done ] || grep -q '^finished:' $RUN/config.yaml 2> /dev/null; then
    [ -d $D ] && { rm -rf $D; ev dr_cleared "task finished: its mirror is deleted"; echo "DR_CLEARED $TASK $(date -u +%T)"; }
    # the local resume state is not needed either once the task is done (checkpoint policy: final weights only)
    ls $RUN/ckpt/global_step_*/actor/*.pt > /dev/null 2>&1 && rm -f $RUN/ckpt/global_step_*/actor/*.pt && echo "DR_LOCAL_RESUME_DELETED $TASK"
    exit 0
  fi
  latest=$(cat $RUN/ckpt/latest_checkpointed_iteration.txt 2> /dev/null || echo 0); [[ $latest =~ ^[0-9]+$ ]] || latest=0
  S=$RUN/ckpt/global_step_$latest
  if [ $latest -gt 0 ] && ls $S/actor/*.pt > /dev/null 2>&1 && [ "$(cat $D/latest 2> /dev/null)" != "$latest" ]; then
    kb=$(du -sk $(dr_root) 2> /dev/null | cut -f1); used_gb=$(( ${kb:-0} / 1000000 ))
    if [ $used_gb -ge $CAP ]; then
      [ "$capped" = $latest ] || { ev dr_cap "DR_ROOT holds ${used_gb} GB >= ${CAP} GB: step $latest not mirrored"; echo "DR_CAP $TASK step $latest (${used_gb} GB)"; capped=$latest; }
    else
      mkdir -p $D; rm -rf $D/global_step_$latest.tmp; t0=$(date +%s)
      if $LOW bash -c "source $RUNBOOK/scripts/env.sh > /dev/null; source $RUNBOOK/scripts/dr.sh; dr_copy $S $D/global_step_$latest.tmp" \
         && ls $D/global_step_$latest.tmp/actor/*.pt > /dev/null 2>&1; then
        rm -rf $D/global_step_$latest && mv $D/global_step_$latest.tmp $D/global_step_$latest && echo $latest > $D/latest.tmp && mv $D/latest.tmp $D/latest
        for o in $D/global_step_*; do [ "$o" = "$D/global_step_$latest" ] || rm -rf "$o"; done
        kb=$(du -sk $D/global_step_$latest 2> /dev/null | cut -f1); mb=$(( ${kb:-0} / 1000 ))
        ev dr_mirrored "step $latest: $mb MB in $(( $(date +%s) - t0 )) s"; echo "DR_MIRRORED $TASK step $latest $mb MB $(date -u +%T)"
      else
        rm -rf $D/global_step_$latest.tmp; ev dr_mirror_failed "step $latest (retried next round)"; echo "DR_MIRROR_FAILED $TASK step $latest"
      fi
    fi
  fi
  # verl's command line carries trainer.experiment_name=<task>: once seen and then gone, this loop ends (the mirror stays)
  # (one more round first after it went: run_arm.sh / run_e12.sh write finished: / .done a moment after verl exits)
  if pgrep -f "experiment_name=$TASK( |$)" > /dev/null; then seen=1; gone=0
  elif [ $seen = 1 ] && [ ${gone:-0} = 0 ]; then gone=1
  elif [ $seen = 1 ] || [ $(( $(date +%s) - t_start )) -gt ${DR_START_GRACE:-1800} ]; then echo "DR_SYNC_EXIT $TASK (training not running; mirror kept)"; exit 0; fi
  sleep ${DR_SYNC_SEC:-120}
done
