#!/usr/bin/env bash
# Stub tests of v2.5.7 disaster recovery (no GPU, network or HF): dr_sync.sh mirrors the newest resume state (no HF
# weights), replaces the older one, respects the cap and clears the mirror when the task is done; dr_restore copies it
# back into an empty CKPT_ROOT and never over a local state; run_e12.sh saves every 25 steps with the resume state, and
# takes a start the container disk lost from the share (P1) or the result repository (E10b starts).
# Usage: bash scripts/test_dr.sh -> DR_TEST PASS/FAIL
set -u
SRC=$(cd "$(dirname "$0")/.." && pwd); fails=0; n=0
ok() { n=$((n + 1)); if eval "$2"; then :; else echo "  FAIL [$1]"; fails=$((fails + 1)); fi; }
setup() {
  T=$(mktemp -d); mkdir -p $T/rb/scripts $T/rb/code/patches $T/base/root/runs $T/base/root/venv/bin $T/ck; : > $T/base/root/venv/bin/activate
  cp $SRC/scripts/dr.sh $SRC/scripts/dr_sync.sh $SRC/scripts/run_e12.sh $T/rb/scripts/
  cat > $T/rb/scripts/env.sh <<EOF
export OPD_ROOT=$T/base/root RUNBOOK=$T/rb CKPT_ROOT=$T/ck STUDENT_17B=$T/m17
has_weights() { ls "\$1"/*.safetensors > /dev/null 2>&1; }
ckpt_link() { local t=\$(basename "\$1"); mkdir -p "\$CKPT_ROOT/\$t/ckpt"; [ -e "\$1/ckpt" ] || ln -s "\$CKPT_ROOT/\$t/ckpt" "\$1/ckpt"; }
EOF
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("control " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/control.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("remote " + " ".join(sys.argv[1:]) + "\\n"); sys.argv[1:2] == ["repo"] and print("me/pub")\n' > $T/rb/scripts/remote.py
  printf 'import sys, os; open(os.environ["CALLS"], "a").write("fetch " + " ".join(sys.argv[1:]) + "\\n")\n' > $T/rb/scripts/fetch_model.py
  export CALLS=$T/calls; : > $CALLS
  R=$T/base/root/runs/E10b-X; mkdir -p $R; (source $T/rb/scripts/env.sh; ckpt_link $R)
  DRD=$T/base/opd-dr/dr/E10b-X
}
step() {   # <n>: a complete resume state at step n (+ HF weights), latest = n
  mkdir -p $R/ckpt/global_step_$1/actor/huggingface; echo pt > $R/ckpt/global_step_$1/actor/model_world_size_1_rank_0.pt
  echo w > $R/ckpt/global_step_$1/actor/huggingface/model.safetensors; echo d > $R/ckpt/global_step_$1/data.pt; echo $1 > $R/ckpt/latest_checkpointed_iteration.txt; }
sync1() { DR_SYNC_SEC=0 DR_START_GRACE=0 bash $T/rb/scripts/dr_sync.sh $R > $T/sync.out 2>&1; }   # one round (no training running)
# 1. mirror, replace, one copy per task, no HF weights
setup
step 25; sync1
ok "step 25 mirrored without HF weights" '[ -f $DRD/global_step_25/actor/model_world_size_1_rank_0.pt ] && [ -f $DRD/global_step_25/data.pt ] && [ ! -e $DRD/global_step_25/actor/huggingface ] && [ "$(cat $DRD/latest)" = 25 ]'
ok "event dr_mirrored" 'grep -q "^control event E10b-X dr_mirrored step 25" $CALLS'
step 50; sync1
ok "step 50 replaces 25" '[ -d $DRD/global_step_50 ] && [ ! -e $DRD/global_step_25 ] && [ ! -e $DRD/global_step_50.tmp ] && [ "$(cat $DRD/latest)" = 50 ]'
ok "loop ends without a training, mirror kept" 'grep -q "DR_SYNC_EXIT E10b-X" $T/sync.out && [ -d $DRD/global_step_50 ]'
# 2. cap: nothing written over DR_CAP_GB
step 75; DR_CAP_GB=0 sync1
ok "over the cap: no mirror, event dr_cap" '[ "$(cat $DRD/latest)" = 50 ] && grep -q "^control event E10b-X dr_cap" $CALLS'
# 3. restore into an empty CKPT_ROOT; a local resume state wins
rm -rf $T/ck/E10b-X/ckpt/*; mkdir -p $R/ckpt
( source $T/rb/scripts/env.sh; source $T/rb/scripts/dr.sh; dr_restore $R ) > $T/r1 2>&1
ok "restored step 50" 'grep -q "DR_RESTORED E10b-X step 50" $T/r1 && [ -f $R/ckpt/global_step_50/actor/model_world_size_1_rank_0.pt ] && [ "$(cat $R/ckpt/latest_checkpointed_iteration.txt)" = 50 ]'
step 60; ( source $T/rb/scripts/env.sh; source $T/rb/scripts/dr.sh; dr_restore $R ) > $T/r2 2>&1
ok "a local state is never overwritten" '! grep -q DR_RESTORED $T/r2 && [ "$(cat $R/ckpt/latest_checkpointed_iteration.txt)" = 60 ]'
( source $T/rb/scripts/env.sh; source $T/rb/scripts/dr.sh; DR=0 dr_restore $R; DR=0 dr_on ) > /dev/null 2>&1; rc=$?
ok "DR=0 is off" '[ $rc != 0 ]'
# 4. done: the mirror is deleted
echo "finished: now" >> $R/config.yaml; sync1
ok "done task: mirror cleared" '[ ! -e $DRD ] && grep -q "^control event E10b-X dr_cleared" $CALLS'
ok "done task: local resume files deleted, weights kept" '! ls $R/ckpt/global_step_*/actor/*.pt > /dev/null 2>&1 && [ -f $R/ckpt/global_step_60/actor/huggingface/model.safetensors ]'
rm -rf $T
# 5. run_e12.sh: every 25 steps with the resume state (DR on), as before with DR=0; GC=0; starts from the share / result repo
setup
mkdir -p $T/m17 $T/base/root/venv-fa/bin; : > $T/base/root/venv-fa/bin/activate; d1=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1); d2=$(DR=0 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1)
d3=$(GC=0 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1); d5=$(GC=0 E12B_GC=1 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1)
ok "E12b DR: save 25 with resume state" '[[ "$d1" == *"steps=100 save=25 contents=['"'"'model'"'"','"'"'optimizer'"'"','"'"'extra'"'"','"'"'hf_model'"'"']"*"gc=True"* ]]'
ok "E12b DR=0: as before" '[[ "$d2" == *"save=100 contents=['"'"'hf_model'"'"']"* ]] && [[ "$d3" == *"gc=True"* ]] && [[ "$d5" == *"gc=False"* ]]'
mkdir -p $T/base/root/runs/SFT-P1 $T/base/opd-dr/keep/SFT-P1/ckpt/global_step_283/huggingface; echo 283 > $T/base/root/runs/SFT-P1/final_step
: > $T/base/opd-dr/keep/SFT-P1/ckpt/global_step_283/huggingface/model.safetensors
d4=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P1 2>&1 | tail -1)
d6=$(VENV=fa DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1); d7=$(VENV=fa E12B_FA=1 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1)
d8=$(SMOKE=1 PROBE_TASK=FA-SPEED-E12-fa_gc0 DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P0 2>&1 | tail -1)   # v2.5.7c
mkdir -p $T/base/root/runs/E10b-P5-s44/ckpt/global_step_150/actor/huggingface_bf16; : > $T/base/root/runs/E10b-P5-s44/ckpt/global_step_150/actor/huggingface_bf16/model.safetensors
d9=$(DRYRUN=1 bash $T/rb/scripts/run_e12.sh 0 E12b-P5-s44 2>&1 | tail -1)   # v2.5.7n
ok "v2.5.7n E12b-P5-s44: 100 steps from E10b-P5-s44" '[[ "$d9" == "DRYRUN E12b-P5-s44 "*"E10b-P5-s44"*"steps=100 "* ]]'
ok "E12b: venv-fa only with E12B_FA=1" '[[ "$d6" == *"venv=venv "* && "$d7" == *"venv=venv-fa "* ]]'
ok "E12b smoke: own dir, 3 steps, no save" '[[ "$d8" == "DRYRUN FA-SPEED-E12-fa_gc0 "*"steps=3 save=-1 "* ]] && [ ! -e $T/ck/FA-SPEED-E12-fa_gc0 ]'
ok "P1 start from the share when the container disk lost it" '[[ "$d4" == *"start=$T/base/opd-dr/keep/SFT-P1/ckpt/global_step_283/huggingface "* ]]'
bash $T/rb/scripts/run_e12.sh 0 E12b-P4lowlr > $T/e4 2>&1
ok "E10b start fetched from the result repository" 'grep -q "^fetch me/pub runs/E10b-P4lowlr/global_step_150/actor/huggingface $T/ck/adhoc_models/result --task E12b-P4lowlr" $CALLS'
rm -rf $T
[ $fails = 0 ] && echo "DR_TEST PASS ($n checks)" || echo "DR_TEST FAIL ($fails of $n)"
