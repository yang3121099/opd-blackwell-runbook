#!/usr/bin/env bash
# One E10b arm on ONE GPU: student Qwen3-1.7B (non-thinking), teacher Qwen3-8B as verl "ref", 150 steps, 4K rollouts.
# Same recipe as the lab's 0.6B E10 arms (run_e10_arm.sh) except: the student, checkpoints every 25 steps with the
# v2.4 save rule below (dedicated node: HF weights at step 150 only, no resume state), and SDPA when flash-attn is absent.
# Idempotent: a finished task (runs/<task>/.done) is skipped. SMOKE=1 runs 3 steps without validation or checkpoints
# (the verification smoke sets STUDENT=$STUDENT_06B). Models: STUDENT_17B / TEACHER from env.sh (models.env).
# Usage: bash run_arm.sh <gpu> <task id>      task ids: E10b-{P2,P3,P4lowlr,P5,P6,P6r,P2noEOS,Pstop,P6rand,P6top5,P4EOSpen}
#        and the seed replicates E10b-P2-s43/-s44, E10b-P6r-s43/-s44; v2.5.7h: E10b-P5top10 / E10b-P5rand10m (50 steps);
#        v2.5.7o: E10d-P2-s43 / E10d-P5-s43 (4B, data.seed 43, flash-attn); v2.5.7p: E10b-P2-t14 / E10b-P5-t14 (14B teacher);
#        v2.5.7r: E10b-P5think (the student in thinking mode too, 8K rollouts); v2.5.7v: E10b-P5self (teacher = the student's base); v2.5.7zg: E10b-P2self
set -euo pipefail
source "$(dirname "$0")/env.sh"; source "$(dirname "$0")/dr.sh"; GPU=$1; TASK=$2
# task id E10b-<arm>[-s<seed>]: the seed changes only the data order (data.seed); the default seed is 42
SEED=42; BASE=$TASK; [[ $TASK =~ -s(43|44)$ ]] && { SEED=${BASH_REMATCH[1]}; BASE=${TASK%-s*}; }
# v2.5.7p (Q-015, reviewer A1 07:1x UTC): E10b-<P2|P5>-t14 = the E10b arm with Qwen3-14B as the teacher ($TEACHER_14B),
# not offloaded (it fits next to the 1.7B student and vLLM); everything else as E10b-<arm> under the node's settings
T14=0; [[ $TASK =~ ^E10b-(P2|P5)-t14$ ]] && { T14=1; BASE=${TASK%-t14}; }
# v2.5.7r (Q-016, reviewer A1 07:5x UTC; 08:2x UTC: 50 steps first, Runming wants a fast answer, against P5 s50; 150 later
# as an optional extension): E10b-P5think = P5 with the student in thinking mode as well (the symmetric setting):
# the rollouts (and validation) use the thinking template, 8K responses (4K would cut most thinking rollouts before
# </think>; not comparable with E10b's 4K), and the teacher scores with the same thinking template. G-OPD renders the
# teacher prompt with data.apply_chat_template_kwargs (ray_trainer -> prepare_ref_model_inputs), so OPD_REF_THINK stays 0:
# the student's response opens with its own <think>, and the patch's extra '<think>\n' (P5's non-thinking student) would
# double it. No change to the installed patch.
STHINK=0; MAXRESP=4096; [ $TASK = E10b-P5think ] && { STHINK=1; MAXRESP=8192; BASE=E10b-P5; }
# v2.5.7v (Q-017, reviewer 08:2x UTC): E10b-P5self = P5 with the teacher replaced by the student's own base (Qwen3-1.7B,
# the same weights the student starts from) on the thinking template: mode without extra knowledge; 50 steps, as P5 s50
SELF=0; [ $TASK = E10b-P5self ] && { SELF=1; BASE=E10b-P5; }
# [v2.5.8e] (reviewer 15:5x UTC 10-20): E10b-P5self-s43 = its second seed (data.seed 43, the seed parse above); E10d-P5self =
# the same with the 4B student, whose own base (Qwen3-4B, thinking template) is the teacher; 50 steps, E10d's batching
[ $TASK = E10b-P5self-s43 ] && { SELF=1; BASE=E10b-P5; }
# v2.5.7zg (Q-019, reviewer 17:0x UTC): E10b-P2self = P2 (non-thinking teacher prompt) with the teacher replaced by the
# student's own base, the other cell of the 2x2 (teacher mode x teacher = self); 50 steps on venv-fa, gated by PROBE-P2SELF
[ $TASK = E10b-P2self ] && { SELF=1; BASE=E10b-P2; }
# v2.5.6f (queue v3 V3-c1): E10c-<arm>-fix4 / -dyn16k = the lab's 0.6B E10 arm on this node (student Qwen3-0.6B; data,
# seeds and hyper-parameters as the lab's run_e10_arm.sh), with a fixed 4 sequences per micro-batch or the lab's dynamic
# token budget of 16384. The batching mode is the experiment: an out-of-memory retry does not change it.
# v2.5.7g (reviewer 18:4x UTC): -dyn6144 = the dynamic token budget at 6144 (dyn16k ran out of memory in the entropy
# without remove-padding; verl needs the budget >= the padded 5120), and -s43 = data.seed 43 (the seed parse above)
MODE=""; [[ $TASK =~ ^E10c-(.+)-(fix4|dyn16k|dyn6144)(-s(43|44))?$ ]] && { BASE=E10b-${BASH_REMATCH[1]}; MODE=${BASH_REMATCH[2]}; }
# v2.5.7b (queue v3 V3-c5, the 4B probe): E10d-<P2|P5> = the E10b arm with the 4B student ($STUDENT_4B); gradient
# checkpointing stays on for it (GC=0 was measured on 1.7B only), the batching is the node's (FIXED_MICRO_BSZ)
SIZE=""; [[ $TASK =~ ^E10d-(P2|P5)(-s43)?$ ]] && { BASE=E10b-${BASH_REMATCH[1]}; SIZE=4B; }   # v2.5.7o: -s43 = data.seed 43
[ $TASK = E10d-P5self ] && { BASE=E10b-P5; SIZE=4B; SELF=1; }   # [v2.5.8e]
# [v2.5.8i] (reviewer 17:4x UTC, Q-029; Runming: explore Qwen3's separate Instruct / Thinking weights): E10d-X1 = P5's recipe
# with the student Qwen3-4B-Instruct-2507 and the teacher Qwen3-4B-Thinking-2507, whose own template already opens <think>
# (so OPD_REF_THINK=0); E10d-X2 = the same student taught by the hybrid Qwen3-4B on the thinking template (as P5). 50 steps;
# both 2507 models from the node's HF cache. G-OPD renders the teacher prompt with the teacher's own tokenizer and template
# (main_ppo loads ref_tokenizer from ref.model.path); the four Qwen3 tokenizer.json files are identical (sha256 aeb13307)
# [v2.5.8l] (Q-032, Runming via the reviewer 06:4x UTC 10-21): E10d-X4 = P5's recipe with the hybrid Qwen3-4B student
# (non-thinking template, as E10d-P5) taught by Qwen3-4B-Instruct-2507 in its own template (no <think> anywhere,
# OPD_REF_THINK=0): a teacher with thinking-level accuracy and non-thinking termination
XARM=""; [[ $TASK =~ ^E10d-(X1|X2|X4)$ ]] && { XARM=${BASH_REMATCH[1]}; BASE=E10b-P5; SIZE=4B; }
# [v2.5.8m] (Q-030, Runming's question 4; reviewer pre-registration 17:5x UTC 10-20): E10b-P5len<S|L|R> = P5's recipe with
# 16 rollouts per prompt of which only the shortest / longest / a random 4 are kept before the old and reference log-probs
# (code/patches/opd_lensel.py through the trainer hook lensel_patch.py): 32 prompts x 4 = 128 sequences per update (P5:
# 128 x 1), one mini-batch, 50 steps, venv-fa; every step logs lensel/* (all 16 against the kept 4: length, truncation,
# EOS, score) and runs/<task>/lensel.jsonl (per prompt). Gated by PROBE-LENSEL.
LENSEL=""; [[ $TASK =~ ^E10b-P5len([SLR])$ ]] && { case ${BASH_REMATCH[1]} in S) LENSEL=short;; L) LENSEL=long;; R) LENSEL=rand;; esac; BASE=E10b-P5; }
case $BASE in
  E10b-P2) ARM=P2; LR=3e-6;; E10b-P3) ARM=P3; LR=3e-6;; E10b-P4lowlr) ARM=P4; LR=6e-7;; E10b-P5) ARM=P5; LR=3e-6;;
  E10b-P6) ARM=P6; LR=3e-6;; E10b-P6r) ARM=P6r; LR=3e-6;; E10b-P2noEOS) ARM=P2noEOS; LR=3e-6;;
  E10b-Pstop) ARM=Pstop; LR=3e-6;; E10b-P6rand) ARM=P6rand; LR=3e-6;; E10b-P6top5) ARM=P6top5; LR=3e-6;;
  E10b-P4EOSpen) ARM=P4EOSpen; LR=3e-6;;
  E10b-P5top10) ARM=P5top10; LR=3e-6;; E10b-P5rand10m) ARM=P5rand10m; LR=3e-6;;   # v2.5.7h (Q-014): token-selective P5
  *) echo "unknown task $TASK"; exit 2;;
esac
[ "${SMOKE:-0}" = 1 ] && TASK=${PROBE_TASK:-smoke-$TASK}   # v2.5.3: FA-PROBE names its own smoke directory
OUT=$OPD_ROOT/runs/$TASK; mkdir -p $OUT
[ -f $OUT/.done ] && { echo "TASK_ALREADY_DONE $TASK"; exit 0; }
STUDENT=${STUDENT:-$STUDENT_17B}   # the verification smoke uses $STUDENT_06B
[ -n "$MODE" ] && STUDENT=$STUDENT_06B
[ "$SIZE" = 4B ] && { STUDENT=$STUDENT_4B; GC=1; }
[ $T14 = 1 ] && { TEACHER=$TEACHER_14B; TEACHER_OFFLOAD=0; }   # v2.5.7p: a node-wide TEACHER_OFFLOAD does not reach it
[ $SELF = 1 ] && { TEACHER=$STUDENT_17B; [ "$SIZE" = 4B ] && TEACHER=$STUDENT_4B; }   # v2.5.7v; [v2.5.8e] the 4B student's own base
if [ -n "$XARM" ]; then   # [v2.5.8i] hub ids, resolved to the local HF cache after the venv is active
  STUDENT=${STUDENT_I2507:-Qwen/Qwen3-4B-Instruct-2507}; TEACHER=${TEACHER_T2507:-Qwen/Qwen3-4B-Thinking-2507}; [ $XARM = X2 ] && TEACHER=$STUDENT_4B
  [ $XARM = X4 ] && { STUDENT=$STUDENT_4B; TEACHER=${STUDENT_I2507:-Qwen/Qwen3-4B-Instruct-2507}; }   # [v2.5.8l]
fi
# [v2.5.8i] an X arm starts only after its gate PROBE-X2507-<X1|X2|X4> passed (prompt templates + a 3-step smoke)
if [ -n "$XARM" ] && [ "${SMOKE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
  grep -q '"result": "PASS"' $OPD_ROOT/runs/PROBE-X2507-$XARM/report.json 2> /dev/null || { echo "PROBE-X2507-$XARM has not passed: $TASK not started"; exit 3; }
fi
# [v2.5.8m] a length-selected arm starts only after PROBE-LENSEL passed (the unit tests on the node + a 2-step smoke)
if [ -n "$LENSEL" ] && [ "${SMOKE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
  grep -q '"result": "PASS"' $OPD_ROOT/runs/PROBE-LENSEL/report.json 2> /dev/null || { echo "PROBE-LENSEL has not passed: $TASK not started"; exit 3; }
fi
# v2.5.7v: E10b-P5self starts only after PROBE-P5SELF passed (a 3-step smoke); v2.5.7zg: E10b-P2self after PROBE-P2SELF
if [ $SELF = 1 ] && [ "${SMOKE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
  SA=${TASK#E10[bd]-}; PG=PROBE-$(echo ${SA%-s4[34]} | tr a-z A-Z); [ "$SIZE" = 4B ] && PG=$PG-4B   # [v2.5.8e] -s43 shares its arm's gate; 4B: PROBE-P5SELF-4B
  grep -q '"result": "PASS"' $OPD_ROOT/runs/$PG/report.json 2> /dev/null || { echo "$PG has not passed: $TASK not started"; exit 3; }
fi
# v2.5.7r: E10b-P5think starts only after PROBE-P5THINK passed (templates + a 3-step smoke whose rollouts open with <think>)
if [ $STHINK = 1 ] && [ "${SMOKE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
  grep -q '"result": "PASS"' $OPD_ROOT/runs/PROBE-P5THINK/report.json 2> /dev/null || { echo "PROBE-P5THINK has not passed: $TASK not started"; exit 3; }
fi
# v2.5.7p: a -t14 arm starts only after PROBE-T14 passed (the 14B fetched to the share + a 3-step smoke under 260 GB)
if [ $T14 = 1 ] && [ "${SMOKE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
  grep -q '"result": "PASS"' $OPD_ROOT/runs/PROBE-T14/report.json 2> /dev/null || { echo "PROBE-T14 has not passed: $TASK not started"; exit 3; }
fi
# v2.5.7e: a 4B arm starts only after PROBE-4B passed (queued ahead, so the node runs them unattended; a FAIL keeps them out)
if [ "$SIZE" = 4B ] && [ "${SMOKE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
  grep -q '"result": "PASS"' $OPD_ROOT/runs/PROBE-4B/report.json 2> /dev/null || { echo "PROBE-4B has not passed: $TASK not started"; exit 3; }
fi
# v2.5.7h: the token-selective arms start only after PROBE-TOP10 passed (the variant unit tests on the node + a 3-step smoke)
if [[ $ARM =~ ^P5(top10|rand10m)$ ]] && [ "${SMOKE:-0}" != 1 ] && [ "${DRYRUN:-0}" != 1 ]; then
  grep -q '"result": "PASS"' $OPD_ROOT/runs/PROBE-TOP10/report.json 2> /dev/null || { echo "PROBE-TOP10 has not passed: $TASK not started"; exit 3; }
fi
case $MODE in fix4) FIXED_MICRO_BSZ=4;; dyn16k) FIXED_MICRO_BSZ=""; MAX_TOK_GPU=16384;; dyn6144) FIXED_MICRO_BSZ=""; MAX_TOK_GPU=6144;; esac
[ -n "$MODE" ] && VENV=   # v2.5.7c: E10c is registered on this node without remove-padding (V3-c1): the venv in use
# v2.5.7d: the 4B ladder stays on SDPA (reviewer 09:0x UTC), FA-SPEED-4B sets E10D_FA=1; v2.5.7o: the 4B seed-43 arms
# (E10d-P2-s43 / E10d-P5-s43) run on flash-attn (reviewer 06:0x UTC: new arms default to FA)
if [ "$SIZE" = 4B ]; then
  if [ $SEED != 42 ]; then VENV=fa; elif [ "${E10D_FA:-0}" != 1 ]; then VENV=; fi
fi
[ $TASK = E10b-P2self ] && VENV=fa   # v2.5.7zg (reviewer: P2self on FA)
[ -n "$LENSEL" ] && VENV=fa   # [v2.5.8m] 512 rollouts per step: flash-attn with remove-padding
case ${VENV:-} in fa) VENV=$OPD_ROOT/venv-fa;; esac   # v2.5.3: setenv VENV=fa (after FA-PROBE passed) = venv-fa with flash-attn
source ${VENV:-$OPD_ROOT/venv}/bin/activate   # FA-PROBE passes its own VENV; default: the venv in use
if [ -n "$XARM" ]; then   # [v2.5.8i] the 2507 models from the HF cache only (the node's proxy blocks the hub)
  loc() { [ -d "$1" ] && { echo "$1"; return 0; }; python -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], local_files_only=True))" "$1" 2> /dev/null; }
  S=$(loc $STUDENT) || { echo "no local copy of $STUDENT"; exit 2; }; T=$(loc $TEACHER) || { echo "no local copy of $TEACHER"; exit 2; }
  STUDENT=$S; TEACHER=$T
fi
export CUDA_VISIBLE_DEVICES=$GPU RAY_TMPDIR=/tmp/ray_gpu$GPU
export PYTHONPATH=$RUNBOOK/code/patches:${PYTHONPATH:-}
export OPD_VARIANT=$ARM OPD_SEED=${OPD_SEED:-1234} OPD_VARIANT_OVERRIDE= OPD_TOKCAT=
# v2.5.7h (Q-014): P5top10 / P5rand10m reach opd_variants.compute through the dispatched hook P6 (opd_variants.OVERRIDE_VIA:
# the installed dp_actor patch is not touched), log the token classes of their selected set, train 50 steps with the P5
# teacher prompt
NSTEPS=150
case $ARM in P5top10|P5rand10m) export OPD_VARIANT=P6 OPD_VARIANT_OVERRIDE=$ARM OPD_TOKCAT=1 OPD_TOKENIZER=$STUDENT; NSTEPS=50;; esac
[ $STHINK = 1 ] && NSTEPS=${P5THINK_STEPS:-50}
[ -n "$XARM" ] && NSTEPS=${X2507_STEPS:-50}   # [v2.5.8i]
[ -n "$LENSEL" ] && NSTEPS=${LENSEL_STEPS:-50}   # [v2.5.8m]
[ $SELF = 1 ] && NSTEPS=${P5SELF_STEPS:-50}   # v2.5.7v: the same point as P5 s50   # v2.5.7r: constant lr, so step 50 is the same point as in a 150-step run
OPD_REF_THINK=0; [[ $ARM =~ ^P5(top10|rand10m)?$ ]] && OPD_REF_THINK=1; [ $STHINK = 1 ] && OPD_REF_THINK=0; [ "$XARM" = X1 ] && OPD_REF_THINK=0; [ "$XARM" = X4 ] && OPD_REF_THINK=0; export OPD_REF_THINK   # [v2.5.8i] X1: the template opens <think>
if python -c "import flash_attn" 2>/dev/null; then RMPAD=True; ATTN=flash_attention_2; TOK_DEFAULT=16384
else RMPAD=False; ATTN=sdpa; TOK_DEFAULT=6144; fi
# without remove-padding every micro-batch is padded to its longest sequence, so the fp32 logits grow with the token
# budget: 16384 ran out of memory on a B200 (183 GB) in the actor update; 6144 keeps the logits under ~10 GB only for
# long answers: the budget counts real tokens while every sequence is padded to the full length, so short answers pack
# ~11 sequences into one micro-batch (32 GiB of logits; E10b-P4lowlr ran out of memory at step 97, 2026-10-17), and
# verl requires the budget >= the padded length (5120), so it cannot be lowered.
# v2.5.2: FIXED_MICRO_BSZ=<n> (setenv, 1-16) uses n sequences per micro-batch in the actor update and in the rollout and
# reference log-probs instead of the token budget: the logits are then bounded by n x 5120 x vocab x 4 B (4: 11.6 GiB).
# Without flash-attn the entropy is computed in chunks (no full-size softmax temporary next to the logits).
if [ -n "${FIXED_MICRO_BSZ:-}" ]; then BATCH="actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$FIXED_MICRO_BSZ"; LOGP_MB=$FIXED_MICRO_BSZ
else BATCH="actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOK_GPU:-$TOK_DEFAULT}"; LOGP_MB=8; fi
ENT=""; [ "$RMPAD" = False ] && ENT="actor_rollout_ref.actor.entropy_from_logits_with_chunking=True"
# v2.5.6o: FUSED=1 (setenv) = verl's fused log-prob / entropy kernels (torch backend, actor and teacher: no full logits);
# GC=0 (setenv) = no gradient checkpointing in the actor. SPEED-PROBE measures both; unset, the ladder runs as before.
KERN=""; [ "${FUSED:-0}" = 1 ] && KERN="actor_rollout_ref.model.use_fused_kernels=True actor_rollout_ref.model.fused_kernel_options.impl_backend=torch"
GCK=True; [ "${GC:-1}" = 0 ] && GCK=False
# [v2.5.8m] prompts per step, rollouts per prompt and the mini-batch in prompts (verl multiplies it by n): P5 128 x 1 in one
# mini-batch; a length-selected arm 32 x 16 with k = 4 kept, so 32 x 4 / 16 = 8 prompts' worth = the 128 kept sequences
TBS=128; RN=1; MBS=128; unset OPD_LENSEL OPD_LENSEL_K OPD_LENSEL_LOG
if [ -n "$LENSEL" ]; then TBS=${LENSEL_PROMPTS:-32}; RN=${LENSEL_N:-16}; LK=${LENSEL_K:-4}; MBS=$(( TBS * LK / RN ))
  export OPD_LENSEL=$LENSEL OPD_LENSEL_K=$LK OPD_LENSEL_LOG=$OUT/lensel.jsonl; fi
UW=$NSTEPS; [ $TASK = E10c-P5-fix4 ] && UW=100,$NSTEPS; [ -n "$XARM" ] && UW=25,$NSTEPS   # [v2.5.8i] X arms: s25 is read too   # [v2.5.8a] (Q-023): its s100 = the step of the lab's 0.6B P5 reading (+19.2)
[ "${DRYRUN:-0}" = 1 ] && { echo "DRYRUN $TASK arm=$ARM lr=$LR seed=$SEED student=$STUDENT batch=[$BATCH] kern=[$KERN] gc=$GCK venv=$(basename ${VENV:-$OPD_ROOT/venv}) rmpad=$RMPAD lensel=${LENSEL:-none}/${TBS}x${RN}/mb$MBS sthink=$STHINK maxresp=$MAXRESP teacher=$TEACHER offload=${TEACHER_OFFLOAD:-1} variant=$OPD_VARIANT override=$OPD_VARIANT_OVERRIDE refthink=$OPD_REF_THINK steps=$NSTEPS"; exit 0; }   # tests
STEPS=$NSTEPS; TEST=10; VALB=True
# save rule (v2.4, Runming 2026-10-17): no archive of resume states. A dedicated node saves only the HF weights at the
# pre-registered reading steps (E10b: step 150, the one checkpoint E11b and E12b read; SAVE=<n> for others), which
# upload_sync turns into bf16, uploads and keeps locally as bf16 only. SPOT=1 (a node that can vanish) also saves the
# resume state every 25 steps, and upload_sync keeps only the newest one.
if [ "${SPOT:-0}" = 1 ]; then SAVE=${SAVE:-25}; CONTENTS="['model','optimizer','extra','hf_model']"
elif dr_on; then SAVE=${SAVE:-${DR_EVERY:-25}}; CONTENTS="['model','optimizer','extra','hf_model']"   # v2.5.7: resume state for dr_sync.sh
else SAVE=${SAVE:-$STEPS}; CONTENTS="['hf_model']"; fi
[ "${SMOKE:-0}" = 1 ] && { STEPS=${SMOKE_STEPS:-3}; SAVE=-1; TEST=${SMOKE_TEST:--1}; VALB=False; }   # v2.5.7b: SMOKE_STEPS (PROBE-4B: 10); v2.5.7f: SMOKE_TEST (FA-SPEED-TWIN: validation every 10)
[ -f $OUT/config.yaml ] && echo "resumed: $(date -u +%FT%TZ)" >> $OUT/config.yaml || cat > $OUT/config.yaml <<CFG
task: $TASK
arm: $ARM
lr: $LR
data_seed: $SEED
steps: $STEPS
student: $STUDENT (no_think)
teacher: $TEACHER (ref; think-template=$OPD_REF_THINK; param_offload=${TEACHER_OFFLOAD:-1})
student_template: $([ $STHINK = 1 ] && echo "thinking (the teacher prompt uses the same template)" || echo non-thinking), max_response_length $MAXRESP
attention: $ATTN (use_remove_padding=$RMPAD)
batching: ${MODE:-default} [$BATCH]
rollouts: $TBS prompts x $RN per step$([ -n "$LENSEL" ] && echo ", length selection $LENSEL: $OPD_LENSEL_K kept per prompt before the log-probs (lensel.jsonl)"), mini-batch $MBS prompts
kernels: fused=${FUSED:-0} gradient_checkpointing=$GCK
dr: $(dr_on && echo "every $SAVE steps" || echo off)
gopd_commit: $GOPD_COMMIT
gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU)
started: $(date -u +%FT%TZ)
CFG
[ "${SMOKE:-0}" = 1 ] || ckpt_link $OUT   # v2.4: checkpoints on $CKPT_ROOT (the large disk)
[ "${SMOKE:-0}" = 1 ] || { dr_restore $OUT || echo "DR_RESTORE_FAILED $TASK"; dr_start_sync $OUT; }   # v2.5.7
[ "${SMOKE:-0}" = 1 ] || pgrep -f "upload_syn[c].sh $OUT\$" >/dev/null || \
  UPLOAD_WEIGHTS=${UPLOAD_WEIGHTS:-$UW} setsid nohup bash $RUNBOOK/scripts/upload_sync.sh $OUT > $OUT/upload.log 2>&1 < /dev/null &
if [ -n "$LENSEL" ]; then python $RUNBOOK/code/patches/lensel_patch.py || { echo "LENSEL_PATCH_FAILED $TASK"; exit 1; }; fi   # [v2.5.8m] idempotent
cd $OPD_ROOT/G-OPD/verl
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.rollout_correction.rollout_is=token algorithm.rollout_correction.rollout_is_threshold=5.0 \
  algorithm.rollout_correction.rollout_rs=null algorithm.rollout_correction.bypass_mode=false \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  data.train_files=$OPD_ROOT/data/dapo_train.parquet data.val_files=$OPD_ROOT/data/math500_100.parquet \
  data.train_batch_size=$TBS data.max_prompt_length=1024 data.max_response_length=$MAXRESP \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True data.seed=$SEED data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=$([ $STHINK = 1 ] && echo True || echo False) \
  actor_rollout_ref.model.path=$STUDENT +actor_rollout_ref.ref.model.path=$TEACHER \
  +actor_rollout_ref.model.override_config.attn_implementation=$ATTN \
  actor_rollout_ref.actor.optim.lr=$LR actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.model.use_remove_padding=$RMPAD \
  actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True actor_rollout_ref.actor.policy_loss.lambda_vals=1.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MBS $BATCH $ENT $KERN \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.checkpoint.save_contents=$CONTENTS \
  actor_rollout_ref.model.enable_gradient_checkpointing=$GCK \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGP_MB actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.5} \
  actor_rollout_ref.rollout.n=$RN actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.8 actor_rollout_ref.rollout.val_kwargs.n=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGP_MB \
  actor_rollout_ref.ref.fsdp_config.param_offload=$([ "${TEACHER_OFFLOAD:-1}" = 1 ] && echo True || echo False) \
  custom_reward_function.path=$RUNBOOK/code/train/reward_mathverify.py custom_reward_function.name=compute_score \
  reward_model.reward_manager=naive trainer.critic_warmup=0 trainer.val_before_train=$VALB \
  trainer.logger='["console"]' trainer.project_name=opd-e10b trainer.experiment_name=$TASK \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq=$SAVE trainer.test_freq=$TEST \
  trainer.resume_mode=auto trainer.total_training_steps=$STEPS trainer.total_epochs=2 \
  trainer.default_local_dir=$OUT/ckpt $([ -n "${ROLLOUT_DUMP:-}" ] && echo trainer.rollout_data_dir=$ROLLOUT_DUMP) \
  2>&1 | tee -a $OUT/train.log
LAST=$(grep -oE "(^|[ )])step:[0-9]+ - " $OUT/train.log | tail -1 | grep -oE "[0-9]+")
echo "completed_steps: ${LAST:-unknown}" >> $OUT/config.yaml
if [ "${LAST:-0}" -ge $STEPS ]; then echo "finished: $(date -u +%FT%TZ)" >> $OUT/config.yaml; touch $OUT/.done
else echo "stopped: $(date -u +%FT%TZ) before the last step (v2.3: no finished: line, so upload_sync keeps running for the retry)" >> $OUT/config.yaml; fi
echo "TASK_FINISHED $TASK steps=${LAST:-unknown}"
