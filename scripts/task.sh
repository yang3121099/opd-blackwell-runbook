#!/usr/bin/env bash
# Whitelist dispatcher: run ONE task on ONE GPU. Only the task IDs matched below are accepted; anything else exits 2.
# Every task writes $OPD_ROOT/runs/<task>/.done when it has finished properly; run_tasks.sh treats a missing .done
# after exit as a failure. task.sh stays the parent process (no exec), so run_tasks.sh can see that a task is running.
# Usage: bash task.sh <gpu> <task id>
set -uo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=$2
# [v2.5.9a] G = one GPU, cpu, or a list a,b (a #gpus task, run_tasks.sh): a list only for the trainings and PROBE-2GPU
[[ $G =~ ^([0-9]+(,[0-9]+){0,3}|cpu)$ ]] || { echo "bad gpu $G"; exit 2; }
case $G in *,*) [[ $T =~ ^(E10[bcd]-|E12[bcd]-|PROBE-2GPU$) ]] || { echo "$T takes one GPU (got $G)"; exit 2; };; esac
# v2.5.4a: a task whose previous attempt ran out of GPU memory (run_tasks.sh writes runs/<task>/.oom) gets FIXED_MICRO_BSZ=4
# for its own retries, unless the key is set already; nothing else changes
[ -f $OPD_ROOT/runs/$T/.oom ] && [ -z "${FIXED_MICRO_BSZ:-}" ] && { export FIXED_MICRO_BSZ=4; echo "OOM_RETRY $T with FIXED_MICRO_BSZ=4"; }
ARMS="P2|P3|P4lowlr|P5|P6|P6r|P2noEOS|Pstop|P6rand|P6top5|P4EOSpen|P5top10|P5rand10m"
case $T in
  E10b-*)  [[ $T =~ ^E10b-(($ARMS)(-s(43|44))?|(P2|P5)-t14|P5think|P5self|P5self-s43|P2self|P5len[SLR]|P5stopmerge|P5tail)$ ]] || exit 2; bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # v2.5.7p: -t14; v2.5.7r: P5think; v2.5.7v: P5self; [v2.5.8e] P5self-s43
  E10c-*)  [[ $T =~ ^E10c-(($ARMS)-(fix4|dyn16k|dyn6144)(-s43)?|P5-polish|P5self)$ ]] || exit 2; bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # v2.5.6f: 0.6B, V3-c1; v2.5.7g: dyn6144, -s43
  E10d-*)  [[ $T =~ ^E10d-((P2|P5)(-s43)?|P5self|X1|X2|X4)$ ]] || exit 2; bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # v2.5.7b: 4B student, V3-c5; v2.5.7o: -s43
  E11b-*)  [[ $T =~ ^E11b-(P0|($ARMS)(-s(43|44))?|(P2|P5)-t14)$ ]] || exit 2; bash $RUNBOOK/scripts/run_eval.sh $G $T;;   # v2.5.6c: seed arms too
  SFT-P1)  bash $RUNBOOK/scripts/run_sft.sh $G $T;;
  E12b-*)  [[ $T =~ ^E12b-(P0|P1|$ARMS|P5-s4[34]|P2-s43|P5self|P5self-s43|P5-kl|P5-stop|P5-stop2|P5-8k|P5-16k)$ ]] || exit 2; bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # v2.5.6j: P5-s43; v2.5.7n: P5-s44; v2.5.7o: P2-s43; [v2.5.8a] P5self; [v2.5.8b] P5-kl; [v2.5.8c] P5-stop; [v2.5.9f] P5-stop2
  E12c-*)  [[ $T =~ ^E12c-(P0|P5)$ ]] || exit 2; bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.8a] the 0.6B GRPO ladder (Q-023)
  E12d-*)  [[ $T =~ ^E12d-(P0|P5)$ ]] || exit 2; bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # v2.5.7o: the 4B GRPO ladder
  E03-MI-l0.0|E03-MI-l0.3|E03-MI-l0.5|E03-MI-l0.7|E03-MI-l1.0) bash $RUNBOOK/scripts/run_zero.sh $G $T;;
  E02-opsa17-c0.5|E02-opsa17-c1.5|E02-opsa4-c-1|E02-opsa4-c0.5|E02-opsa4-c1.5) bash $RUNBOOK/scripts/run_zero.sh $G $T;;
  E02-opsa17-c-1|E02-opsa17-c2|E02-opsa4-c2) bash $RUNBOOK/scripts/run_zero.sh $G $T;;   # v2.5.6j: the scaling ends (V3-c3 a)
  E01-opsa17-think|E01-opsa17-nothink|E01-base17-think) bash $RUNBOOK/scripts/run_zero.sh $G $T;;   # v2.5.6j: E01 remainder (V3-c3 d)
  E05-fstop-base06|E05-fstop-art06) bash $RUNBOOK/scripts/run_zero.sh $G $T;;   # v2.5.6l: E05-p3 forced stop (V3-c3 c)
  FA-PROBE) bash $RUNBOOK/scripts/fa_probe.sh $G;;
  SPEED-PROBE) bash $RUNBOOK/scripts/speed_probe.sh $G;;
  PROBE-4B) bash $RUNBOOK/scripts/probe_4b.sh $G;;
  PROBE-TERM|PROBE-TERM2|PROBE-TERM3|PROBE-TERM4|PROBE-TERM4-4B|PROBE-TERM4-06B) bash $RUNBOOK/scripts/term_probe.sh $G $T;;   # v2.5.7zk: -4B; [v2.5.9v] -06B
  STRICT-C1) bash $RUNBOOK/scripts/strict_c1.sh $G $T;;
  PROBE-T14) bash $RUNBOOK/scripts/probe_t14.sh $G;;   # v2.5.7p (Q-015): the 14B teacher's gate
  PROBE-P5THINK) bash $RUNBOOK/scripts/probe_p5think.sh $G;;   # v2.5.7r (Q-016): the thinking student's gate
  PROBE-P2SELF) bash $RUNBOOK/scripts/probe_p2self.sh $G;;   # v2.5.7zg (Q-019): the P2 self-teacher's gate
  PROBE-E12C) bash $RUNBOOK/scripts/probe_e12c.sh $G;;   # [v2.5.8a] (Q-023): the gate of E12c (a 3-step smoke of E12c-P0)
  PROBE-E12B-KL) bash $RUNBOOK/scripts/probe_e12var.sh $G kl;;   # [v2.5.8b] (Q-020): the gate of E12b-P5-kl
  PROBE-E12B-STOP2) bash $RUNBOOK/scripts/probe_e12var.sh $G stop2;;   # [v2.5.9f] (T1): the gate of E12b-P5-stop2
  PROBE-E12B-STOP) bash $RUNBOOK/scripts/probe_e12var.sh $G stop;;   # [v2.5.8c] (Q-021): the gate of E12b-P5-stop
  PROBE-E12B-8K) bash $RUNBOOK/scripts/probe_e12var.sh $G 8k;;   # [v2.5.8h] (Q-028): the gate of E12b-P5-8k
  PROBE-E12B-16K) bash $RUNBOOK/scripts/probe_e12var.sh $G 16k;;   # [v2.5.8h] (Q-028): the gate of E12b-P5-16k
  PROBE-P5SELF-4B) bash $RUNBOOK/scripts/probe_p5self.sh $G 4b;;   # [v2.5.8e] the gate of E10d-P5self
  PROBE-P5SELF-06B) bash $RUNBOOK/scripts/probe_p5self.sh $G 06b;;   # [v2.5.9v] (Q-047) the gate of E10c-P5self
  PROBE-NET) bash $RUNBOOK/scripts/probe_net.sh $G;;   # [v2.5.9c] the routes to the model hubs (the CPU lane)
  PROBE-NET2) bash $RUNBOOK/scripts/probe_net2.sh $G;;   # [v2.5.9h] per-connection or per-host limit, parallel Range connections (the CPU lane)
  PROBE-ROLLOUT-U60|PROBE-ROLLOUT-U75|PROBE-ROLLOUT-U85) bash $RUNBOOK/scripts/probe_rollout.sh $G ${T#PROBE-ROLLOUT-};;   # [v2.5.9i] vLLM memory vs E12b step time (one GPU each)
  PROBE-TOOL) bash $RUNBOOK/scripts/probe_tool.sh $G;;   # [v2.5.9b] the tool-call OPD code path on the synthetic samples
  PROBE-2GPU) bash $RUNBOOK/scripts/probe_2gpu.sh $G;;   # [v2.5.9a] the gate of the two-GPU slot (G = a,b from '#gpus PROBE-2GPU 2')
  PROBE-LENSEL) bash $RUNBOOK/scripts/probe_lensel.sh $G;;   # [v2.5.8m] (Q-030): the gate of E10b-P5lenS / L / R
  PROBE-STOPMERGE) bash $RUNBOOK/scripts/probe_stopmerge.sh $G;;   # [v2.5.9l] (Q-038): the gate of E10b-P5stopmerge
  PROBE-TAIL) bash $RUNBOOK/scripts/probe_tail.sh $G;;   # [v2.5.9n] (T4): the gate of E10b-P5tail
  PROBE-POLISH) bash $RUNBOOK/scripts/probe_polish.sh $G;;   # [v2.5.9u] (T2): the gate of E10c-P5-polish
  PROBE-X2507-X1|PROBE-X2507-X2|PROBE-X2507-X4) bash $RUNBOOK/scripts/probe_x2507.sh $G ${T#PROBE-X2507-};;   # [v2.5.8i] (Q-029): the gates of E10d-X1 / X2
  PROBE-P5SELF) bash $RUNBOOK/scripts/probe_p5self.sh $G;;   # v2.5.7v (Q-017): the self-teacher's gate   # v2.5.7o: C1 strict-contract rejudge of the MATH500 readings, CPU only   # v2.5.7g: termination-quality probe, forward passes only (#term lines)
  PROBE-TOP10) bash $RUNBOOK/scripts/probe_top10.sh $G;;   # v2.5.7h: gate of E10b-P5top10 / P5rand10m (unit tests + smokes)
  FA-SPEED-17|FA-SPEED-4B|FA-SPEED-E12|FA-SPEED-TWIN) bash $RUNBOOK/scripts/fa_speed.sh $G $T;;   # v2.5.7c: flash-attn equivalence + speed   # v2.5.7b: 10-step smoke of the 4B student   # v2.5.6o: fused kernels / no gradient checkpointing
  EVAL-*)  [[ $T =~ ^EVAL-[A-Za-z0-9._-]+$ ]] || exit 2; bash $RUNBOOK/scripts/run_queue_eval.sh $G $T;;   # v2.5.4a: #eval lines of queue.txt   # v2.5.3: can flash-attn 2 varlen run here (no E10b directory touched)
  *) echo "NOT_WHITELISTED $T"; exit 2;;
esac
