#!/usr/bin/env bash
# Whitelist dispatcher: run ONE task on ONE GPU. Only the task IDs matched below are accepted; anything else exits 2.
# Every task writes $OPD_ROOT/runs/<task>/.done when it has finished properly; run_tasks.sh treats a missing .done
# after exit as a failure. task.sh stays the parent process (no exec), so run_tasks.sh can see that a task is running.
# Usage: bash task.sh <gpu> <task id>
set -uo pipefail
source "$(dirname "$0")/env.sh"; G=$1; T=$2
# [v2.5.9a] G = one GPU, cpu, or a list a,b (a #gpus task, run_tasks.sh): a list only for the trainings and PROBE-2GPU
[[ $G =~ ^([0-9]+(,[0-9]+){0,3}|cpu)$ ]] || { echo "bad gpu $G"; exit 2; }
case $G in *,*) [[ $T =~ ^(E10[bcde]-|E12[bcde]-|PROBE-2GPU$|PROBE-8B$|PROBE-E12E-8K$|PROBE-E12E-16K$|PROBE-TAIL-8B$) ]] || { echo "$T takes one GPU (got $G)"; exit 2; };; esac
# v2.5.4a: a task whose previous attempt ran out of GPU memory (run_tasks.sh writes runs/<task>/.oom) gets FIXED_MICRO_BSZ=4
# for its own retries, unless the key is set already; nothing else changes
[ -f $OPD_ROOT/runs/$T/.oom ] && [ -z "${FIXED_MICRO_BSZ:-}" ] && { export FIXED_MICRO_BSZ=4; echo "OOM_RETRY $T with FIXED_MICRO_BSZ=4"; }
# [v2.5.9z44] (ops-lead 10-22: PROBE-T32 peaked at 253.9 of 268.6 GB in a 3-step smoke with short answers) E10d-P5-t32 after an
# out of memory (.oom: run_tasks.sh's test, the gate's regex) retries with the actor's optimizer offloaded too (run_arm.sh
# T32_OPT_OFFLOAD=1: speed only, a resumed run's results unchanged), unless the key is set; an event t32_opt_offload_on_retry
[ $T = E10d-P5-t32 ] && [ -f $OPD_ROOT/runs/$T/.oom ] && [ -z "${T32_OPT_OFFLOAD:-}" ] && { export T32_OPT_OFFLOAD=1; echo "OOM_RETRY $T with T32_OPT_OFFLOAD=1"; python3 $RUNBOOK/scripts/control.py event $T t32_opt_offload_on_retry "the previous attempt ran out of GPU memory: the actor's optimizer offloaded" > /dev/null 2>&1; }
ARMS="P2|P3|P4lowlr|P5|P6|P6r|P2noEOS|Pstop|P6rand|P6top5|P4EOSpen|P5top10|P5rand10m"
case $T in
  E12b-P5tail-16k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z92] (Q-074d, Fable's pre-registration research_program 3g) E12b-P5-16k's recipe from E10b-P5tail s50, #gpus 4; above E12b-* (z84b)
  E10d-P5tail-s43) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z92] (Q-075c, 3g) the 4B tail swap with data.seed 43, one GPU; gated by PROBE-TAIL and PROBE-TAIL-4B
  E12c-P5tail-8k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z92] (Q-074f, 3g) E12c-P5-8k's recipe from E10c-P5tail-c50 s50 (its global_step_30), one GPU; above E12c-*
  E12e-P5tail-8k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z92] (Q-074e, 3g) the 8B method chain: the 8K RL recipe from E10e-P5tail s50, #gpus 4; gated by PROBE-E12B-8K / PROBE-E12E-8K as Q-081, starts once E10e-P5tail's s50 is uploaded (no wait for Q-075b's readings)
  E10b-P5-t32) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z91] (Q-082, Fable's pre-registration research_program 3e) E10d-P5-t32's recipe for the 1.7B student (the 32B thinking teacher offloaded, vLLM 0.4, the optimizer offloaded as PROBE-T32 needed), one GPU; gated by PROBE-T32
  E10e-P5tail) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z91] (Q-075b, research_program 3g) E10b-P5tail's recipe for the 8B student taught by itself (self + tail swap), #gpus 4; gated by PROBE-8B, PROBE-TAIL and PROBE-TAIL-8B
  E12b-P5tail-8k-s43) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z91] (Q-074c, 3g) the method's chain, second seed: the 8K RL arm from E10b-P5tail-s43 s50; above E12b-* (z84b)
  E12e-P5self-8k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z91] (Q-081, 3e) the 8B window arm: the 8K RL recipe from E10e-P5self s50, #gpus 4; gated by PROBE-E12B-8K and PROBE-E12E-8K
  E12e-P5tail-16k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z97] (Q-074j, Fable's research_program 3h, a conditional arm) the 8B method chain at the 16K window: E12e's 16K recipe from E10e-P5tail s50, #gpus 4; gated by PROBE-E12B-16K / PROBE-E12E-16K, starts once E10e-P5tail's s50 is uploaded
  PROBE-E12E-16K) PEV_LADDER=E12e bash $RUNBOOK/scripts/probe_e12var.sh $G 16k;;   # [v2.5.9z97] (Q-074j): its gate, the 8B 16K 3-step smoke of E12e-P5tail-16k on four GPUs (#gpus 4)
  PROBE-E12E-8K) PEV_LADDER=E12e bash $RUNBOOK/scripts/probe_e12var.sh $G 8k;;   # [v2.5.9z91] (Q-081): the gate of E12e-P5self-8k, a 4-GPU 3-step smoke (#gpus 4)
  PROBE-TAIL-8B) PTL_PROBE=PROBE-TAIL-8B PTL_ARM=E10e-P5tail bash $RUNBOOK/scripts/probe_tail.sh $G;;   # [v2.5.9z91] (Q-075b): the 8B tail arm's smoke, four GPUs (#gpus 4)
  E10b-P5tail-s44) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z93] (Q-074g, Fable's pre-registration research_program 3g) tail-swap from the base, data.seed 44 (the third start), one GPU; gated by PROBE-TAIL
  E12b-P5tail-8k-s44) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z93] (Q-074g) the 8K RL arm from E10b-P5tail-s44 s50, one GPU; above E12b-*
  E10b-P5tail-s43) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z41] (padding ruling 10-22) tail-swap, data.seed 43
  E10b-P5route-inv) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z45] (Q-063) route reversed: easy -> thinking teacher
  E10b-P5selfroute) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z82] (Q-072) route with the probe's out-of-fold labels
  E10b-P5mix60) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z47] mix with the thinking teacher on p = 0.6 of the prompts
  E16-X5|E16-X5e) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z46] (Q-057) a student without a thinking mode (Qwen2.5-Math-1.5B-Instruct <- R1-Distill-1.5B)
  E10b-P5-8k|E10b-P5smerge-8k) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z66] (Q-059) P5 / stop-merged P5 with 8K rollouts
  E10b-P5bud2k|E10b-P5bud3k) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z73] (Q-061) the tail swap at min(the first box, N)
  E10b-P5softmix50) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z83] (Q-073) the two readings soft-mixed (0.5 / 0.5) at every position
  E10b-P5-dense|E10b-P5-akl) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z76] (Q-065) the per-token KL over the vocabulary
  E10b-P5-akl2|E10b-P5-fkl|E10b-P5-dense-k256|E10b-P5-akl-mu03|E10b-P5-akl-mu07|E10b-P5self-akl|E10b-P5selfroute-c100) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z85] (Q-079) the AKL series; (Q-072b) selfroute continued to s100
  E10e-P5self) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z68] (Q-068) the 8B student taught by itself, four GPUs (#gpus 4)
  PROBE-8B) bash $RUNBOOK/scripts/probe_8b.sh $G;;   # [v2.5.9z68] (Q-068): the gate of E10e-P5self, a 4-GPU 3-step smoke
  E10b-*)  [[ $T =~ ^E10b-(($ARMS)(-s(43|44))?|(P2|P5)-t14|P5think|P5self|P5self-s43|P2self|P5len[SLR]|P5stopmerge|P5tail|P5route|P5mix|P5mix75|P5mix90)$ ]] || exit 2; bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # v2.5.7p: -t14; v2.5.7r: P5think; v2.5.7v: P5self; [v2.5.8e] P5self-s43
  E10d-P5tail) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z84a] (Q-075) E10b-P5tail's recipe on the 4B student
  E10d-P5-akl) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z85] (Q-079) AKL for the 4B student
  E10c-P5tail) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z70] (Q-055) the 0.6B P5 student with the tail-swapped 8B teacher, 20 steps from E10c-P5-fix4 s150
  E10c-P5tail-c50) bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z82] (Q-055b) E10c-P5tail continued: 30 steps from its s20 weights
  E10c-*)  [[ $T =~ ^E10c-(($ARMS)-(fix4|dyn16k|dyn6144)(-s43)?|P5-polish|P5self)$ ]] || exit 2; bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # v2.5.6f: 0.6B, V3-c1; v2.5.7g: dyn6144, -s43
  E10d-*)  [[ $T =~ ^E10d-((P2|P5)(-s43)?|P5self(-s43)?|X1|X2|X4|P5-t32)$ ]] || exit 2; bash $RUNBOOK/scripts/run_arm.sh $G $T;;   # [v2.5.9z77] P5self-s43   # v2.5.7b: 4B student, V3-c5; v2.5.7o: -s43
  E11b-*)  [[ $T =~ ^E11b-(P0|($ARMS)(-s(43|44))?|(P2|P5)-t14)$ ]] || exit 2; bash $RUNBOOK/scripts/run_eval.sh $G $T;;   # v2.5.6c: seed arms too
  SFT-P1)  bash $RUNBOOK/scripts/run_sft.sh $G $T;;
  E12b-P5-8k-s44) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z41] (padding ruling 10-22) the 8K arm from E10b-P5-s44 s150
  E12b-P5-prog) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z43] (Q-058) the 4K recipe from E12b-P5-8k s25 (shrink the window)
  E12b-P5-prog-up) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z43] (Q-058) the 8K recipe from E12b-P5-s44 s25 (grow the window)
  E12b-P5tail-8k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z84a] (Q-074) the 8K RL arm from E10b-P5tail s50 (the method's chain); [v2.5.9z84b] above E12b-*, which took it first (exit 2, no log line)
  E12b-P5-16k-s44) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z84] (Q-076) the 1.7B 16K arm from E10b-P5-s44 s150, #gpus 2; [v2.5.9z84b] above E12b-*
  E12b-*)  [[ $T =~ ^E12b-(P0|P1|$ARMS|P5-s4[34]|P2-s43|P5self|P5self-s43|P5-kl|P5-stop|P5-stop2|P5-8k|P5-16k|P5-r3|P5-16k-kl)$ ]] || exit 2; bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # v2.5.6j: P5-s43; v2.5.7n: P5-s44; v2.5.7o: P2-s43; [v2.5.8a] P5self; [v2.5.8b] P5-kl; [v2.5.8c] P5-stop; [v2.5.9f] P5-stop2
  E12c-P5-8k|E12c-P5-16k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z85] (Q-077) the 0.6B window arms from E10c-P5-fix4 s150 (above E12c-*)
  E12c-*)  [[ $T =~ ^E12c-(P0|P5)$ ]] || exit 2; bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.8a] the 0.6B GRPO ladder (Q-023)
  E12d-P5-8k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z60] (Q-066) E12d-P5 (4B, from E10d-P5 s150) at the 8K limit, 50 steps, gated by PROBE-E12B-8K
  E12d-P5-8k-s43) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z77] (Q-071) the 8K arm from E10d-P5-s43 s150
  E12d-P5-16k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z67] (Q-069) E12d-P5 (4B, from E10d-P5 s150) at the 16K limit on two GPUs (#gpus 2), 50 steps, gated by PROBE-E12B-16K
  E12d-P5-16k-s43) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z84] (Q-076) the 4B 16K arm from E10d-P5-s43 s150, #gpus 4
  E12d-P5tail-16k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z95] (Q-074h, Fable's research_program 3g) the 4B method chain at the 16K window: E12d-P5-16k's recipe from E10d-P5tail s50, #gpus 4; above E12d-*
  E12d-P5tail-8k-s43) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z97] (Q-074i, Fable's research_program 3h) the 4B method chain's second seed: E12d-P5tail-8k's recipe from E10d-P5tail-s43 s50 (Q-075c), #gpus 2; above E12d-*
  E12d-P5tail-8k) bash $RUNBOOK/scripts/run_e12.sh $G $T;;   # [v2.5.9z90] (Q-074b, Fable's pre-registration research_program 3e) the method's chain at 4B: the 8K RL arm from E10d-P5tail s50, #gpus 2; above E12d-* (z84b)
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
  PROBE-T32) bash $RUNBOOK/scripts/probe_t32.sh $G;;   # [v2.5.9z9] (Q-053): the gate of E10d-P5-t32
  PROBE-T14) bash $RUNBOOK/scripts/probe_t14.sh $G;;   # v2.5.7p (Q-015): the 14B teacher's gate
  PROBE-P5THINK) bash $RUNBOOK/scripts/probe_p5think.sh $G;;   # v2.5.7r (Q-016): the thinking student's gate
  PROBE-P2SELF) bash $RUNBOOK/scripts/probe_p2self.sh $G;;   # v2.5.7zg (Q-019): the P2 self-teacher's gate
  PROBE-E12C) bash $RUNBOOK/scripts/probe_e12c.sh $G;;   # [v2.5.8a] (Q-023): the gate of E12c (a 3-step smoke of E12c-P0)
  PROBE-E12B-KL) bash $RUNBOOK/scripts/probe_e12var.sh $G kl;;   # [v2.5.8b] (Q-020): the gate of E12b-P5-kl
  PROBE-E12B-STOP2) bash $RUNBOOK/scripts/probe_e12var.sh $G stop2;;   # [v2.5.9f] (T1): the gate of E12b-P5-stop2
  PROBE-E12B-R3) bash $RUNBOOK/scripts/probe_e12var.sh $G r3;;   # [v2.5.9w] (Q-039) the gate of E12b-P5-r3
  PROBE-E12B-STOP) bash $RUNBOOK/scripts/probe_e12var.sh $G stop;;   # [v2.5.8c] (Q-021): the gate of E12b-P5-stop
  PROBE-E12B-8K) bash $RUNBOOK/scripts/probe_e12var.sh $G 8k;;   # [v2.5.8h] (Q-028): the gate of E12b-P5-8k
  PROBE-E12B-16K) bash $RUNBOOK/scripts/probe_e12var.sh $G 16k;;   # [v2.5.8h] (Q-028): the gate of E12b-P5-16k
  PROBE-E12C-8K) PEV_LADDER=E12c bash $RUNBOOK/scripts/probe_e12var.sh $G 8k;;   # [v2.5.9z85] (Q-077): the gate of E12c-P5-8k
  PROBE-E12C-16K) PEV_LADDER=E12c bash $RUNBOOK/scripts/probe_e12var.sh $G 16k;;   # [v2.5.9z85] (Q-077): the gate of E12c-P5-16k
  PROBE-P5SELF-4B) bash $RUNBOOK/scripts/probe_p5self.sh $G 4b;;   # [v2.5.8e] the gate of E10d-P5self
  PROBE-P5SELF-06B) bash $RUNBOOK/scripts/probe_p5self.sh $G 06b;;   # [v2.5.9v] (Q-047) the gate of E10c-P5self
  PROBE-NET) bash $RUNBOOK/scripts/probe_net.sh $G;;   # [v2.5.9c] the routes to the model hubs (the CPU lane)
  PROBE-NET2) bash $RUNBOOK/scripts/probe_net2.sh $G;;   # [v2.5.9h] per-connection or per-host limit, parallel Range connections (the CPU lane)
  BUILD-VENV-VL) bash $RUNBOOK/scripts/build_venv_vl.sh;;   # [v2.5.9y1] (Q-051) venv-vl for the Qwen3-VL arms (the CPU lane)
  BUILD-VENV-IFEVAL) bash $RUNBOOK/scripts/build_venv_ifeval.sh;;   # [v2.5.9z16] venv-ifeval: Google's IFEval checker, grading only (the CPU lane)
  CORPUS-RECORD) bash $RUNBOOK/scripts/dryrun_json.sh --all;;   # [v2.6 S0] (Fable F7, ops-lead O7: the only structural exception during the v2.6 migration) the golden corpus of 2.5.9's behaviour, CPU lane
  HEARTBEAT-UNSTICK) bash $RUNBOOK/scripts/heartbeat_unstick.sh;;   # [v2.5.9z88f] (Fable / ops-lead 10-26) the watchdog's gave_up marker removed once no sandbox process is left, CPU lane
  READINGS-EXTRA) bash $RUNBOOK/scripts/readings_extra.sh;;   # [v2.5.9x1] MATH-500 splits and the frontier fill (the CPU lane)
  PROBE-VLLM-VL) bash $RUNBOOK/scripts/probe_vllm_vl.sh $G;;   # [v2.5.9y1] venv-vl's GPU gate: vLLM 0.11.0 + Qwen3-VL-2B on one Geometry3K image
  PROBE-ROLLOUT-U60|PROBE-ROLLOUT-U75|PROBE-ROLLOUT-U85) bash $RUNBOOK/scripts/probe_rollout.sh $G ${T#PROBE-ROLLOUT-};;   # [v2.5.9i] vLLM memory vs E12b step time (one GPU each)
  PROBE-TOOL) bash $RUNBOOK/scripts/probe_tool.sh $G;;   # [v2.5.9b] the tool-call OPD code path on the synthetic samples
  PROBE-2GPU) bash $RUNBOOK/scripts/probe_2gpu.sh $G;;   # [v2.5.9a] the gate of the two-GPU slot (G = a,b from '#gpus PROBE-2GPU 2')
  PROBE-LENSEL) bash $RUNBOOK/scripts/probe_lensel.sh $G;;   # [v2.5.8m] (Q-030): the gate of E10b-P5lenS / L / R
  PROBE-STOPMERGE) bash $RUNBOOK/scripts/probe_stopmerge.sh $G;;   # [v2.5.9l] (Q-038): the gate of E10b-P5stopmerge
  PROBE-WDIFF) bash $RUNBOOK/scripts/probe_wdiff.sh cpu;;   # [v2.5.9z18] (Q-031): per-layer weight differences of five 4B models vs Qwen3-4B (the CPU lane)
  PROBE-STOPMERGE-UNIT) PSM_UNIT_ONLY=1 bash $RUNBOOK/scripts/probe_stopmerge.sh cpu;;   # [v2.5.9z6] its unit tests + the installed hooks' guards only (the CPU lane)
  PROBE-P5-8K) bash $RUNBOOK/scripts/probe_w8k.sh $G;;   # [v2.5.9z66] (Q-059): the gate of E10b-P5-8k, a 3-step smoke
  PROBE-SMERGE-8K) PSM_PROBE=PROBE-SMERGE-8K PSM_ARM=E10b-P5smerge-8k PSM_MAXRESP=8192 PSM_PY=$OPD_ROOT/venv-fa/bin/python bash $RUNBOOK/scripts/probe_stopmerge.sh $G;;   # [v2.5.9z66]: the gate of E10b-P5smerge-8k
  TOKCHECK-X5) bash $RUNBOOK/scripts/tok_check.sh;;
  PROBE-MIXP) bash $RUNBOOK/scripts/probe_mixp.sh;;   # [v2.5.9z38] (Q-060): the gate of E10b-P5mix75 / -P5mix90, the coin at p (the CPU lane)
  PROBE-PROG) bash $RUNBOOK/scripts/probe_prog.sh;;   # [v2.5.9z43] (Q-058): the gate of E12b-P5-prog / -prog-up, the continuation file and the starts (the CPU lane)
  VENV-VL-NVRTC) bash $RUNBOOK/scripts/venv_vl_nvrtc.sh;;   # [v2.5.9z31] (ops-lead, Q-051): CUDA 12.9's NVRTC into venv-vl (sm_103), the CPU lane   # [v2.5.9z24] (ops-lead, Q-057): vocab / special tokens / chat template of the X5 trio once pulled (the CPU lane)
  PROBE-TAIL) bash $RUNBOOK/scripts/probe_tail.sh $G;;   # [v2.5.9n] (T4): the gate of E10b-P5tail
  PROBE-TAIL-4B) PTL_PROBE=PROBE-TAIL-4B PTL_ARM=E10d-P5tail bash $RUNBOOK/scripts/probe_tail.sh $G;;   # [v2.5.9z84a] (Q-075): the 4B tail arm's smoke
  PROBE-SOFTMIX) PTL_PROBE=PROBE-SOFTMIX PTL_ARM=E10b-P5softmix50 PTL_SOFTMIX=0.5 bash $RUNBOOK/scripts/probe_tail.sh $G;;   # [v2.5.9z83] (Q-073): the soft mix's gate
  PROBE-TAIL-BUD) PTL_PROBE=PROBE-TAIL-BUD PTL_ARM=E10b-P5bud2k PTL_BUD=2048 bash $RUNBOOK/scripts/probe_tail.sh $G;;   # [v2.5.9z73] (Q-061): the budget arms' gate
  PROBE-POLISH) bash $RUNBOOK/scripts/probe_polish.sh $G;;   # [v2.5.9u] (T2): the gate of E10c-P5-polish
  PROBE-DIFF17) bash $RUNBOOK/scripts/probe_diff17.sh $G;;   # [v2.5.9x] (Q-041) difficulty labels, the 1.7B base
  PROBE-DIFF17-4B) bash $RUNBOOK/scripts/probe_diff17.sh $G 4b;;   # [v2.5.9x] the k_4b column (Q-050), a pad
  PROBE-DIFFPROBE) bash $RUNBOOK/scripts/probe_diffprobe.sh $G;;   # [v2.5.9z74] (Q-062) the difficulty probe on the 1.7B base's hidden states
  PROBE-SELFROUTE) bash $RUNBOOK/scripts/probe_selfroute.sh $G;;   # [v2.5.9z82] (Q-072) the gate of E10b-P5selfroute: the probe's labels and their check
  PROBE-PSTAR) bash $RUNBOOK/scripts/probe_pstar.sh $G;;   # [v2.5.9z75] (Q-064) p* from the 8B teacher's sequence log-probs
  PROBE-DENSE) bash $RUNBOOK/scripts/probe_dense.sh $G;;   # [v2.5.9z76] (Q-065) the dense arms' gate
  PROBE-DENSE-4B) PDN_PROBE=PROBE-DENSE-4B PDN_P5=E10d-P5 PDN_ARM=E10d-P5-akl PDN_MODE=akl bash $RUNBOOK/scripts/probe_dense.sh $G;;   # [v2.5.9z85] (Q-079) the 4B AKL arm's smoke
  PROBE-ROUTE) bash $RUNBOOK/scripts/probe_route.sh $G;;   # [v2.5.9x] the gate of E10b-P5route / -P5mix
  PROBE-ROUTE-INV) PRT_TASK=PROBE-ROUTE-INV PRT_ARM=E10b-P5route-inv PRT_INVERT=1 bash $RUNBOOK/scripts/probe_route.sh $G;;   # [v2.5.9z45] (Q-063) the gate of E10b-P5route-inv: PROBE-ROUTE's checks, the rule reversed
  PROBE-X5) bash $RUNBOOK/scripts/probe_x5.sh $G;;   # [v2.5.9z46] (Q-057) the gate of E16-X5: unit tests, the teacher's completions, its stop probability under two prefixes, a 3-step smoke
  PROBE-VLM) bash $RUNBOOK/scripts/probe_vlm.sh $G;;   # [v2.5.9z] (Q-051) the gate of E15-VLM-P5
  E15-VLM-P2lo|E15-VLM-P5tail-lo) bash $RUNBOOK/scripts/run_vlm.sh $G $T;;   # [v2.5.9z89] (Q-080a / b, Fable's research_program 3f) the VLM retry arms (run_vlm.sh)
  PROBE-VLM-COMPLETE) bash $RUNBOOK/scripts/probe_vlm_complete.sh $G;;   # [v2.5.9z89] (Q-080a) the gate of E15-VLM-P2lo: the Instruct teacher's completion rate
  PROBE-VLM-TAIL) PTL_PROBE=PROBE-VLM-TAIL PTL_ARM=E15-VLM-P5tail-lo PTL_RUNNER=run_vlm.sh PTL_PY=$OPD_ROOT/venv-vl/bin/python PTL_UNIT_MODULES='test_opd_tailswap test_opd_finished' PTL_FINISHED=1 bash $RUNBOOK/scripts/probe_tail.sh $G;;   # [v2.5.9z89] (Q-080b) the gate of E15-VLM-P5tail-lo
  DIAG-VLM-TRUNC) bash $RUNBOOK/scripts/diag_vlm_trunc.sh;;
  VERIFY-PRIV-LFS) bash $RUNBOOK/scripts/verify_priv_lfs.sh;;   # [v2.5.9z96a] (Fable 10-27 02:5x UTC) the CPU lane, read-only: the private repository's 759 raw jsonl checked on the node   # [v2.5.9z89] (Q-080's diagnostic) the CPU lane: the truncated VLM answers' classes
  E15-VLM-P5) bash $RUNBOOK/scripts/run_vlm.sh $G $T;;   # [v2.5.9z] (Q-051) P5's OPD for Qwen3-VL-2B-Instruct from Qwen3-VL-4B-Thinking ([v2.5.9z28], was 8B), venv-vl
  PROBE-X2507-X1|PROBE-X2507-X2|PROBE-X2507-X4) bash $RUNBOOK/scripts/probe_x2507.sh $G ${T#PROBE-X2507-};;   # [v2.5.8i] (Q-029): the gates of E10d-X1 / X2
  PROBE-P5SELF) bash $RUNBOOK/scripts/probe_p5self.sh $G;;   # v2.5.7v (Q-017): the self-teacher's gate   # v2.5.7o: C1 strict-contract rejudge of the MATH500 readings, CPU only   # v2.5.7g: termination-quality probe, forward passes only (#term lines)
  PROBE-TOP10) bash $RUNBOOK/scripts/probe_top10.sh $G;;   # v2.5.7h: gate of E10b-P5top10 / P5rand10m (unit tests + smokes)
  FA-SPEED-17|FA-SPEED-4B|FA-SPEED-E12|FA-SPEED-TWIN) bash $RUNBOOK/scripts/fa_speed.sh $G $T;;   # v2.5.7c: flash-attn equivalence + speed   # v2.5.7b: 10-step smoke of the 4B student   # v2.5.6o: fused kernels / no gradient checkpointing
  EVAL-*)  [[ $T =~ ^EVAL-[A-Za-z0-9._-]+$ ]] || exit 2; bash $RUNBOOK/scripts/run_queue_eval.sh $G $T;;   # v2.5.4a: #eval lines of queue.txt   # v2.5.3: can flash-attn 2 varlen run here (no E10b directory touched)
  *) echo "NOT_WHITELISTED $T"; exit 2;;
esac
