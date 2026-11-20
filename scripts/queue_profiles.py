"""[v2.5.9z137] (lab WO-001, the v3 workflow 11-09 section 7.2) newnode-v2: one queue per machine size, P0 first.

The next machine's size is unknown, so tasks/queue_<profile>.txt is written here for five profiles from one row source,
tasks/rows_newnode_v1.txt (z136's live queue, kept unchanged for provenance), plus the rows WO-001 adds (the same-machine
anchors, the 1.7B chain's AIME 32K anchor, the 8B base model's thinking-mode TERM4). tasks/queue.txt is a copy of one of them
(1g until a machine arrives); every node-side reader keeps reading tasks/queue.txt, nothing on the node changes.

  1g    one GPU (the default; a B200 ~180 GB with profiles/140g.env): P0 in series, the trainings interleaved with the readings
        (a, d-train, c, b, e-train, d / e readings, f, g, the AIME anchor), then P1-a; no multi-GPU task
  4g    2-4 GPUs, or 4 under 80 GB: P0 in parallel (gates, then the two trainings, then the readings), then P1-a
  4g80  4 x 80 GB: 4g + P1-b (the 8B chain Q-074e on 4 GPUs, its gate PROBE-E12E-8K)
  8g    8 GPUs under 80 GB: P0 + P1-a, then P2 without 4B training (Q-074f, Q-074g)
  8g80  8 x 80 GB: P0 + P1-a, P1-b on 4 GPUs, P2 fills the idle GPUs in P2 order (Q-074f, Q-074h, Q-074g)
Under 80 GB no 4B and no 8B training (P0 trains 1.7B only, so 1g serves both).

Usage: python3 scripts/queue_profiles.py              write tasks/queue_<profile>.txt for every profile
       python3 scripts/queue_profiles.py --check      exit 1 if a committed profile file differs from what this writes
[v3.0] (Fable 11-17) --select is gone: tasks/queue.txt opens nothing; a node is given its profile with the control verb
`queue <profile>` (rules 2.5.116), which a node's status confirms, instead of one shared file every clone would run.
"""
import os
import sys

RB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(RB, "tasks", "rows_newnode_v1.txt")

# stage -> its task specs (task[:dep+dep]); "src:<task>" copies that task's spec from the row source
STAGES = {
    # P0-a the local gates (PROBE-TAIL = the canary); PROBE-E12B-STOP2 = Q-090's reward smoke
    "p0a": ["src:PROBE-TAIL", "src:PROBE-E12B-8K", "src:PROBE-E12B-STOP2"],
    # P0-c audit FAIL-1: the 1.7B 8K RL-only arm's CSR; + the 8B base model's thinking-mode TERM4 (scoring only, its 32K gens in -pub)
    "p0c": ["src:PROBE-TERM4-NRL8K", "PROBE-TERM4-N8BTHINK", "PROBE-TERM4-NP5H200"],   # [z144] + the chain arm start E10b-P5 s150 on an H200 (Fable 11-11: NRL8K paired on one machine)
    # P0-b same-machine anchors: the 1.7B chain s50 and the 1.7B RL-only s50 at 16K T0.6 / T1.0, the 4B chain s50 at 16K (table 1's CI)
    "p0b": ["EVAL-e12b-P5tail-8k-nn:E12b-P5tail-8k", "EVAL-e12b-P5-8k-nn:E12b-P5-8k", "src:EVAL-e12d-P5tail-8k-nn",
            "EVAL-ext32k-4Bp5tail8k-nn:E12d-P5tail-8k"],   # [z146] (Fable D9-5, review A4) + the 4B chain s50's AIME24 / 25 32K on the same machine
    # P0-d Q-091 E12b-P0-8k + its readings (research_program 3s)
    "p0d": ["src:E12b-P0-8k"],
    "p0dr": ["src:EVAL-e12b-P0-8k", "src:EVAL-budA-e12bP0-8k", "src:EVAL-ext32k-17p0l8k", "PROBE-TERM4-N17P0H200", "src:PROBE-TERM4-N091"],
    # [z146] N17P0H200 = the 1.7B base (non-thinking) TERM4 on this machine: Q-091 H3' pairs N091 with it (Fable D9-1)
    # P0-e Q-090 E12b-P5-8k-lenpen + its readings
    "p0e": ["src:E12b-P5-8k-lenpen"],
    "p0er": ["src:EVAL-e12b-P5-8k-lenpen", "src:EVAL-budA-e12bP5-8k-lenpen", "src:EVAL-ext32k-17p5l8klp", "src:PROBE-TERM4-N090"],
    # P0-f Q-074i's three readings (the 4B second start) + its TERM4
    "p0f": ["src:EVAL-e12d-P5tail-8k-s43-nn", "src:EVAL-ext32k-4Bp5tail8k-s43-nn", "src:EVAL-budA-e12dP5tail-8k-s43-nn",
            "src:PROBE-TERM4-N074I"],
    # P0-g Q-075c on one machine: E10d-P5tail s50 and -s43 s50 at 16K and 32K, the s43 TERM4 (H2)
    "p0g": ["src:EVAL-e10d-P5tail-nn", "src:EVAL-budA-e10dP5tail-nn", "src:EVAL-e10d-P5tail-s43-nn", "src:EVAL-budA-e10dP5tail-s43-nn",
            "src:PROBE-TERM4-N075C"],
    # P0 tail: the 1.7B chain s50 AIME24 / 25 32K on the same machine (Q-090 H4 compares with the chain at AIME 32K)
    "p0x": ["EVAL-ext32k-17p5tail8k-nn:E12b-P5tail-8k"],
    # [z146] Q-093 (Fable D10; lab/registry/Q-093.md): 8K GRPO in the thinking template from the base; its gate, the arm, then the
    # 16K readings and the TERM4 first, budA 32K and AIME 32K last; one GPU (no #gpus line)
    # [z148] (Fable 13:5x; registry rows_20261111 13:5x) Q-090 / Q-093 moved to B200s at s25 (Runming 12:4x / 13:0x): their pairing
    # references re-read on a B200 - the chain s50 16K, the 1.7B base thinking 16K (Q-093's start) and its TERM4, E10b-P5 s150's
    # TERM4 (Q-090's start); after the arm, before its own readings; each B200 keeps its own rows (the others paused there)
    # [z152] (WO-005; Fable D12-C / D15; registry Q-093 判读补充, rows_20261111) each B200 reads its own arm's 16K + TERM4 first, then its
    # references and the new rows: us-6 (Q-093) the base thinking 16K + N17THB2, Q-093 s25 16K, N093FT (Q-093 s50 on the base thinking
    # text); us-3 (Q-090, whose 16K / N090 are p0er, earlier) the chain s50 -b2, the RL-only s50 -nn-b2, NP5B2, NN2S0 (N2 step 0: Tail-Swap
    # s50 on P5 s150's text), N090FT (Q-090 s50 on the same text); budA / AIME last. The shares pick each machine's rows.
    "p1q": ["PROBE-E12B-P0THINK", "E12b-P0think-8k:PROBE-E12B-P0THINK",
            "EVAL-e12b-P0think-8k:E12b-P0think-8k", "PROBE-TERM4-N093:EVAL-e12b-P0think-8k",
            "EVAL-think16k-17base-b2", "PROBE-TERM4-N17THB2:EVAL-think16k-17base-b2",
            # [z152d] (Fable 18:2x) CONDITIONAL, kept out of every share KEEP (paused): Q-093 s25 here and the s43 chain's s25 / s50 below are read
            # only if Q-093's 16K length ratio to the chain falls outside [0.85, 1.15] (registry Q-093 判读补充); the lead resumes them then.
            # [z154] (Fable D25) the s43 chain's s50 row is unconditional now: Q-093-s43's same-machine reference
            "EVAL-e12b-P0think-8k-s25:E12b-P0think-8k", "PROBE-TERM4-N093FT:EVAL-think16k-17base-b2+E12b-P0think-8k",
            "EVAL-e12b-P5tail-8k-nn-b2:E12b-P5tail-8k", "EVAL-e12b-P5-8k-nn-b2:E12b-P5-8k",
            # [z152c] (Q-093 判读补充: the chain's s25 point; seed 42 never saved s25, so the s43 chain's s25 and s50 on a B200, pending Fable)
            "EVAL-e12b-P5tail-8k-s43-s25-b2:E12b-P5tail-8k-s43", "EVAL-e12b-P5tail-8k-s43-s50-b2:E12b-P5tail-8k-s43",
            "PROBE-TERM4-NP5B2", "PROBE-TERM4-NN2S0:PROBE-TERM4-NP5B2", "PROBE-TERM4-N090FT:PROBE-TERM4-NP5B2+E12b-P5-8k-lenpen",
            "EVAL-budA-e12bP0think-8k:E12b-P0think-8k", "EVAL-ext32k-17p0think8k:E12b-P0think-8k"],
    # [z152] N5 (lab/registry/N5.md; Fable D13 / D15, the lead 11-11 17:4x: en-2 first) the same-machine frontier: the thinking mode
    # at 16K on MATH-500 (n 4) and AIME24 / 25 (n 16), 1.7B and 4B, each forced to answer at 2K / 4K / 8K / 16K (#forced, forced_stop.py
    # budB, on the task's own reading); the 4B Instruct-2507 point at 16K. All zero-training readings, one GPU each, the en H200s
    "pn5": ["EVAL-n5-17think", "EVAL-n5-4Bthink", "EVAL-n5-17think-aime", "EVAL-n5-4Bthink-aime", "EVAL-n5-4Binst2507"],
    # P1-a the 2 x 2 beyond arms + readings + TERM4
    "p1a": ["src:PROBE-FRET", "src:E10b-P5fret", "src:E10b-P5topd", "src:E10b-P5fret-topd", "src:EVAL-e10b-P5fret", "src:EVAL-e10b-P5topd",
            "src:EVAL-e10b-P5fret-topd", "src:PROBE-TERM4-FRET"],
    # [z149] (Q-093b, Fable 14:2x; lab/registry/Q-093b.md) Q-093's recipe on the 4B base: the thinking template, 8K GRPO, 50 steps, four GPUs
    # on venv-fa; its 4-GPU gate, the arm, then the 16K readings and the TERM4 first, budA 32K and AIME 32K last (80 GB profiles only)
    "p1q4": ["PROBE-E12D-P0THINK", "E12d-P0think-8k:PROBE-E12D-P0THINK", "EVAL-e12d-P0think-8k:E12d-P0think-8k",
             "PROBE-TERM4-N093B:EVAL-e12d-P0think-8k",
             # [z154] (Q-093c.md 读数; Fable D25) Q-093b's same-machine references after its own 16K + TERM4: the 4B chain s50 16K and the
             # 4B base thinking 16K + its TERM4, re-read beside it
             "EVAL-e12d-P5tail-8k-nn-b2:E12d-P5tail-8k", "EVAL-think16k-4Bbase-b2", "PROBE-TERM4-N4BTHB2:EVAL-think16k-4Bbase-b2",
             "EVAL-budA-e12dP0think-8k:E12d-P0think-8k", "EVAL-ext32k-4p0think8k:E12d-P0think-8k"],
    # [z154] Q-093c (lab/registry/Q-093c.md; Fable D25 06:1x) E12e-P0think-8k: the 8B base, thinking template, 8K GRPO, 50 steps from data
    # position 0, all 8 GPUs; its gate, the arm, its 16K + TERM4, the same-machine 8B base thinking 16K + TERM4, budA, AIME last
    "p1q8": ["PROBE-E12E-P0THINK", "E12e-P0think-8k:PROBE-E12E-P0THINK", "EVAL-e12e-P0think-8k:E12e-P0think-8k",
             "PROBE-TERM4-N093C:EVAL-e12e-P0think-8k", "EVAL-think16k-8Bbase-b2", "PROBE-TERM4-N8BTHB2:EVAL-think16k-8Bbase-b2",
             "EVAL-budA-e12eP0think-8k:E12e-P0think-8k", "EVAL-ext32k-8p0think8k:E12e-P0think-8k",
             # [z154] (Fable D26) the 8B base thinking AIME 32K on the same machine; after the training (a reference never blocks it).
             # EVAL-think16k-8Bbase-b2 and this one keep the names of the lead's ad hoc runs on me-8a, so their .done is seen
             "EVAL-ext32k-8Bthink-b2:E12e-P0think-8k"],
    # [z154] Q-093-s43: E12b-P0think-8k with data.seed 43, all 8 GPUs (its own 8-GPU gate); its 16K + TERM4, then budA / AIME; the chain s43's
    # s50 reference is p1q's EVAL-e12b-P5tail-8k-s43-s50-b2 (unconditional since D25); + the seed-42 chain s50's AIME 32K with z145
    # outcomes, to pair with Q-093's AIME
    "p1qs43": ["PROBE-E12B-P0THINK-8G", "E12b-P0think-8k-s43:PROBE-E12B-P0THINK-8G", "EVAL-e12b-P0think-8k-s43:E12b-P0think-8k-s43",
               "PROBE-TERM4-N093S43:EVAL-e12b-P0think-8k-s43", "EVAL-budA-e12bP0think-8k-s43:E12b-P0think-8k-s43",
               "EVAL-ext32k-17p0think8k-s43:E12b-P0think-8k-s43", "EVAL-ext32k-17p5tail8k-oc:E12b-P5tail-8k"],
    # [z156] (lab WO-012; Fable D50) Q-093b-s43: E12d-P0think-8k with data.seed 43, all 8 GPUs (its own 8-GPU gate: Q-093b's passed on
    # 4 x B200, this one smokes the s43 arm on its H200s); the gate and the training first, then Q-093b's readings in Q-093b's order
    "p1q4s43": ["PROBE-E12D-P0THINK-8G", "E12d-P0think-8k-s43:PROBE-E12D-P0THINK-8G", "EVAL-e12d-P0think-8k-s43:E12d-P0think-8k-s43",
                "PROBE-TERM4-N093BS43:EVAL-e12d-P0think-8k-s43", "EVAL-budA-e12dP0think-8k-s43:E12d-P0think-8k-s43",
                "EVAL-ext32k-4p0think8k-s43:E12d-P0think-8k-s43"],
    # [z156b] (Fable 20:3x) H200 same-machine references under NEW names (the -b2 names hold the B200 results in the ops repo, a rerun
    # under them would overwrite): the 8B base thinking 16K + its TERM4, and the 8B chain's s50 (E12e-P5tail-8k-nn's global_step_25 =
    # the chain's s50, as the e12e_P5tail-8k_s50 rows read it) at 16K T0.6 / T1.0; no training dependency
    "p1h2": ["EVAL-think16k-8Bbase-h2", "PROBE-TERM4-N8BTHH2:EVAL-think16k-8Bbase-h2", "EVAL-e12e-P5tail-8k-nn-h2"],
    # [z157] (Fable D87 11-14 21:3x, registry Q-093c 11-14 addendum) B300 same-machine references under NEW names (-b3; the -b2 / -h2
    # names hold the B200 / H200 results in the ops repo): for Q-093b-s43 the 4B chain's s50 + the 4B base thinking 16K (+ TERM4), for
    # Q-093c the 8B chain's s50 (E12e-P5tail-8k-nn global_step_25) + the 8B base thinking 16K (+ TERM4) + its AIME24 / 25 32K, for
    # Q-093-s43 the 1.7B s43 chain's s50 + the 1.7B base thinking 16K (+ TERM4); the AIME row last; no training dependency
    # [z158b] (Fable D95 / D96 / D98) Q-093-s44 (1.7B, data.seed 44) and Q-096 A (the training window 4K / 16K), each arm on 4 GPUs (D98: two
    # 1.7B arms side by side on one 8-GPU node) behind its own 3-step gate; readings as Q-093's / Q-093-s43's; the same-machine reread of
    # Q-093 s50 (EVAL-e12b-P0think-8k-b3 + TERM4) is the pairing partner of all three (no training dependency)
    "p1qs44": ["PROBE-E12B-P0THINK-S44", "E12b-P0think-8k-s44:PROBE-E12B-P0THINK-S44", "EVAL-e12b-P0think-8k-s44:E12b-P0think-8k-s44",
               "PROBE-TERM4-N093S44:EVAL-e12b-P0think-8k-s44", "EVAL-budA-e12bP0think-8k-s44:E12b-P0think-8k-s44",
               "EVAL-ext32k-17p0think8k-s44:E12b-P0think-8k-s44"],
    "p1q096a": ["PROBE-E12B-P0THINK-4K", "E12b-P0think-4k:PROBE-E12B-P0THINK-4K", "EVAL-e12b-P0think-4k:E12b-P0think-4k",
                "PROBE-TERM4-N096A4K:EVAL-e12b-P0think-4k", "PROBE-E12B-P0THINK-16K", "E12b-P0think-16k:PROBE-E12B-P0THINK-16K",
                "EVAL-e12b-P0think-16k:E12b-P0think-16k", "PROBE-TERM4-N096A16K:EVAL-e12b-P0think-16k"],
    "p1b3q093": ["EVAL-e12b-P0think-8k-b3", "PROBE-TERM4-N093B3:EVAL-e12b-P0think-8k-b3"],
    # [z158c] (Fable D94, lab/registry/Q-095.md) the brevity-transfer arms: P5self's recipe with Q-093 s50 as the teacher (A 1.7B, B 4B student),
    # gates on one GPU, arms on 4; readings 16K T0.6 n4 (no_think) + TERM4; the same-machine rereads of the two controls (-b3); the teacher's
    # reread is p1b3q093's EVAL-e12b-P0think-8k-b3
    "p1q095": ["PROBE-P5Q093T", "E10b-P5q093t:PROBE-P5Q093T", "EVAL-e10b-P5q093t:E10b-P5q093t", "PROBE-TERM4-N095A:EVAL-e10b-P5q093t",
               "PROBE-P5Q093T-4B", "E10d-P5q093t:PROBE-P5Q093T-4B", "EVAL-e10d-P5q093t:E10d-P5q093t", "PROBE-TERM4-N095B:EVAL-e10d-P5q093t",
               "EVAL-e10b-P5self-b3", "EVAL-e10d-P5self-b3"],
    # [z158c] (Fable D96, lab/registry/Q-096.md C) out-of-domain readings: the 1.7B base (thinking), the 1.7B chain's s50 (its own non-thinking
    # template) and Q-093 s50 (thinking) on OlympiadBench (674), Minerva Math (272) and MMLU-Pro 1K (non-maths), 16K n4 T0.6
    "p1q096c": ["EVAL-ood-17think-b3", "EVAL-ood-17chain-b3", "EVAL-ood-17q093-b3"],
    # [z158d] (Fable 02:3x, lab/registry/Q-094.md) the prefilled openings on the non-thinking template: T = "Okay" (32313, the self-distilled
    # students' first token, 100 %), N = "We" (1654, the bases' first token, 98.5 / 99.5 %); base and student (E10b-P5self s50 / E10d-P5self
    # s50) under each, both sizes, plus the descriptive cell E10b-P5 s150 (the chain's actual first stage) under T and N; MATH-500 16K T0.6 n4
    "p1q094": ["EVAL-q094-17base-T", "EVAL-q094-17base-N", "EVAL-q094-17self-T", "EVAL-q094-17self-N", "EVAL-q094-4base-T", "EVAL-q094-4base-N",
               "EVAL-q094-4self-T", "EVAL-q094-4self-N", "EVAL-q094-17p5-T", "EVAL-q094-17p5-N",
               # [z158e] (Fable 03:0x, Q-094 check V1) the bases' own non-thinking readings on the same machine (-b3), the same wave
               "EVAL-nothink16k-17base-b3", "EVAL-nothink16k-4Bbase-b3"],
    # [z158f] (Fable D96 / D99, lab/registry/Q-096.md B) Q-093 continued from s50 to s100 (weights only, a fresh optimizer, data from step 51),
    # 4 GPUs behind PROBE-E12B-P0THINK-S44 (the same recipe on 4 GPUs); readings s75 / s100 16K T0.6 n4 + TERM4 on the reading machine, paired
    # with Q-093 s50's reread there (EVAL-e12b-P0think-8k-b3)
    "p1q096b": ["E12b-P0think-8k-c100:PROBE-E12B-P0THINK-S44", "EVAL-e12b-P0think-8k-c100:E12b-P0think-8k-c100",
                "PROBE-TERM4-N096B75:EVAL-e12b-P0think-8k-c100", "PROBE-TERM4-N096BC:EVAL-e12b-P0think-8k-c100"],
    # [z159c] (Fable 11:17 / 11:30, lab/registry/Q-097.md "运行记录与裁决") the released arms: Q-097's D arm E12b-P0think-6k (the base, 6K = the log
    # midpoint of 4K and 8K; W_s = 7K gave no discriminating S arm, so D went back to the registered 6K) and Q-096 A's 12K point E12b-P0think-12k
    # (the base, MAXR 12288), 50 steps each, 4 GPUs behind their own gates
    "p1q097": ["PROBE-E12B-P0THINK-6K", "E12b-P0think-6k:PROBE-E12B-P0THINK-6K", "PROBE-E12B-P0THINK-12K", "E12b-P0think-12k:PROBE-E12B-P0THINK-12K"],
    # [z159c] PAUSED = in no profile (Q-097 11-15 11:17: not trained): A4 (Q-093 s50 + 50 steps at 4K) and the W_s = 7K arm; release = add
    # "p1q097p" to PROFILES["8gs"]
    "p1q097p": ["PROBE-E12B-P0THINK-8K50-4K", "E12b-P0think-8k50-4k:PROBE-E12B-P0THINK-8K50-4K", "PROBE-E12B-P0THINK-7K",
                "E12b-P0think-7k:PROBE-E12B-P0THINK-7K"],
    # [z162] (Fable board v2, 11-15) the RECORD of reading machine R1 (uk-share), as its ad hoc queue ran it (Fable's first reads 10:30-10:44,
    # then /var/tmp/fable_adhoc/queue.txt under Fable / executor C): run_queue_eval's settings (n 4, T0.6, top_p 0.95, top_k 0, 16K), the
    # tags verbatim (so outcomes / TERM4 align); us=p1|p2 = Q-098's prompt baselines, seed=1000 = the independent repeats, budgets= on the
    # forced answers. In NO profile: a record, never dispatched (the readings ran ad hoc); add "p1u1r" to a profile only for a rerun
    "p1u1r": ["EVAL-u1-refs", "EVAL-u1-ood", "EVAL-u1-t4k", "EVAL-u1-q098p", "EVAL-u1-seed", "EVAL-u1-x10", "EVAL-u1-f2:EVAL-u1-refs", "PROBE-TERM4-N17THU1:EVAL-u1-refs", "PROBE-TERM4-N4THU1:EVAL-u1-refs", "PROBE-TERM4-N093U1:EVAL-u1-refs", "PROBE-TERM4-N096A4KU1:EVAL-u1-refs", "PROBE-TERM4-N093BU1:EVAL-u1-refs", "PROBE-TERM4-N095AU1:EVAL-u1-refs", "PROBE-TERM4-NT4KU1:EVAL-u1-t4k"],
    # [z163] (Fable board v7, lab/registry/Q-101.md) E10b-P5selfp2 = E10b-P5self + Q-098's P2 on the TEACHER's prompt only; gate on one GPU
    # (the suffix reached every teacher row and no student prompt), the arm on 4 GPUs
    "p1q101": ["PROBE-P5SELFP2", "E10b-P5selfp2:PROBE-P5SELFP2"],
    # [v3.0] (Fable 11-17; lab/registry/Q-107.md; Runming: a classmate's 8 x H200, maybe two machines, GitHub mode, light on the HF API)
    # h2g: Q-107's own gate (4 GPUs; until v3.12 with the dispatch canary in front). q107: the two 3K window arms from the base, 4 GPUs each, side by side.
    # q107r: the four registered points, one EVAL task each (four GPUs in parallel): MATH-500 16K thinking T0.6 n 4, the 16K forced
    # remedy, outcomes (the stop at </think>) as every reading. q107z: zero-training readings of weights already in the result
    # repository - the direct 4K arm's s25 (owed) and its s50 reread on this machine (the cross-machine anchor: the same weights were
    # read on the B300 reading machine on 11-15). q107z6: the direct 6K arm's s25 (owed; its weights reached the repository on 11-17).
    # [v3.12] (Runming 11-18: "can the 20-30 minutes before the arms start be shortened?") h2g is the gate alone: the arms depend
    # on it and it runs their recipe on their four GPUs with venv-fa, so a separate canary (PROBE-TAIL: one more 3-step smoke and
    # 8 GB of weights to fetch first) adds nothing in front of it; profiles/h200.env says DISPATCH_CANARY=none. h2gb keeps it.
    "h2g": ["PROBE-E12B-P0THINK-3K"],
    "h2gb": ["PROBE-TAIL"],
    "q107": ["E12b-P0think-3k:PROBE-E12B-P0THINK-3K", "E12b-P0think-3k-s44:PROBE-E12B-P0THINK-3K"],
    "q107r": ["EVAL-q107-3k-s50:E12b-P0think-3k", "EVAL-q107-3k-s44-s50:E12b-P0think-3k-s44",
              "EVAL-q107-3k-s25:E12b-P0think-3k", "EVAL-q107-3k-s44-s25:E12b-P0think-3k-s44"],
    "q107z": ["EVAL-q107-4k-s50", "EVAL-q107-4k-s25"],
    "q107z6": ["EVAL-q107-6k-s25"],
    # [v3.14] (Fable 11-19; lab/registry/Q-108.md; the independent reviewer's list of 11-17: out of domain only the direct RL arm was read)
    # q108: OlympiadBench (674 problems, 16K, T0.6, n 4, the 16K forced remedy, outcomes) of the base, the direct 8K / 4K arms, the
    # OPD student (Q-101) and the OPD -> RL arms (8K: three seeds, 4K: two), one EVAL task and one GPU each; inference only
    "q108": ["EVAL-q108-base17", "EVAL-q108-d8k", "EVAL-q108-d4k", "EVAL-q108-opd", "EVAL-q108-c8k", "EVAL-q108-c8k-s4343", "EVAL-q108-c8k-s4444", "EVAL-q108-c4k", "EVAL-q108-c4k-s4444"],
    # [v3.15] (Fable 11-19; lab/registry/Q-109.md, the reviewer's matrix A7) q109g: the 4B chain's own gate (4 GPUs). q109: E12d-P5selfp2-8k = the
    # 8K RL arm from the 4B OPD student E10d-P5selfp2 s50, 4 GPUs. q109z: the same-machine references, zero training (the 4B direct 8K
    # arm s50 and the 4B OPD student s50). q109r: the arm's s50 / s25. (lab/registry/Q-110.md) q110: Q-107's two 3K arms continued from
    # their own s50 to s100, 4 GPUs each (the 3K gate passed on this node on 11-19; run_e12.sh checks it). q110r: their s100 / s75
    "q109g": ["PROBE-E12D-P5SELFP2-8K"],
    "q109": ["E12d-P5selfp2-8k:PROBE-E12D-P5SELFP2-8K"],
    "q110": ["E12b-P0think-3k-c100", "E12b-P0think-3k-s44-c100"],
    "q109z": ["EVAL-q109-d8k-4b", "EVAL-q109-opd-4b"],
    "q110r": ["EVAL-q110-3k-s100:E12b-P0think-3k-c100", "EVAL-q110-3k-s44-s100:E12b-P0think-3k-s44-c100",
              "EVAL-q110-3k-s75:E12b-P0think-3k-c100", "EVAL-q110-3k-s44-s75:E12b-P0think-3k-s44-c100"],
    "q109r": ["EVAL-q109-4b-s50:E12d-P5selfp2-8k", "EVAL-q109-4b-s25:E12d-P5selfp2-8k"],
    # [v3.16] (Fable 11-19; lab/registry/Q-111.md) q111: OlympiadBench (as q108) of the two direct 3K arms at s50 and s100 (s100 = the
    # continued arm's global_step_50), one GPU each; inference only
    "q111": ["EVAL-q111-3k-s100:E12b-P0think-3k-c100", "EVAL-q111-3k-s44-s100:E12b-P0think-3k-s44-c100", "EVAL-q111-3k-s50", "EVAL-q111-3k-s44-s50"],
    # [v3.17] (Fable 11-19; lab/registry/Q-112.md) q112g: the 4B thinking arm's own 4-GPU smoke on this node. q112: E12d-P0think-8k-c100 = the
    # 4B direct 8K arm continued from its own s50 to s100, 4 GPUs. q112r: its s100 / s75 (MATH-500, 16K, the forced remedy, outcomes)
    "q112g": ["PROBE-E12D-P0THINK"],
    "q112": ["E12d-P0think-8k-c100:PROBE-E12D-P0THINK"],
    "q112r": ["EVAL-q112-4b-s100:E12d-P0think-8k-c100", "EVAL-q112-4b-s75:E12d-P0think-8k-c100"],
    # [v3.18] (Fable 11-20; lab/registry/Q-111.md and Q-112.md addenda, Q-113.md) the cloud 8 x H200 that finishes what the borrowed machine
    # left. q111b: Q-111's two s50 readings and the card-type anchor (the s100 of seed 4242 read again). q112za / q112zb: the same-machine
    # references of Q-112 (the 4B chain s50, the 4B direct 8K s50). q113: Minerva Math (272) of the base, the direct 3K arms' s100, the
    # OPD -> RL 4K arms and the direct 4K arm. Everything after the first four readings waits for the 4B gate, so four GPUs stay free for it
    "q111b": ["EVAL-q111-3k-s50", "EVAL-q111-3k-s44-s50", "EVAL-q111-3k-s100-en2"],
    "q112za": ["EVAL-q112-chain-s50"],
    "q112zb": ["EVAL-q112-d8k-s50:PROBE-E12D-P0THINK"],
    "q113": ["EVAL-q113-3k-s100:PROBE-E12D-P0THINK", "EVAL-q113-3k-s44-s100:PROBE-E12D-P0THINK", "EVAL-q113-c4k:PROBE-E12D-P0THINK", "EVAL-q113-c4k-s4444:PROBE-E12D-P0THINK", "EVAL-q113-base17:PROBE-E12D-P0THINK", "EVAL-q113-d4k:PROBE-E12D-P0THINK"],
    # [z160] (Fable D105 / 10:44, lab/registry/Q-097.md T4K) E10b-P5q4kt = Q-095 A's recipe with the 4K arm s50 as the teacher; gate on one GPU, arm on 4
    "p1q097t": ["PROBE-P5Q4KT", "E10b-P5q4kt:PROBE-P5Q4KT"],
    "p1b3": ["EVAL-e12d-P5tail-8k-nn-b3", "EVAL-think16k-4Bbase-b3", "PROBE-TERM4-N4BTHB3:EVAL-think16k-4Bbase-b3",
             "EVAL-e12e-P5tail-8k-nn-b3", "EVAL-think16k-8Bbase-b3", "PROBE-TERM4-N8BTHB3:EVAL-think16k-8Bbase-b3",
             "EVAL-e12b-P5tail-8k-s43-s50-b3", "EVAL-think16k-17base-b3", "PROBE-TERM4-N17THB3:EVAL-think16k-17base-b3",
             "EVAL-ext32k-8Bthink-b3"],
    # P1-b Q-074e the 8B chain from s25 (4 x 80 GB)
    "p1b": ["src:PROBE-E12E-8K", "src:E12e-P5tail-8k-nn", "src:EVAL-e12e-P5tail-8k-nn", "src:EVAL-budA-e12eP5tail-8k-nn",
            "src:EVAL-ext32k-8Bp5tail8k-nn", "src:PROBE-TERM4-N074E"],
    # P2 (only on 8 GPUs, filling idle ones): Q-074f the 0.6B chain, Q-074h the 4B 16K chain (80 GB), Q-074g the 1.7B third start
    "p2f": ["src:PROBE-E12C-8K", "src:EVAL-e12c-P5-8k", "src:E12c-P5tail-8k-nn", "src:EVAL-e12c-P5tail-8k-nn", "src:EVAL-budA-e12cP5tail-8k-nn",
            "src:EVAL-ext32k-06p5tail8k-nn", "src:PROBE-TERM4-N074F"],
    "p2h": ["src:PROBE-E12B-16K", "src:PROBE-E12D-16K", "src:E12d-P5tail-16k", "src:EVAL-e12d-P5tail-16k", "src:EVAL-budA-e12dP5tail-16k",
            "src:EVAL-ext32k-4Bp5tail16k", "src:PROBE-TERM4-N074H"],
    "p2g": ["src:E10b-P5tail-s44", "src:E12b-P5tail-8k-s44", "src:EVAL-tail-s44", "src:EVAL-ext32k-17tail-s44", "src:EVAL-e12b-P5tail-8k-s44",
            "src:EVAL-budA-e12bP5tail-8k-s44", "src:EVAL-ext32k-17p5tail8k-s44"],
}
# [11-09 21:5x, the lead: GPUs are self-paid, no idle GPU; readings and trainings interleaved] one GPU: the Q-091 training first (it
# starts from the base model, so nothing waits on a download), then c and b while the next starts are local, then the Q-090
# training, then the two trainings' readings (their same-machine anchors already there), f, g; v3's a > c > b > d > e > f > g kept
# for the readings. On several GPUs the two trainings take GPUs right after the gates, the readings fill the others.
P0_SERIAL = ["p0a", "p0d", "p0c", "p0b", "p0e", "p0dr", "p0er", "p0f", "p0g", "p0x"]
P0_PARALLEL = ["p0a", "p0d", "p0e", "p0c", "p0b", "p0dr", "p0er", "p0f", "p0g", "p0x"]
# B200 GPU-hours per task (one B200 ~ one B300 card; the B300's measured gen_seconds: 1.7B MATH-500 16K n4 ~0.7 h, 4B ~0.5 h, AIME
# 32K avg@16 0.7-1.6 h a benchmark, budA 32K 0.6-1.2 h; trainings 50 steps on one card from the H800-era 5 h at ~1.9x the H800's
# generation speed; gates from v3's 1.5 H800 GPU-h). Written as a comment over each stage; env-probe's step time replaces them.
EST = {"PROBE-TAIL": 0.3, "PROBE-E12B-8K": 0.3, "PROBE-E12B-STOP2": 0.3, "E12b-P0-8k": 3.0, "PROBE-TERM4-NRL8K": 1.0,
       "PROBE-TERM4-N8BTHINK": 0.3, "PROBE-TERM4-NP5H200": 0.5, "EVAL-e12b-P5tail-8k-nn": 1.2, "EVAL-e12b-P5-8k-nn": 1.2, "EVAL-e12d-P5tail-8k-nn": 0.5,
       "E12b-P5-8k-lenpen": 3.0, "EVAL-e12b-P0-8k": 1.2, "EVAL-budA-e12bP0-8k": 1.0, "EVAL-ext32k-17p0l8k": 1.5, "PROBE-TERM4-N091": 0.3,
       "EVAL-e12b-P5-8k-lenpen": 1.0, "EVAL-budA-e12bP5-8k-lenpen": 0.8, "EVAL-ext32k-17p5l8klp": 1.3, "PROBE-TERM4-N090": 0.3,
       "EVAL-e12d-P5tail-8k-s43-nn": 1.5, "EVAL-ext32k-4Bp5tail8k-s43-nn": 1.3, "EVAL-budA-e12dP5tail-8k-s43-nn": 0.6,
       "PROBE-TERM4-N074I": 0.3, "EVAL-e10d-P5tail-nn": 0.75, "EVAL-budA-e10dP5tail-nn": 1.2, "EVAL-e10d-P5tail-s43-nn": 0.75,
       "EVAL-budA-e10dP5tail-s43-nn": 1.2, "PROBE-TERM4-N075C": 1.0, "EVAL-ext32k-17p5tail8k-nn": 1.5,
       "PROBE-FRET": 0.5, "E10b-P5fret": 2.5, "E10b-P5topd": 2.5, "E10b-P5fret-topd": 2.5, "EVAL-e10b-P5fret": 0.7,
       "EVAL-e10b-P5topd": 0.7, "EVAL-e10b-P5fret-topd": 0.7, "PROBE-TERM4-FRET": 0.5,
       "EVAL-ext32k-4Bp5tail8k-nn": 1.2, "PROBE-TERM4-N17P0H200": 1.0,   # [z146] the 4B AIME pair (B300 ~0.6 h a benchmark); gen + score
       "PROBE-E12B-P0THINK": 0.5, "E12b-P0think-8k": 4.5, "EVAL-e12b-P0think-8k": 1.8, "PROBE-TERM4-N093": 0.3,
       "EVAL-budA-e12bP0think-8k": 1.5, "EVAL-ext32k-17p0think8k": 3.0,
       "EVAL-e12b-P5tail-8k-nn-b2": 1.2, "EVAL-think16k-17base-b2": 1.0, "PROBE-TERM4-N17THB2": 0.3, "PROBE-TERM4-NP5B2": 0.5,
       "PROBE-E12D-P0THINK": 0.6, "E12d-P0think-8k": 7.5, "EVAL-e12d-P0think-8k": 2.0, "PROBE-TERM4-N093B": 0.4,   # [z149] Q-093b (H200 x 4)
       "EVAL-budA-e12dP0think-8k": 1.6, "EVAL-ext32k-4p0think8k": 3.0,
       "EVAL-n5-17think": 1.3, "EVAL-n5-4Bthink": 1.9, "EVAL-n5-17think-aime": 1.5, "EVAL-n5-4Bthink-aime": 2.2, "EVAL-n5-4Binst2507": 0.4,
       "EVAL-e12b-P0think-8k-s25": 1.0, "PROBE-TERM4-N093FT": 0.3, "EVAL-e12b-P5-8k-nn-b2": 1.2, "PROBE-TERM4-NN2S0": 0.3, "PROBE-TERM4-N090FT": 0.3,
       "EVAL-e12b-P5tail-8k-s43-s25-b2": 0.7, "EVAL-e12b-P5tail-8k-s43-s50-b2": 0.7,
       "EVAL-e12d-P5tail-8k-nn-b2": 1.0, "EVAL-think16k-4Bbase-b2": 1.5, "PROBE-TERM4-N4BTHB2": 0.4,   # [z154] Q-093b's references
       "PROBE-E12E-P0THINK": 1.2, "E12e-P0think-8k": 22.0, "EVAL-e12e-P0think-8k": 2.0, "PROBE-TERM4-N093C": 0.5, "EVAL-think16k-8Bbase-b2": 1.5,
       "PROBE-TERM4-N8BTHB2": 0.5, "EVAL-budA-e12eP0think-8k": 2.0, "EVAL-ext32k-8p0think8k": 3.0,   # [z154] Q-093c: 50 x 195 s x 8 GPUs
       "PROBE-E12B-P0THINK-8G": 0.8, "E12b-P0think-8k-s43": 16.0, "EVAL-e12b-P0think-8k-s43": 1.8, "PROBE-TERM4-N093S43": 0.3,
       "EVAL-budA-e12bP0think-8k-s43": 1.5, "EVAL-ext32k-17p0think8k-s43": 3.0, "EVAL-ext32k-17p5tail8k-oc": 1.5, "EVAL-ext32k-8Bthink-b2": 3.0,
       "PROBE-E12D-P0THINK-8G": 0.6, "E12d-P0think-8k-s43": 7.5, "EVAL-e12d-P0think-8k-s43": 2.0, "PROBE-TERM4-N093BS43": 0.4,   # [z156] as Q-093b
       "EVAL-budA-e12dP0think-8k-s43": 1.6, "EVAL-ext32k-4p0think8k-s43": 3.0,
       "EVAL-think16k-8Bbase-h2": 1.5, "PROBE-TERM4-N8BTHH2": 0.5, "EVAL-e12e-P5tail-8k-nn-h2": 1.2,
       "EVAL-e12d-P5tail-8k-nn-b3": 1.0, "EVAL-think16k-4Bbase-b3": 1.5, "PROBE-TERM4-N4BTHB3": 0.4, "EVAL-e12e-P5tail-8k-nn-b3": 1.2,
       "EVAL-think16k-8Bbase-b3": 1.5, "PROBE-TERM4-N8BTHB3": 0.5, "EVAL-e12b-P5tail-8k-s43-s50-b3": 0.7, "EVAL-think16k-17base-b3": 1.0,
       "PROBE-TERM4-N17THB3": 0.3, "EVAL-ext32k-8Bthink-b3": 3.0,
       "PROBE-E12B-P0THINK-S44": 0.4, "E12b-P0think-8k-s44": 8.0, "EVAL-e12b-P0think-8k-s44": 1.8, "PROBE-TERM4-N093S44": 0.3,
       "EVAL-budA-e12bP0think-8k-s44": 1.5, "EVAL-ext32k-17p0think8k-s44": 3.0, "PROBE-E12B-P0THINK-4K": 0.3, "E12b-P0think-4k": 4.0,
       "EVAL-e12b-P0think-4k": 0.9, "PROBE-TERM4-N096A4K": 0.3, "PROBE-E12B-P0THINK-16K": 0.6, "E12b-P0think-16k": 14.0,
       "EVAL-e12b-P0think-16k": 0.9, "PROBE-TERM4-N096A16K": 0.3, "EVAL-e12b-P0think-8k-b3": 1.8, "PROBE-TERM4-N093B3": 0.3,
       "PROBE-P5Q093T": 0.3, "E10b-P5q093t": 3.5, "EVAL-e10b-P5q093t": 0.6, "PROBE-TERM4-N095A": 0.3, "PROBE-P5Q093T-4B": 0.4, "E10d-P5q093t": 4.5,
       "EVAL-e10d-P5q093t": 0.8, "PROBE-TERM4-N095B": 0.3, "EVAL-e10b-P5self-b3": 0.6, "EVAL-e10d-P5self-b3": 0.8,
       "EVAL-ood-17think-b3": 3.0, "EVAL-ood-17chain-b3": 1.5, "EVAL-ood-17q093-b3": 3.0,
       "EVAL-q094-17base-T": 0.6, "EVAL-q094-17base-N": 0.3, "EVAL-q094-17self-T": 0.6, "EVAL-q094-17self-N": 0.6, "EVAL-q094-4base-T": 0.8,
       "EVAL-q094-4base-N": 0.4, "EVAL-q094-4self-T": 0.8, "EVAL-q094-4self-N": 0.8, "EVAL-q094-17p5-T": 0.6, "EVAL-q094-17p5-N": 0.6,
       "EVAL-nothink16k-17base-b3": 0.3, "EVAL-nothink16k-4Bbase-b3": 0.4,
       "E12b-P0think-8k-c100": 8.0, "EVAL-e12b-P0think-8k-c100": 1.8, "PROBE-TERM4-N096B75": 0.3, "PROBE-TERM4-N096BC": 0.3,
       "PROBE-E12B-P0THINK-8K50-4K": 0.3, "E12b-P0think-8k50-4k": 4.0, "PROBE-E12B-P0THINK-7K": 0.5, "E12b-P0think-7k": 7.0, "PROBE-E12B-P0THINK-6K": 0.5, "E12b-P0think-6k": 6.0, "PROBE-E12B-P0THINK-12K": 0.6, "E12b-P0think-12k": 11.0, "PROBE-P5Q4KT": 0.3, "E10b-P5q4kt": 3.5, "PROBE-P5SELFP2": 0.3, "E10b-P5selfp2": 3.5}   # [z159] A4 as the 4K arm, 6K between 4K and 8K   # [z158b] as Q-093-s43 (the 16K arm ~1.8 x the 8K step)   # [z156b] as the -b2 references   # [z146] Q-093: thinking answers fill the 8K window (training ~1.5 x Q-091)   # P1-a arms: #steptime 180 s x 50 steps
PROFILES = {
    "1g": P0_SERIAL + ["pn5", "p1q", "p1a"],   # [z152] N5 after P0 (1gt inherits it, 1gs does not)   # [z146] Q-093 after P0, before P1-a (1gt inherits it; 1gs does not: a 1gs / 1gt pair shares the gates only)
    "4g": P0_PARALLEL + ["p1a"],
    "4g80": P0_PARALLEL + ["p1a", "p1b", "p1q4"],   # [z149] Q-093b after P1-b
    "8g": P0_PARALLEL + ["p1a", "p2f", "p2g"],
    "8g80": P0_PARALLEL + ["p1a", "p1b", "p1q4", "p1q", "p2f", "p2h", "p2g"],   # [z149] Q-093b after P1-b; [z150] p1q too (an 8 x B200 reads
    # Q-090 / Q-093 and their B200 references beside P1-b: Fable 11-11 15:5x; the share keeps the rows, the 1.7B arm stays paused there)
}
# [z139] (the lead 11-10) two machines at once, each with its own disk and runs/: a 1 x H200 that has run p0a and is in p0c, and an
# 8 x H200. 1gs = the 1-GPU box's share (the readings of historical weights, which need no new training); 8gs = 8g80 without those
# stages (every new training and its readings on one machine). Together they are 8g80's stages, only p0a (the local gates) on both.
# The 8-GPU node reads tasks/queue_8gs.txt through control/queue_name (env.sh exports QUEUE_FILE); the 1-GPU node keeps queue.txt.
SHARE_1G = ["p0a", "p0c", "p0f", "p0g", "p0x"]
PROFILES["1gs"] = SHARE_1G
PROFILES["8gs"] = [s for s in PROFILES["8g80"] if s == "p0a" or s not in SHARE_1G] + ["p1q8", "p1qs43", "p1q4s43", "p1h2", "p1b3", "p1b3q093", "p1qs44", "p1q096a", "p1q095", "p1q096c", "p1q094", "p1q096b", "p1q097", "p1q097t", "p1q101"]   # [z163] p1q101   # [z160] p1q097t   # [z159] p1q097   # [z154] 8-GPU-only stages; [z156] p1q4s43; [z156b] p1h2; [z157] p1b3; [z158b] p1b3q093, p1qs44, p1q096a
# [z142] (the lead 11-10: the 8 x H200 reclaimed, a second 1 x H200 back) 1gt = the 1-GPU trainings and their readings beside a 1gs node:
# 1g without the 1gs stages (p0d, p0e, p0b, p0dr, p0er, P1-a); with 1gs it is 1g's stages, only p0a on both
PROFILES["1gt"] = [s for s in PROFILES["1g"] if s == "p0a" or s not in SHARE_1G]
# [v3.0] (Fable 11-17) the borrowed 8 x H200 machines. h2a = one machine does everything: the gates, Q-107's two arms, their readings,
# then the zero-training readings (an `auto` stage: it opens once the stages before it have ended). h2b = a second machine, if there is
# one: the canary and the zero-training readings at once; h2a then finds their .done in the result repository and skips them
PROFILES["h2a"] = ["h2g", "q107", "q107r", "q107z", "q107z6"]
PROFILES["h2b"] = ["h2gb", "q107z", "q107z6"]
# [v3.14] h2c = the same borrowed machine on its second day (`queue h2c`): readings and arms registered after Q-107
PROFILES["h2c"] = ["q108", "q109g", "q109", "q110", "q109z", "q110r", "q109r", "q111", "q112g", "q112", "q112r"]   # [v3.15] Q-109 / Q-110 behind Q-108's readings
# [v3.18] en2 = a cloud 8 x H200 (`queue en2`): Q-111's s50 readings, Q-112 (gate, arm, readings, same-machine references), Q-113
PROFILES["en2"] = ["q111b", "q112za", "q112g", "q112", "q112zb", "q113", "q112r"]
AUTO = {("h2a", "q107z"), ("h2a", "q107z6")}   # (profile, stage) written as `auto`, every other stage `open`
# each stage's starts pulled ahead (#fetch, from T0 + 8 min): the source's 14 lines plus the 1.7B chain's s50, in stage order
FETCH = {
    "p0a": ["E10b-P5/global_step_150", "E10b-P5tail/global_step_50"],
    "p0c": ["E12b-P5-8k/global_step_50"],
    "p0b": ["E12b-P5tail-8k/global_step_50", "E12d-P5tail-8k/global_step_50"],
    "p0f": ["E12d-P5tail-8k-s43/global_step_50"],
    "p0g": ["E10d-P5tail/global_step_50", "E10d-P5tail-s43/global_step_50"],
    "p1b": ["E10e-P5self/global_step_50", "E12e-P5tail-8k/global_step_25", "E10e-P5tail/global_step_50"],
    "p2f": ["E10c-P5-fix4/global_step_150", "E12c-P5-8k/global_step_25", "E12c-P5-8k/global_step_50", "E10c-P5tail-c50/global_step_30"],
    "p1b3q093": ["E12b-P0think-8k/global_step_50"],   # [z158c] Q-093 s50: the reread and Q-095's teacher
    "p1q095": ["E10b-P5self/global_step_50", "E10d-P5self/global_step_50"],
    "p1q094": ["E10b-P5self/global_step_50", "E10d-P5self/global_step_50", "E10b-P5/global_step_150"],   # [z158d]
    "p1q096b": ["E12b-P0think-8k/global_step_50"],   # [z158f] the start of Q-096 B
    "p1q097p": ["E12b-P0think-8k/global_step_50"],   # [z159] the start of A4 (paused)
    "p1q097t": ["E12b-P0think-4k/global_step_50"],   # [z160] T4K's teacher
    "h2gb": ["E10b-P5/global_step_150", "E10b-P5tail/global_step_50"],
    "q107z": ["E12b-P0think-4k/global_step_50", "E12b-P0think-4k/global_step_25"],
    "q107z6": ["E12b-P0think-6k/global_step_25"],
    "q109g": ["E10d-P5selfp2/global_step_50"],
    "q110": ["E12b-P0think-3k/global_step_50", "E12b-P0think-3k-s44/global_step_50"],
    "q109z": ["E12d-P0think-8k/global_step_50"],
    "q111": ["E12b-P0think-3k/global_step_50", "E12b-P0think-3k-s44/global_step_50"],
    "q112": ["E12d-P0think-8k/global_step_50"],
    "q111b": ["E12b-P0think-3k/global_step_50", "E12b-P0think-3k-s44/global_step_50", "E12b-P0think-3k-c100/global_step_50"],
    "q112za": ["E12d-P5selfp2-8k/global_step_50"],
    "q113": ["E12b-P0think-3k-s44-c100/global_step_50", "E12b-P5selfp2-4k/global_step_50", "E12b-P5selfp2-4k-s4444/global_step_50", "E12b-P0think-4k/global_step_50"],
    "q108": ["E12b-P0think-8k/global_step_50", "E12b-P0think-4k/global_step_50", "E10b-P5selfp2/global_step_50", "E12b-P5selfp2-8k/global_step_50", "E12b-P5selfp2-8k-s4343/global_step_50", "E12b-P5selfp2-8k-s4444/global_step_50", "E12b-P5selfp2-4k/global_step_50", "E12b-P5selfp2-4k-s4444/global_step_50"],
}
DFETCH = {   # [z158c] (Q-096 C) the multiple-choice bench's data, pulled by run_tasks' fetch_pass into $OPD_ROOT/data/hf
    "p1q096c": ["TIGER-Lab/MMLU-Pro"],
}
NEW_ROWS = """\
#gpus PROBE-E12B-P0THINK-3K 4
#gpus E12b-P0think-3k 4
#gpus E12b-P0think-3k-s44 4
#eval EVAL-q107-3k-s50 q107_e12b_P0think-3k_s50_h2_math500_T0.6 result:runs/E12b-P0think-3k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q107-3k-s50 q107_e12b_P0think-3k_s50_h2_math500_T0.6 result:runs/E12b-P0think-3k/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q107-3k-s44-s50 q107_e12b_P0think-3k-s44_s50_h2_math500_T0.6 result:runs/E12b-P0think-3k-s44/global_step_50/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q107-3k-s44-s50 q107_e12b_P0think-3k-s44_s50_h2_math500_T0.6 result:runs/E12b-P0think-3k-s44/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q107-3k-s25 q107_e12b_P0think-3k_s25_h2_math500_T0.6 result:runs/E12b-P0think-3k/global_step_25/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q107-3k-s25 q107_e12b_P0think-3k_s25_h2_math500_T0.6 result:runs/E12b-P0think-3k/global_step_25/actor/huggingface think budgets=16384
#eval EVAL-q107-3k-s44-s25 q107_e12b_P0think-3k-s44_s25_h2_math500_T0.6 result:runs/E12b-P0think-3k-s44/global_step_25/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q107-3k-s44-s25 q107_e12b_P0think-3k-s44_s25_h2_math500_T0.6 result:runs/E12b-P0think-3k-s44/global_step_25/actor/huggingface think budgets=16384
#eval EVAL-q107-4k-s50 q107_e12b_P0think-4k_s50_h2_math500_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q107-4k-s50 q107_e12b_P0think-4k_s50_h2_math500_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q107-4k-s25 q107_e12b_P0think-4k_s25_h2_math500_T0.6 result:runs/E12b-P0think-4k/global_step_25/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q107-4k-s25 q107_e12b_P0think-4k_s25_h2_math500_T0.6 result:runs/E12b-P0think-4k/global_step_25/actor/huggingface think budgets=16384
#eval EVAL-q107-6k-s25 q107_e12b_P0think-6k_s25_h2_math500_T0.6 result:runs/E12b-P0think-6k/global_step_25/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q107-6k-s25 q107_e12b_P0think-6k_s25_h2_math500_T0.6 result:runs/E12b-P0think-6k/global_step_25/actor/huggingface think budgets=16384
#gpus PROBE-E12D-P5SELFP2-8K 4
#gpus E12d-P5selfp2-8k 4
#gpus E12b-P0think-3k-c100 4
#gpus E12b-P0think-3k-s44-c100 4
#eval EVAL-q109-4b-s50 q109_e12d_P5selfp2-8k_s50_h2_math500_T0.6 result:runs/E12d-P5selfp2-8k/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#forced EVAL-q109-4b-s50 q109_e12d_P5selfp2-8k_s50_h2_math500_T0.6 result:runs/E12d-P5selfp2-8k/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q109-4b-s25 q109_e12d_P5selfp2-8k_s25_h2_math500_T0.6 result:runs/E12d-P5selfp2-8k/global_step_25/actor/huggingface math500 4 0.6 16384 no_think
#forced EVAL-q109-4b-s25 q109_e12d_P5selfp2-8k_s25_h2_math500_T0.6 result:runs/E12d-P5selfp2-8k/global_step_25/actor/huggingface no_think budgets=16384
#eval EVAL-q109-d8k-4b q109_e12d_P0think-8k_s50_h2_math500_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q109-d8k-4b q109_e12d_P0think-8k_s50_h2_math500_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q109-opd-4b q109_e10d_P5selfp2_s50_h2_math500_T0.6 result:runs/E10d-P5selfp2/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#forced EVAL-q109-opd-4b q109_e10d_P5selfp2_s50_h2_math500_T0.6 result:runs/E10d-P5selfp2/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q111-3k-s100 q111_3k_s100_h2_olympiadbench_T0.6 result:runs/E12b-P0think-3k-c100/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#forced EVAL-q111-3k-s100 q111_3k_s100_h2_olympiadbench_T0.6 result:runs/E12b-P0think-3k-c100/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q111-3k-s44-s100 q111_3k-s44_s100_h2_olympiadbench_T0.6 result:runs/E12b-P0think-3k-s44-c100/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#forced EVAL-q111-3k-s44-s100 q111_3k-s44_s100_h2_olympiadbench_T0.6 result:runs/E12b-P0think-3k-s44-c100/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q111-3k-s50 q111_3k_s50_h2_olympiadbench_T0.6 result:runs/E12b-P0think-3k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#forced EVAL-q111-3k-s50 q111_3k_s50_h2_olympiadbench_T0.6 result:runs/E12b-P0think-3k/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q111-3k-s44-s50 q111_3k-s44_s50_h2_olympiadbench_T0.6 result:runs/E12b-P0think-3k-s44/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#forced EVAL-q111-3k-s44-s50 q111_3k-s44_s50_h2_olympiadbench_T0.6 result:runs/E12b-P0think-3k-s44/global_step_50/actor/huggingface think budgets=16384
#gpus E12d-P0think-8k-c100 4
#eval EVAL-q112-4b-s100 q112_e12d_P0think-8k-c100_s100_h2_math500_T0.6 result:runs/E12d-P0think-8k-c100/global_step_50/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q112-4b-s100 q112_e12d_P0think-8k-c100_s100_h2_math500_T0.6 result:runs/E12d-P0think-8k-c100/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q112-4b-s75 q112_e12d_P0think-8k-c100_s75_h2_math500_T0.6 result:runs/E12d-P0think-8k-c100/global_step_25/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q112-4b-s75 q112_e12d_P0think-8k-c100_s75_h2_math500_T0.6 result:runs/E12d-P0think-8k-c100/global_step_25/actor/huggingface think budgets=16384
#eval EVAL-q111-3k-s100-en2 q111_3k_s100_en2_olympiadbench_T0.6 result:runs/E12b-P0think-3k-c100/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#forced EVAL-q111-3k-s100-en2 q111_3k_s100_en2_olympiadbench_T0.6 result:runs/E12b-P0think-3k-c100/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q112-chain-s50 q112_e12d_P5selfp2-8k_s50_en2_math500_T0.6 result:runs/E12d-P5selfp2-8k/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#forced EVAL-q112-chain-s50 q112_e12d_P5selfp2-8k_s50_en2_math500_T0.6 result:runs/E12d-P5selfp2-8k/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q112-d8k-s50 q112_e12d_P0think-8k_s50_en2_math500_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q112-d8k-s50 q112_e12d_P0think-8k_s50_en2_math500_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q113-base17 q113_base17_en2_minervamath_T0.6 Qwen/Qwen3-1.7B minervamath 4 0.6 16384 think
#forced EVAL-q113-base17 q113_base17_en2_minervamath_T0.6 Qwen/Qwen3-1.7B think budgets=16384
#eval EVAL-q113-3k-s100 q113_3k_s100_en2_minervamath_T0.6 result:runs/E12b-P0think-3k-c100/global_step_50/actor/huggingface minervamath 4 0.6 16384 think
#forced EVAL-q113-3k-s100 q113_3k_s100_en2_minervamath_T0.6 result:runs/E12b-P0think-3k-c100/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q113-3k-s44-s100 q113_3k-s44_s100_en2_minervamath_T0.6 result:runs/E12b-P0think-3k-s44-c100/global_step_50/actor/huggingface minervamath 4 0.6 16384 think
#forced EVAL-q113-3k-s44-s100 q113_3k-s44_s100_en2_minervamath_T0.6 result:runs/E12b-P0think-3k-s44-c100/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q113-c4k q113_c4k_en2_minervamath_T0.6 result:runs/E12b-P5selfp2-4k/global_step_50/actor/huggingface minervamath 4 0.6 16384 no_think
#forced EVAL-q113-c4k q113_c4k_en2_minervamath_T0.6 result:runs/E12b-P5selfp2-4k/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q113-c4k-s4444 q113_c4k-s4444_en2_minervamath_T0.6 result:runs/E12b-P5selfp2-4k-s4444/global_step_50/actor/huggingface minervamath 4 0.6 16384 no_think
#forced EVAL-q113-c4k-s4444 q113_c4k-s4444_en2_minervamath_T0.6 result:runs/E12b-P5selfp2-4k-s4444/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q113-d4k q113_d4k_en2_minervamath_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface minervamath 4 0.6 16384 think
#forced EVAL-q113-d4k q113_d4k_en2_minervamath_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q110-3k-s100 q110_e12b_P0think-3k-c100_s100_h2_math500_T0.6 result:runs/E12b-P0think-3k-c100/global_step_50/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q110-3k-s100 q110_e12b_P0think-3k-c100_s100_h2_math500_T0.6 result:runs/E12b-P0think-3k-c100/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q110-3k-s75 q110_e12b_P0think-3k-c100_s75_h2_math500_T0.6 result:runs/E12b-P0think-3k-c100/global_step_25/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q110-3k-s75 q110_e12b_P0think-3k-c100_s75_h2_math500_T0.6 result:runs/E12b-P0think-3k-c100/global_step_25/actor/huggingface think budgets=16384
#eval EVAL-q110-3k-s44-s100 q110_e12b_P0think-3k-s44-c100_s100_h2_math500_T0.6 result:runs/E12b-P0think-3k-s44-c100/global_step_50/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q110-3k-s44-s100 q110_e12b_P0think-3k-s44-c100_s100_h2_math500_T0.6 result:runs/E12b-P0think-3k-s44-c100/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q110-3k-s44-s75 q110_e12b_P0think-3k-s44-c100_s75_h2_math500_T0.6 result:runs/E12b-P0think-3k-s44-c100/global_step_25/actor/huggingface math500 4 0.6 16384 think
#forced EVAL-q110-3k-s44-s75 q110_e12b_P0think-3k-s44-c100_s75_h2_math500_T0.6 result:runs/E12b-P0think-3k-s44-c100/global_step_25/actor/huggingface think budgets=16384
#eval EVAL-q108-base17 q108_base17_h2_olympiadbench_T0.6 Qwen/Qwen3-1.7B olympiadbench 4 0.6 16384 think
#forced EVAL-q108-base17 q108_base17_h2_olympiadbench_T0.6 Qwen/Qwen3-1.7B think budgets=16384
#eval EVAL-q108-d8k q108_d8k_h2_olympiadbench_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#forced EVAL-q108-d8k q108_d8k_h2_olympiadbench_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q108-d4k q108_d4k_h2_olympiadbench_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#forced EVAL-q108-d4k q108_d4k_h2_olympiadbench_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface think budgets=16384
#eval EVAL-q108-opd q108_opd_h2_olympiadbench_T0.6 result:runs/E10b-P5selfp2/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 no_think
#forced EVAL-q108-opd q108_opd_h2_olympiadbench_T0.6 result:runs/E10b-P5selfp2/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q108-c8k q108_c8k_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-8k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 no_think
#forced EVAL-q108-c8k q108_c8k_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-8k/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q108-c8k-s4343 q108_c8k-s4343_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-8k-s4343/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 no_think
#forced EVAL-q108-c8k-s4343 q108_c8k-s4343_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-8k-s4343/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q108-c8k-s4444 q108_c8k-s4444_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-8k-s4444/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 no_think
#forced EVAL-q108-c8k-s4444 q108_c8k-s4444_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-8k-s4444/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q108-c4k q108_c4k_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-4k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 no_think
#forced EVAL-q108-c4k q108_c4k_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-4k/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-q108-c4k-s4444 q108_c4k-s4444_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-4k-s4444/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 no_think
#forced EVAL-q108-c4k-s4444 q108_c4k-s4444_h2_olympiadbench_T0.6 result:runs/E12b-P5selfp2-4k-s4444/global_step_50/actor/huggingface no_think budgets=16384
#eval EVAL-e12b-P5tail-8k-nn e12b_P5tail-8k_s50_math500_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e12b-P5tail-8k-nn e12b_P5tail-8k_s50_math500_T1.0 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface math500 4 1.0
#eval EVAL-e12b-P5-8k-nn e12b_P5-8k_s50_math500_T0.6 result:runs/E12b-P5-8k/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e12b-P5-8k-nn e12b_P5-8k_s50_math500_T1.0 result:runs/E12b-P5-8k/global_step_50/actor/huggingface math500 4 1.0
#eval EVAL-ext32k-17p5tail8k-nn ext32k17_E12bP5tail8k_s50_aime24_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface aime24 16 0.6 32768
#eval EVAL-ext32k-17p5tail8k-nn ext32k17_E12bP5tail8k_s50_aime25_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface aime25 16 0.6 32768
#term PROBE-TERM4-N8BTHINK 8B_P0think Qwen/Qwen3-8B EVAL-budA-8Bthink/budA8B_think_math500_T0.6_32k_math500_think_t0.6.jsonl think
#term PROBE-TERM4-NP5H200 17_P5s150 result:runs/E10b-P5/global_step_150/actor/huggingface gen
#eval EVAL-ext32k-4Bp5tail8k-nn ext32k4B_E12dP5tail8k_s50_aime24_T0.6 result:runs/E12d-P5tail-8k/global_step_50/actor/huggingface aime24 16 0.6 32768
#eval EVAL-ext32k-4Bp5tail8k-nn ext32k4B_E12dP5tail8k_s50_aime25_T0.6 result:runs/E12d-P5tail-8k/global_step_50/actor/huggingface aime25 16 0.6 32768
#term PROBE-TERM4-N17P0H200 17_P0 Qwen/Qwen3-1.7B gen
#eval EVAL-e12b-P0think-8k e12b_P0think-8k_s50_math500_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-e12b-P0think-8k e12b_P0think-8k_s50_math500_T1.0 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface math500 4 1.0 16384 think
#term PROBE-TERM4-N093 17_P0think8k result:runs/E12b-P0think-8k/global_step_50/actor/huggingface EVAL-e12b-P0think-8k/e12b_P0think-8k_s50_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-budA-e12bP0think-8k budAe12bP0think-8k_s50_math500_T0.6_32k result:runs/E12b-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 32768 think
#eval EVAL-ext32k-17p0think8k ext32k17_E12bP0think8k_s50_aime24_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface aime24 16 0.6 32768 think
#eval EVAL-ext32k-17p0think8k ext32k17_E12bP0think8k_s50_aime25_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface aime25 16 0.6 32768 think
#eval EVAL-e12b-P5tail-8k-nn-b2 e12b_P5tail-8k_s50_b2_math500_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e12b-P5tail-8k-nn-b2 e12b_P5tail-8k_s50_b2_math500_T1.0 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface math500 4 1.0
#eval EVAL-think16k-17base-b2 base17_think16k_b2_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 think
#term PROBE-TERM4-N17THB2 17_base_think Qwen/Qwen3-1.7B EVAL-think16k-17base-b2/base17_think16k_b2_math500_T0.6_math500_think_t0.6.jsonl think
#term PROBE-TERM4-NP5B2 17_P5s150 result:runs/E10b-P5/global_step_150/actor/huggingface gen
#gpus PROBE-E12D-P0THINK 4
#gpus E12d-P0think-8k 4
#eval EVAL-e12d-P0think-8k e12d_P0think-8k_s50_math500_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-e12d-P0think-8k e12d_P0think-8k_s50_math500_T1.0 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface math500 4 1.0 16384 think
#term PROBE-TERM4-N093B 4_P0think8k result:runs/E12d-P0think-8k/global_step_50/actor/huggingface EVAL-e12d-P0think-8k/e12d_P0think-8k_s50_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-budA-e12dP0think-8k budAe12dP0think-8k_s50_math500_T0.6_32k result:runs/E12d-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 32768 think
#eval EVAL-ext32k-4p0think8k ext32k4_E12dP0think8k_s50_aime24_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface aime24 16 0.6 32768 think
#eval EVAL-ext32k-4p0think8k ext32k4_E12dP0think8k_s50_aime25_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface aime25 16 0.6 32768 think
#eval EVAL-n5-17think n5_17think_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 think
#forced EVAL-n5-17think n5_17think_math500_T0.6 Qwen/Qwen3-1.7B think
#eval EVAL-n5-4Bthink n5_4Bthink_math500_T0.6 local:models/Qwen3-4B math500 4 0.6 16384 think
#forced EVAL-n5-4Bthink n5_4Bthink_math500_T0.6 local:models/Qwen3-4B think
#eval EVAL-n5-17think-aime n5_17think_aime24_T0.6 Qwen/Qwen3-1.7B aime24 16 0.6 16384 think
#eval EVAL-n5-17think-aime n5_17think_aime25_T0.6 Qwen/Qwen3-1.7B aime25 16 0.6 16384 think
#forced EVAL-n5-17think-aime n5_17think_aime24_T0.6 Qwen/Qwen3-1.7B think
#forced EVAL-n5-17think-aime n5_17think_aime25_T0.6 Qwen/Qwen3-1.7B think
#eval EVAL-n5-4Bthink-aime n5_4Bthink_aime24_T0.6 local:models/Qwen3-4B aime24 16 0.6 16384 think
#eval EVAL-n5-4Bthink-aime n5_4Bthink_aime25_T0.6 local:models/Qwen3-4B aime25 16 0.6 16384 think
#forced EVAL-n5-4Bthink-aime n5_4Bthink_aime24_T0.6 local:models/Qwen3-4B think
#forced EVAL-n5-4Bthink-aime n5_4Bthink_aime25_T0.6 local:models/Qwen3-4B think
#eval EVAL-n5-4Binst2507 n5_4Binst2507_math500_T0.6 local:models/Qwen3-4B-Instruct-2507 math500 4 0.6 16384 no_think
#eval EVAL-e12b-P0think-8k-s25 e12b_P0think-8k_s25_math500_T0.6 result:runs/E12b-P0think-8k/global_step_25/actor/huggingface math500 4 0.6 16384 think
#term PROBE-TERM4-N093FT 17_P0think8k-on-basethink result:runs/E12b-P0think-8k/global_step_50/actor/huggingface EVAL-think16k-17base-b2/base17_think16k_b2_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-e12b-P5-8k-nn-b2 e12b_P5-8k_s50_b2_math500_T0.6 result:runs/E12b-P5-8k/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e12b-P5-8k-nn-b2 e12b_P5-8k_s50_b2_math500_T1.0 result:runs/E12b-P5-8k/global_step_50/actor/huggingface math500 4 1.0
#term PROBE-TERM4-NN2S0 17_P5tail-on-P5text result:runs/E10b-P5tail/global_step_50/actor/huggingface PROBE-TERM4-NP5B2/gen/term_17_P5s150_math500_no_think_t0.6.jsonl
#term PROBE-TERM4-N090FT 17_P5l8klp-on-P5text result:runs/E12b-P5-8k-lenpen/global_step_50/actor/huggingface PROBE-TERM4-NP5B2/gen/term_17_P5s150_math500_no_think_t0.6.jsonl
#eval EVAL-e12b-P5tail-8k-s43-s25-b2 e12b_P5tail-8k-s43_s25_b2_math500_T0.6 result:runs/E12b-P5tail-8k-s43/global_step_25/actor/huggingface math500 4 0.6
#eval EVAL-e12b-P5tail-8k-s43-s50-b2 e12b_P5tail-8k-s43_s50_b2_math500_T0.6 result:runs/E12b-P5tail-8k-s43/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e12b-P5tail-8k-s43-s50-b2 e12b_P5tail-8k-s43_s50_b2_math500_T1.0 result:runs/E12b-P5tail-8k-s43/global_step_50/actor/huggingface math500 4 1.0
#eval EVAL-e12d-P5tail-8k-nn-b2 e12d_P5tail-8k_s50_b2_math500_T0.6 result:runs/E12d-P5tail-8k/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e12d-P5tail-8k-nn-b2 e12d_P5tail-8k_s50_b2_math500_T1.0 result:runs/E12d-P5tail-8k/global_step_50/actor/huggingface math500 4 1.0
#eval EVAL-think16k-4Bbase-b2 base4_think16k_b2_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 think
#term PROBE-TERM4-N4BTHB2 4B_base_think Qwen/Qwen3-4B EVAL-think16k-4Bbase-b2/base4_think16k_b2_math500_T0.6_math500_think_t0.6.jsonl think
#gpus PROBE-E12E-P0THINK 8
#gpus E12e-P0think-8k 8
#steptime E12e-P0think-8k 195
#eval EVAL-e12e-P0think-8k e12e_P0think-8k_s50_math500_T0.6 result:runs/E12e-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-e12e-P0think-8k e12e_P0think-8k_s50_math500_T1.0 result:runs/E12e-P0think-8k/global_step_50/actor/huggingface math500 4 1.0 16384 think
#term PROBE-TERM4-N093C 8B_P0think8k result:runs/E12e-P0think-8k/global_step_50/actor/huggingface EVAL-e12e-P0think-8k/e12e_P0think-8k_s50_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-think16k-8Bbase-b2 base8_think16k_b2_math500_T0.6 Qwen/Qwen3-8B math500 4 0.6 16384 think
#term PROBE-TERM4-N8BTHB2 8B_base_think Qwen/Qwen3-8B EVAL-think16k-8Bbase-b2/base8_think16k_b2_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-budA-e12eP0think-8k budAe12eP0think-8k_s50_math500_T0.6_32k result:runs/E12e-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 32768 think
#eval EVAL-ext32k-8p0think8k ext32k8B_E12eP0think8k_s50_aime24_T0.6 result:runs/E12e-P0think-8k/global_step_50/actor/huggingface aime24 16 0.6 32768 think
#eval EVAL-ext32k-8p0think8k ext32k8B_E12eP0think8k_s50_aime25_T0.6 result:runs/E12e-P0think-8k/global_step_50/actor/huggingface aime25 16 0.6 32768 think
#eval EVAL-ext32k-8Bthink-b2 ext32k8_base8think_b2_aime24_T0.6 Qwen/Qwen3-8B aime24 16 0.6 32768 think
#eval EVAL-ext32k-8Bthink-b2 ext32k8_base8think_b2_aime25_T0.6 Qwen/Qwen3-8B aime25 16 0.6 32768 think
#gpus PROBE-E12B-P0THINK-8G 8
#gpus E12b-P0think-8k-s43 8
#steptime E12b-P0think-8k-s43 120
#eval EVAL-e12b-P0think-8k-s43 e12b_P0think-8k-s43_s50_math500_T0.6 result:runs/E12b-P0think-8k-s43/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-e12b-P0think-8k-s43 e12b_P0think-8k-s43_s50_math500_T1.0 result:runs/E12b-P0think-8k-s43/global_step_50/actor/huggingface math500 4 1.0 16384 think
#term PROBE-TERM4-N093S43 17_P0think8ks43 result:runs/E12b-P0think-8k-s43/global_step_50/actor/huggingface EVAL-e12b-P0think-8k-s43/e12b_P0think-8k-s43_s50_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-budA-e12bP0think-8k-s43 budAe12bP0think-8ks43_s50_math500_T0.6_32k result:runs/E12b-P0think-8k-s43/global_step_50/actor/huggingface math500 4 0.6 32768 think
#eval EVAL-ext32k-17p0think8k-s43 ext32k17_E12bP0think8ks43_s50_aime24_T0.6 result:runs/E12b-P0think-8k-s43/global_step_50/actor/huggingface aime24 16 0.6 32768 think
#eval EVAL-ext32k-17p0think8k-s43 ext32k17_E12bP0think8ks43_s50_aime25_T0.6 result:runs/E12b-P0think-8k-s43/global_step_50/actor/huggingface aime25 16 0.6 32768 think
#gpus PROBE-E12D-P0THINK-8G 8
#gpus E12d-P0think-8k-s43 8
#eval EVAL-e12d-P0think-8k-s43 e12d_P0think-8k-s43_s50_math500_T0.6 result:runs/E12d-P0think-8k-s43/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-e12d-P0think-8k-s43 e12d_P0think-8k-s43_s50_math500_T1.0 result:runs/E12d-P0think-8k-s43/global_step_50/actor/huggingface math500 4 1.0 16384 think
#term PROBE-TERM4-N093BS43 4_P0think8ks43 result:runs/E12d-P0think-8k-s43/global_step_50/actor/huggingface EVAL-e12d-P0think-8k-s43/e12d_P0think-8k-s43_s50_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-budA-e12dP0think-8k-s43 budAe12dP0think-8ks43_s50_math500_T0.6_32k result:runs/E12d-P0think-8k-s43/global_step_50/actor/huggingface math500 4 0.6 32768 think
#eval EVAL-ext32k-4p0think8k-s43 ext32k4_E12dP0think8ks43_s50_aime24_T0.6 result:runs/E12d-P0think-8k-s43/global_step_50/actor/huggingface aime24 16 0.6 32768 think
#eval EVAL-ext32k-4p0think8k-s43 ext32k4_E12dP0think8ks43_s50_aime25_T0.6 result:runs/E12d-P0think-8k-s43/global_step_50/actor/huggingface aime25 16 0.6 32768 think
#eval EVAL-think16k-8Bbase-h2 base8_think16k_h2_math500_T0.6 Qwen/Qwen3-8B math500 4 0.6 16384 think
#term PROBE-TERM4-N8BTHH2 8B_base_think Qwen/Qwen3-8B EVAL-think16k-8Bbase-h2/base8_think16k_h2_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-e12e-P5tail-8k-nn-h2 e12e_P5tail-8k_s50_h2_math500_T0.6 result:runs/E12e-P5tail-8k-nn/global_step_25/actor/huggingface math500 4 0.6
#eval EVAL-e12e-P5tail-8k-nn-h2 e12e_P5tail-8k_s50_h2_math500_T1.0 result:runs/E12e-P5tail-8k-nn/global_step_25/actor/huggingface math500 4 1.0
#eval EVAL-e12d-P5tail-8k-nn-b3 e12d_P5tail-8k_s50_b3_math500_T0.6 result:runs/E12d-P5tail-8k/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e12d-P5tail-8k-nn-b3 e12d_P5tail-8k_s50_b3_math500_T1.0 result:runs/E12d-P5tail-8k/global_step_50/actor/huggingface math500 4 1.0
#eval EVAL-think16k-4Bbase-b3 base4_think16k_b3_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 think
#term PROBE-TERM4-N4BTHB3 4B_base_think Qwen/Qwen3-4B EVAL-think16k-4Bbase-b3/base4_think16k_b3_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-e12e-P5tail-8k-nn-b3 e12e_P5tail-8k_s50_b3_math500_T0.6 result:runs/E12e-P5tail-8k-nn/global_step_25/actor/huggingface math500 4 0.6
#eval EVAL-e12e-P5tail-8k-nn-b3 e12e_P5tail-8k_s50_b3_math500_T1.0 result:runs/E12e-P5tail-8k-nn/global_step_25/actor/huggingface math500 4 1.0
#eval EVAL-think16k-8Bbase-b3 base8_think16k_b3_math500_T0.6 Qwen/Qwen3-8B math500 4 0.6 16384 think
#term PROBE-TERM4-N8BTHB3 8B_base_think Qwen/Qwen3-8B EVAL-think16k-8Bbase-b3/base8_think16k_b3_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-e12b-P5tail-8k-s43-s50-b3 e12b_P5tail-8k-s43_s50_b3_math500_T0.6 result:runs/E12b-P5tail-8k-s43/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e12b-P5tail-8k-s43-s50-b3 e12b_P5tail-8k-s43_s50_b3_math500_T1.0 result:runs/E12b-P5tail-8k-s43/global_step_50/actor/huggingface math500 4 1.0
#eval EVAL-think16k-17base-b3 base17_think16k_b3_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 think
#term PROBE-TERM4-N17THB3 17_base_think Qwen/Qwen3-1.7B EVAL-think16k-17base-b3/base17_think16k_b3_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-ext32k-8Bthink-b3 ext32k8_base8think_b3_aime24_T0.6 Qwen/Qwen3-8B aime24 16 0.6 32768 think
#eval EVAL-ext32k-8Bthink-b3 ext32k8_base8think_b3_aime25_T0.6 Qwen/Qwen3-8B aime25 16 0.6 32768 think
#eval EVAL-e10b-P5q093t e10b_P5q093t_s50_math500_T0.6 result:runs/E10b-P5q093t/global_step_50/actor/huggingface math500 4 0.6
#term PROBE-TERM4-N095A 17_P5q093t result:runs/E10b-P5q093t/global_step_50/actor/huggingface EVAL-e10b-P5q093t/e10b_P5q093t_s50_math500_T0.6_math500_no_think_t0.6.jsonl
#eval EVAL-e10d-P5q093t e10d_P5q093t_s50_math500_T0.6 result:runs/E10d-P5q093t/global_step_50/actor/huggingface math500 4 0.6
#term PROBE-TERM4-N095B 4_P5q093t result:runs/E10d-P5q093t/global_step_50/actor/huggingface EVAL-e10d-P5q093t/e10d_P5q093t_s50_math500_T0.6_math500_no_think_t0.6.jsonl
#eval EVAL-e10b-P5self-b3 e10b_P5self_s50_b3_math500_T0.6 result:runs/E10b-P5self/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-e10d-P5self-b3 e10d_P5self_s50_b3_math500_T0.6 result:runs/E10d-P5self/global_step_50/actor/huggingface math500 4 0.6
#eval EVAL-ood-17think-b3 ood_base17think_b3_olympiadbench_T0.6 Qwen/Qwen3-1.7B olympiadbench 4 0.6 16384 think
#eval EVAL-ood-17think-b3 ood_base17think_b3_minervamath_T0.6 Qwen/Qwen3-1.7B minervamath 4 0.6 16384 think
#eval EVAL-ood-17think-b3 ood_base17think_b3_mmlupro1k_T0.6 Qwen/Qwen3-1.7B mmlupro1k 4 0.6 16384 think
#eval EVAL-ood-17chain-b3 ood_e12bP5tail8k_s50_b3_olympiadbench_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384
#eval EVAL-ood-17chain-b3 ood_e12bP5tail8k_s50_b3_minervamath_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface minervamath 4 0.6 16384
#eval EVAL-ood-17chain-b3 ood_e12bP5tail8k_s50_b3_mmlupro1k_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface mmlupro1k 4 0.6 16384
#eval EVAL-ood-17q093-b3 ood_e12bP0think8k_s50_b3_olympiadbench_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#eval EVAL-ood-17q093-b3 ood_e12bP0think8k_s50_b3_minervamath_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface minervamath 4 0.6 16384 think
#eval EVAL-ood-17q093-b3 ood_e12bP0think8k_s50_b3_mmlupro1k_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface mmlupro1k 4 0.6 16384 think
#gpus E10b-P5q093t 4
#gpus E10d-P5q093t 4
#gpus PROBE-P5Q093T-4B 4
#gpus PROBE-E12B-P0THINK-S44 4
#gpus E12b-P0think-8k-s44 4
#gpus PROBE-E12B-P0THINK-4K 4
#gpus E12b-P0think-4k 4
#gpus PROBE-E12B-P0THINK-16K 4
#gpus E12b-P0think-16k 4
#eval EVAL-e12b-P0think-8k-s44 e12b_P0think-8k-s44_s50_math500_T0.6 result:runs/E12b-P0think-8k-s44/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-e12b-P0think-8k-s44 e12b_P0think-8k-s44_s50_math500_T1.0 result:runs/E12b-P0think-8k-s44/global_step_50/actor/huggingface math500 4 1.0 16384 think
#term PROBE-TERM4-N093S44 17_P0think8ks44 result:runs/E12b-P0think-8k-s44/global_step_50/actor/huggingface EVAL-e12b-P0think-8k-s44/e12b_P0think-8k-s44_s50_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-budA-e12bP0think-8k-s44 budAe12bP0think-8ks44_s50_math500_T0.6_32k result:runs/E12b-P0think-8k-s44/global_step_50/actor/huggingface math500 4 0.6 32768 think
#eval EVAL-ext32k-17p0think8k-s44 ext32k17_E12bP0think8ks44_s50_aime24_T0.6 result:runs/E12b-P0think-8k-s44/global_step_50/actor/huggingface aime24 16 0.6 32768 think
#eval EVAL-ext32k-17p0think8k-s44 ext32k17_E12bP0think8ks44_s50_aime25_T0.6 result:runs/E12b-P0think-8k-s44/global_step_50/actor/huggingface aime25 16 0.6 32768 think
#eval EVAL-e12b-P0think-4k e12b_P0think-4k_s50_math500_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#term PROBE-TERM4-N096A4K 17_P0think4k result:runs/E12b-P0think-4k/global_step_50/actor/huggingface EVAL-e12b-P0think-4k/e12b_P0think-4k_s50_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-e12b-P0think-16k e12b_P0think-16k_s50_math500_T0.6 result:runs/E12b-P0think-16k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#term PROBE-TERM4-N096A16K 17_P0think16k result:runs/E12b-P0think-16k/global_step_50/actor/huggingface EVAL-e12b-P0think-16k/e12b_P0think-16k_s50_math500_T0.6_math500_think_t0.6.jsonl think
#eval EVAL-e12b-P0think-8k-b3 e12b_P0think-8k_s50_b3_math500_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-e12b-P0think-8k-b3 e12b_P0think-8k_s50_b3_math500_T1.0 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface math500 4 1.0 16384 think
#term PROBE-TERM4-N093B3 17_P0think8k result:runs/E12b-P0think-8k/global_step_50/actor/huggingface EVAL-e12b-P0think-8k-b3/e12b_P0think-8k_s50_b3_math500_T0.6_math500_think_t0.6.jsonl think
#fetch result:runs/E12b-P0think-8k/global_step_50/actor/huggingface
#eval EVAL-ext32k-17p5tail8k-oc ext32k17_E12bP5tail8k_s50oc_aime24_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface aime24 16 0.6 32768
#eval EVAL-ext32k-17p5tail8k-oc ext32k17_E12bP5tail8k_s50oc_aime25_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface aime25 16 0.6 32768
#eval EVAL-q094-17base-T q094_base17_pfT_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 no_think pf=32313
#eval EVAL-q094-17base-N q094_base17_pfN_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 no_think pf=1654
#eval EVAL-q094-17self-T q094_e10bP5self_s50_pfT_math500_T0.6 result:runs/E10b-P5self/global_step_50/actor/huggingface math500 4 0.6 16384 no_think pf=32313
#eval EVAL-q094-17self-N q094_e10bP5self_s50_pfN_math500_T0.6 result:runs/E10b-P5self/global_step_50/actor/huggingface math500 4 0.6 16384 no_think pf=1654
#eval EVAL-q094-4base-T q094_base4B_pfT_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 no_think pf=32313
#eval EVAL-q094-4base-N q094_base4B_pfN_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 no_think pf=1654
#eval EVAL-q094-4self-T q094_e10dP5self_s50_pfT_math500_T0.6 result:runs/E10d-P5self/global_step_50/actor/huggingface math500 4 0.6 16384 no_think pf=32313
#eval EVAL-q094-4self-N q094_e10dP5self_s50_pfN_math500_T0.6 result:runs/E10d-P5self/global_step_50/actor/huggingface math500 4 0.6 16384 no_think pf=1654
#eval EVAL-q094-17p5-T q094_e10bP5_s150_pfT_math500_T0.6 result:runs/E10b-P5/global_step_150/actor/huggingface math500 4 0.6 16384 no_think pf=32313
#eval EVAL-q094-17p5-N q094_e10bP5_s150_pfN_math500_T0.6 result:runs/E10b-P5/global_step_150/actor/huggingface math500 4 0.6 16384 no_think pf=1654
#eval EVAL-nothink16k-17base-b3 base17_nothink16k_b3_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 no_think
#eval EVAL-nothink16k-4Bbase-b3 base4_nothink16k_b3_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 no_think
#gpus E12b-P0think-8k-c100 4
#gpus PROBE-E12B-P0THINK-8K50-4K 4
#gpus E12b-P0think-8k50-4k 4
#gpus PROBE-E12B-P0THINK-7K 4
#gpus E12b-P0think-7k 4
#gpus PROBE-E12B-P0THINK-6K 4
#gpus E12b-P0think-6k 4
#gpus PROBE-E12B-P0THINK-12K 4
#gpus E12b-P0think-12k 4
#gpus E10b-P5q4kt 4
#gpus E10b-P5selfp2 4
#eval EVAL-u1-refs base17_think_u1_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 think
#eval EVAL-u1-refs base4_think_u1_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 think
#eval EVAL-u1-ood base17_think_u1_minervamath_T0.6 Qwen/Qwen3-1.7B minervamath 4 0.6 16384 think
#eval EVAL-u1-refs base17_nothink_u1_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 no_think
#eval EVAL-u1-refs base4_nothink_u1_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 no_think
#eval EVAL-u1-refs e12b_P0think-8k_s50_u1_math500_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-u1-refs e12b_P5tail-8k_s50_u1_math500_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#eval EVAL-u1-refs e10b_P5self_s50_u1_math500_T0.6 result:runs/E10b-P5self/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#eval EVAL-u1-refs e10b_P5q093t_s50_u1_math500_T0.6 result:runs/E10b-P5q093t/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#eval EVAL-u1-t4k e10b_P5q4kt_s50_u1_math500_T0.6 result:runs/E10b-P5q4kt/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#eval EVAL-u1-q098p base17_think_p1_u1_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 think us=p1
#eval EVAL-u1-q098p base17_think_p2_u1_math500_T0.6 Qwen/Qwen3-1.7B math500 4 0.6 16384 think us=p2
#eval EVAL-u1-q098p base4_think_p1_u1_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 think us=p1
#eval EVAL-u1-q098p base4_think_p2_u1_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 think us=p2
#eval EVAL-u1-refs e12b_P0think-4k_s50_u1_math500_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-u1-refs e10d_P5self_s50_u1_math500_T0.6 result:runs/E10d-P5self/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#eval EVAL-u1-refs e12d_P0think-8k_s50_u1_math500_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-u1-refs e12d_P5tail-8k_s50_u1_math500_T0.6 result:runs/E12d-P5tail-8k/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#eval EVAL-u1-seed e12b_P0think-8k_s50_u1s1000_math500_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 think seed=1000
#eval EVAL-u1-seed e12b_P5tail-8k_s50_u1s1000_math500_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface math500 4 0.6 16384 no_think seed=1000
#eval EVAL-u1-seed base4_think_u1s1000_math500_T0.6 Qwen/Qwen3-4B math500 4 0.6 16384 think seed=1000
#eval EVAL-u1-seed base8_think_u1s1000_math500_T0.6 Qwen/Qwen3-8B math500 4 0.6 16384 think seed=1000
#eval EVAL-u1-ood base17_think_u1_olympiadbench_T0.6 Qwen/Qwen3-1.7B olympiadbench 4 0.6 16384 think
#eval EVAL-u1-ood ood_17q093_u1_minervamath_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface minervamath 4 0.6 16384 think
#eval EVAL-u1-ood ood_17chain_u1_minervamath_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface minervamath 4 0.6 16384 no_think
#eval EVAL-u1-ood ood_17q093_u1_olympiadbench_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 think
#eval EVAL-u1-ood ood_17chain_u1_olympiadbench_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface olympiadbench 4 0.6 16384 no_think
#eval EVAL-u1-ood ood_17base_u1_mmlupro1k_T0.6 Qwen/Qwen3-1.7B mmlupro1k 4 0.6 16384 think
#eval EVAL-u1-ood ood_17q093_u1_mmlupro1k_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface mmlupro1k 4 0.6 16384 think
#eval EVAL-u1-ood ood_17chain_u1_mmlupro1k_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface mmlupro1k 4 0.6 16384 no_think
#eval EVAL-u1-ood ood_17base_u1_aime24_T0.6 Qwen/Qwen3-1.7B aime24 4 0.6 16384 think
#eval EVAL-u1-ood ood_17q093_u1_aime24_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface aime24 4 0.6 16384 think
#eval EVAL-u1-ood ood_17chain_u1_aime24_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface aime24 4 0.6 16384 no_think
#eval EVAL-u1-ood ood_17base_u1_aime25_T0.6 Qwen/Qwen3-1.7B aime25 4 0.6 16384 think
#eval EVAL-u1-ood ood_17q093_u1_aime25_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface aime25 4 0.6 16384 think
#eval EVAL-u1-ood ood_17chain_u1_aime25_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface aime25 4 0.6 16384 no_think
#eval EVAL-u1-ood ood_17base_u1_amc23_T0.6 Qwen/Qwen3-1.7B amc23 4 0.6 16384 think
#eval EVAL-u1-ood ood_17q093_u1_amc23_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface amc23 4 0.6 16384 think
#eval EVAL-u1-ood ood_17chain_u1_amc23_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface amc23 4 0.6 16384 no_think
#eval EVAL-u1-x10 x10_17q093_nothink_u1_math500_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#eval EVAL-u1-x10 x10_17q4k_nothink_u1_math500_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#eval EVAL-u1-x10 x10_4q093b_nothink_u1_math500_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface math500 4 0.6 16384 no_think
#forced EVAL-u1-t4k e10b_P5q4kt_s50_u1_math500_T0.6 result:runs/E10b-P5q4kt/global_step_50/actor/huggingface no_think budgets=16384
#forced EVAL-u1-q098p base17_think_p1_u1_math500_T0.6 Qwen/Qwen3-1.7B think budgets=16384
#forced EVAL-u1-q098p base17_think_p2_u1_math500_T0.6 Qwen/Qwen3-1.7B think budgets=16384
#forced EVAL-u1-q098p base4_think_p1_u1_math500_T0.6 Qwen/Qwen3-4B think budgets=16384
#forced EVAL-u1-q098p base4_think_p2_u1_math500_T0.6 Qwen/Qwen3-4B think budgets=16384
#forced EVAL-u1-refs base4_think_u1_math500_T0.6 Qwen/Qwen3-4B think budgets=1024,2048,3072,4096,6144,8192,16384
#forced EVAL-u1-refs e12b_P0think-8k_s50_u1_math500_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface think budgets=16384
#forced EVAL-u1-refs e12b_P5tail-8k_s50_u1_math500_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface no_think budgets=16384
#forced EVAL-u1-refs e12b_P0think-4k_s50_u1_math500_T0.6 result:runs/E12b-P0think-4k/global_step_50/actor/huggingface think budgets=16384
#forced EVAL-u1-refs e10b_P5q093t_s50_u1_math500_T0.6 result:runs/E10b-P5q093t/global_step_50/actor/huggingface no_think budgets=16384
#forced EVAL-u1-refs e10b_P5self_s50_u1_math500_T0.6 result:runs/E10b-P5self/global_step_50/actor/huggingface no_think budgets=16384
#forced EVAL-u1-f2 EVAL-u1-refs/e12b_P0think-8k_s50_u1_math500_T0.6_math500_think_t0.6.jsonl result:runs/E12b-P0think-8k/global_step_50/actor/huggingface think budgets=1024,2048,3072,4096,6144,8192,16384
#forced EVAL-u1-f2 EVAL-u1-refs/e12b_P0think-4k_s50_u1_math500_T0.6_math500_think_t0.6.jsonl result:runs/E12b-P0think-4k/global_step_50/actor/huggingface think budgets=1024,2048,3072,4096,6144,8192,16384
#forced EVAL-u1-f2 EVAL-u1-refs/e10b_P5q093t_s50_u1_math500_T0.6_math500_no_think_t0.6.jsonl result:runs/E10b-P5q093t/global_step_50/actor/huggingface no_think budgets=1024,2048,3072,4096,6144,8192,16384
#forced EVAL-u1-refs e10d_P5self_s50_u1_math500_T0.6 result:runs/E10d-P5self/global_step_50/actor/huggingface no_think budgets=16384
#forced EVAL-u1-refs e12d_P0think-8k_s50_u1_math500_T0.6 result:runs/E12d-P0think-8k/global_step_50/actor/huggingface think budgets=16384
#forced EVAL-u1-refs e12d_P5tail-8k_s50_u1_math500_T0.6 result:runs/E12d-P5tail-8k/global_step_50/actor/huggingface no_think budgets=16384
#forced EVAL-u1-ood base17_think_u1_minervamath_T0.6 Qwen/Qwen3-1.7B think budgets=1024,2048,3072,4096,6144,8192,16384
#forced EVAL-u1-ood ood_17q093_u1_minervamath_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface think budgets=16384
#forced EVAL-u1-ood ood_17chain_u1_minervamath_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface no_think budgets=16384
#forced EVAL-u1-ood base17_think_u1_olympiadbench_T0.6 Qwen/Qwen3-1.7B think budgets=1024,2048,3072,4096,6144,8192,16384
#forced EVAL-u1-ood ood_17q093_u1_olympiadbench_T0.6 result:runs/E12b-P0think-8k/global_step_50/actor/huggingface think budgets=16384
#forced EVAL-u1-ood ood_17chain_u1_olympiadbench_T0.6 result:runs/E12b-P5tail-8k/global_step_50/actor/huggingface no_think budgets=16384
#term PROBE-TERM4-N17THU1 17_base_think Qwen/Qwen3-1.7B EVAL-u1-refs/base17_think_u1_math500_T0.6_math500_think_t0.6.jsonl think
#term PROBE-TERM4-N4THU1 4B_base_think Qwen/Qwen3-4B EVAL-u1-refs/base4_think_u1_math500_T0.6_math500_think_t0.6.jsonl think
#term PROBE-TERM4-N093U1 17_P0think8k result:runs/E12b-P0think-8k/global_step_50/actor/huggingface EVAL-u1-refs/e12b_P0think-8k_s50_u1_math500_T0.6_math500_think_t0.6.jsonl think
#term PROBE-TERM4-N096A4KU1 17_P0think4k result:runs/E12b-P0think-4k/global_step_50/actor/huggingface EVAL-u1-refs/e12b_P0think-4k_s50_u1_math500_T0.6_math500_think_t0.6.jsonl think
#term PROBE-TERM4-N093BU1 4_P0think8k result:runs/E12d-P0think-8k/global_step_50/actor/huggingface EVAL-u1-refs/e12d_P0think-8k_s50_u1_math500_T0.6_math500_think_t0.6.jsonl think
#term PROBE-TERM4-N095AU1 17_P5q093t result:runs/E10b-P5q093t/global_step_50/actor/huggingface EVAL-u1-refs/e10b_P5q093t_s50_u1_math500_T0.6_math500_no_think_t0.6.jsonl no_think
#term PROBE-TERM4-NT4KU1 17_P5q4kt result:runs/E10b-P5q4kt/global_step_50/actor/huggingface EVAL-u1-t4k/e10b_P5q4kt_s50_u1_math500_T0.6_math500_no_think_t0.6.jsonl no_think
#eval EVAL-e12b-P0think-8k-c100 e12b_P0think-8k-c100_s75_math500_T0.6 result:runs/E12b-P0think-8k-c100/global_step_25/actor/huggingface math500 4 0.6 16384 think
#eval EVAL-e12b-P0think-8k-c100 e12b_P0think-8k-c100_s100_math500_T0.6 result:runs/E12b-P0think-8k-c100/global_step_50/actor/huggingface math500 4 0.6 16384 think
#term PROBE-TERM4-N096B75 17_P0think8kc100s75 result:runs/E12b-P0think-8k-c100/global_step_25/actor/huggingface EVAL-e12b-P0think-8k-c100/e12b_P0think-8k-c100_s75_math500_T0.6_math500_think_t0.6.jsonl think
#term PROBE-TERM4-N096BC 17_P0think8kc100s100 result:runs/E12b-P0think-8k-c100/global_step_50/actor/huggingface EVAL-e12b-P0think-8k-c100/e12b_P0think-8k-c100_s100_math500_T0.6_math500_think_t0.6.jsonl think
"""
ROW_KINDS = ("#eval", "#pair", "#budget", "#term", "#gpus", "#steptime")
# [z151] (Fable D12 11-11 16:3x) per-profile GPU counts: on 8gs (the me-west1 8 x B200s) each long training and its gate take all
# 8 GPUs, one training per machine (P1-b on one, Q-093b on the other); the other profiles keep the source's 4
GPUS_BY_PROFILE = {"8gs": {"PROBE-E12E-8K": 8, "E12e-P5tail-8k-nn": 8, "PROBE-E12D-P0THINK": 8, "E12d-P0think-8k": 8,
                           "PROBE-E12E-P0THINK": 8, "E12e-P0think-8k": 8, "PROBE-E12B-P0THINK-8G": 8, "E12b-P0think-8k-s43": 8,
                           "PROBE-E12D-P0THINK-8G": 8, "E12d-P0think-8k-s43": 8}}   # [z154]; [z156]
HEADER = """\
# Remote-node task queue, version 2. newnode-v2, profile {p}: {what}
# Written by scripts/queue_profiles.py (lab WO-001; v3 workflow 7.2) from tasks/rows_newnode_v1.txt + its new rows: edit the
# generator, not this file. Profiles 1g / 4g / 4g80 / 8g / 8g80 (+ the split pair 1gs / 8gs, z139); a node reads this file once the project
# sends it `queue {p}` (control verb, v3.0; tasks/queue.txt itself opens nothing). Under 80 GB: no 4B / 8B training.
# Stages: p0a gates, p0b same-machine anchors, p0c TERM4 (NRL8K, 8B think), p0d / p0dr Q-091 training / readings, p0e / p0er Q-090
# training / readings, p0f Q-074i, p0g Q-075c, p0x the 1.7B chain's AIME 32K anchor, p1a 2 x 2 beyond arms, p1b Q-074e (8B, 4 GPUs),
# p2f Q-074f, p2h Q-074h, p2g Q-074g; "# <stage> ~N B200 GPU-h" over each stage. Open stages start in file order, one task per free
# GPU set; a waiting multi-GPU task holds the GPUs as they free.
# Memory keys: one card >= 140 GB (B200 ~180 GB) -> bootstrap.sh --profile 140g, or those export lines into $OPD_ROOT/control.env by
# hand before setup (Fable 11-03: GC=0, LOGP_MB=4, VLLM_UTIL=0.5 on the B300 numerics); 80 GB cards -> --profile 80g.
# Hard conditions of every arm (as in v1): its starts local (.fetch_ok; the #fetch block), its local gate PASS (the deps), stage open;
# a multi-GPU arm only after the running ones show 5 steps without OOM and a median step <= 2 x its #steptime.
# -nn names (Fable 11-03): the lost H800 may still upload under the old names; before T0 the executor checks -pub once: an original's
# s50 .done already there -> pause the -nn row and point its readings back at the original (a one-word diff, pre-approved).
"""
WHAT = {"1g": "one GPU (B200 default, profiles/140g.env), P0 in series with the trainings interleaved, then Q-093, then P1-a",
        "4g": "2-4 GPUs (or 4 under 80 GB), P0 in parallel, then P1-a",
        "4g80": "4 x 80 GB, P0 in parallel, then P1-a and P1-b (8B on 4 GPUs)",
        "8g": "8 GPUs under 80 GB, P0 + P1-a, then P2 without 4B training",
        "8g80": "8 x 80 GB, P0 + P1-a, P1-b on 4 GPUs, P2 fills idle GPUs",
        "1gs": "one GPU beside an 8gs node: gates, P0-c, the historical-weight readings P0-f / P0-g / P0-x",
        "8gs": "8 x >= 80 GB beside a 1gs node (control/queue_name = 8gs): 8g80 without P0-c / f / g / x",
        "1gt": "one GPU beside a 1gs node (control/queue_name = 1gt): 1g's trainings and their readings (P0-d / e / b / dr / er, Q-093, P1-a)",
        "h2a": "a borrowed 8 x H200 (profiles/h200.env, `queue h2a`): the canary, Q-107's two 3K arms on 4 GPUs each, their readings, then the 4K / 6K zero-training readings",
        "h2b": "a second borrowed 8 x H200 (`queue h2b`): the canary and the 4K / 6K zero-training readings; nothing h2a trains",
        "h2c": "the borrowed 8-GPU machine after Q-107 (`queue h2c`): Q-108's OlympiadBench readings, the 4B chain (Q-109: gate, arm, readings) and the 3K arms continued to s100 (Q-110), the 3K arms' OlympiadBench readings (Q-111), then the 4B direct 8K arm continued to s100 (Q-112: gate, arm, readings)",
        "en2": "a cloud 8 x H200 (`queue en2`): Q-111's two s50 readings with a card-type anchor, Q-112 (the 4B direct 8K arm continued to s100: gate, arm, readings, same-machine references) and Q-113's Minerva Math readings"}


def source():
    lines = open(SRC).read().splitlines()
    spec = {}
    for l in lines:
        w = l.split()
        if len(w) >= 3 and not l.startswith("#") and w[1] in ("open", "auto", "locked"):
            for s in w[2:]:
                spec[s.split(":")[0]] = s
    rows = [l for l in lines if l.split()[:1] and l.split()[0] in ROW_KINDS] + NEW_ROWS.splitlines()
    return spec, rows


def stage_specs(name, spec):
    return [spec[s[4:]] if s.startswith("src:") else s for s in STAGES[name]]


def render(p):
    spec, rows = source()
    out = [HEADER.format(p=p, what=WHAT[p]).rstrip("\n"), "version: 2"]
    tasks = []
    for st in PROFILES[p]:
        ss = stage_specs(st, spec)
        tasks += [s.split(":")[0] for s in ss]
        est = [(s.split(":")[0], EST[s.split(":")[0]]) for s in ss if s.split(":")[0] in EST]
        if est:
            out.append(f"# {st} ~{sum(h for _, h in est):.1f} B200 GPU-h: " + ", ".join(f"{t} {h:g}" for t, h in est))
        out.append(f"{st:<10} {'auto' if (p, st) in AUTO else 'open'}   " + " ".join(ss))
    have = set(tasks)
    out.append("")
    g8 = GPUS_BY_PROFILE.get(p, {})
    out += [f"#gpus {r.split()[1]} {g8[r.split()[1]]}" if r.split()[0] == "#gpus" and r.split()[1] in g8 else r
            for r in rows if r.split()[0] in ("#gpus", "#steptime") and r.split()[1] in have]
    out.append("")
    seen = []
    for st in PROFILES[p]:
        seen += [f for f in FETCH.get(st, []) if f not in seen]
    out += [f"#fetch result:runs/{f}/actor/huggingface" for f in seen]
    out += [f"#fetch dataset:{d}" for st in PROFILES[p] for d in DFETCH.get(st, [])]   # [z158c] datasets a stage reads from data/hf
    out.append("")
    out += [r for r in rows if r.split()[0] not in ("#gpus", "#steptime") and r.split()[1] in have]
    return "\n".join(out) + "\n"


def path(p):
    return os.path.join(RB, "tasks", f"queue_{p}.txt")


def main(argv):
    if argv[:1] == ["--select"]:
        p = argv[1] if len(argv) > 1 else ""
        if p not in PROFILES:
            sys.exit(f"profile must be one of {' '.join(PROFILES)}")
        sys.exit(f"--select is gone (v3.0): push tasks/queue_{p}.txt and send the node `queue {p}`; tasks/queue.txt opens nothing")
    bad = []
    for p in PROFILES:
        text = render(p)
        if argv[:1] == ["--check"]:
            if not os.path.exists(path(p)) or open(path(p)).read() != text:
                bad.append(p)
        else:
            open(path(p), "w").write(text)
    if bad:
        print("stale profile files: " + " ".join(bad)); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
