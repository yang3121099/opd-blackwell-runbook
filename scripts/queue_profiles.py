"""[v2.5.9z137] (lab WO-001, the v3 workflow 11-09 section 7.2) newnode-v2: one queue per machine size, P0 first.

The next machine's size is unknown, so tasks/queue_<profile>.txt is written here for five profiles from one row source,
tasks/queue_newnode_v1.txt (z136's live queue, kept unchanged for provenance), plus the rows WO-001 adds (the same-machine
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
       python3 scripts/queue_profiles.py --select P   copy tasks/queue_P.txt to tasks/queue.txt (at T0, before cold_start.sh)
"""
import os
import sys

RB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(RB, "tasks", "queue_newnode_v1.txt")

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
    "p1q": ["PROBE-E12B-P0THINK", "E12b-P0think-8k:PROBE-E12B-P0THINK",
            "EVAL-e12b-P5tail-8k-nn-b2:E12b-P5tail-8k", "EVAL-think16k-17base-b2", "PROBE-TERM4-N17THB2:EVAL-think16k-17base-b2",
            "PROBE-TERM4-NP5B2", "EVAL-e12b-P0think-8k:E12b-P0think-8k",
            "PROBE-TERM4-N093:EVAL-e12b-P0think-8k", "EVAL-budA-e12bP0think-8k:E12b-P0think-8k", "EVAL-ext32k-17p0think8k:E12b-P0think-8k"],
    # P1-a the 2 x 2 beyond arms + readings + TERM4
    "p1a": ["src:PROBE-FRET", "src:E10b-P5fret", "src:E10b-P5topd", "src:E10b-P5fret-topd", "src:EVAL-e10b-P5fret", "src:EVAL-e10b-P5topd",
            "src:EVAL-e10b-P5fret-topd", "src:PROBE-TERM4-FRET"],
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
       "EVAL-e12b-P5tail-8k-nn-b2": 1.2, "EVAL-think16k-17base-b2": 1.0, "PROBE-TERM4-N17THB2": 0.3, "PROBE-TERM4-NP5B2": 0.5}   # [z146] Q-093: thinking answers fill the 8K window (training ~1.5 x Q-091)   # P1-a arms: #steptime 180 s x 50 steps
PROFILES = {
    "1g": P0_SERIAL + ["p1q", "p1a"],   # [z146] Q-093 after P0, before P1-a (1gt inherits it; 1gs does not: a 1gs / 1gt pair shares the gates only)
    "4g": P0_PARALLEL + ["p1a"],
    "4g80": P0_PARALLEL + ["p1a", "p1b"],
    "8g": P0_PARALLEL + ["p1a", "p2f", "p2g"],
    "8g80": P0_PARALLEL + ["p1a", "p1b", "p2f", "p2h", "p2g"],
}
# [z139] (the lead 11-10) two machines at once, each with its own disk and runs/: a 1 x H200 that has run p0a and is in p0c, and an
# 8 x H200. 1gs = the 1-GPU box's share (the readings of historical weights, which need no new training); 8gs = 8g80 without those
# stages (every new training and its readings on one machine). Together they are 8g80's stages, only p0a (the local gates) on both.
# The 8-GPU node reads tasks/queue_8gs.txt through control/queue_name (env.sh exports QUEUE_FILE); the 1-GPU node keeps queue.txt.
SHARE_1G = ["p0a", "p0c", "p0f", "p0g", "p0x"]
PROFILES["1gs"] = SHARE_1G
PROFILES["8gs"] = [s for s in PROFILES["8g80"] if s == "p0a" or s not in SHARE_1G]
# [z142] (the lead 11-10: the 8 x H200 reclaimed, a second 1 x H200 back) 1gt = the 1-GPU trainings and their readings beside a 1gs node:
# 1g without the 1gs stages (p0d, p0e, p0b, p0dr, p0er, P1-a); with 1gs it is 1g's stages, only p0a on both
PROFILES["1gt"] = [s for s in PROFILES["1g"] if s == "p0a" or s not in SHARE_1G]
# each stage's starts pulled ahead (#fetch, from T0 + 8 min): the source's 14 lines plus the 1.7B chain's s50, in stage order
FETCH = {
    "p0a": ["E10b-P5/global_step_150", "E10b-P5tail/global_step_50"],
    "p0c": ["E12b-P5-8k/global_step_50"],
    "p0b": ["E12b-P5tail-8k/global_step_50", "E12d-P5tail-8k/global_step_50"],
    "p0f": ["E12d-P5tail-8k-s43/global_step_50"],
    "p0g": ["E10d-P5tail/global_step_50", "E10d-P5tail-s43/global_step_50"],
    "p1b": ["E10e-P5self/global_step_50", "E12e-P5tail-8k/global_step_25", "E10e-P5tail/global_step_50"],
    "p2f": ["E10c-P5-fix4/global_step_150", "E12c-P5-8k/global_step_25", "E12c-P5-8k/global_step_50", "E10c-P5tail-c50/global_step_30"],
}
NEW_ROWS = """\
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
"""
ROW_KINDS = ("#eval", "#pair", "#budget", "#term", "#gpus", "#steptime")
HEADER = """\
# Remote-node task queue, version 2. newnode-v2, profile {p}: {what}
# Written by scripts/queue_profiles.py (lab WO-001; v3 workflow 7.2) from tasks/queue_newnode_v1.txt + its new rows: edit the
# generator, not this file. Profiles 1g / 4g / 4g80 / 8g / 8g80 (+ the split pair 1gs / 8gs, z139); at T0, before cold_start.sh: `python3 scripts/queue_profiles.py
# --select <profile>` copies tasks/queue_<profile>.txt to tasks/queue.txt (one commit, main + hf). Under 80 GB: no 4B / 8B training.
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
        "1gt": "one GPU beside a 1gs node (control/queue_name = 1gt): 1g's trainings and their readings (P0-d / e / b / dr / er, Q-093, P1-a)"}


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
        out.append(f"{st:<10} open   " + " ".join(ss))
    have = set(tasks)
    out.append("")
    out += [r for r in rows if r.split()[0] in ("#gpus", "#steptime") and r.split()[1] in have]
    out.append("")
    seen = []
    for st in PROFILES[p]:
        seen += [f for f in FETCH.get(st, []) if f not in seen]
    out += [f"#fetch result:runs/{f}/actor/huggingface" for f in seen]
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
        open(os.path.join(RB, "tasks", "queue.txt"), "w").write(open(path(p)).read())
        print(f"tasks/queue.txt = tasks/queue_{p}.txt")
        return 0
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
