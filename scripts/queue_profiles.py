"""[v2.5.9z137] (lab WO-001, the v3 workflow 11-09 section 7.2) newnode-v2: one queue per machine size, P0 first.

The next machine's size is unknown, so tasks/queue_<profile>.txt is written here for five profiles from one row source,
tasks/queue_newnode_v1.txt (z136's live queue, kept unchanged for provenance), plus the rows WO-001 adds (the same-machine
anchors, the 1.7B chain's AIME 32K anchor, the 8B base model's thinking-mode TERM4). tasks/queue.txt is a copy of one of them
(1g until a machine arrives); every node-side reader keeps reading tasks/queue.txt, nothing on the node changes.

  1g    one GPU: P0 in series a > c > b > d > e > f > g (+ the AIME anchor), then P1-a; no multi-GPU task
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
    "p0c": ["src:PROBE-TERM4-NRL8K", "PROBE-TERM4-N8BTHINK"],
    # P0-b same-machine anchors: the 1.7B chain s50 and the 1.7B RL-only s50 at 16K T0.6 / T1.0, the 4B chain s50 at 16K (table 1's CI)
    "p0b": ["EVAL-e12b-P5tail-8k-nn:E12b-P5tail-8k", "EVAL-e12b-P5-8k-nn:E12b-P5-8k", "src:EVAL-e12d-P5tail-8k-nn"],
    # P0-d Q-091 E12b-P0-8k + its readings (research_program 3s)
    "p0d": ["src:E12b-P0-8k", "src:EVAL-e12b-P0-8k", "src:EVAL-budA-e12bP0-8k", "src:EVAL-ext32k-17p0l8k", "src:PROBE-TERM4-N091"],
    # P0-e Q-090 E12b-P5-8k-lenpen + its readings
    "p0e": ["src:E12b-P5-8k-lenpen", "src:EVAL-e12b-P5-8k-lenpen", "src:EVAL-budA-e12bP5-8k-lenpen", "src:EVAL-ext32k-17p5l8klp",
            "src:PROBE-TERM4-N090"],
    # P0-f Q-074i's three readings (the 4B second start) + its TERM4
    "p0f": ["src:EVAL-e12d-P5tail-8k-s43-nn", "src:EVAL-ext32k-4Bp5tail8k-s43-nn", "src:EVAL-budA-e12dP5tail-8k-s43-nn",
            "src:PROBE-TERM4-N074I"],
    # P0-g Q-075c on one machine: E10d-P5tail s50 and -s43 s50 at 16K and 32K, the s43 TERM4 (H2)
    "p0g": ["src:EVAL-e10d-P5tail-nn", "src:EVAL-budA-e10dP5tail-nn", "src:EVAL-e10d-P5tail-s43-nn", "src:EVAL-budA-e10dP5tail-s43-nn",
            "src:PROBE-TERM4-N075C"],
    # P0 tail: the 1.7B chain s50 AIME24 / 25 32K on the same machine (Q-090 H4 compares with the chain at AIME 32K)
    "p0x": ["EVAL-ext32k-17p5tail8k-nn:E12b-P5tail-8k"],
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
P0_SERIAL = ["p0a", "p0c", "p0b", "p0d", "p0e", "p0f", "p0g", "p0x"]   # v3 7.2: a > c > b > d > e > f > g
P0_PARALLEL = ["p0a", "p0d", "p0e", "p0c", "p0b", "p0f", "p0g", "p0x"]  # the two ~8 GPU-h trainings take GPUs right after the gates
PROFILES = {
    "1g": P0_SERIAL + ["p1a"],
    "4g": P0_PARALLEL + ["p1a"],
    "4g80": P0_PARALLEL + ["p1a", "p1b"],
    "8g": P0_PARALLEL + ["p1a", "p2f", "p2g"],
    "8g80": P0_PARALLEL + ["p1a", "p1b", "p2f", "p2h", "p2g"],
}
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
"""
ROW_KINDS = ("#eval", "#pair", "#budget", "#term", "#gpus", "#steptime")
HEADER = """\
# Remote-node task queue, version 2. newnode-v2, profile {p}: {what}
# Written by scripts/queue_profiles.py (lab WO-001; v3 workflow 7.2) from tasks/queue_newnode_v1.txt + its new rows: edit the
# generator, not this file. Profiles 1g / 4g / 4g80 / 8g / 8g80; at T0, before cold_start.sh: `python3 scripts/queue_profiles.py
# --select <profile>` copies tasks/queue_<profile>.txt to tasks/queue.txt (one commit, main + hf). Under 80 GB: no 4B / 8B training.
# Stages: p0a gates, p0b same-machine anchors, p0c TERM4 (NRL8K, 8B think), p0d Q-091, p0e Q-090, p0f Q-074i, p0g Q-075c, p0x the
# 1.7B chain's AIME 32K anchor, p1a 2 x 2 beyond arms, p1b Q-074e (8B, 4 GPUs), p2f Q-074f, p2h Q-074h, p2g Q-074g. Open stages
# start in file order, one task per free GPU set; a waiting multi-GPU task holds the GPUs as they free.
# Hard conditions of every arm (as in v1): its starts local (.fetch_ok; the #fetch block), its local gate PASS (the deps), stage open;
# a multi-GPU arm only after the running ones show 5 steps without OOM and a median step <= 2 x its #steptime.
# -nn names (Fable 11-03): the lost H800 may still upload under the old names; before T0 the executor checks -pub once: an original's
# s50 .done already there -> pause the -nn row and point its readings back at the original (a one-word diff, pre-approved).
"""
WHAT = {"1g": "one GPU, P0 in series a > c > b > d > e > f > g, then P1-a",
        "4g": "2-4 GPUs (or 4 under 80 GB), P0 in parallel, then P1-a",
        "4g80": "4 x 80 GB, P0 in parallel, then P1-a and P1-b (8B on 4 GPUs)",
        "8g": "8 GPUs under 80 GB, P0 + P1-a, then P2 without 4B training",
        "8g80": "8 x 80 GB, P0 + P1-a, P1-b on 4 GPUs, P2 fills idle GPUs"}


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
