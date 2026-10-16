# Reference numbers (metrics only)

These are the project's own runs of the same recipes with the smaller student, kept here so the node's results can be
read against them. Single runs; trends, not conclusions.

| File / table | What it is |
|---|---|
| `E10b-P2_reference_H100_first10.json` | Steps 0-10 of the 1.7B P2 pilot on 1x H100 (same recipe as `E10b-P2`; step-0 validation 74.0, 256-513 s per step). `scripts/compare_first_arm.py` puts the node's first 10 steps next to it. |

**0.6B student, 16K re-evaluation (E11), MATH500 avg@4 at T=0.6, non-thinking** (the 1.7B stage E11b repeats this):

| Start | Accuracy |
|---|---|
| P0 (base Qwen3-0.6B) | 51.05 |
| P2 (OPD) step 150 | 61.8 (+10.75 over P0; paired 95% interval 8.5-13.0) |
| P6 (top-20% \|A\|) step 150 | 62.2 |
| P6r (other 80%) step 150 | 60.95 |
| P4-lowlr step 150 | 10.95 (collapsed to ~29-token answers) |

**0.6B E12 (GRPO), in-training validation (MATH500 first 100, avg@4, 4K)**: P0 56.25 at step 0, 59.75 at step 25;
mean response length grew from about 1.2K to 2.2K tokens and truncation from about 1% to 16% (pure RL lengthens too).
