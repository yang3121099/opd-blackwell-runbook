"""First-arm comparison: the node's E10b-P2 steps 0-10 side by side with the reference H100 run of the same recipe
(reference/E10b-P2_reference_H100_first10.json). Written to $OPD_ROOT/logs/compare_<host>.json; heartbeat.sh uploads
it once to status/compare_<host>.json when E10b-P2 has reached step 10.
What to expect: the same data order (seed 42), so the same prompts at each step; but rollouts and validation are
sampled (T=1.0 / T=0.7) and the kernels differ (SDPA here, flash-attn there), so values are close, not identical.
Step 0 validation should agree within sampling noise (about +-7 points on 100 problems x 2 samples); step 1 metrics
should be in the same range; later steps may drift apart. Usage: python compare_first_arm.py"""
import json, os, re, socket, sys

root, runbook = os.environ["OPD_ROOT"], os.environ["RUNBOOK"]
ref = json.load(open(os.path.join(runbook, "reference", "E10b-P2_reference_H100_first10.json")))
log = os.path.join(root, "runs", "E10b-P2", "train.log")
KEYS = ["val-core/math500/reward/mean@2", "actor/pg_loss", "actor/entropy", "response_length/mean",
        "response_length/clip_ratio", "critic/score/mean", "opd/adv_abs_mean", "timing_s/step"]
mine = {}
for line in open(log, errors="ignore"):
    m = re.search(r"(?:^|[ )])step:(\d+) - (.*)$", line)
    if not m or int(m.group(1)) > 10:
        continue
    row = {}
    for kv in m.group(2).split(" - "):
        k, _, v = kv.strip().rpartition(":")
        if k in KEYS:
            try:
                row[k] = round(float(v), 6)
            except ValueError:
                pass
    mine[m.group(1)] = row
if "10" not in mine:
    sys.exit("E10b-P2 has not reached step 10 yet")
table = {s: {k: {"node": mine.get(s, {}).get(k), "reference_h100": ref["steps"].get(s, {}).get(k)} for k in KEYS
             if k in mine.get(s, {}) or k in ref["steps"].get(s, {})} for s in sorted(set(mine) | set(ref["steps"]), key=int)}
v0n, v0r = mine.get("0", {}).get(KEYS[0]), ref["steps"].get("0", {}).get(KEYS[0])
out = {"host": socket.gethostname(), "reference": ref["run"], "reference_hardware": ref["hardware"],
       "step0_val": {"node": v0n, "reference_h100": v0r,
                     "within_noise": None if v0n is None or v0r is None else abs(v0n - v0r) <= 0.07},
       "steps": table}
dst = os.path.join(root, "logs", f"compare_{socket.gethostname().split('.')[0]}.json")
json.dump(out, open(dst, "w"), indent=1)
print(dst, "step-0 val node", v0n, "reference", v0r)
