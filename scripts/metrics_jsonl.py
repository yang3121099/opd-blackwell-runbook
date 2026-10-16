"""One JSON object per training step from a verl console log ("step:N - key:value - key:value ...").
Usage: python metrics_jsonl.py train.log > metrics.jsonl"""
import json, re, sys
for line in open(sys.argv[1], errors="ignore"):
    m = re.search(r"(?:^|[ )])step:(\d+) - (.*)$", line)
    if not m:
        continue
    row = {"step": int(m.group(1))}
    for kv in m.group(2).split(" - "):
        k, _, v = kv.strip().rpartition(":")
        try:
            row[k] = float(v)
        except ValueError:
            pass
    print(json.dumps(row))
