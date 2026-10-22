"""[v2.5.9z38] (Q-060, the reviewer's phase step; ops-lead: on the node, not by watching) per-step training length and clip ratio of a
training run, and the first step whose mean response length reaches 0.9 x the response cap (the 4K wall: P5 s16, route s32, mix never).

    python len_wall.py <run dir> <task>

Reads <run>/train.log (verl's "step:N - key:value - ..." lines, as metrics_jsonl.py) and the cap from <run>/config.yaml
("max_response_length N", else "max_response_length=N", else 4096). Writes <run>/len_by_step.csv (step, mean_len, clip_ratio) and,
at the first step with mean_len >= 0.9 x cap, <run>/wall.json (task, step, mean_len, clip_ratio, cap, utc) once; prints
"WALL <task> s<step> len <mean_len> clip <clip_ratio> cap <cap>" only when it writes wall.json. Exit 0 (no train.log: nothing).
upload_sync.sh calls it every round of an E10b / E10c / E10d run: the CSV goes up every LEN_EVERY (5) new steps, wall.json and an
event wall_cross when it appears."""
import json
import os
import re
import sys
import time

STEP = re.compile(r"(?:^|[ )\]])step:(\d+) - (.*)$")


def rows_of(text):
    """{step: (mean_len, clip_ratio)} from a verl console log; a step logged twice (a resume) keeps its last line."""
    out = {}
    for line in text.splitlines():
        m = STEP.search(line)
        if not m:
            continue
        kv = {}
        for part in m.group(2).split(" - "):
            k, _, v = part.strip().rpartition(":")
            try:
                kv[k] = float(v)
            except ValueError:
                pass
        if "response_length/mean" in kv:
            out[int(m.group(1))] = (kv["response_length/mean"], kv.get("response_length/clip_ratio"))
    return out


def cap_of(cfg):
    m = re.search(r"max_response_length[ =]+(\d+)", cfg or "")
    return int(m.group(1)) if m else 4096


def main(run, task):
    try:
        text = open(os.path.join(run, "train.log"), errors="ignore").read()
    except OSError:
        return 0
    try:
        cfg = open(os.path.join(run, "config.yaml"), errors="ignore").read()
    except OSError:
        cfg = ""
    rows, cap = rows_of(text), cap_of(cfg)
    tmp = os.path.join(run, "len_by_step.csv.tmp")
    with open(tmp, "w") as f:
        f.write("step,mean_len,clip_ratio\n")
        for s in sorted(rows):
            L, c = rows[s]
            f.write(f"{s},{round(L, 1)},{'' if c is None else round(c, 4)}\n")
    os.replace(tmp, os.path.join(run, "len_by_step.csv"))
    wall = os.path.join(run, "wall.json")
    first = next((s for s in sorted(rows) if rows[s][0] >= 0.9 * cap), None)
    if first is not None and not os.path.exists(wall):
        L, c = rows[first]
        json.dump({"task": task, "step": first, "mean_len": round(L, 1), "clip_ratio": c, "cap": cap, "threshold": 0.9 * cap,
                   "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(wall, "w"), indent=1)
        print(f"WALL {task} s{first} len {round(L)} clip {c} cap {cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
