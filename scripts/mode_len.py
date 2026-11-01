"""[v2.5.9z108] (Q-083 E10b-P5mode; research_program §3i: "len_by_step 按 token 分开统计") per-step response length of each mode token.

    python mode_len.py <run dir> <task> [--tokenizer <path>]

Reads verl's rollout dump of the run (<run>/mode_dump/<step>.jsonl: "input", "output" per row; run_arm.sh sets ROLLOUT_DUMP there for
E10b-P5mode), finds each row's token in its input ('[long] ' or '[short] ', the first one; neither: none) and counts the output's tokens
with the student's tokenizer (--tokenizer, else STUDENT_17B). Per step and token: rows, mean length, the share at the cap (>= 0.98 x
cap: a re-tokenized truncated response); the wall of each token = its first step with mean >= 0.9 x cap, as len_wall.py (cap from
<run>/config.yaml, else 4096). Writes <run>/len_by_token.json (task, cap, rows, wall_long, wall_short, utc); a step file already
counted (same size) is read from <run>/.mode_len_cache.json. Prints "MODE_LEN <task> steps <n> wall_long <s> wall_short <s>".
Exit 0 (no dump: nothing written).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

TOKENS = ("[long]", "[short]")


def token_of(text):
    """'[long]', '[short]' or None: whichever of the two (followed by a space) comes first in the rendered prompt."""
    t = str(text); best = None
    for tok in TOKENS:
        i = t.find(tok + " ")
        if i >= 0 and (best is None or i < best[0]):
            best = (i, tok)
    return best[1] if best else None


def cap_of(run):
    try:
        s = open(os.path.join(run, "config.yaml")).read()
    except OSError:
        return 4096
    m = re.search(r"max_response_length[ =:]+(\d+)", s)
    return int(m.group(1)) if m else 4096


def step_stats(lines, count):
    """{token: [n, sum of lengths, n at cap-candidate lengths list]} for one step's dump lines."""
    out = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tok = token_of(r.get("input", "")) or "none"
        out.setdefault(tok, []).append(count(r.get("output", "")))
    return out


def summarize(per_step, cap):
    """rows (step, per token n / mean_len / trunc) and each token's wall step (mean >= 0.9 x cap)."""
    rows, walls = [], {"[long]": None, "[short]": None}
    for step in sorted(per_step):
        row = {"step": step}
        for tok, lens in sorted(per_step[step].items()):
            name = tok.strip("[]")
            mean = sum(lens) / len(lens)
            row[name] = {"n": len(lens), "mean_len": round(mean, 1), "trunc": round(sum(1 for x in lens if x >= 0.98 * cap) / len(lens), 4)}
            if tok in walls and walls[tok] is None and mean >= 0.9 * cap:
                walls[tok] = step
        rows.append(row)
    return rows, walls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("task"); ap.add_argument("--tokenizer", default=os.environ.get("STUDENT_17B", ""))
    a = ap.parse_args()
    files = glob.glob(os.path.join(a.run, "mode_dump", "*.jsonl"))
    if not files:
        return 0
    cache_p = os.path.join(a.run, ".mode_len_cache.json")
    try:
        cache = json.load(open(cache_p))
    except (OSError, ValueError):
        cache = {}
    count = None
    per_step = {}
    for f in files:
        name = os.path.basename(f)
        m = re.fullmatch(r"(\d+)\.jsonl", name)
        if not m:
            continue
        size = os.path.getsize(f)
        c = cache.get(name)
        if not (c and c.get("size") == size):
            if count is None:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(a.tokenizer)
                count = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
            c = {"size": size, "lens": step_stats(open(f).read().splitlines(), count)}
            cache[name] = c
        per_step[int(m.group(1))] = c["lens"]
    tmp = f"{cache_p}.tmp.{os.getpid()}"
    json.dump(cache, open(tmp, "w")); os.replace(tmp, cache_p)
    cap = cap_of(a.run)
    rows, walls = summarize(per_step, cap)
    rec = {"task": a.task, "cap": cap, "wall_long": walls["[long]"], "wall_short": walls["[short]"], "rows": rows,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "mode_len.py (verl rollout dump, re-tokenized)"}
    out = os.path.join(a.run, "len_by_token.json"); tmp = f"{out}.tmp.{os.getpid()}"
    json.dump(rec, open(tmp, "w"), indent=1); os.replace(tmp, out)
    print(f"MODE_LEN {a.task} steps {len(rows)} wall_long {walls['[long]']} wall_short {walls['[short]']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
