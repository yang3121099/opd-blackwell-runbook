"""[v2.5.9z100] The cold start manifest (Fable's requirements v26 §11, 10-27, after the B300 was lost): what an empty machine
needs to run the queue again, derived from git alone (tasks/queue.txt and the runners' defaults). v2.6's compiler will write
the same format. Nothing is fetched here; cold_start.sh reads the manifest on the new machine.

  models     the four base models (always) and, each with the open tasks that use it, the VLM models, the 32B teacher and every
             Hugging Face id or local:models/<name> an open stage's #eval row names (cold_start.sh skips one whose tasks are all done)
  envs       the four venvs, each with its recipe and the import that checks it
  data       the training and evaluation data with their preparation
  artifacts  the probe labels: fetched from the private repository and checked (sha256), else the probes rerun as a NEW label
             version (Fable 10-27: not the same labels as Q-072's out-of-fold ones)
  weights    every result:runs/<task>/global_step_<n>/actor/huggingface an open stage's #eval / #term row reads, and every
             open E12 arm's start (run_e12.sh's naming), each with the tasks that need it
  pairs      every reference an open stage's #pair / #budget row and every #strict / #term row reads (<task dir>/<file>.jsonl)

Usage: python coldstart_manifest.py <queue.txt> <out.json>
"""
import json
import re
import sys

E12_PARSE = {"P5-kl": "P5", "P5-stop": "P5", "P5-stop2": "P5", "P5-8k": "P5", "P5-16k": "P5", "P5-r3": "P5", "P5-16k-kl": "P5",
             "P5-8k-s44": "P5-s44", "P5tail-8k": "P5tail", "P5tail-8k-s43": "P5tail-s43", "P5tail-8k-s44": "P5tail-s44",
             "P5tail-16k": "P5tail", "P5self-8k": "P5self", "P5-16k-s44": "P5-s44", "P5-16k-s43": "P5-s43", "P5-8k-s43": "P5-s43",
             "P5-prog-up": "P5-prog-up"}   # run_e12.sh lines 29-39
ENVS = [
    {"name": "venv", "recipe": "scripts/install_uv.sh (overnight.sh step 3)", "check": "import torch, vllm, verl"},
    {"name": "venv-fa", "recipe": "scripts/fa_probe.sh <gpu> (a copy of venv + the pinned flash-attn wheel; GPU smoke); if flash_attn "
                                  "does not import (old glibc): scripts/fa_build.sh (from source, ~40 min), then FA-PROBE again",
     "needs": "venv", "check": "import flash_attn"},
    {"name": "venv-vl", "recipe": "scripts/build_venv_vl.sh (vLLM 0.11.0, transformers 4.57)", "needs": "venv", "check": "import vllm, transformers"},
    {"name": "venv-ifeval", "recipe": "scripts/build_venv_ifeval.sh (no torch)", "check": "import langdetect"},
]
DATA = [
    {"name": "data/dapo_train.parquet (+ MATH-500)", "recipe": "overnight.sh step 4 (code/train/prep_data.py)"},
    {"name": "data/dapo_train_skip1600_s4242.parquet", "recipe": "the progressive-window arms' skip file (run_e12.sh PROG)", "when": "E12b-P5-prog"},
    {"name": "data/mmk12_*.parquet", "recipe": "code/train/prep_mmk12.py", "when": "E15-"},
    {"name": "benchmarks", "recipe": "gen_eval.py loads them from Hugging Face datasets at a pinned revision when a reading starts"},
]
ARTIFACTS = [
    {"name": "labels_17.jsonl", "repo_path": "runs/PROBE-DIFF17/labels_17.jsonl", "node_path": "runs/PROBE-DIFF17/labels_17.jsonl",
     "rerun": "PROBE-DIFF17 (one GPU)", "used_by": "E10b-P5route*, READINGS-EXTRA splits, DIAG-PROBE-GATE"},
    {"name": "labels_probe_17.jsonl", "repo_path": "runs/PROBE-SELFROUTE/labels_probe_17.jsonl", "node_path": "runs/PROBE-SELFROUTE/labels_probe_17.jsonl",
     "rerun": "PROBE-SELFROUTE (one GPU, after PROBE-DIFF17)", "used_by": "E10b-P5selfroute*, DIAG-PROBE-GATE, DIAG-WALL-BIMODAL"},
]
RESULT = re.compile(r"^result:(runs/[A-Za-z0-9._-]+/global_step_\d+/actor/huggingface)$")


def open_tasks(lines):
    out = []
    for l in lines:
        m = re.match(r"^([a-z0-9]+)\s+open\s+(.*)$", l)
        if m:
            out += [t.split(":")[0] for t in m.group(2).split()]
    return out


def e12_start(task):
    """run_e12.sh's start of an E12 arm: (repository weight path or None, note)."""
    m = re.match(r"^(E12[bcde])-(.+)$", task)
    if not m:
        return None, ""
    ladder, a = m.groups()
    a = E12_PARSE.get(a, a)
    if a in ("P0", "P1"):
        return None, "base model" if a == "P0" else "SFT-P1's final (opd-dr keep or the public repository)"
    src = {"E12b": "E10b", "E12c": "E10c", "E12d": "E10d", "E12e": "E10e"}[ladder]
    srun = f"E10c-{a}-fix4" if ladder == "E12c" else f"{src}-{a}"
    ss = 50 if a in ("P5self", "P5self-s43", "P5tail", "P5tail-s43", "P5tail-s44") else 150
    if task == "E12c-P5tail-8k":
        srun, ss = "E10c-P5tail-c50", 30
    if task == "E12b-P5-prog":
        srun, ss = "E12b-P5-8k", 25
    return f"runs/{srun}/global_step_{ss}/actor/huggingface", ""


def build(queue_text):
    lines = queue_text.split("\n")
    opened = open_tasks(lines); op = set(opened)
    models = {"Qwen3-0.6B": {"hf": "Qwen/Qwen3-0.6B", "role": "0.6B student", "always": True, "by": "overnight.sh"},
              "Qwen3-1.7B": {"hf": "Qwen/Qwen3-1.7B", "role": "1.7B student", "always": True, "by": "overnight.sh"},
              "Qwen3-4B": {"hf": "Qwen/Qwen3-4B", "role": "4B student", "always": True},
              "Qwen3-8B": {"hf": "Qwen/Qwen3-8B", "role": "teacher; 8B student", "always": True, "by": "overnight.sh"}}
    users = {}
    def use(name, task, hf=None, role="an #eval row"):
        models.setdefault(name, {"hf": hf or f"Qwen/{name}", "role": role}); users.setdefault(name, set()).add(task)
    for t in op:   # the runners' own models beyond the four (run_vlm.sh, run_arm.sh's t32 teacher)
        if re.match(r"^(E15-|PROBE-VLM|DIAG-VLM)", t):
            for n, r in (("Qwen3-VL-2B-Instruct", "VLM student"), ("Qwen3-VL-4B-Thinking", "VLM teacher"), ("Qwen3-VL-4B-Instruct", "VLM teacher (Q-080a)")):
                use(n, t, role=r)
        if re.search(r"-t32$", t):
            use("Qwen3-32B", t, role="the 32B thinking teacher")
    weights, pairs = {}, {}
    def need(d, key, task):
        d.setdefault(key, set()).add(task)
    for l in lines:
        p = l.split()
        if not p:
            continue
        if p[0] == "#eval" and len(p) > 3 and p[1] in op:
            src = p[3]; m = RESULT.match(src)
            if m:
                need(weights, m.group(1), p[1])
            elif src.startswith("local:models/"):
                use(src.split("/", 1)[1], p[1])
            elif re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", src):
                use(src.split("/")[1], p[1], hf=src)
        elif p[0] in ("#pair", "#budget") and len(p) > 3 and p[1] in op:
            for r in p[3:]:
                if "/" in r and r.endswith(".jsonl"):
                    need(pairs, r, p[1])
        elif p[0] == "#strict" and len(p) > 3:
            for r in p[2:]:
                need(pairs, r, "STRICT-C1")
        elif p[0] == "#term" and len(p) > 4:
            m = RESULT.match(p[3])
            if m:
                need(weights, m.group(1), p[1])
            need(pairs, p[4], p[1])
    notes = {}
    for t in opened:
        w, note = e12_start(t)
        if w:
            need(weights, w, t)
        elif note:
            notes[t] = note
    data = [d for d in DATA if "when" not in d or any(t.startswith(d["when"]) for t in op)]
    return {"schema": "coldstart/1", "source": "tasks/queue.txt (v2.5 grammar; v2.6's compiler writes the same format)",
            "open_tasks": len(opened),
            "models": [dict(name=n, **v, tasks=sorted(users.get(n, ()))) for n, v in sorted(models.items())],
            "envs": ENVS, "data": data, "artifacts": ARTIFACTS,
            "weights": [{"path": k, "tasks": sorted(v)} for k, v in sorted(weights.items())],
            "starts_without_weights": notes,
            "pairs": [{"path": k, "readings": sorted(v)} for k, v in sorted(pairs.items())]}


def dumps(m):
    """One entry per line (a reviewable diff), still plain JSON."""
    parts = []
    for k, v in m.items():
        if isinstance(v, list):
            body = ",\n".join("  " + json.dumps(x, ensure_ascii=False) for x in v)
            parts.append(f" {json.dumps(k)}: [\n{body}\n ]" if v else f" {json.dumps(k)}: []")
        else:
            parts.append(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def main():
    q, out = sys.argv[1:3]
    m = build(open(q, encoding="utf-8").read())
    open(out, "w").write(dumps(m))
    print(f"COLDSTART_MANIFEST {len(m['models'])} models, {len(m['weights'])} weights, {len(m['pairs'])} pair references, {m['open_tasks']} open tasks")


if __name__ == "__main__":
    main()
