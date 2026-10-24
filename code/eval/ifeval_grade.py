"""[v2.5.9z16] IFEval grading: Google's instruction checker over one gen_eval.py reading of the ifeval bench. It serves the
reviewer's general-ability checks of 10-22; Fable / ops-lead agreed on a grading-only venv.

The checker is code/eval/ifeval/instruction_following_eval/ (google-research e6890f8, vendored unchanged; see ifeval/SOURCE.md).
run_queue_eval.sh runs this file in $OPD_ROOT/venv-ifeval, which BUILD-VENV-IFEVAL builds: nltk 3.9.1, langdetect 1.0.9,
absl-py 2.1.0, immutabledict 4.2.1, and punkt_tab in the venv's own nltk_data. No GPU, no network.

    python ifeval_grade.py --data <ifeval_input_data.jsonl> --jsonl <stem>.jsonl --metrics <stem>.metrics.json --out <stem>.ifeval.json
    python ifeval_grade.py --selftest     BUILD-VENV-IFEVAL's check: Google's unit tests and the KNOWN responses below

What is graded:
- the reading's text as written;
- for a text with a thinking block, the answer after its last </think>, with the whitespace around it stripped
  (Qwen writes "</think>\\n\\n");
- a thinking-mode sample whose thinking never closed has no answer: it is graded as "" (fails every instruction; think_unclosed).

Each sample goes through evaluation_lib.test_instruction_following_strict and _loose, Google's functions, unchanged. Its
instructions are built from the data's kwargs, with null values dropped (the HF jsonl has none; its parquet copy fills every key
with null).
- prompt level: every instruction of the prompt followed;
- instruction level: the share of instructions followed;
- both are averaged over the samples (n = 1 greedy gives Google's own report).

Fixed here, where Google leaves them open:
- nltk reads only the venv's own nltk_data (sys.prefix/nltk_data), never a stray ~/nltk_data;
- langdetect's seed is 0 (DetectorFactory.seed; detection is random per call otherwise);
- random.seed(0) before each sample.

<stem>.ifeval.json holds the four accuracies (%), the counts, by_group / by_instruction (strict and loose), the per-sample
verdicts and the versions. Its metrics file gets, written atomically before the .ifeval.json (its presence means graded):
- avg_acc = prompt-level strict;
- pass_at_k computed from it;
- the block ifeval {prompt_strict, prompt_loose, inst_strict, inst_loose}.
Exit 2 when the reading does not cover the data (a pid missing, unknown, or with another sample count).
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "ifeval")
UPSTREAM = "google-research/google-research instruction_following_eval @ e6890f85757dd84e27ca6df2dd30651dafad28e0"
FOUR = ("prompt_strict", "prompt_loose", "inst_strict", "inst_loose")


def checker():
    """evaluation_lib, after nltk is pointed at the venv's own nltk_data only and langdetect is seeded."""
    import nltk
    nltk.data.path[:] = [os.path.join(sys.prefix, "nltk_data")]
    import langdetect
    langdetect.DetectorFactory.seed = 0
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    from instruction_following_eval import evaluation_lib
    return evaluation_lib


def read_inputs(path, E):
    """Google's read_prompt_list, with null kwargs dropped."""
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            ex = json.loads(line)
            out.append(E.InputExample(key=ex["key"], instruction_id_list=ex["instruction_id_list"], prompt=ex["prompt"],
                                      kwargs=[{k: v for k, v in kw.items() if v is not None} for kw in ex["kwargs"]]))
    return out


def response_of(text, mode):
    """(the answer graded, thinking never closed)."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), False
    if mode == "think":
        return "", True
    return text, False


def coverage(inputs, recs):
    """None, or why the reading does not cover the data: every prompt's pid ifeval-<key> with the same sample count."""
    want = {f"ifeval-{i.key}" for i in inputs}
    n = defaultdict(int)
    for r in recs:
        n[r["pid"]] += 1
    unknown, missing = sorted(set(n) - want), sorted(want - set(n))
    if unknown or missing:
        return f"{len(missing)} prompts missing ({missing[:3]}), {len(unknown)} unknown ({unknown[:3]})"
    if len(set(n.values())) != 1:
        return f"sample counts differ: {sorted(set(n.values()))}"
    return None


def grade_samples(inputs, recs, mode, E):
    by_pid = {f"ifeval-{i.key}": i for i in inputs}
    rows = []
    for r in recs:
        inp = by_pid[r["pid"]]
        resp, unclosed = response_of(r["text"], mode)
        random.seed(0)
        s = E.test_instruction_following_strict(inp, {inp.prompt: resp})
        random.seed(0)
        lo = E.test_instruction_following_loose(inp, {inp.prompt: resp})
        rows.append({"pid": r["pid"], "sample": r.get("sample", 0), "strict": bool(s.follow_all_instructions),
                     "loose": bool(lo.follow_all_instructions), "strict_list": [bool(x) for x in s.follow_instruction_list],
                     "loose_list": [bool(x) for x in lo.follow_instruction_list], "instruction_id_list": list(inp.instruction_id_list),
                     "think_unclosed": unclosed, "empty": not resp.strip(), "truncated": r.get("finish_reason") == "length"})
    return rows


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def summarize(rows):
    pct = lambda a, b: round(100 * a / b, 2) if b else None
    n_inst = sum(len(r["strict_list"]) for r in rows)
    out = {"prompt_strict": pct(sum(r["strict"] for r in rows), len(rows)), "prompt_loose": pct(sum(r["loose"] for r in rows), len(rows)),
           "inst_strict": pct(sum(sum(r["strict_list"]) for r in rows), n_inst), "inst_loose": pct(sum(sum(r["loose_list"]) for r in rows), n_inst)}
    per = defaultdict(list)
    for r in rows:
        per[r["pid"]].append(r["strict"])
    n = len(next(iter(per.values()))) if per else 0
    out["pass_at_k"] = {k: round(100 * sum(pass_at_k(len(v), sum(v), k) for v in per.values()) / len(per), 2)
                        for k in [1, 2, 4, 8, 16, 32, 64] if k <= n}
    groups = {"by_instruction": lambda i: i, "by_group": lambda i: i.split(":")[0]}   # Google's print_report: tier 1 / tier 0
    for name, key in groups.items():
        acc = defaultdict(lambda: [0, 0, 0])
        for r in rows:
            for i, s, lo in zip(r["instruction_id_list"], r["strict_list"], r["loose_list"]):
                a = acc[key(i)]; a[0] += 1; a[1] += s; a[2] += lo
        out[name] = {k: {"n": a[0], "strict": pct(a[1], a[0]), "loose": pct(a[2], a[0])} for k, a in sorted(acc.items())}
    out.update(n_prompts=len(per), n_samples=len(rows), n_instructions=n_inst, think_unclosed=sum(r["think_unclosed"] for r in rows),
               empty_responses=sum(r["empty"] for r in rows), truncated=sum(r["truncated"] for r in rows))
    return out


def versions():
    import importlib.metadata as md
    v = {}
    for p in ("nltk", "langdetect", "absl-py", "immutabledict"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    return v


def write_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def fold(metrics_path, s):
    """The metrics file of the reading: avg_acc = prompt-level strict, pass_at_k from it, the block ifeval."""
    m = json.load(open(metrics_path))
    m["avg_acc"], m["pass_at_k"] = s["prompt_strict"], s["pass_at_k"]
    m["ifeval"] = {k: s[k] for k in FOUR}
    write_json(metrics_path, m)
    return m


def grade_file(data, jsonl, metrics, out):
    E = checker()
    inputs = read_inputs(data, E)
    recs = [json.loads(l) for l in open(jsonl, encoding="utf-8") if l.strip()]
    m = json.load(open(metrics))
    why = coverage(inputs, recs)
    if why:
        print(f"IFEVAL_COVERAGE {os.path.basename(jsonl)}: {why}", file=sys.stderr)
        sys.exit(2)
    rows = grade_samples(inputs, recs, m.get("mode"), E)
    s = summarize(rows)
    fold(metrics, s)
    write_json(out, {**s, "grader": UPSTREAM, "versions": versions(), "mode": m.get("mode"), "reading": os.path.basename(jsonl),
                     "per_sample": rows})
    print(f"IFEVAL {os.path.basename(jsonl)[:-6]} " + " ".join(f"{k}={s[k]}" for k in FOUR) + f" think_unclosed={s['think_unclosed']}")
    return s


# Known responses (prompt, instruction ids, kwargs, response, strict, loose): a check of the whole path in the venv. The
# punctuation / sentence / case rows need nltk's punkt_tab and langdetect; the last two pass only loose (the "**" closing line,
# the asterisks around a JSON block).
KNOWN = [
    ("Write two short sentences about cats. Do not use any commas.", ["punctuation:no_comma", "length_constraints:number_sentences"],
     [{}, {"relation": "less than", "num_sentences": 3}], "Cats are calm animals. They sleep for most of the day.", True, True),
    ("Write two short sentences about cats. Do not use any commas.", ["punctuation:no_comma", "length_constraints:number_sentences"],
     [{}, {"relation": "less than", "num_sentences": 3}], "Cats, as a rule, are calm. They sleep. They eat. They play.", False, False),
    ("Answer in English and in lowercase letters only.", ["change_case:english_lowercase", "language:response_language"],
     [{}, {"language": "en"}], "this is a short answer written in plain english words, all of them in lowercase.", True, True),
    ("End your answer with the exact phrase: Is there anything else I can help with?", ["startend:end_checker"],
     [{"end_phrase": "Is there anything else I can help with?"}], "Here is the plan.\nIs there anything else I can help with?\n**", False, True),
    ("Wrap the whole answer in JSON.", ["detectable_format:json_format"], [{}], '**{"answer": 42}**', False, True),
]


GOOGLE_TESTS = ("instructions_util_test", "instructions_test")
RUN_TEST = ("import importlib, os, sys; vendor, t = sys.argv[1:3]; sys.argv = sys.argv[:1]; sys.path.insert(0, vendor); import nltk; "
            "nltk.data.path[:] = [os.path.join(sys.prefix, 'nltk_data')]; from absl.testing import absltest; "
            "absltest.main(module=importlib.import_module('instruction_following_eval.' + t))")


def selftest():
    E = checker()
    from instruction_following_eval import instructions_util as U
    punkt = os.path.isdir(os.path.join(sys.prefix, "nltk_data", "tokenizers", "punkt_tab", "english"))
    res = {"versions": versions(), "punkt_tab_in_venv": punkt, "count_sentences_3": U.count_sentences("One here. Two here. Three here."),
           "google_tests": {}, "known": []}
    ok = punkt and res["count_sentences_3"] == 3
    for t in GOOGLE_TESTS:   # absltest, as Google runs them; each in its own interpreter
        p = subprocess.run([sys.executable, "-I", "-c", RUN_TEST, VENDOR, t], capture_output=True, text=True, timeout=600)
        tail = [l for l in p.stderr.splitlines() if l.startswith(("Ran ", "OK", "FAILED"))]
        res["google_tests"][t] = {"rc": p.returncode, "summary": " ".join(tail)[:200]}
        ok &= p.returncode == 0
    for i, (prompt, ids, kw, resp, want_s, want_l) in enumerate(KNOWN):
        inp = E.InputExample(key=i, instruction_id_list=ids, prompt=prompt, kwargs=kw)
        random.seed(0); s = E.test_instruction_following_strict(inp, {prompt: resp}).follow_all_instructions
        random.seed(0); lo = E.test_instruction_following_loose(inp, {prompt: resp}).follow_all_instructions
        res["known"].append({"ids": ids, "strict": s, "loose": lo, "ok": (s, lo) == (want_s, want_l)})
        ok &= (s, lo) == (want_s, want_l)
    res["ok"] = bool(ok)
    print("IFEVAL_SELFTEST " + json.dumps(res))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data")
    ap.add_argument("--jsonl")
    ap.add_argument("--metrics")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not (a.data and a.jsonl and a.metrics and a.out):
        ap.error("--data, --jsonl, --metrics and --out (or --selftest)")
    grade_file(a.data, a.jsonl, a.metrics, a.out)


if __name__ == "__main__":
    main()
