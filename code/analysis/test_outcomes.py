"""[v2.5.9z101] outcomes.py: a reading's and a rollout dump's outcomes (no text), and the backfill (download, write, delete;
resumable; the tiers / tasks filters; a failure listed)."""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import outcomes as O  # noqa: E402


def write(path, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return str(path)


def lines(p):
    return [json.loads(l) for l in open(p)]


def test_reading_outcomes_keep_no_text(tmp_path):
    src = write(tmp_path / "r.jsonl", [{"pid": "math500-0", "sample": 0, "correct": True, "n_tokens": 120, "finish_reason": "stop", "text": "so \\boxed{4}", "token_ids": [1, 2]},
                                       {"pid": "math500-0", "sample": 1, "correct": False, "n_tokens": 16384, "finish_reason": "length", "text": "still going"}])
    out = str(tmp_path / "r.outcomes.jsonl")
    assert O.reading(src, out) == 2
    assert lines(out) == [{"id": "math500-0", "s": 0, "ok": True, "n": 120, "trunc": False, "box": True},
                          {"id": "math500-0", "s": 1, "ok": False, "n": 16384, "trunc": True, "box": False}]
    assert "boxed" not in open(out).read() and "still going" not in open(out).read()


def test_rollout_outcomes_with_key_and_tokens(tmp_path):
    inp = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    src = write(tmp_path / "rollouts" / "7.jsonl", [{"input": inp, "output": "four \\boxed{4}", "score": 1.0, "step": 7}])
    out = str(tmp_path / "7.outcomes.jsonl")
    assert O.rollouts(src, out, lambda texts: [len(t.split()) for t in texts]) == 1
    (r,) = lines(out)
    assert r == {"step": 7, "key": O.key(inp), "score": 1.0, "chars": 14, "box": True, "n": 2}
    import hashlib
    assert r["key"] == hashlib.sha1(b"What is 2+2?").hexdigest()[:16]                    # diff_labels.key's identity


def test_backfill_downloads_writes_deletes_and_resumes(tmp_path):
    rows = [{"tier": "D-train", "task": "E12b-x", "filename": "runs/E12b-x/rollouts/3.jsonl"},
            {"tier": "C-chain", "task": "EVAL-a", "filename": "runs/EVAL-a/a_math500_no_think_t0.6.jsonl"},
            {"tier": "B-closed", "task": "EVAL-b", "filename": "runs/EVAL-b/b.jsonl"},
            {"tier": "C-chain", "task": "EVAL-bad", "filename": "runs/EVAL-bad/bad.jsonl"}]
    c = tmp_path / "list.csv"
    with open(c, "w", newline="") as f:
        w = csv.DictWriter(f, ["tier", "task", "filename"]); w.writeheader(); [w.writerow(r) for r in rows]
    fetched, dirs = [], []
    def fetch(repo, fn, d):
        fetched.append(fn); dirs.append(d)
        if "bad" in fn:
            raise IOError("gone")
        rec = {"input": "<|im_start|>user\nq<|im_end|>", "output": "x", "score": 0, "step": 3} if "rollouts" in fn else {"pid": "math500-1", "sample": 0, "correct": True, "n_tokens": 5, "finish_reason": "stop", "text": "\\boxed{1}"}
        return write(os.path.join(d, "f.jsonl"), [rec])
    out = tmp_path / "out"
    r = O.backfill(str(c), "me/ops", str(out), tiers={"D", "C"}, fetch=fetch)
    assert r["done"] == 2 and r["kept"] == 0 and [x["filename"] for x in r["failed"]] == ["runs/EVAL-bad/bad.jsonl"]
    assert "runs/EVAL-b/b.jsonl" not in fetched                                           # tier B not asked for
    assert os.path.exists(out / "E12b-x" / "rollouts" / "3.outcomes.jsonl") and os.path.exists(out / "EVAL-a" / "a_math500_no_think_t0.6.outcomes.jsonl")
    assert not any(os.path.exists(d) for d in dirs)                                        # every download deleted
    fetched.clear()
    r2 = O.backfill(str(c), "me/ops", str(out), tasks={"EVAL-a"}, fetch=fetch)
    assert r2 == {"done": 0, "kept": 1, "failed": []} and fetched == []                    # resumable
    assert len(lines(out / "backfill_log.jsonl")) == 2


def test_run_queue_eval_writes_outcomes_for_finished_readings(tmp_path):   # the hook block of run_queue_eval.sh, run as it is
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__)); rb = os.path.dirname(os.path.dirname(here))
    sh = open(os.path.join(rb, "scripts", "run_queue_eval.sh")).read()
    block = sh[sh.index("for J in $OUT/*.jsonl; do"):]; block = block[:block.index("\ndone\n") + 6]
    out = tmp_path / "EVAL-a"; out.mkdir()
    rec = {"pid": "math500-0", "sample": 0, "correct": True, "n_tokens": 9, "finish_reason": "stop", "text": "\\boxed{1}"}
    write(str(out / "a_math500_no_think_t0.6.jsonl"), [rec]); (out / "a_math500_no_think_t0.6.metrics.json").write_text("{}")
    write(str(out / "b_math500_no_think_t0.6.jsonl"), [rec])                                   # no metrics yet: not finished
    write(str(out / "a_math500_no_think_t0.6.forced.jsonl"), [rec])
    r = subprocess.run(["bash", "-c", block], env=dict(os.environ, OUT=str(out), RUNBOOK=rb), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in out.glob("*.outcomes.jsonl")) == ["a_math500_no_think_t0.6.outcomes.jsonl"]
    assert lines(out / "a_math500_no_think_t0.6.outcomes.jsonl") == [{"id": "math500-0", "s": 0, "ok": True, "n": 9, "trunc": False, "box": True}]
    assert sh.count("grep -Ev '\\.(forced|outcomes)\\.jsonl$'") == 3                # the readers never pick an outcomes file (ERE: BSD grep too)
