"""[v2.5.9z108] (Q-083) mode_len.py: the token of a rendered prompt, per-step and per-token lengths from a rollout dump, each token's
wall (mean >= 0.9 x cap), the cache of counted step files. No transformers: a stand-in tokenizer. Run: python -m pytest -q scripts/test_mode_len.py"""
import json
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mode_len as M  # noqa: E402


def test_token_of():
    assert M.token_of("system\nYou are\nuser\n[long] Find x.\nassistant\n") == "[long]"
    assert M.token_of("user\n[short] Find x. Then [long] y") == "[short]"      # the first one
    assert M.token_of("user\nFind [long]x") is None and M.token_of("") is None


def test_step_stats_and_walls():
    dump = {1: [("[long] a", 10), ("[short] b", 2), ("c", 5)], 2: [("[long] a", 95), ("[short] b", 3)], 3: [("[long] a", 100), ("[short] b", 99)]}
    per = {s: M.step_stats([json.dumps({"input": "user\n" + i, "output": "w " * n}) for i, n in rows], lambda t: len(t.split()))
           for s, rows in dump.items()}
    rows, walls = M.summarize(per, 100)
    assert rows[0]["long"] == {"n": 1, "mean_len": 10.0, "trunc": 0.0} and rows[0]["none"]["n"] == 1
    assert rows[2]["long"]["trunc"] == 1.0 and rows[2]["short"]["trunc"] == 1.0   # >= 0.98 x cap
    assert walls == {"[long]": 2, "[short]": 3}                                     # 95 >= 90 at s2; short 99 at s3


def test_main_with_a_stand_in_tokenizer(tmp_path):
    run = tmp_path / "run"; (run / "mode_dump").mkdir(parents=True)
    (run / "config.yaml").write_text("data:\n  max_response_length=10\n")
    for s, (nl, ns) in {1: (3, 1), 2: (9, 2)}.items():
        (run / "mode_dump" / f"{s}.jsonl").write_text("\n".join(json.dumps({"input": f"user\n[{t}] q", "output": "w " * n})
                                                               for t, n in (("long", nl), ("short", ns))) + "\n")
    fake = tmp_path / "fake"; (fake / "transformers").mkdir(parents=True)
    (fake / "transformers" / "__init__.py").write_text(
        "class AutoTokenizer:\n    @staticmethod\n    def from_pretrained(p):\n"
        "        return lambda s, add_special_tokens=False: {'input_ids': s.split()}\n")
    env = dict(os.environ, PYTHONPATH=str(fake))
    r = subprocess.run([sys.executable, os.path.join(HERE, "mode_len.py"), str(run), "E10b-P5mode", "--tokenizer", "x"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == "MODE_LEN E10b-P5mode steps 2 wall_long 2 wall_short None", r.stdout + r.stderr
    rec = json.load(open(run / "len_by_token.json"))
    assert rec["cap"] == 10 and [x["long"]["mean_len"] for x in rec["rows"]] == [3.0, 9.0] and rec["wall_long"] == 2
    cache = json.load(open(run / ".mode_len_cache.json"))
    assert set(cache) == {"1.jsonl", "2.jsonl"}
    env["PYTHONPATH"] = str(tmp_path / "none")                                       # cached: no tokenizer needed
    r = subprocess.run([sys.executable, os.path.join(HERE, "mode_len.py"), str(run), "E10b-P5mode"], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "steps 2" in r.stdout
    empty = tmp_path / "empty"; empty.mkdir()
    r = subprocess.run([sys.executable, os.path.join(HERE, "mode_len.py"), str(empty), "X"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == "" and not (empty / "len_by_token.json").exists()
