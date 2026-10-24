# code/eval/ifeval: Google's IFEval instruction checker, vendored unchanged (runbook v2.5.9z16)

Upstream: github.com/google-research/google-research, directory `instruction_following_eval/`, commit
`e6890f85757dd84e27ca6df2dd30651dafad28e0` (2026-07-22, "Adding type suppressions for pyrefly"; the directory's latest commit
when this copy was made, 2026-10-24). Licence: Apache License 2.0 (`LICENSE`, the google-research repository's own file); every
file keeps its copyright and licence header.

The files under `instruction_following_eval/` are byte-identical to that commit (git blob ids as GitHub lists them):

| file | git blob |
|---|---|
| evaluation_lib.py | c87422822f9e1565ca7574ac0b3cee5389c9f44e |
| instructions.py | 0007359bd37c7b9c1d9557f8937cb6586d2788a7 |
| instructions_registry.py | 6b42adcc4c8cc827810ab3efa8a06f02bb48c397 |
| instructions_util.py | d50702400738df4378fe286c0090a416f9a172fa |
| instructions_test.py | 9877688496a1b9f805d8ba7da6febb6d5fd823ef |
| instructions_util_test.py | 9846ea9e0478cc2b2ac9e11bb11b53b4b6a71670 |

`git hash-object <file>` reproduces each id. The directory keeps its upstream name, so the files' own imports
(`from instruction_following_eval import ...`) work unchanged; `code/eval/ifeval_grade.py` puts this folder on `sys.path`. Not
copied: `evaluation_main.py` (absl's command line; `ifeval_grade.py` calls `evaluation_lib` directly), `run.sh`, `README.md`,
`requirements.txt` (absl, langdetect, nltk, immutabledict, unpinned) and `data/` (the prompts come from the node's #fetch copy of
google/IFEval, `ifeval_input_data.jsonl`).

The two test files are Google's own unit tests. They need absl / langdetect / nltk, so they run only in `$OPD_ROOT/venv-ifeval`:
BUILD-VENV-IFEVAL runs them through `ifeval_grade.py --selftest`, and `code/eval/test_ifeval.py` runs them when `IFEVAL_TEST_PY`
names such a python. `conftest.py` keeps a plain pytest run of code/eval from collecting them.

Versions (BUILD-VENV-IFEVAL): nltk 3.9.1, langdetect 1.0.9, absl-py 2.1.0, immutabledict 4.2.1, and nltk_data's
`tokenizers/punkt_tab` (nltk_data commit 4f15a3d89eefe9748ec1c05be495d91289197155, punkt_tab.zip 4,319,076 bytes, sha256
e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106). `instructions_util.count_sentences` loads
`nltk:tokenizers/punkt/english.pickle`. nltk 3.9 answers that request with its pickle-free `PunktTokenizer` on punkt_tab, so no
pickle is ever loaded (CVE-2024-39705). The Mac check of 2026-10-24 compared it with nltk 3.8.1 and the pickled punkt (nltk_data
1d3c34b4, sha256 51c30789...660ec) on the 541 IFEval prompts and their 791 non-empty lines. Sentence counts and `word_tokenize`
tokens were identical on all 1,332 texts, and Google's 48 unit tests pass with both.

What the grader fixes that Google leaves open (in `ifeval_grade.py`, never in these files):
- nltk reads only the venv's own `nltk_data`;
- langdetect's seed is 0, since its `detect` is random per call otherwise;
- `random.seed(0)` before each sample.
