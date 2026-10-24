"""[v2.5.9z16] Google's own unit tests (instruction_following_eval/*_test.py, vendored unchanged) need absl / langdetect / nltk: they
run in venv-ifeval only (ifeval_grade.py --selftest; code/eval/test_ifeval.py with IFEVAL_TEST_PY), never in a plain pytest run."""
collect_ignore_glob = ["instruction_following_eval/*_test.py"]
