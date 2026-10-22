"""[v2.5.9z38] len_wall.py (Q-060's phase step): per-step length / clip from a verl log, the cap from the config, the first step at
mean length >= 0.9 x cap written once (wall.json) and printed once."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import len_wall as W  # noqa: E402


def _run(lines, cfg="student_template: non-thinking, max_response_length 4096\n"):
    d = tempfile.mkdtemp()
    open(os.path.join(d, "train.log"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(d, "config.yaml"), "w").write(cfg)
    return d


def line(s, L, c):
    return f"(TaskRunner pid=1) step:{s} - critic/score/mean:0.5 - response_length/mean:{L} - response_length/clip_ratio:{c} - timing_s/step:300"


def test_rows_resume_and_cap():
    rows = W.rows_of("\n".join([line(1, 1569, 0.016), line(2, 2000, 0.1), line(2, 2100, 0.12), "noise step:x"]))
    assert rows == {1: (1569.0, 0.016), 2: (2100.0, 0.12)}   # a step logged twice (a resume) keeps its last line
    assert W.cap_of("max_response_length 8192") == 8192 and W.cap_of("data.max_response_length=2048 x") == 2048 and W.cap_of("") == 4096


def test_main_csv_and_one_wall(capsys):
    d = _run([line(1, 1569, 0.016), line(2, 3600, 0.8), line(3, 3700, 0.9), line(4, 4000, 0.95)])
    assert W.main(d, "E10b-P5mix90") == 0
    out = capsys.readouterr().out
    assert out.strip() == "WALL E10b-P5mix90 s3 len 3700 clip 0.9 cap 4096"   # 3700 >= 0.9 x 4096 = 3686.4; s2's 3600 is not
    w = json.load(open(os.path.join(d, "wall.json")))
    assert w["step"] == 3 and w["mean_len"] == 3700.0 and w["cap"] == 4096 and w["task"] == "E10b-P5mix90"
    assert open(os.path.join(d, "len_by_step.csv")).read().splitlines() == ["step,mean_len,clip_ratio", "1,1569.0,0.016", "2,3600.0,0.8", "3,3700.0,0.9", "4,4000.0,0.95"]
    W.main(d, "E10b-P5mix90")
    assert capsys.readouterr().out == ""   # printed once: wall.json is the marker


def test_no_wall_no_file_and_no_log():
    d = _run([line(1, 1569, 0.016), line(10, 3201, 0.516)])
    W.main(d, "E10b-P5mix")
    assert not os.path.exists(os.path.join(d, "wall.json")) and len(open(os.path.join(d, "len_by_step.csv")).read().splitlines()) == 3
    assert W.main(tempfile.mkdtemp(), "x") == 0
