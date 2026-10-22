"""[v2.5.9z33] prep_mmk12 on small frames: '<image>' first (once), the answer instruction, the image kept, rows without an image or
with an empty answer dropped and counted, the validation draw fixed, the files written."""
import json
import os
import tempfile

import pandas as pd

import prep_mmk12 as P


def _df():
    img = lambda b: {"bytes": b, "path": None}
    return pd.DataFrame([{"id": "a", "question": "Find x.", "answer": "3", "subject": "math", "image": img(b"i0")},
                         {"id": "b", "question": "<image>How many?", "answer": " 12 ", "subject": "physics", "image": img(b"i1")},
                         {"id": "c", "question": "No picture.", "answer": "1", "subject": "math", "image": None},
                         {"id": "d", "question": "Empty bytes.", "answer": "1", "subject": "math", "image": {"bytes": b"", "path": None}},
                         {"id": "e", "question": "No answer.", "answer": "  ", "subject": "math", "image": img(b"i4")},
                         {"id": "f", "question": "None answer.", "answer": None, "subject": "biology", "image": img(b"i5")}])


def test_rows_of_prompt_image_drops():
    rows, c = P.rows_of(_df(), "train")
    assert [r["extra_info"]["id"] for r in rows] == ["a", "b"]
    assert rows[0]["prompt"] == [{"role": "user", "content": "<image>Find x." + P.SUFFIX}]
    assert rows[1]["prompt"][0]["content"] == "<image>How many?" + P.SUFFIX and rows[1]["reward_model"]["ground_truth"] == "12"
    assert rows[0]["images"] == [{"bytes": b"i0", "path": None}] and rows[1]["extra_info"] == {"split": "train", "index": 1, "id": "b", "subject": "physics"}
    assert all(r["data_source"] == "FanqingM/MMK12" and r["ability"] == "math" for r in rows)
    assert c == {"kept": 2, "subject:math": 1, "subject:physics": 1, "dropped_no_image": 2, "dropped_empty_answer": 2}


def test_val_rows_fixed_draw():
    rows = [{"i": k} for k in range(1000)]
    a, b = P.val_rows(rows, 300), P.val_rows(rows, 300)
    assert len(a) == 300 and a == b and a == sorted(a, key=lambda r: r["i"]) and P.val_rows(rows[:10], 300) == rows[:10]


def test_main_writes_train_and_val():   # parquet needs pyarrow: the node (the Mac skips it)
    import pytest
    pytest.importorskip("pyarrow")
    src, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    os.makedirs(os.path.join(src, "data"))
    _df().to_parquet(os.path.join(src, "data", "train-00000-of-00001.parquet")); _df().to_parquet(os.path.join(src, "data", "test-00000-of-00001.parquet"))
    c = P.main(src, out, val_n=1)
    assert c["train"]["kept"] == 2 and c["test"]["val_rows"] == 1
    assert len(pd.read_parquet(os.path.join(out, "mmk12_train.parquet"))) == 2 and len(pd.read_parquet(os.path.join(out, "mmk12_val.parquet"))) == 1
    json.dumps(c)


def _fake_io(monkeypatch, src, frames, fail_on=None):
    """read_parquet / to_parquet without pyarrow: frames by file name; to_parquet records the path (or raises for fail_on)."""
    import pandas as pd
    written = []
    monkeypatch.setattr(pd, "read_parquet", lambda f: frames[os.path.basename(f)])
    def to_parquet(self, path, *a, **k):
        if fail_on and fail_on in path:
            open(path, "w").write("half"); raise OSError("interrupted")
        open(path, "w").write(str(len(self))); written.append(os.path.basename(path))
    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)
    for name in frames:
        os.makedirs(os.path.join(src, "data"), exist_ok=True); open(os.path.join(src, "data", name), "w").close()
    return written


def test_main_atomic_and_val_first(monkeypatch):   # [ops-lead's A1 condition a]
    src, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    written = _fake_io(monkeypatch, src, {"train-00000-of-00001.parquet": _df(), "test-00000-of-00001.parquet": _df()})
    P.main(src, out, val_n=1)
    assert written == ["mmk12_val.parquet.tmp", "mmk12_train.parquet.tmp"] and sorted(os.listdir(out)) == ["mmk12_train.parquet", "mmk12_val.parquet"]


def test_main_interrupted_train_write_leaves_no_train_file(monkeypatch):
    import pytest
    src, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    _fake_io(monkeypatch, src, {"train-00000-of-00001.parquet": _df(), "test-00000-of-00001.parquet": _df()}, fail_on="mmk12_train")
    with pytest.raises(OSError):
        P.main(src, out, val_n=1)
    assert not os.path.exists(os.path.join(out, "mmk12_train.parquet"))   # only the .tmp: run_vlm.sh makes it again


def test_main_fails_without_train_rows_or_split(monkeypatch):
    import pytest
    src, out = tempfile.mkdtemp(), tempfile.mkdtemp()
    empty = _df().assign(image=None)
    _fake_io(monkeypatch, src, {"train-00000-of-00001.parquet": empty, "test-00000-of-00001.parquet": _df()})
    with pytest.raises(SystemExit) as e:
        P.main(src, out)
    assert "no train row kept" in str(e.value) and os.listdir(out) == []
    src2, out2 = tempfile.mkdtemp(), tempfile.mkdtemp()
    _fake_io(monkeypatch, src2, {"test-00000-of-00001.parquet": _df()})
    with pytest.raises(SystemExit) as e:
        P.main(src2, out2)
    assert "no train-*.parquet" in str(e.value) and os.listdir(out2) == []
