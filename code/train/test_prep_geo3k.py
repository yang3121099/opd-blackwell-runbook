"""[v2.5.9z] prep_geo3k.rows_of on a small frame: the '<image>' placeholder first, the answer instruction, images kept, the verl fields."""
import pandas as pd

import prep_geo3k as P


def test_rows_of():
    df = pd.DataFrame([{"problem": "<image>Find x.", "answer": "3", "images": [{"bytes": b"img0", "path": None}]},
                       {"problem": "Find y.", "answer": " 8\\sqrt{3} ", "images": [{"bytes": b"img1", "path": None}]}])
    rows = P.rows_of(df, "train")
    assert rows[0]["prompt"] == [{"role": "user", "content": "<image>Find x." + P.SUFFIX}]
    assert rows[1]["prompt"][0]["content"].startswith("<image>Find y.") and rows[1]["reward_model"]["ground_truth"] == "8\\sqrt{3}"
    assert rows[0]["images"][0]["bytes"] == b"img0" and rows[1]["extra_info"] == {"split": "train", "index": 1}
    assert all(r["data_source"] == "hiyouga/geometry3k" and r["ability"] == "math" for r in rows)
