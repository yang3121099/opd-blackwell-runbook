"""[v2.5.9z46] gen_eval.model_len: max_tokens + 2048, capped at a local model's max_position_embeddings (no rope_scaling)."""
import json
import os
import tempfile

import gen_eval as G


def _model(cfg):
    d = tempfile.mkdtemp(); json.dump(cfg, open(os.path.join(d, "config.json"), "w")); return d


def test_model_len():
    assert G.model_len(_model({"max_position_embeddings": 4096}), 3072) == 4096          # Qwen2.5-Math-1.5B-Instruct
    assert G.model_len(_model({"max_position_embeddings": 40960}), 32768) == 34816       # Qwen3 as before
    assert G.model_len(_model({"max_position_embeddings": 131072}), 16384) == 18432      # R1-Distill
    assert G.model_len(_model({"max_position_embeddings": 4096, "rope_scaling": {"type": "yarn", "factor": 4}}), 16384) == 18432
    assert G.model_len(_model({}), 1000) == 3048 and G.model_len("Qwen/Qwen3-1.7B", 16384) == 18432   # no config here: as before


def test_one_bos():
    class T:
        bos_token, add_bos_token = "<｜begin▁of▁sentence｜>", True
    class Q:
        bos_token, add_bos_token = None, False
    p = "<｜begin▁of▁sentence｜><｜User｜>x<｜Assistant｜><think>\n"
    assert G.one_bos(T(), p) == "<｜User｜>x<｜Assistant｜><think>\n" and G.one_bos(T(), "<｜User｜>x") == "<｜User｜>x"
    assert G.one_bos(Q(), "<|im_start|>user\nx") == "<|im_start|>user\nx"
    T.add_bos_token = False; assert G.one_bos(T(), p) == p   # a tokenizer that adds none: the template's stays
