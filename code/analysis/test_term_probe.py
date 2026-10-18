"""Tests of term_probe.py's anchor, newline and marker logic (runbook v2.5.7g) on byte pieces (no model)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import term_probe as TP  # noqa: E402

EOS = TP.EOS_IDS[0]


def P(*texts):
    """pieces and ids: one token per text, None = <|im_end|>."""
    pieces = [b"" if t is None else t.encode() for t in texts]
    ids = [EOS if t is None else 1000 + i for i, t in enumerate(texts)]
    return pieces, ids


def test_first_boxed_end_balanced():
    b = b"x \\boxed{\\frac{1}{2}} y \\boxed{3}"
    assert b[:TP.first_boxed_end(b)].endswith(b"{2}}") and TP.first_boxed_end(b"\\boxed{1") is None


def test_anchor_newline_after_box():
    pieces, ids = P("So ", "\\boxed{", "4", "}", "$.", "\n\n", "Wait", ", let", " me check", ".", None)
    assert TP.anchor_positions(pieces, ids) == (4, 5, "nl")


def test_anchor_eos_first_and_none():
    pieces, ids = P("\\boxed{", "4", "}", "$", None)
    assert TP.anchor_positions(pieces, ids) == (3, 4, "eos_first")
    pieces, ids = P("\\boxed{", "4", "}", " and more")
    assert TP.anchor_positions(pieces, ids) == (3, None, "none")
    assert TP.anchor_positions(*P("no answer", "\n")) is None


def test_markers_share_of_continuation():
    pieces, ids = P("\\boxed{", "4", "}", "\n\n", "Wait", ", let", " me check", ": ", "4", ".", None)
    end = TP.first_boxed_end(b"".join(pieces))
    ct, mt = TP.markers(pieces, end)
    assert ct == 7 and mt == 3   # "\n\n" "Wait" ", let" " me check" ": " "4" "."; markers Wait + let me check (2 tokens)
    pieces, ids = P("\\boxed{", "4", "}", "$", None)
    assert TP.markers(pieces, TP.first_boxed_end(b"".join(pieces))) == (1, 0)
    pieces, _ = P("\\boxed{", "4", "}", " awaited", " however")   # word bounded
    assert TP.markers(pieces, TP.first_boxed_end(b"".join(pieces))) == (2, 1)


def test_paired_metric_and_describe():
    A = {1: [0.5, 0.7], 2: [0.1], 3: []}; B = {1: [0.2], 2: [0.1], 4: [0.9]}
    r = TP.paired_metric(A, B)
    assert r["problems"] == 2 and r["diff"] == 0.2 and r["ci95"][0] <= 0.2 <= r["ci95"][1]
    d = TP.describe([0.1, None, 0.3], {1: [0.1], 2: [0.3]})
    assert d["n"] == 2 and d["mean"] == 0.2 and "mean_ci95_problems" in d and TP.describe([None], {}) == {"n": 0}
    assert "mean_ci95_problems" not in TP.describe([0.5], {})


def test_token_bytes_byte_level():
    import types
    enc = lambda t: "".join(TP.bytes_to_unicode()[x] for x in t.encode())
    vocab = {5: enc(" x²"), 6: enc("\n\n"), 7: "<|im_end|>"}
    tok = types.SimpleNamespace(added_tokens_decoder={7: None}, all_special_ids=[7], convert_ids_to_tokens=lambda ids: [vocab[i] for i in ids])
    assert TP.token_bytes(tok, [5, 6, 7]) == [" x²".encode(), b"\n\n", b""]
    assert len(TP.bytes_to_unicode()) == 256 and TP.bytes_to_unicode()[ord(" ")] == "Ġ" and TP.bytes_to_unicode()[ord("\n")] == "Ċ"


def test_bare_verify_recheck():   # reviewer A1 19:2x UTC
    pieces, _ = P("\\boxed{", "4", "}", " To", " verify", ",", " re", "-check", ".")
    assert TP.markers(pieces, TP.first_boxed_end(b"".join(pieces))) == (6, 3)
