"""[v2.5.9z74] (Q-062) the difficulty probe's numerics without a GPU: auc (ties, one class), the L2 logistic fit against a known
solution, the layer probe's CV choice and test AUC on separable / random features, the route length ratio, the pre-registered lines."""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diff_probe as P  # noqa: E402


def test_auc():
    assert P.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert P.auc([0, 1], [0.5, 0.5]) == 0.5                         # a tie counts half
    assert P.auc([1, 1], [0.1, 0.2]) is None                        # one class only
    assert P.auc([0, 1, 0, 1], [3, 2, 1, 0]) == 0.25


def test_auc_against_sklearn():
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return
    rs = np.random.RandomState(1); y = rs.rand(400) > 0.6; s = np.round(rs.rand(400) + 0.3 * y, 2)   # with ties
    assert abs(P.auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_fit_matches_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return
    rs = np.random.RandomState(2); X = rs.randn(300, 5); y = (X @ np.array([1.0, -2, 0.5, 0, 0]) + 0.3 * rs.randn(300) > 0).astype(float)
    l2 = 0.01; w, b = P.fit_logreg(X, y, l2)
    sk = LogisticRegression(C=1 / (l2 * len(y)), tol=1e-10, max_iter=10000).fit(X, y)   # C sum loss + |w|^2 / 2 = n (mean + l2 / 2 |w|^2)
    assert np.allclose(w, sk.coef_[0], atol=1e-4) and abs(b - sk.intercept_[0]) < 1e-4


def test_layer_probe_separable_and_random():
    rs = np.random.RandomState(3)
    Xtr = rs.randn(500, 8); ytr = (Xtr[:, 0] > 0).astype(float); Xte = rs.randn(200, 8); yte = (Xte[:, 0] > 0).astype(float)
    r = P.layer_probe(Xtr, ytr, Xte, yte)
    assert r["cv_auc"] > 0.97 and r["test_auc"] > 0.97 and r["l2"] in P.L2_GRID and len(r["p_test"]) == 200
    r = P.layer_probe(rs.randn(500, 8), (rs.rand(500) > 0.5).astype(float), rs.randn(200, 8), (rs.rand(200) > 0.5).astype(float))
    assert 0.35 < r["test_auc"] < 0.65                                 # noise stays near 0.5


def test_length_ratio_and_route_file():
    d = tempfile.mkdtemp(); p = os.path.join(d, "r.jsonl")
    with open(p, "w") as f:
        for i, n in ((0, 100), (0, 300), (1, 1000), (2, 600)):
            f.write(json.dumps({"pid": f"math500-{i}", "n_tokens": n, "correct": True}) + "\n")
    recs = P.read_route(p)
    assert P.length_ratio(recs, {0: True, 1: False, 2: False}) == round(800 / 200, 4)
    assert P.length_ratio(recs, {0: True, 1: True, 2: True}) is None


def test_judge():
    L = [{"cv_auc": 0.7, "test_auc": 0.85}, {"cv_auc": 0.82, "test_auc": 0.79}, {"cv_auc": 0.75, "test_auc": 0.9}]
    j = P.judge(L, 0.6, 0.65, {"ratio_true": 2.0, "ratio_probe": 1.85})
    assert (j["h1_layer"], j["h1_auc"], j["H1"]) == (1, 0.79, False)    # H1's layer by train CV, not by MATH-500
    assert (j["best_test_layer"], j["best_test_auc"]) == (2, 0.9) and j["H2"] is True and j["H3"] is True
    assert P.judge(L, 0.8, None, {})["H2"] is False and P.judge(L, 0.8, None, {})["H3"] is None
