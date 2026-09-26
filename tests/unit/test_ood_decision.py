from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from coffeeguard.inference.decision import QualityThresholds, decide
from coffeeguard.inference.ood import OODState, _l2n, score
from coffeeguard.inference.quality import quality_metrics
from tests.fixtures.synthetic import make_leaf

CLASSES = ["Healthy", "Cercospora", "Leaf Rust", "Phoma"]
QT = QualityThresholds(
    min_brightness=60, max_brightness=230, min_contrast=10, min_sharpness=20, min_green_frac=0.02
)


def _state(rng):
    means = rng.normal(0, 1, (4, 16)).astype(np.float32) * 5
    bank = np.concatenate([m + rng.normal(0, 0.3, (50, 16)) for m in means])
    return OODState(
        temperature=1.0,
        knn_bank=_l2n(bank).astype(np.float16),
        class_means=means,
        precision=np.eye(16, dtype=np.float32),
    ), means


def test_feature_scorers_rank_training_like_embeddings_as_less_ood():
    rng = np.random.default_rng(0)
    st, means = _state(rng)
    id_emb = means[[0, 1, 2, 3]] + rng.normal(0, 0.3, (4, 16))
    ood_emb = rng.normal(0, 5, (4, 16)) + 20  # far from every class mean
    logits = np.zeros((4, 4))
    for name in ("mahalanobis", "knn"):
        s_id, s_ood = score(name, logits, id_emb, st), score(name, logits, ood_emb, st)
        assert s_id.max() < s_ood.min(), name


def test_logit_scorers_prefer_confident_logits():
    st = OODState(temperature=1.0)
    confident, flat = np.array([[8.0, 0, 0, 0]]), np.array([[0.1, 0, 0, 0]])
    emb = np.zeros((1, 16))
    for name in ("msp", "energy"):
        assert score(name, confident, emb, st)[0] < score(name, flat, emb, st)[0], name
    with pytest.raises(ValueError):
        score("nope", confident, emb, st)


def test_decision_paths():
    p_conf = np.array([0.97, 0.01, 0.01, 0.01])
    p_split = np.array([0.02, 0.50, 0.46, 0.02])
    kw = {"tau_ood": 1.0, "qhat": 0.7, "tau_conf": 0.9}
    d = decide(CLASSES, p_conf, 0.5, [], **kw)
    assert (d.status, d.reason, d.label, d.prediction_set) == (
        "accepted",
        None,
        "Healthy",
        ["Healthy"],
    )
    d = decide(CLASSES, p_split, 0.5, [], **kw)
    assert (d.status, d.reason) == ("uncertain", "low_confidence")
    assert d.prediction_set == ["Cercospora", "Leaf Rust"] and d.advice
    d = decide(CLASSES, p_conf, 2.0, [], **kw)
    assert (d.status, d.reason, d.label) == ("rejected", "ood", None)
    d = decide(CLASSES, p_conf, 0.5, ["too_dark"], **kw)  # quality is checked first
    assert (d.status, d.reason, d.issues) == ("rejected", "low_quality", ["too_dark"])


def test_quality_gate_rejects_blank_and_dark_but_passes_a_leaf():
    blank = Image.new("RGB", (384, 288), (128, 128, 128))
    issues = QT.check(quality_metrics(blank))
    assert "too_blurry" in issues and "no_leaf" in issues
    leaf = make_leaf("Leaf Rust", seed=3, size=(384, 288))
    assert QT.check(quality_metrics(leaf)) == []
    dark = Image.eval(leaf, lambda v: v // 5)
    assert "too_dark" in QT.check(quality_metrics(dark))
