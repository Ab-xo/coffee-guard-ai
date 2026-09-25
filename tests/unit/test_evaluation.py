from __future__ import annotations

import numpy as np
import pytest

from coffeeguard.evaluation.calibration import ece, fit_temperature
from coffeeguard.evaluation.conformal import fit_qhat, prediction_sets, set_metrics
from coffeeguard.evaluation.metrics import bootstrap_ci, paired_bootstrap_diff
from coffeeguard.inference.predictor import softmax


def _calibrated_logits(n: int, seed: int, scale: float = 3.0):
    """Labels drawn from softmax(logits): perfectly calibrated at T = 1 by construction."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, scale, (n, 4))
    p = softmax(logits)
    labels = np.array([rng.choice(4, p=row) for row in p])
    return logits, labels


def test_temperature_recovers_known_miscalibration():
    logits, labels = _calibrated_logits(4000, seed=0)
    # a model that outputs logits x2 is over-confident; the fitted T should undo it (~2)
    t = fit_temperature(logits * 2.0, labels)
    assert t == pytest.approx(2.0, rel=0.1)
    assert ece(softmax(logits * 2.0, t), labels) < ece(softmax(logits * 2.0), labels)


def test_ece_near_zero_when_calibrated_and_large_when_not():
    logits, labels = _calibrated_logits(4000, seed=1)
    assert ece(softmax(logits), labels) < 0.03
    assert ece(softmax(logits * 4.0), labels) > 0.1


def test_conformal_coverage_holds_on_fresh_data():
    cal_logits, cal_y = _calibrated_logits(2000, seed=2)
    test_logits, test_y = _calibrated_logits(4000, seed=3)
    for alpha in (0.1, 0.02):
        q = fit_qhat(softmax(cal_logits), cal_y, alpha)
        m = set_metrics(prediction_sets(softmax(test_logits), q), test_y, list("ABCD"))
        assert m["coverage"] >= 1 - alpha - 0.02  # marginal guarantee, minus sampling noise


def test_prediction_sets_never_empty():
    probs = np.array([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    sets = prediction_sets(probs, qhat=0.0)  # threshold 1.0: only the forced top class
    assert sets.sum(1).tolist() == [1, 1]


def test_bootstrap_ci_brackets_point_estimate_and_paired_diff_of_identical_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    pred = np.where(rng.random(400) < 0.9, y, (y + 1) % 4)
    ci = bootstrap_ci(y, pred, n_boot=300)
    acc = (y == pred).mean()
    assert ci["accuracy"]["lo"] <= acc <= ci["accuracy"]["hi"]
    d = paired_bootstrap_diff(y, pred, pred, n_boot=200)
    assert d["diff"] == d["lo"] == d["hi"] == 0.0
