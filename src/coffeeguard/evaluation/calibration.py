"""Confidence calibration: temperature scaling and expected calibration error (ECE)."""

from __future__ import annotations

import numpy as np

from coffeeguard.inference.predictor import softmax


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Single scalar T minimising validation NLL (Guo et al., 2017).

    Searched over log T so T stays positive; T > 1 softens over-confident models, T < 1
    sharpens under-confident ones (label smoothing makes models under-confident).
    """
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        lambda log_t: nll(logits, labels, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(res.x))


def reliability_bins(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> dict[str, np.ndarray]:
    """Per confidence bin: mean confidence, accuracy and count (equal-width bins)."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.bincount(idx, conf, n_bins) / count
        acc = np.bincount(idx, correct, n_bins) / count
    return {"edges": edges, "confidence": mean_conf, "accuracy": acc, "count": count}


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: count-weighted |accuracy - confidence| over bins."""
    b = reliability_bins(probs, labels, n_bins)
    m = b["count"] > 0
    return float(
        np.sum(b["count"][m] * np.abs(b["accuracy"][m] - b["confidence"][m])) / len(labels)
    )
