"""Split conformal prediction with the LAC score (Sadinle et al., 2019).

Fit on validation: score = 1 - p(true class); q̂ = the ⌈(n+1)(1-α)⌉/n quantile.
Predict: the set of classes with p ≥ 1 - q̂. With exchangeable data the set contains
the true class with probability ≥ 1 - α. A set of size > 1 is how the UI can say
"Leaf Rust or Cercospora - retake the photo".
"""

from __future__ import annotations

import numpy as np


def fit_qhat(probs: np.ndarray, labels: np.ndarray, alpha: float = 0.10) -> float:
    n = len(labels)
    scores = 1.0 - probs[np.arange(n), labels]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean ``(N, C)`` membership; the top class is always included (no empty sets)."""
    sets = probs >= 1.0 - qhat
    sets[np.arange(len(probs)), probs.argmax(1)] = True
    return sets


def set_metrics(sets: np.ndarray, labels: np.ndarray, classes: list[str]) -> dict:
    covered = sets[np.arange(len(labels)), labels]
    size = sets.sum(1)
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "per_class": {
            c: {
                "coverage": float(covered[labels == k].mean()),
                "avg_set_size": float(size[labels == k].mean()),
            }
            for k, c in enumerate(classes)
            if (labels == k).any()
        },
    }
