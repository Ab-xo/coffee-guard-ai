"""Classification metrics. Macro-F1 is the primary metric throughout."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def core_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs for accuracy and macro-F1 (resampling images)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        acc[b] = (y_true[i] == y_pred[i]).mean()
        f1[b] = _macro_f1(y_true[i], y_pred[i])
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {
        "accuracy": {"lo": float(np.percentile(acc, lo)), "hi": float(np.percentile(acc, hi))},
        "macro_f1": {"lo": float(np.percentile(f1, lo)), "hi": float(np.percentile(f1, hi))},
    }


def paired_bootstrap_diff(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Macro-F1(A) - macro-F1(B) on the same resampled images: mean, 95% CI, P(A <= B)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d[b] = _macro_f1(y_true[i], pred_a[i]) - _macro_f1(y_true[i], pred_b[i])
    return {
        "diff": _macro_f1(y_true, pred_a) - _macro_f1(y_true, pred_b),
        "lo": float(np.percentile(d, 2.5)),
        "hi": float(np.percentile(d, 97.5)),
        "p_a_not_better": float((d <= 0).mean()),
    }


def full_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: Sequence[str], probs: np.ndarray | None = None
) -> dict:
    labels = list(range(len(classes)))
    out: dict = core_metrics(y_true, y_pred)
    out["per_class"] = classification_report(
        y_true, y_pred, labels=labels, target_names=list(classes), output_dict=True, zero_division=0
    )
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    if probs is not None:
        from sklearn.metrics import roc_auc_score

        try:
            out["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=labels)
            )
        except ValueError:  # a class missing from y_true
            out["roc_auc_ovr_macro"] = None
    return out
