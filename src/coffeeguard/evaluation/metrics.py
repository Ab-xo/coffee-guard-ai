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
