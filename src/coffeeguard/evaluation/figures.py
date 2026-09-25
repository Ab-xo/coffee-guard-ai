"""Phase 4 figures: confusion matrix, reliability diagram, selective accuracy, error gallery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from coffeeguard.config import DataConfig
from coffeeguard.evaluation.calibration import reliability_bins
from coffeeguard.inference.predictor import softmax
from coffeeguard.utils.plotting import GRID, SERIES, TEXT, TEXT_MUTED, apply_style, save


def fig_confusion(cm: np.ndarray, classes: list[str], title: str, out: Path) -> Path:
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if norm[i, j] > 0.6 else TEXT
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", color=color)
    ax.set_xticks(range(len(classes)), classes, rotation=20)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    return save(fig, out)


def fig_reliability(data: dict, temperature: float, title: str, out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, split in zip(axes, ("val", "test"), strict=True):
        d = data[split]
        for probs, label, color in (
            (softmax(d["logits"]), "before", SERIES[1]),
            (softmax(d["logits"], temperature), f"after (T={temperature:.2f})", SERIES[0]),
        ):
            b = reliability_bins(probs, d["labels"], 10)
            m = b["count"] > 0
            ax.plot(
                b["confidence"][m], b["accuracy"][m], marker="o", ms=4, color=color, label=label
            )
        ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
        ax.set_title(f"{split}")
        ax.set_xlabel("confidence")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("accuracy")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, out)


def fig_selective(sel: dict, title: str, out: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(sel["coverage"], sel["accuracy"], color=SERIES[0])
    ax.set_xlabel("coverage (share of test images answered, most confident first)")
    ax.set_ylabel("accuracy on answered images")
    ax.set_title(title)
    fig.tight_layout()
    return save(fig, out)


def fig_error_gallery(frame, cfg: DataConfig, title: str, out: Path, max_n: int = 12) -> Path:
    import matplotlib.pyplot as plt

    err = frame[~frame["correct"]].sort_values("confidence", ascending=False).head(max_n)
    n = max(len(err), 1)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.7), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), err.iterrows(), strict=False):
        with Image.open(cfg.processed_dir / r["image"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(
            f"true {r['label']}\npred {r['pred_label']} ({r['confidence']:.2f})",
            fontsize=8,
            color=TEXT_MUTED,
            fontweight="normal",
        )
    if err.empty:
        axes[0, 0].text(0.5, 0.5, "no errors", ha="center")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return save(fig, out)


def make_figures(
    name: str, classes: list[str], data: dict, res: dict, out_dir: Path, cfg: DataConfig
) -> None:
    apply_style()
    cm = np.array(res["test"]["confusion_matrix"])
    fig_confusion(cm, classes, f"{name} — test confusion", out_dir / "confusion_test.png")
    fig_reliability(data, res["temperature"], f"{name} — reliability", out_dir / "reliability.png")
    fig_selective(
        res["test"]["selective"],
        f"{name} — selective accuracy (test)",
        out_dir / "selective_test.png",
    )
    fig_error_gallery(
        data["test"]["frame"],
        cfg,
        f"{name} — test errors (most confident first)",
        out_dir / "errors_test.png",
    )
