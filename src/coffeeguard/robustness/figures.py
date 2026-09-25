"""Robustness figures: degradation curves and shortcut-test examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from coffeeguard.robustness.leafmask import apply_mask
from coffeeguard.utils.plotting import GRID, SERIES, TEXT_MUTED, apply_style, save


def fig_degradation(res: dict, out: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    names = list(res["corruptions"])
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)
    clean = res["clean"]["macro_f1"]
    for ax, c in zip(axes.ravel(), names, strict=True):
        sev = [0, *[int(s) for s in res["corruptions"][c]]]
        f1 = [clean, *[v["macro_f1"] for v in res["corruptions"][c].values()]]
        conf = [
            res["clean"]["mean_confidence"],
            *[v["mean_confidence"] for v in res["corruptions"][c].values()],
        ]
        ax.plot(sev, f1, marker="o", ms=3, color=SERIES[0], label="macro-F1")
        ax.plot(
            sev, conf, marker="o", ms=3, color=SERIES[1], label="mean confidence", linewidth=1.2
        )
        ax.axhline(0.25, color=GRID, linestyle="--", linewidth=1)
        ax.set_title(c.replace("_", " "), fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.set_xticks(range(6))
    for ax in axes[-1]:
        ax.set_xlabel("severity (0 = clean)")
    axes[0, 0].legend(loc="lower left")
    rr = res["relative_robustness"]
    fig.suptitle(
        f"{res['bundle']} — test macro-F1 under corruption "
        f"(relative robustness sev≤3: {rr['severity_1_3']:.3f}; dashed = chance)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, out)


def fig_shortcut_examples(
    images: list[Image.Image], masks: list[np.ndarray], test: pd.DataFrame, out: Path
) -> Path:
    """Two test images per class: original, leaf-only, background-only (mask sanity check)."""
    import matplotlib.pyplot as plt

    apply_style()
    idx = test.groupby("label").sample(2, random_state=0).index.tolist()
    fig, axes = plt.subplots(len(idx), 3, figsize=(6.6, len(idx) * 2.0))
    for r, i in enumerate(idx):
        views = (
            images[i],
            apply_mask(images[i], masks[i], True),
            apply_mask(images[i], masks[i], False),
        )
        for c, (ax, im) in enumerate(zip(axes[r], views, strict=True)):
            ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(("original", "leaf only", "background only")[c], fontsize=9)
        axes[r, 0].set_ylabel(test.loc[i, "label"], fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    return save(fig, out)
