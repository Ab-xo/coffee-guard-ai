"""Disease-severity *proxy* and model behaviour by severity (the dataset has no severity labels).

Severity proxy = share of the leaf covered by visible lesions, from colour alone (so it
does not depend on the model being judged): inside the colour leaf mask, a pixel counts as
lesion if it is yellow / orange / brown (rust pustules, Cercospora halos) or much darker
than the leaf's typical brightness (necrotic Cercospora / Phoma patches). Only photos with
a separable background are used (in field photos the mask covers surrounding foliage).

This is an indication, not a validated severity score: veins, glare, shadows and dirt can
count as lesions, and very early infection may show no colour change at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from coffeeguard.config import DataConfig
from coffeeguard.data.dataset import read_split
from coffeeguard.inference.pipeline import Pipeline
from coffeeguard.robustness.leafmask import leaf_mask
from coffeeguard.robustness.sweep import MAX_LEAF_FRACTION
from coffeeguard.utils.io import write_json
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)


def lesion_map(img: Image.Image, mask: np.ndarray) -> np.ndarray:
    """Boolean map of lesion-coloured pixels inside the leaf mask."""
    import cv2

    hsv = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2HSV)  # H in [0, 180)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    if not mask.any():
        return np.zeros(mask.shape, bool)
    v_leaf = np.median(v[mask])
    warm = (h >= 5) & (h <= 30) & (s >= 70) & (v >= 60)  # yellow / orange / brown
    necrotic = v < 0.5 * v_leaf  # much darker than the rest of this leaf
    return mask & (warm | necrotic)


def lesion_fraction(img: Image.Image) -> tuple[float, float]:
    """(lesion share of the leaf, leaf share of the image)."""
    mask = leaf_mask(img)
    if not mask.any():
        return 0.0, 0.0
    return float(lesion_map(img, mask).sum() / mask.sum()), float(mask.mean())


def run_severity(bundle_dir: Path, cfg: DataConfig, out_root: Path) -> dict:
    pipe = Pipeline(bundle_dir)
    rows = []
    for split in ("val", "test"):
        for _, r in read_split(cfg.splits_dir, split).iterrows():
            with Image.open(cfg.processed_dir / r["image"]) as im:
                img = im.convert("RGB")
            frac, leaf = lesion_fraction(img)
            d = pipe.run(img).decision
            rows.append(
                {
                    "split": split,
                    "image": r["image"],
                    "label": r["label"],
                    "lesion_frac": frac,
                    "leaf_frac": leaf,
                    "status": d.status,
                    "pred": d.label,
                    "confidence": d.confidence,
                }
            )
    df = pd.DataFrame(rows)
    df = df[(df["leaf_frac"] > 0.02) & (df["leaf_frac"] <= MAX_LEAF_FRACTION)].copy()
    df["answered_wrong"] = (df["status"] == "accepted") & (df["pred"] != df["label"])
    df["wrong_top_class"] = df["pred"].notna() & (df["pred"] != df["label"])

    res: dict = {
        "bundle": bundle_dir.name,
        "n_used": len(df),
        "note": "severity proxy = colour-based lesion share of the leaf; photos on paper only",
        "lesion_frac_by_class": df.groupby("label")["lesion_frac"]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .round(4)
        .to_dict("index"),
        "by_tercile": {},
    }
    for label in ("Cercospora", "Leaf Rust", "Phoma"):
        sub = df[df["label"] == label].copy()
        sub["tercile"] = pd.qcut(
            sub["lesion_frac"].rank(method="first"), 3, labels=["mild", "moderate", "severe"]
        )
        tab = {}
        for t, g in sub.groupby("tercile", observed=True):
            tab[str(t)] = {
                "n": len(g),
                "lesion_frac_median": float(g["lesion_frac"].median()),
                "accepted": float((g["status"] == "accepted").mean()),
                "uncertain": float((g["status"] == "uncertain").mean()),
                "rejected": float((g["status"] == "rejected").mean()),
                "wrong_top_class": float(g["wrong_top_class"].mean()),
                "called_healthy": float((g["pred"] == "Healthy").mean()),
                "answered_wrong": float(g["answered_wrong"].mean()),
                "mean_confidence": float(g["confidence"].mean()),
            }
        res["by_tercile"][label] = tab
    out_dir = out_root / bundle_dir.name
    write_json(out_dir / "severity.json", res)
    df.to_csv(out_dir / "severity_per_image.csv", index=False)
    _figures(df, cfg, res, out_dir / "figures")
    return res


def _figures(df: pd.DataFrame, cfg: DataConfig, res: dict, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    from coffeeguard.utils.plotting import SERIES, TEXT_MUTED, apply_style, save

    apply_style()
    # 1. rates by severity tercile
    labels = list(res["by_tercile"])
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3.4), sharey=True)
    for ax, label in zip(axes, labels, strict=True):
        tab = res["by_tercile"][label]
        x = np.arange(len(tab))
        for i, (key, name) in enumerate(
            (
                ("uncertain", "uncertain"),
                ("wrong_top_class", "wrong top class"),
                ("called_healthy", "called Healthy"),
            )
        ):
            ax.bar(
                x + (i - 1) * 0.27,
                [v[key] for v in tab.values()],
                0.27,
                color=SERIES[i],
                label=name,
            )
        ax.set_xticks(x, [f"{k}\n({v['lesion_frac_median']:.0%})" for k, v in tab.items()])
        ax.set_title(label)
    axes[0].set_ylabel("share of photos")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Model behaviour by lesion-coverage tercile (val + test; proxy, see text)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, out_dir / "severity_rates.png")

    # 2. proxy sanity check: mildest and most severe examples with the lesion map
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.4))
    for r_i, label in enumerate(labels):
        sub = df[df["label"] == label].sort_values("lesion_frac")
        picks = list(sub.head(3).itertuples()) + list(sub.tail(3).itertuples())
        for c_i, (ax, row) in enumerate(zip(axes[r_i], picks, strict=True)):
            with Image.open(cfg.processed_dir / row.image) as im:
                img = im.convert("RGB")
            les = lesion_map(img, leaf_mask(img))
            a = np.asarray(img).copy()
            a[les] = (255, 0, 255)  # lesion pixels in magenta
            ax.imshow(a)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(
                f"{'mild' if c_i < 3 else 'severe'} {row.lesion_frac:.0%}",
                fontsize=8,
                color=TEXT_MUTED,
                fontweight="normal",
            )
        axes[r_i, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(
        "Severity proxy check: least vs. most lesion coverage (lesions in magenta)",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, out_dir / "severity_examples.png")
