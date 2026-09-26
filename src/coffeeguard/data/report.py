"""``coffeeguard data report``: EDA + data-quality figures and ``eda_summary.json``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from coffeeguard.config import DataConfig
from coffeeguard.data.split import SPLITS
from coffeeguard.utils.io import read_json, write_json
from coffeeguard.utils.log import get_logger
from coffeeguard.utils.paths import portable_path
from coffeeguard.utils.plotting import PRIMARY, SPLIT_COLORS, TEXT, TEXT_MUTED, apply_style, save

log = get_logger(__name__)

QUALITY_COLUMNS = ["sharpness", "brightness", "contrast", "green_frac"]
QUALITY_TITLES = {
    "sharpness": "Sharpness (Laplacian variance, log scale)",
    "brightness": "Brightness (mean luma, 0-255)",
    "contrast": "Contrast (RMS, luma std)",
    "green_frac": "Green-pixel fraction",
}


def _describe(series: pd.Series) -> dict[str, float]:
    q = series.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "mean": float(series.mean()),
        "p01": float(q[0.01]),
        "p25": float(q[0.25]),
        "median": float(q[0.5]),
        "p75": float(q[0.75]),
        "p99": float(q[0.99]),
    }


def build_summary(m: pd.DataFrame, cfg: DataConfig) -> dict:
    valid = m[m["is_valid"]]
    ok = m[m["status"] == "ok"]
    summary: dict = {
        "files": len(m),
        "valid": int(m["is_valid"].sum()),
        "status_counts": m["status"].value_counts().to_dict(),
        "raw_class_counts": m["raw_label"].value_counts().to_dict(),
        "source_split_by_class": pd.crosstab(m["label"].fillna("?"), m["source_split"]).to_dict(
            orient="index"
        ),
        "clean_class_counts": ok["label"]
        .value_counts()
        .reindex(cfg.classes, fill_value=0)
        .to_dict(),
        "imbalance_ratio": float(
            ok["label"].value_counts().max() / max(ok["label"].value_counts().min(), 1)
        ),
        "formats": valid["format"].value_counts().to_dict(),
        "modes": valid["mode"].value_counts().to_dict(),
        "exif_rotated": int(valid["exif_rotated"].sum()),
        "aspect_warnings": int(valid["warnings"].str.contains("aspect").sum()),
        "width": _describe(valid["width"]),
        "height": _describe(valid["height"]),
        "most_common_sizes": (
            valid["width"].astype(int).astype(str) + "x" + valid["height"].astype(int).astype(str)
        )
        .value_counts()
        .head(10)
        .to_dict(),
        "quality_by_class": {
            col: {cls: _describe(g[col]) for cls, g in ok.groupby("label")}
            for col in QUALITY_COLUMNS
        },
        # Shortcut / confounding checks: if resolution, file size or file-name source
        # pattern differ strongly by class, a model can learn them instead of disease.
        "resolution_by_class": {
            cls: {
                "long_side": _describe(g[["width", "height"]].max(axis=1)),
                "kb": _describe(g["bytes"] / 1024),
            }
            for cls, g in ok.groupby("label")
        },
        "name_pattern_by_class": pd.crosstab(ok["name_pattern"], ok["label"])
        .loc[lambda t: t.sum(axis=1) >= 5]
        .to_dict(orient="index"),
    }
    split_info = cfg.splits_dir / "split_info.json"
    if split_info.exists():
        info = read_json(split_info)
        summary["dedup"] = info.get("dedup")
        summary["fingerprint"] = info.get("fingerprint")
        summary["balance"] = info.get("balance")
    return summary


# ------------------------------------------------------------------------ figures


def fig_class_distribution(m: pd.DataFrame, cfg: DataConfig, out: Path) -> Path:
    import matplotlib.pyplot as plt

    ok = m[m["status"] == "ok"]
    counts = pd.crosstab(ok["label"], ok["split"]).reindex(
        index=cfg.classes, columns=list(SPLITS), fill_value=0
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(cfg.classes))
    width = 0.26
    for i, split in enumerate(SPLITS):
        bars = ax.bar(
            x + (i - 1) * (width + 0.01),
            counts[split],
            width,
            label=split,
            color=SPLIT_COLORS[split],
            linewidth=0,
        )
        ax.bar_label(bars, fmt="%d", fontsize=7, color=TEXT_MUTED, padding=2)
    totals = counts.sum(axis=1)
    ax.set_xticks(x, [f"{c}\n(n={t})" for c, t in zip(cfg.classes, totals, strict=True)])
    ax.set_ylabel("images")
    ax.grid(axis="x", visible=False)
    ax.set_title("Clean images per class and split")
    ax.legend(ncols=3, loc="upper right")
    return save(fig, out)


def fig_sample_grid(
    m: pd.DataFrame, cfg: DataConfig, label: str, out: Path, n: int = 5, seed: int = 0
) -> Path:
    import matplotlib.pyplot as plt

    pool = m[(m["status"] == "ok") & (m["label"] == label)]
    take = pool.sample(min(n * n, len(pool)), random_state=seed)
    fig, axes = plt.subplots(n, n, figsize=(n * 1.6, n * 1.6))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, row) in zip(axes.ravel(), take.iterrows(), strict=False):
        with Image.open(cfg.processed_dir / row["cache_path"]) as im:
            ax.imshow(im.convert("RGB"))
    fig.suptitle(
        f"{label} — random clean samples",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, out)


def fig_augmentation_preview(
    train_df: pd.DataFrame,
    cfg: DataConfig,
    augment,
    img_size: int,
    out: Path,
    per_class: int = 2,
    n_aug: int = 5,
    seed: int = 0,
) -> Path:
    """Original + ``n_aug`` training augmentations for a few images per class.

    Used to check by eye that lesions (small, colour-coded) survive augmentation.
    """
    import random

    import matplotlib.pyplot as plt
    import torch

    from coffeeguard.data.transforms import train_transform

    random.seed(seed)
    torch.manual_seed(seed)
    # mean 0 / std 1 → the tensor is the augmented image in [0, 1], ready to plot
    tf = train_transform(img_size, augment, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    swap = None
    if augment.bg_swap_p > 0:
        from coffeeguard.data.bgswap import build_background_swap

        donors = []
        for rel in train_df["image"].sample(min(80, len(train_df)), random_state=seed):
            with Image.open(cfg.processed_dir / rel) as im:
                donors.append(im.convert("RGB"))
        swap = build_background_swap(donors, augment.bg_swap_p, seed=seed)
    rows = (
        train_df.groupby("label", sort=True)
        .sample(per_class, random_state=seed)
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(len(rows), n_aug + 1, figsize=((n_aug + 1) * 1.6, len(rows) * 1.6))
    for r, row in rows.iterrows():
        with Image.open(cfg.processed_dir / row["image"]) as im:
            img = im.convert("RGB")
        axes[r, 0].imshow(img.resize((img_size, img_size), Image.Resampling.BICUBIC))
        axes[r, 0].set_ylabel(row["label"], color=TEXT, fontsize=8)
        for c in range(1, n_aug + 1):
            aug_in = swap(img) if swap is not None else img
            axes[r, c].imshow(tf(aug_in).clamp(0, 1).permute(1, 2, 0).numpy())
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    axes[0, 0].set_title("original", fontsize=8, fontweight="normal")
    for c in range(1, n_aug + 1):
        axes[0, c].set_title(f"augmented {c}", fontsize=8, fontweight="normal")
    fig.suptitle(
        "Training augmentation preview",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, out)


def fig_image_sizes(m: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    v = m[m["is_valid"]]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    long_side = v[["width", "height"]].max(axis=1)
    axes[0].hist(long_side, bins=40, color=PRIMARY, edgecolor="white", linewidth=0.5)
    axes[0].set_title("Long side (px)")
    axes[0].set_ylabel("images")
    axes[1].hist(v["width"] / v["height"], bins=40, color=PRIMARY, edgecolor="white", linewidth=0.5)
    axes[1].set_title("Aspect ratio (w / h)")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return save(fig, out)


def fig_quality_by_class(m: pd.DataFrame, cfg: DataConfig, out: Path) -> Path:
    import matplotlib.pyplot as plt

    ok = m[m["status"] == "ok"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, col in zip(axes.ravel(), QUALITY_COLUMNS, strict=True):
        data = [ok.loc[ok["label"] == c, col].to_numpy() for c in cfg.classes]
        ax.boxplot(
            data,
            tick_labels=cfg.classes,
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#cde2fb", "edgecolor": PRIMARY, "linewidth": 1},
            medianprops={"color": PRIMARY, "linewidth": 2},
            whiskerprops={"color": PRIMARY, "linewidth": 1},
            capprops={"color": PRIMARY, "linewidth": 1},
        )
        if col == "sharpness":
            ax.set_yscale("log")
        ax.set_title(QUALITY_TITLES[col], fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Image quality by class (whiskers = 1.5 IQR, outliers hidden)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, out)


def fig_resolution_by_class(m: pd.DataFrame, cfg: DataConfig, out: Path) -> Path:
    """Original resolution and file size per class — a shortcut-risk check."""
    import matplotlib.pyplot as plt

    ok = m[m["status"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    series = {
        "Original long side (px, log)": ok[["width", "height"]].max(axis=1),
        "Original file size (KB, log)": ok["bytes"] / 1024,
    }
    for ax, (title, values) in zip(axes, series.items(), strict=True):
        data = [values[ok["label"] == c].to_numpy() for c in cfg.classes]
        ax.boxplot(
            data,
            tick_labels=cfg.classes,
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#cde2fb", "edgecolor": PRIMARY, "linewidth": 1},
            medianprops={"color": PRIMARY, "linewidth": 2},
            whiskerprops={"color": PRIMARY, "linewidth": 1},
            capprops={"color": PRIMARY, "linewidth": 1},
        )
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Resolution and file size by class (differences = possible shortcut)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, out)


def fig_duplicate_groups(
    m: pd.DataFrame, cfg: DataConfig, out: Path, n_groups: int = 6, per_group: int = 5
) -> Path | None:
    import matplotlib.pyplot as plt

    ok = m[m["status"] == "ok"]
    sizes = ok["group_id"].value_counts()
    multi = sizes[sizes > 1]
    if multi.empty:
        return None
    # show a spread of group sizes: largest few + some typical ones
    chosen = list(multi.index[: n_groups // 2]) + list(
        multi.sample(min(n_groups - n_groups // 2, len(multi)), random_state=0).index
    )
    chosen = list(dict.fromkeys(chosen))[:n_groups]
    fig, axes = plt.subplots(len(chosen), per_group, figsize=(per_group * 1.7, len(chosen) * 1.8))
    axes = np.atleast_2d(axes)
    for r, gid in enumerate(chosen):
        rows = ok[ok["group_id"] == gid].head(per_group)
        for c in range(per_group):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(rows):
                row = rows.iloc[c]
                with Image.open(cfg.processed_dir / row["cache_path"]) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(row["label"], fontsize=7, color=TEXT_MUTED)
        axes[r, 0].text(
            -0.08,
            0.5,
            f"group {gid}\nsize {sizes[gid]}",
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=7,
            color=TEXT_MUTED,
        )
    fig.suptitle(
        "Examples of duplicate / same-leaf groups (kept together in one split)",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, out)


def make_report(cfg: DataConfig) -> dict:
    apply_style()
    m = pd.read_parquet(cfg.manifest_path)
    figs = cfg.figures_dir / "eda"
    written = [
        fig_class_distribution(m, cfg, figs / "class_distribution.png"),
        fig_image_sizes(m, figs / "image_sizes.png"),
        fig_quality_by_class(m, cfg, figs / "quality_by_class.png"),
        fig_resolution_by_class(m, cfg, figs / "resolution_by_class.png"),
    ]
    from coffeeguard.data.scan import label_slug

    for label in cfg.classes:
        written.append(fig_sample_grid(m, cfg, label, figs / f"samples_{label_slug(label)}.png"))
    dup = fig_duplicate_groups(m, cfg, figs / "duplicate_groups.png")
    if dup:
        written.append(dup)

    summary = build_summary(m, cfg)
    summary["figures"] = [portable_path(p) for p in written]
    write_json(cfg.reports_dir / "eda_summary.json", summary)

    invalid = m[m["status"] != "ok"][["path", "label", "status", "issues", "duplicate_of"]]
    invalid.to_csv(cfg.reports_dir / "data_quality_removed.csv", index=False)
    log.info("Report written: %d figures, %s", len(written), cfg.reports_dir / "eda_summary.json")
    return summary
