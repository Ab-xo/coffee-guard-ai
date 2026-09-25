"""Stage 3 of ``data prepare``: stratified, group-aware train/val/test split.

``StratifiedGroupKFold`` with ``n_folds`` folds keeps every duplicate group inside a
single fold while balancing class proportions across folds. Whole folds are then
assigned to splits (default 20 folds: 3 test + 3 val + 14 train = 15/15/70).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from coffeeguard.config import SplitConfig

SPLITS = ("train", "val", "test")
SPLIT_COLUMNS = ["image", "label", "class_id", "group_id", "sha256", "path"]


def assign_splits(df: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """Add a ``split`` column to rows with ``status == 'ok'`` (others get '')."""
    df = df.copy()
    df["split"] = ""
    ok = df.index[df["status"] == "ok"]
    y = df.loc[ok, "class_id"].to_numpy()
    groups = df.loc[ok, "group_id"].to_numpy()

    n_test = round(cfg.test * cfg.n_folds)
    n_val = round(cfg.val * cfg.n_folds)
    sgkf = StratifiedGroupKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    fold_of = np.empty(len(ok), dtype=int)
    for fold, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(ok)), y, groups)):
        fold_of[test_idx] = fold

    split = np.where(fold_of < n_test, "test", np.where(fold_of < n_test + n_val, "val", "train"))
    df.loc[ok, "split"] = split
    return df


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-split frames with the columns training needs (``image`` = cache path)."""
    frames = {}
    for name in SPLITS:
        part = df[df["split"] == name].rename(columns={"cache_path": "image"})
        frames[name] = part[SPLIT_COLUMNS].sort_values("image").reset_index(drop=True)
    return frames


def check_no_leakage(frames: dict[str, pd.DataFrame]) -> None:
    """Raise if any SHA-256 or group appears in more than one split."""
    for key in ("sha256", "group_id"):
        seen: dict = {}
        for name, frame in frames.items():
            for value in frame[key].unique():
                if value in seen and seen[value] != name:
                    raise AssertionError(f"{key}={value} is in both {seen[value]} and {name}")
                seen[value] = name


def balance_table(frames: dict[str, pd.DataFrame], classes: list[str]) -> pd.DataFrame:
    """Counts and percentages per class and split."""
    counts = pd.DataFrame(
        {
            name: frame["label"].value_counts().reindex(classes, fill_value=0)
            for name, frame in frames.items()
        }
    )
    counts["total"] = counts.sum(axis=1)
    counts.loc["total"] = counts.sum()
    return counts
