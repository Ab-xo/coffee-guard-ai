from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from coffeeguard.config import DataConfig, SplitConfig
from coffeeguard.data.dedup import assign_groups, embedding_pairs, phash_pairs
from coffeeguard.data.prepare import prepare
from coffeeguard.data.scan import scan_dataset
from coffeeguard.data.split import assign_splits, check_no_leakage, split_frames
from tests.fixtures.synthetic import make_leaf


def _cfg(raw: Path, tmp: Path, **kw) -> DataConfig:
    return DataConfig(
        class_aliases={"Cerscospora": "Cercospora", "Leaf rust": "Leaf Rust"},
        raw_dir=raw,
        processed_dir=tmp / "processed",
        manifest_path=tmp / "manifests" / "manifest.parquet",
        splits_dir=tmp / "splits",
        embeddings_path=tmp / "manifests" / "embeddings.npz",
        **kw,
    )


# ------------------------------------------------------------------------- scan


def test_scan_flags_bad_files(synthetic_raw: Path, tmp_path: Path):
    healthy = synthetic_raw / "train" / "Healthy"
    (healthy / "empty.jpg").write_bytes(b"")
    good = sorted(healthy.glob("healthy_*.jpg"))[0].read_bytes()
    (healthy / "truncated.jpg").write_bytes(good[: len(good) // 2])
    make_leaf("Healthy", seed=99, size=(60, 40)).save(healthy / "tiny.jpg")
    (synthetic_raw / "train" / "Maize").mkdir()
    make_leaf("Healthy", seed=98).save(synthetic_raw / "train" / "Maize" / "m.jpg")
    (synthetic_raw / ".ipynb_checkpoints").mkdir()
    make_leaf("Healthy", seed=97).save(synthetic_raw / ".ipynb_checkpoints" / "x.jpg")

    df = scan_dataset(_cfg(synthetic_raw, tmp_path), workers=1).set_index("path")

    assert df.loc["train/Healthy/empty.jpg", "issues"] == "empty_file"
    assert df.loc["train/Healthy/truncated.jpg", "issues"].startswith("unreadable")
    assert df.loc["train/Healthy/tiny.jpg", "issues"] == "too_small"
    assert df.loc["train/Maize/m.jpg", "issues"] == "unknown_label"
    assert not any(p.startswith(".ipynb_checkpoints") for p in df.index)
    ok = df[df["is_valid"]]
    assert len(ok) == 24
    assert set(ok["label"]) == {"Healthy", "Cercospora", "Leaf Rust", "Phoma"}
    assert ok["cache_path"].notna().all()
    assert {"sharpness", "brightness", "green_frac"} <= set(df.columns)


def test_cache_is_resized_and_content_addressed(synthetic_raw: Path, tmp_path: Path):
    cfg = _cfg(synthetic_raw, tmp_path, cache_long_side=128)
    df = scan_dataset(cfg, workers=1)
    row = df[df["is_valid"]].iloc[0]
    with Image.open(cfg.processed_dir / row["cache_path"]) as im:
        assert max(im.size) == 128
    assert row["sha256"][:20] in row["cache_path"]


# ------------------------------------------------------------------------ dedup


def test_phash_and_embedding_pairs():
    hashes = np.array([0b0, 0b111, 0xFFFF_FFFF_FFFF_FFFF], dtype=np.uint64)
    assert phash_pairs(hashes, max_distance=3).tolist() == [[0, 1]]
    emb = np.array([[1, 0], [0.99, 0.05], [0, 1]], dtype=np.float32)
    assert embedding_pairs(emb, 0.95).tolist() == [[0, 1]]


def test_dedup_exact_near_and_conflicts(synthetic_raw: Path, tmp_path: Path):
    healthy = synthetic_raw / "train" / "Healthy"
    phoma = synthetic_raw / "train" / "Phoma"
    src = sorted(healthy.glob("*.jpg"))[0]
    shutil.copy(src, healthy / "zz_copy.jpg")  # exact dup, same class
    shutil.copy(sorted(healthy.glob("*.jpg"))[1], phoma / "x.jpg")  # exact dup, other class
    with Image.open(sorted(healthy.glob("*.jpg"))[2]) as im:  # near dup (re-encoded smaller)
        im.resize((200, 150)).save(healthy / "zz_near.jpg", quality=70)

    df = scan_dataset(_cfg(synthetic_raw, tmp_path), workers=1)
    out, summary = assign_groups(df, phash_max_distance=5)
    by_path = out.set_index("path")

    assert by_path.loc["train/Healthy/zz_copy.jpg", "status"] == "duplicate"
    assert by_path.loc["train/Phoma/x.jpg", "status"] == "conflict"
    assert summary["exact_conflicts_removed"] == 2
    near_src = sorted(p.name for p in healthy.glob("healthy_*.jpg"))[2]
    assert (
        by_path.loc[f"train/Healthy/{near_src}", "group_id"]
        == by_path.loc["train/Healthy/zz_near.jpg", "group_id"]
    )


def test_mirrored_copy_is_grouped_and_aug_files_excluded(synthetic_raw: Path, tmp_path: Path):
    healthy = synthetic_raw / "train" / "Healthy"
    src = sorted(healthy.glob("healthy_*.jpg"))[0]
    with Image.open(src) as im:
        im.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(healthy / "mirror.jpg", quality=90)
        im.save(healthy / "aug_1_2.jpeg", quality=90)

    df = scan_dataset(_cfg(synthetic_raw, tmp_path), workers=1)
    assert df.set_index("path").loc["train/Healthy/mirror.jpg", "name_pattern"] == "mirror"
    out, summary = assign_groups(df, phash_max_distance=5, exclude_name_patterns=["^aug_"])
    by_path = out.set_index("path")

    assert by_path.loc["train/Healthy/aug_1_2.jpeg", "status"] == "excluded_pattern"
    assert summary["excluded_by_name_pattern"] == 1
    assert (
        by_path.loc[f"train/Healthy/{src.name}", "group_id"]
        == by_path.loc["train/Healthy/mirror.jpg", "group_id"]
    )


def test_name_pattern():
    from coffeeguard.data.scan import name_pattern

    assert name_pattern("C10P10H1.jpg") == "C#P#H#"
    assert name_pattern("20231123_102720 (2).jpg") == "#_#"
    assert name_pattern("aug_12_3.jpeg") == "aug_#_#"


# ------------------------------------------------------------------------ split


def test_group_split_has_no_leakage_and_right_proportions():
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {
            "status": "ok",
            "class_id": rng.integers(0, 4, n),
            "group_id": rng.integers(0, 700, n),  # many multi-image groups
            "sha256": [f"{i:064x}" for i in range(n)],
            "path": [f"p{i}.jpg" for i in range(n)],
            "cache_path": [f"c{i}.jpg" for i in range(n)],
        }
    )
    df["label"] = df["class_id"].map({0: "Healthy", 1: "Cercospora", 2: "Leaf Rust", 3: "Phoma"})
    out = assign_splits(df, SplitConfig())
    frames = split_frames(out)
    check_no_leakage(frames)
    fractions = {k: len(v) / n for k, v in frames.items()}
    assert fractions["train"] == pytest.approx(0.70, abs=0.03)
    assert fractions["val"] == pytest.approx(0.15, abs=0.03)
    assert fractions["test"] == pytest.approx(0.15, abs=0.03)
    for frame in frames.values():  # stratification: each class within ±4 pp of overall share
        share = frame["class_id"].value_counts(normalize=True).sort_index()
        overall = df["class_id"].value_counts(normalize=True).sort_index()
        assert (share - overall).abs().max() < 0.04


def test_check_no_leakage_detects_shared_group():
    a = pd.DataFrame({"sha256": ["x"], "group_id": [1]})
    b = pd.DataFrame({"sha256": ["y"], "group_id": [1]})
    with pytest.raises(AssertionError):
        check_no_leakage({"train": a, "test": b})


# ---------------------------------------------------------------------- prepare


def test_prepare_end_to_end_without_embeddings(synthetic_raw: Path, tmp_path: Path):
    cfg = _cfg(
        synthetic_raw, tmp_path, split=SplitConfig(n_folds=4, train=0.5, val=0.25, test=0.25)
    )
    summary = prepare(cfg, stages=("scan", "group", "split"), workers=1)
    for name in ("train", "val", "test"):
        assert (cfg.splits_dir / f"{name}.csv").exists()
    assert (cfg.splits_dir / "split_info.json").exists()
    assert len(summary["fingerprint"]) == 16
    # idempotent: a second run reuses the scan and gives the same fingerprint
    again = prepare(cfg, stages=("group", "split"), workers=1)
    assert again["fingerprint"] == summary["fingerprint"]
