"""Stage 1 of ``data prepare``: ingest + validate + measure + cache, one pass per image.

Each image is opened exactly once. The pass records identity (path, label, SHA-256),
structure (format, size, EXIF), validity, quality measures and a perceptual hash, and
writes a resized copy to the processed cache. Cache files are content-addressed
(named by SHA-256), so re-running is cheap and exact duplicates share one cache file.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from coffeeguard.config import DataConfig
from coffeeguard.inference.quality import quality_metrics, resize_long_side, to_rgb
from coffeeguard.utils.hashing import sha256_file
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SKIP_DIRS = {".ipynb_checkpoints", "__MACOSX", ".git"}
SOURCE_SPLITS = {"train", "test", "val", "valid", "validation"}

# Guard against decompression bombs (a 12 MP phone photo is ~12e6 pixels).
Image.MAX_IMAGE_PIXELS = 80_000_000


@dataclass(frozen=True)
class ScanSettings:
    raw_dir: Path
    processed_dir: Path
    allowed_formats: tuple[str, ...]
    min_side: int
    aspect_warn: tuple[float, float]
    cache_long_side: int
    cache_quality: int
    labels: dict[str, str | None]  # raw folder name -> canonical label (None = unknown)
    class_to_id: dict[str, int]


def label_slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def list_image_files(raw_dir: Path) -> list[Path]:
    """All image files under ``raw_dir`` (sorted, skipping notebook/OS junk folders)."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(raw_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(p)
    return files


def _source_split(rel: Path) -> str:
    for part in rel.parts[:-1]:
        if part.lower().split()[0] in SOURCE_SPLITS:  # e.g. "train aug"
            return part.lower()
    return ""


def name_pattern(filename: str) -> str:
    """Filename with digit runs replaced by '#' and copy suffixes removed.

    e.g. ``C10P10H1.jpg`` → ``C#P#H#``, ``20231123_102720 (2).jpg`` → ``#_#``. Different
    patterns usually mean different acquisition sources, which the EDA cross-tabs
    against class to spot source/class confounding.
    """
    stem = re.sub(r"\s*\(\d+\)$", "", Path(filename).stem)
    return re.sub(r"\d+", "#", stem)


_D4 = [
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
    Image.Transpose.FLIP_LEFT_RIGHT,
    Image.Transpose.FLIP_TOP_BOTTOM,
    Image.Transpose.TRANSPOSE,
    Image.Transpose.TRANSVERSE,
]


def dihedral_phashes(img: Image.Image) -> list[str]:
    """pHash of the image under all 8 rotations/flips (identity first).

    Comparing one image's identity hash with another's 8 variants detects copies that
    were mirrored or rotated by 90° — common in pre-augmented datasets.
    """
    import imagehash

    small = resize_long_side(img, 128).convert("L")
    return [str(imagehash.phash(small if t is None else small.transpose(t))) for t in _D4]


def scan_one(path: Path, s: ScanSettings) -> dict[str, Any]:
    rel = path.relative_to(s.raw_dir)
    raw_label = path.parent.name
    label = s.labels.get(raw_label)
    row: dict[str, Any] = {
        "path": rel.as_posix(),
        "source_split": _source_split(rel),
        "raw_label": raw_label,
        "label": label,
        "class_id": s.class_to_id[label] if label else -1,
        "bytes": path.stat().st_size,
        "sha256": None,
        "format": None,
        "mode": None,
        "width": None,
        "height": None,
        "exif_rotated": False,
        "phash": None,
        "phash_d4": None,
        "cache_path": None,
        "name_pattern": name_pattern(path.name),
    }
    issues: list[str] = []
    warnings: list[str] = []

    if row["bytes"] == 0:
        issues.append("empty_file")
    else:
        row["sha256"] = sha256_file(path)
        try:
            with Image.open(path) as im:
                row["format"] = im.format
                row["mode"] = im.mode
                im.load()  # full decode: catches truncated/corrupt files that verify() misses
                orientation = im.getexif().get(0x0112, 1)
                rgb = to_rgb(im)
            row["exif_rotated"] = orientation not in (None, 1)
            row["width"], row["height"] = rgb.size
        except (
            OSError,
            SyntaxError,
            ValueError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
        ) as exc:
            issues.append(f"unreadable:{type(exc).__name__}")
            rgb = None

        if rgb is not None:
            if row["format"] not in s.allowed_formats:
                issues.append(f"format:{row['format']}")
            w, h = rgb.size
            if min(w, h) < s.min_side:
                issues.append("too_small")
            aspect = w / h
            if not s.aspect_warn[0] <= aspect <= s.aspect_warn[1]:
                warnings.append("aspect")
            row.update(quality_metrics(rgb))

            variants = dihedral_phashes(rgb)
            row["phash"] = variants[0]
            row["phash_d4"] = ",".join(variants)

            if label is not None:
                cache_rel = Path(label_slug(label)) / f"{row['sha256'][:20]}.jpg"
                cache_abs = s.processed_dir / cache_rel
                if not cache_abs.exists():
                    cache_abs.parent.mkdir(parents=True, exist_ok=True)
                    small = resize_long_side(rgb, s.cache_long_side, upscale=False)
                    small.save(cache_abs, "JPEG", quality=s.cache_quality, optimize=True)
                row["cache_path"] = cache_rel.as_posix()

    if label is None:
        issues.append("unknown_label")
    row["is_valid"] = not issues
    row["issues"] = ";".join(issues)
    row["warnings"] = ";".join(warnings)
    return row


def _scan_star(args: tuple[Path, ScanSettings]) -> dict[str, Any]:
    return scan_one(*args)


def scan_dataset(cfg: DataConfig, workers: int | None = None) -> pd.DataFrame:
    """Scan every raw image in parallel and return the raw manifest."""
    files = list_image_files(cfg.raw_dir)
    if not files:
        raise FileNotFoundError(f"No images under {cfg.raw_dir}. Run `coffeeguard data download`.")

    folders = sorted({p.parent.name for p in files})
    labels = {f: cfg.canonical_label(f) for f in folders}
    unknown = [f for f, lab in labels.items() if lab is None]
    if unknown:
        log.warning("Folders with no class mapping (their images are marked invalid): %s", unknown)
    log.info("Folder -> class map: %s", {f: lab for f, lab in labels.items() if lab})

    settings = ScanSettings(
        raw_dir=cfg.raw_dir,
        processed_dir=cfg.processed_dir,
        allowed_formats=tuple(cfg.allowed_formats),
        min_side=cfg.min_side,
        aspect_warn=cfg.aspect_warn,
        cache_long_side=cfg.cache_long_side,
        cache_quality=cfg.cache_quality,
        labels=labels,
        class_to_id=cfg.class_to_id,
    )
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    log.info("Scanning %d files with %d workers", len(files), workers)
    tasks = [(p, settings) for p in files]
    if workers == 1:
        rows = [scan_one(p, settings) for p in tqdm(files, desc="scan")]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(
                tqdm(pool.map(_scan_star, tasks, chunksize=16), total=len(tasks), desc="scan")
            )
    return pd.DataFrame(rows)
