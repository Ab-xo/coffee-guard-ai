"""Tiny synthetic 'leaf' dataset so tests never need the real 2 GB Kaggle data.

Each image is a green ellipse ('leaf') on a soil-coloured background, with
class-specific spots: none (Healthy), small brown dots (Cercospora), orange blotches
(Leaf Rust), dark large patches (Phoma). The raw folder names mimic the Kaggle
layout, including the 'Cerscospora' typo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RAW_FOLDERS = {
    "Healthy": "Healthy",
    "Cercospora": "Cerscospora",
    "Leaf Rust": "Leaf rust",
    "Phoma": "Phoma",
}

_SPOTS = {
    "Healthy": None,
    "Cercospora": ((110, 70, 40), 4, 12),
    "Leaf Rust": ((230, 130, 30), 7, 8),
    "Phoma": ((50, 35, 25), 14, 4),
}


def make_leaf(label: str, seed: int, size: tuple[int, int] = (256, 192)) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    bg = tuple(int(c) for c in rng.integers([70, 50, 30], [150, 120, 90]))
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    # background clutter so perceptual hashes of different images differ
    for _ in range(12):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(8, 40))
        shade = tuple(int(c) for c in rng.integers(20, 200, 3))
        draw.rectangle([x - r, y - r, x + r, y + r], fill=shade)
    green = tuple(int(c) for c in rng.integers([40, 110, 30], [70, 160, 60]))
    cx, cy = rng.uniform(0.4, 0.6) * w, rng.uniform(0.4, 0.6) * h
    rx, ry = rng.uniform(0.25, 0.4) * w, rng.uniform(0.2, 0.35) * h
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=green)
    spec = _SPOTS[label]
    if spec is not None:
        color, radius, count = spec
        for _ in range(count):
            cx = int(rng.integers(w * 0.3, w * 0.7))
            cy = int(rng.integers(h * 0.3, h * 0.7))
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
    noise = rng.normal(0, 4, (h, w, 3))
    arr = np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_raw_dataset(root: Path, per_class: int = 6, seed: int = 0) -> dict[str, list[Path]]:
    """Write a Kaggle-like tree ``root/<split>/<Folder>/*.jpg``; return paths per class."""
    written: dict[str, list[Path]] = {c: [] for c in RAW_FOLDERS}
    for ci, (label, folder) in enumerate(RAW_FOLDERS.items()):
        for i in range(per_class):
            split = "train" if i < per_class - 2 else "test"
            out = root / split / folder / f"{folder.lower()}_{i:03d}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            make_leaf(label, seed=seed * 1000 + ci * 100 + i).save(out, quality=92)
            written[label].append(out)
    return written
