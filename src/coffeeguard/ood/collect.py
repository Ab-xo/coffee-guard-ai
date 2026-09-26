"""Collect the out-of-distribution (OOD) image set used to fit and test the OOD gate.

Images come from public Kaggle datasets (each downloaded once as a zip, a random sample
kept, the zip deleted) plus locally generated synthetic frames. They are split **by
source** into ``cal`` (fits the threshold) and ``test`` (reported only), so the gate is
tested on kinds of images it has never been tuned on. Stored as 384 px JPEGs (like the
training cache) in ``data/ood/<split>/<near|far>/<source>/`` (git-ignored).
"""

from __future__ import annotations

import io
import random
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from coffeeguard.inference.preprocess import CACHE_SIDE
from coffeeguard.inference.quality import resize_long_side
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)
IMAGE_EXT = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Source:
    name: str
    ref: str  # Kaggle dataset
    split: str  # cal | test
    kind: str  # near (other plant leaves) | far (not a leaf)
    license: str
    n: int = 50
    exclude: str | None = None  # regex on the file path (e.g. skip augmented copies)


SOURCES = [  # Kaggle (one zip each; only small datasets: the connection here is slow)
    Source("tomato_leaves", "kaustubhb999/tomatoleaf", "cal", "near", "CC0"),
    Source(
        "banana_leaves_lsd",
        "shifatearman/bananalsd",
        "cal",
        "near",
        "CC BY-SA 4.0",
        exclude="(?i)aug",
    ),
]


@dataclass(frozen=True)
class HFSource:
    """One file (parquet with an image column, or a zip of images) from a HF dataset."""

    name: str
    repo: str
    file: str
    split: str
    kind: str
    license: str
    n: int = 50


HF_SOURCES = [
    HFSource(
        "animals",
        "Francesco/animals-ij5d2",
        "data/validation-00000-of-00001-876de533d76d48c6.parquet",
        "cal",
        "far",
        "CC BY 4.0",
    ),
    HFSource(
        "bean_leaves",
        "AI-Lab-Makerere/beans",
        "data/validation-00000-of-00001.parquet",
        "test",
        "near",
        "MIT",
    ),
    HFSource(
        "indoor_scenes",
        "keremberke/indoor-scene-classification",
        "data/valid-mini.zip",
        "test",
        "far",
        "CC BY 4.0",
    ),
    HFSource(
        "rendered_text",
        "nateraw/rendered-sst2",
        "data/validation-00000-of-00001-83818a0b65a0bbd4.parquet",
        "test",
        "far",
        "see dataset card",
    ),
]
WALLPAPERS = Path("C:/Windows/Web")  # local landscape photos (far-OOD, cal); not redistributed


def _hf_images(src: HFSource) -> list[bytes]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(src.repo, src.file, repo_type="dataset"))
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            return [z.read(n) for n in z.namelist() if n.lower().endswith(IMAGE_EXT)]
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    col = next(c for c in table.column_names if c in ("image", "img"))
    return [cell["bytes"] for cell in table.column(col).to_pylist() if cell and cell.get("bytes")]


def _sample_save(raws: list[bytes], dst_dir: Path, n: int, seed: int) -> int:
    random.Random(seed).shuffle(raws)
    saved = 0
    for raw in raws:
        if saved >= n:
            break
        if _save(raw, dst_dir / f"{saved:03d}.jpg"):
            saved += 1
    return saved


def _save(raw: bytes, dst: Path) -> bool:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            img = resize_long_side(im.convert("RGB"), CACHE_SIDE, upscale=False)
    except Exception:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, "JPEG", quality=95)
    return True


def _download(api, src: Source, out: Path, seed: int) -> int:
    """One request per dataset (per-file downloads hit Kaggle's rate limit): fetch the
    zip, sample ``n`` images straight from it, delete the zip."""
    dst_dir = out / src.split / src.kind / src.name
    if dst_dir.exists() and len(list(dst_dir.glob("*.jpg"))) >= src.n:
        return len(list(dst_dir.glob("*.jpg")))
    zdir = out / "_zips"
    zdir.mkdir(parents=True, exist_ok=True)
    for attempt in range(8):  # Kaggle rate-limits bursts (HTTP 429): back off and retry
        try:
            api.dataset_download_files(src.ref, path=str(zdir), unzip=False, quiet=True)
            break
        except Exception as exc:
            if "429" not in str(exc) or attempt == 7:
                raise
            wait = 60 * 2**attempt
            log.warning("%s: rate-limited by Kaggle, retrying in %d s", src.name, wait)
            time.sleep(wait)
    zpath = zdir / f"{src.ref.split('/')[1]}.zip"
    saved = 0
    with zipfile.ZipFile(zpath) as z:
        names = [
            n
            for n in z.namelist()
            if n.lower().endswith(IMAGE_EXT) and not (src.exclude and re.search(src.exclude, n))
        ]
        random.Random(seed).shuffle(names)
        for n in names:
            if saved >= src.n:
                break
            if _save(z.read(n), dst_dir / f"{saved:03d}.jpg"):
                saved += 1
    zpath.unlink()
    log.info("%s (%s/%s): %d images", src.name, src.split, src.kind, saved)
    return saved


def synthetic(out: Path, n_each: int = 10, seed: int = 0) -> int:
    """Far-OOD frames a phone camera can produce: blank, noise, gradients, screenshots."""
    rng = np.random.default_rng(seed)
    d = out / "test" / "far" / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    w, h, i = 384, 288, 0
    for _ in range(n_each):
        Image.new("RGB", (w, h), tuple(int(v) for v in rng.integers(0, 256, 3))).save(
            d / f"{i:03d}.jpg"
        )
        i += 1
        Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(d / f"{i:03d}.jpg")
        i += 1
        c0, c1 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, w)[None, :, None]
        Image.fromarray((c0 * (1 - t) + c1 * t).repeat(h, 0).astype(np.uint8)).save(
            d / f"{i:03d}.jpg"
        )
        i += 1
        img = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(6, 14))):  # text lines and boxes, like a screenshot
            y, x = int(rng.integers(0, h - 12)), int(rng.integers(0, w // 3))
            if rng.random() < 0.6:
                draw.text((x, y), "Lorem ipsum 12:30 " * int(rng.integers(1, 3)), fill=(30, 30, 30))
            else:
                draw.rectangle(
                    [x, y, x + int(rng.integers(40, 200)), y + int(rng.integers(10, 60))],
                    fill=tuple(int(v) for v in rng.integers(0, 256, 3)),
                )
        img.save(d / f"{i:03d}.jpg")
        i += 1
    return i


def collect(out: Path, seed: int = 0) -> dict[str, int]:
    from coffeeguard.training.remote import _api

    api = _api()
    counts = {s.name: _download(api, s, out, seed) for s in SOURCES}
    for src in HF_SOURCES:
        dst = out / src.split / src.kind / src.name
        if not (dst.exists() and len(list(dst.glob("*.jpg"))) >= src.n):
            counts[src.name] = _sample_save(_hf_images(src), dst, src.n, seed)
            log.info("%s (%s/%s): %d images", src.name, src.split, src.kind, counts[src.name])
    if WALLPAPERS.exists():
        raws = [p.read_bytes() for p in WALLPAPERS.rglob("*") if p.suffix.lower() in IMAGE_EXT]
        counts["wallpapers"] = _sample_save(raws, out / "cal" / "far" / "wallpapers", 50, seed)
    counts["synthetic"] = synthetic(out, seed=seed)
    return counts
