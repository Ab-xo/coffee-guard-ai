"""DINOv2 image embeddings for duplicate grouping, label auditing and a probe baseline.

Embeddings are stored keyed by image SHA-256 in an ``.npz`` file, so re-runs only
embed new images and rows with identical content share one vector.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from coffeeguard.utils.log import get_logger

log = get_logger(__name__)

EMBED_SIZE = 224


def load_store(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=False)
    return dict(zip(data["sha256"].tolist(), data["emb"], strict=True))


def save_store(path: Path, store: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.array(sorted(store))
    emb = np.stack([store[k] for k in keys]).astype(np.float16)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, sha256=keys, emb=emb)
    tmp.replace(path)


def extract_embeddings(
    images: pd.DataFrame,
    image_root: Path,
    store_path: Path,
    model_name: str,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """Embed every row of ``images`` (columns ``sha256``, ``image``) not yet in the store."""
    import timm
    import torch

    store = load_store(store_path)
    todo = images.drop_duplicates("sha256")
    todo = todo[~todo["sha256"].isin(store)]
    if todo.empty:
        log.info("All %d embeddings already cached in %s", len(store), store_path.name)
        return store

    torch.set_num_threads(os.cpu_count() or 4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=EMBED_SIZE)
    model.eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)

    log.info("Embedding %d images with %s on %s", len(todo), model_name, device)
    rows = list(todo.itertuples(index=False))
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), batch_size), desc="embed"):
            batch = rows[start : start + batch_size]
            arrs = []
            for r in batch:
                with Image.open(image_root / r.image) as im:
                    im = im.convert("RGB").resize(
                        (EMBED_SIZE, EMBED_SIZE), Image.Resampling.BICUBIC
                    )
                    arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
            x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
            x = ((x - mean) / std).to(device)
            feats = model(x).float().cpu().numpy()
            for r, f in zip(batch, feats, strict=True):
                store[r.sha256] = f
            if (start // batch_size) % 50 == 49:  # checkpoint periodically
                save_store(store_path, store)
    save_store(store_path, store)
    return store


def aligned_embeddings(df: pd.DataFrame, store: dict[str, np.ndarray]) -> np.ndarray | None:
    """Matrix row-aligned with ``df``; rows without an embedding get zeros."""
    if not store:
        return None
    dim = len(next(iter(store.values())))
    out = np.zeros((len(df), dim), dtype=np.float32)
    for i, sha in enumerate(df["sha256"].to_numpy()):
        vec = store.get(sha) if isinstance(sha, str) else None
        if vec is not None:
            out[i] = vec
    return out
