"""PyTorch dataset over a split CSV (rows point at cached images)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class LeafDataset(Dataset):
    """Returns ``(image_tensor, class_id)``.

    ``preload=True`` decodes every image once into memory (as small RGB PIL images);
    worthwhile on Kaggle where RAM is plentiful and the cache is ~384 px.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        preload: bool = False,
        pre_transform: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.paths = [root / p for p in frame["image"]]
        self.labels = frame["class_id"].astype(int).tolist()
        self.transform = transform
        self.pre_transform = pre_transform  # e.g. background swap, before the torch transforms
        self._cache: list[Image.Image] | None = None
        if preload:
            self._cache = [self._load(p) for p in self.paths]

    @staticmethod
    def _load(path: Path) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("RGB")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = self._cache[i] if self._cache is not None else self._load(self.paths[i])
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        return self.transform(img), self.labels[i]


def read_split(splits_dir: Path, name: str) -> pd.DataFrame:
    path = splits_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `coffeeguard data prepare` first")
    return pd.read_csv(path)
