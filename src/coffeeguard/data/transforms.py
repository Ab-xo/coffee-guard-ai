"""Train/eval image transforms (torchvision ``transforms.v2``).

Eval uses the canonical NumPy preprocessing from ``coffeeguard.inference.preprocess``
so offline metrics and the served model see identical pixels. Train adds mild,
disease-preserving augmentation on top of the same normalisation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from PIL import Image
from torchvision.transforms import v2

from coffeeguard.config import AugmentConfig
from coffeeguard.inference.preprocess import preprocess


def train_transform(
    img_size: int, aug: AugmentConfig, mean: Sequence[float], std: Sequence[float]
) -> Callable[[Image.Image], torch.Tensor]:
    ops: list = [v2.RandomRotation(aug.rotation, interpolation=v2.InterpolationMode.BILINEAR)]
    ops.append(
        v2.RandomResizedCrop(
            img_size,
            scale=(aug.scale_min, 1.0),
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True,
        )
    )
    if aug.hflip:
        ops.append(v2.RandomHorizontalFlip(aug.hflip))
    if aug.vflip:
        ops.append(v2.RandomVerticalFlip(aug.vflip))
    ops.append(
        v2.ColorJitter(
            brightness=aug.brightness, contrast=aug.contrast, saturation=aug.saturation, hue=0.0
        )
    )
    if aug.blur_p:
        ops.append(v2.RandomApply([v2.GaussianBlur(3, sigma=(0.1, 1.5))], p=aug.blur_p))
    ops += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ]
    return v2.Compose(ops)


class EvalTransform:
    """Picklable (Windows DataLoader workers use spawn) wrapper around ``preprocess``."""

    def __init__(self, img_size: int, mean: Sequence[float], std: Sequence[float]) -> None:
        self.img_size, self.mean, self.std = img_size, tuple(mean), tuple(std)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return torch.from_numpy(preprocess(img, self.img_size, self.mean, self.std))


def eval_transform(
    img_size: int, mean: Sequence[float], std: Sequence[float]
) -> Callable[[Image.Image], torch.Tensor]:
    return EvalTransform(img_size, mean, std)
