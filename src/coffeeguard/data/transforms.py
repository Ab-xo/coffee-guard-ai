"""Train/eval image transforms (torchvision ``transforms.v2``).

Eval uses the canonical NumPy preprocessing from ``coffeeguard.inference.preprocess``
so offline metrics and the served model see identical pixels. Train adds mild,
disease-preserving augmentation on top of the same normalisation.
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from coffeeguard.config import AugmentConfig
from coffeeguard.inference.preprocess import preprocess


class QualityDegrade:
    """Randomly lower image quality: downscale→upscale and/or JPEG re-compression.

    Degradation only goes one way (a blurry photo can't be sharpened), so it pulls the
    high-quality classes towards the low-quality ones and breaks the class↔quality
    shortcut found in EDA. Uses Python's ``random``, which PyTorch seeds per worker.
    Picklable for Windows (spawn) DataLoader workers.
    """

    def __init__(
        self,
        downscale_p: float = 0.5,
        downscale_min: float = 0.35,
        jpeg_p: float = 0.5,
        jpeg_quality_min: int = 30,
    ) -> None:
        self.downscale_p, self.downscale_min = downscale_p, downscale_min
        self.jpeg_p, self.jpeg_quality_min = jpeg_p, jpeg_quality_min

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.downscale_p:
            w, h = img.size
            f = random.uniform(self.downscale_min, 1.0)
            small = img.resize(
                (max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.BILINEAR
            )
            img = small.resize((w, h), Image.Resampling.BILINEAR)
        if random.random() < self.jpeg_p:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=random.randint(self.jpeg_quality_min, 95))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return img


class RandomRotateFill:
    """Rotate by a uniform angle in ``[-degrees, degrees]``, filling the exposed corners
    with the image's mean border colour instead of black.

    Black wedges never occur at eval time and would be a train-only artefact; the border
    colour continues the background (paper, soil, foliage) instead.
    """

    def __init__(self, degrees: float) -> None:
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.degrees:
            return img
        img = img.convert("RGB")
        a = np.asarray(img)
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        fill = tuple(int(v) for v in border.mean(axis=0))
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)


def train_transform(
    img_size: int, aug: AugmentConfig, mean: Sequence[float], std: Sequence[float]
) -> Callable[[Image.Image], torch.Tensor]:
    ops: list = []
    if aug.downscale_p or aug.jpeg_p:
        ops.append(
            QualityDegrade(aug.downscale_p, aug.downscale_min, aug.jpeg_p, aug.jpeg_quality_min)
        )
    ops.append(RandomRotateFill(aug.rotation))
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
