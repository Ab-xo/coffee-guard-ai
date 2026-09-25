"""The single, canonical eval-time preprocessing (NumPy + Pillow only).

Training-time evaluation, ONNX export checks and the API all call this function, so
the served model sees exactly the pixels it was evaluated on. Steps:

1. EXIF-orient + RGB
2. downscale so the long side is ``CACHE_SIDE`` (what the training cache stores)
3. resize to ``img_size × img_size`` (bicubic)
4. scale to [0, 1], normalise with the model's mean/std, HWC → CHW float32
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from coffeeguard.inference.quality import resize_long_side, to_rgb

CACHE_SIDE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_for_model(img: Image.Image, img_size: int) -> Image.Image:
    """Steps 1–3: returns the RGB image the model will see (useful for CAM overlays)."""
    img = resize_long_side(to_rgb(img), CACHE_SIDE, upscale=False)
    return img.resize((img_size, img_size), Image.Resampling.BICUBIC)


def normalize(
    img: Image.Image, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD
) -> np.ndarray:
    """Step 4: ``(3, H, W)`` float32."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess(
    img: Image.Image,
    img_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> np.ndarray:
    return normalize(resize_for_model(img, img_size), mean, std)
