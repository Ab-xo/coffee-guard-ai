"""Image quality measures (NumPy + Pillow only).

Used twice with identical code: offline, to describe every dataset image (EDA, error
analysis, gate thresholds), and online, in the API's quality gate. Measures are
computed on the image resized so its long side is ``QUALITY_SIDE`` pixels, which makes
them comparable across camera resolutions.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

QUALITY_SIDE = 384

# Hue window for "green" in PIL's 0-255 HSV scale (~45°-170°), with minimum
# saturation/value so grey or black pixels are not counted.
_GREEN_HUE = (32, 120)
_MIN_SAT = 40
_MIN_VAL = 30


def to_rgb(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation and convert to 3-channel RGB."""
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def resize_long_side(img: Image.Image, side: int, upscale: bool = True) -> Image.Image:
    w, h = img.size
    scale = side / max(w, h)
    if scale >= 1 and not upscale:
        return img
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(new, Image.Resampling.BICUBIC)


def quality_metrics(img: Image.Image) -> dict[str, float]:
    """Brightness, contrast, sharpness and colour statistics of an RGB image.

    - ``brightness``: mean luma (0-255)
    - ``contrast``: RMS contrast = std of luma
    - ``sharpness``: variance of the 4-neighbour Laplacian (low = blurry)
    - ``green_frac``: fraction of saturated green pixels (proxy for leaf coverage)
    - ``dark_frac`` / ``bright_frac``: fraction of near-black / near-white pixels
    """
    small = resize_long_side(img, QUALITY_SIDE)
    luma = np.asarray(small.convert("L"), dtype=np.float32)

    lap = (
        luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4.0 * luma[1:-1, 1:-1]
    )
    hsv = np.asarray(small.convert("HSV"), dtype=np.uint8)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (hue >= _GREEN_HUE[0]) & (hue <= _GREEN_HUE[1]) & (sat >= _MIN_SAT) & (val >= _MIN_VAL)

    return {
        "brightness": float(luma.mean()),
        "contrast": float(luma.std()),
        "sharpness": float(lap.var()),
        "green_frac": float(green.mean()),
        "dark_frac": float((luma < 20).mean()),
        "bright_frac": float((luma > 240).mean()),
    }
