"""Deterministic image corruptions, 5 severities each (ImageNet-C style, field-motivated).

Applied to the cached 384 px image (what the model's preprocessing starts from). Each
function takes ``(img, severity 1-5, rng)`` and returns an RGB PIL image; randomness
(noise, patch positions, blur direction) comes only from ``rng`` so sweeps are repeatable.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

Corruption = Callable[[Image.Image, int, np.random.Generator], Image.Image]


def _pick(levels: tuple, severity: int):
    return levels[severity - 1]


def brightness(img, s, rng):  # darker photos: shade, dusk, under-exposure
    return ImageEnhance.Brightness(img).enhance(_pick((0.8, 0.65, 0.5, 0.35, 0.25), s))


def contrast(img, s, rng):  # haze, flat light
    return ImageEnhance.Contrast(img).enhance(_pick((0.75, 0.55, 0.4, 0.3, 0.2), s))


def gaussian_blur(img, s, rng):  # out of focus
    return img.filter(ImageFilter.GaussianBlur(_pick((1.0, 2.0, 3.0, 4.5, 6.0), s)))


def motion_blur(img, s, rng):  # hand shake
    import cv2

    length = _pick((5, 9, 13, 17, 23), s)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    angle = float(rng.uniform(0, 180))
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= max(k.sum(), 1e-6)
    return Image.fromarray(cv2.filter2D(np.asarray(img), -1, k))


def gaussian_noise(img, s, rng):  # sensor noise in low light
    a = np.asarray(img, np.float32) / 255.0
    a = a + rng.normal(0, _pick((0.04, 0.08, 0.12, 0.18, 0.26), s), a.shape)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def jpeg(img, s, rng):  # messaging-app compression
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=_pick((40, 25, 15, 10, 5), s))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def occlusion(img, s, rng):  # fingers, other leaves, glare
    a = np.asarray(img).copy()
    h, w = a.shape[:2]
    target, covered = _pick((0.05, 0.1, 0.2, 0.3, 0.4), s) * h * w, 0
    while covered < target:
        ph, pw = int(rng.uniform(0.1, 0.3) * h), int(rng.uniform(0.1, 0.3) * w)
        y, x = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
        a[y : y + ph, x : x + pw] = 128
        covered += ph * pw
    return Image.fromarray(a)


def crop_zoom(img, s, rng):  # photo taken too close / leaf partly out of frame
    w, h = img.size
    f = np.sqrt(_pick((0.8, 0.65, 0.5, 0.4, 0.3), s))  # kept area fraction -> side factor
    cw, ch = int(w * f), int(h * f)
    x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    return img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.Resampling.BICUBIC)


def rotation(img, s, rng):  # any orientation; corners filled with the border colour
    a = np.asarray(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    fill = tuple(int(v) for v in border.mean(0))
    angle = _pick((10, 20, 30, 45, 90), s) * (1 if rng.random() < 0.5 else -1)
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)


CORRUPTIONS: dict[str, Corruption] = {
    "brightness": brightness,
    "contrast": contrast,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg": jpeg,
    "occlusion": occlusion,
    "crop_zoom": crop_zoom,
    "rotation": rotation,
}
