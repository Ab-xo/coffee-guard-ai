"""Background-swap augmentation against the photo-setup shortcut (Phase 5 finding).

Each class was photographed in its own setup (blue paper for Cercospora/Leaf Rust,
white paper for Healthy/Phoma), and the shortcut test showed backgrounds alone carry
class information. With probability ``p`` a training leaf is cut out with the colour
leaf mask and pasted (soft edge) onto either the background of another training photo
(its own leaf filled in with its background colour and blurred away) or a random plain
background. Field photos, where the mask is the whole frame, are left unchanged.
Uses Python's ``random`` (seeded per DataLoader worker by PyTorch); picklable.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageFilter

from coffeeguard.robustness.leafmask import leaf_mask

MIN_LEAF, MAX_LEAF = 0.02, 0.8  # outside this, the mask is unreliable (empty / field photo)

# plain backgrounds: paper white, grey, pale blue, soil brown, foliage green (RGB ranges)
_PALETTE = [
    ((200, 200, 200), (250, 250, 250)),
    ((110, 110, 110), (190, 190, 190)),
    ((150, 170, 200), (200, 215, 245)),
    ((90, 65, 40), (150, 115, 80)),
    ((40, 80, 30), (100, 150, 80)),
]


def leaf_only_background(img: Image.Image) -> Image.Image | None:
    """The photo with its leaf painted over in its median background colour, blurred."""
    mask = leaf_mask(img)
    if not MIN_LEAF <= mask.mean() <= MAX_LEAF:
        return None
    a = np.asarray(img.convert("RGB")).copy()
    a[mask] = np.median(a[~mask], axis=0).astype(np.uint8)
    return Image.fromarray(a).filter(ImageFilter.GaussianBlur(6))


def _plain_background(size: tuple[int, int]) -> Image.Image:
    lo, hi = random.choice(_PALETTE)
    color = np.array([random.randint(a, b) for a, b in zip(lo, hi, strict=True)], np.float32)
    w, h = size
    # a gentle brightness gradient + fine noise, so the background isn't perfectly flat
    grad = np.linspace(random.uniform(0.85, 1.0), random.uniform(1.0, 1.15), w, dtype=np.float32)
    a = color[None, None, :] * grad[None, :, None]
    a = a + np.random.default_rng(random.getrandbits(32)).normal(0, 5, (h, w, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))


class BackgroundSwap:
    def __init__(self, donors: Sequence[Image.Image], p: float = 0.5, plain_p: float = 0.4):
        self.donors, self.p, self.plain_p = list(donors), p, plain_p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        mask = leaf_mask(img)
        if not MIN_LEAF <= mask.mean() <= MAX_LEAF:
            return img
        if self.donors and random.random() >= self.plain_p:
            bg = random.choice(self.donors).resize(img.size, Image.Resampling.BILINEAR)
            if random.random() < 0.5:
                bg = bg.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            bg = _plain_background(img.size)
        soft = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
        return Image.composite(img.convert("RGB"), bg, soft)


def build_background_swap(
    images: Sequence[Image.Image], p: float, max_donors: int = 400, seed: int = 0
) -> BackgroundSwap:
    """Donor backgrounds from a random subset of training images (memory: ~0.3 MB each)."""
    rng = random.Random(seed)
    pool = rng.sample(list(images), min(max_donors, len(images)))
    donors = [bg for im in pool if (bg := leaf_only_background(im)) is not None]
    return BackgroundSwap(donors, p)
