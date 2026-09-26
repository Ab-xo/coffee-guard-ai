"""The augmented KNN reference bank and the external-evaluation photo conditions."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

from coffeeguard.ood.external import CONDITIONS, small_jpeg
from coffeeguard.ood.fit import BANK_VARIANTS, bank_variants, degrade


def _leaf(w: int = 400, h: int = 200) -> Image.Image:
    a = np.zeros((h, w, 3), np.uint8)
    a[..., 1] = np.linspace(60, 200, w, dtype=np.uint8)[None, :]
    return Image.fromarray(a)


def test_degrade_is_small_and_deterministic():
    img = _leaf()
    a, b = degrade(img, random.Random(3)), degrade(img, random.Random(3))
    assert max(a.size) <= 384 and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_bank_variants_cover_every_photo_in_every_variant():
    imgs = [_leaf(), _leaf(300, 150)]
    v = bank_variants(imgs, seed=0)
    assert tuple(v) == BANK_VARIANTS
    assert all(len(v[k]) == len(imgs) for k in BANK_VARIANTS)
    assert v["original"][0] is imgs[0]
    assert v["rot90"][0].size == (200, 400)  # turned: landscape -> portrait
    assert v["rot90+degraded"][0].height > v["rot90+degraded"][0].width
    assert max(v["degraded"][0].size) <= 384


def test_external_conditions():
    img = _leaf(2048, 1024)
    assert max(small_jpeg(img, 150).size) == 150
    assert CONDITIONS["leaf vertical"](img).size == (1024, 2048)
    out = CONDITIONS["vertical + 300 px JPEG"](img)
    assert out.size == (150, 300)
