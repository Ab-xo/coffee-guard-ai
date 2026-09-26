from __future__ import annotations

import pickle
import random

import numpy as np
import pytest
from PIL import Image

from coffeeguard.config import AugmentConfig
from coffeeguard.data.transforms import (
    QualityDegrade,
    RandomRotateFill,
    train_transform,
)
from coffeeguard.inference.quality import quality_metrics
from tests.fixtures.synthetic import make_leaf

pytestmark = pytest.mark.ml


def test_quality_degrade_lowers_sharpness_and_keeps_size():
    random.seed(0)
    img = make_leaf("Phoma", seed=3, size=(384, 288))
    always = QualityDegrade(downscale_p=1.0, downscale_min=0.3, jpeg_p=1.0, jpeg_quality_min=30)
    sharp = quality_metrics(img)["sharpness"]
    degraded = [always(img) for _ in range(5)]
    assert all(d.size == img.size and d.mode == "RGB" for d in degraded)
    assert np.mean([quality_metrics(d)["sharpness"] for d in degraded]) < sharp * 0.8


def test_quality_degrade_disabled_is_identity():
    img = make_leaf("Healthy", seed=4, size=(200, 150))
    out = QualityDegrade(downscale_p=0.0, jpeg_p=0.0)(img)
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_rotation_fills_corners_with_border_colour_not_black():
    random.seed(1)
    img = Image.new("RGB", (100, 100), (200, 210, 230))  # pale "paper" background
    rot = RandomRotateFill(30)
    for _ in range(5):
        a = np.asarray(rot(img)).astype(int)
        assert a.min() > 150  # no black wedges anywhere


def test_train_transform_is_picklable_and_shapes_are_right():
    tf = train_transform(64, AugmentConfig(), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tf = pickle.loads(pickle.dumps(tf))  # Windows DataLoader workers use spawn
    x = tf(make_leaf("Leaf Rust", seed=5, size=(384, 288)))
    assert tuple(x.shape) == (3, 64, 64)


def test_background_swap_keeps_leaf_changes_background_and_skips_field_photos():
    from coffeeguard.data.bgswap import BackgroundSwap, build_background_swap

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:120, :160]
    leaf = ((yy - 60) / 35) ** 2 + ((xx - 80) / 60) ** 2 <= 1

    def photo(paper):
        a = np.empty((120, 160, 3), np.uint8)
        a[:] = paper
        a[leaf] = (40, 110, 40)
        return Image.fromarray(a)

    donor = photo((240, 240, 240))  # white paper
    swap = build_background_swap([donor], p=1.0)
    swap.plain_p = 0.0  # always use the donor background
    swap = pickle.loads(pickle.dumps(swap))  # Windows DataLoader workers use spawn
    blue = photo((170, 190, 230))
    out = np.asarray(swap(blue)).astype(int)
    inner = leaf & (((yy - 60) / 30) ** 2 + ((xx - 80) / 52) ** 2 <= 1)  # away from the soft edge
    assert (np.abs(out[inner] - (40, 110, 40)).max(1) <= 2).all()  # leaf kept
    corner = out[:10, :10].reshape(-1, 3).mean(0)
    assert abs(corner[2] - 240) < 20 and abs(corner[0] - 240) < 20  # background is now white
    field = Image.fromarray(rng.integers(30, 120, (120, 160, 3)).astype(np.uint8))
    field_arr = np.asarray(field).copy()
    field_arr[..., 1] = 140  # mostly green everywhere: mask = whole frame
    field = Image.fromarray(field_arr)
    assert np.array_equal(np.asarray(BackgroundSwap([], p=1.0)(field)), field_arr)
