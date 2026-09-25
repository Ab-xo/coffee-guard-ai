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
