from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from coffeeguard.inference.cam import class_activation_map, overlay
from coffeeguard.models.factory import build_model
from coffeeguard.robustness.corruptions import CORRUPTIONS
from coffeeguard.robustness.leafmask import apply_mask, leaf_mask
from tests.fixtures.synthetic import make_leaf

pytestmark = pytest.mark.ml


def test_numpy_cam_matches_pytorch_grad_cam():
    """For a pool→linear head, CAM from features × weights equals Grad-CAM (up to scale)."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    torch.manual_seed(0)
    model = build_model("efficientnet_b0", num_classes=4, pretrained=False).eval()
    torch.nn.init.normal_(model.get_classifier().weight, std=0.05)  # zero-init by default
    x = torch.randn(1, 3, 96, 96)
    with torch.no_grad():
        fmap = model.forward_features(x)[0].numpy()
    weight = model.get_classifier().weight.detach().numpy()
    k = int(model(x).argmax())

    ours = class_activation_map(fmap, weight, k, 96)
    with GradCAM(model=model, target_layers=[model.bn2]) as gc:
        ref = gc(input_tensor=x, targets=[ClassifierOutputTarget(k)])[0]
    assert np.corrcoef(ours.ravel(), ref.ravel())[0, 1] > 0.99


def test_cam_range_and_overlay_shape():
    fmap = np.random.default_rng(0).random((8, 7, 7)).astype(np.float32)
    weight = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)
    cam = class_activation_map(fmap, weight, 2, 64)
    assert cam.shape == (64, 64) and cam.min() >= 0 and cam.max() <= 1
    assert overlay(Image.new("RGB", (100, 80), (0, 128, 0)), cam).size == (64, 64)


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_corruptions_are_deterministic_and_keep_size(name):
    img = make_leaf("Leaf Rust", seed=7, size=(160, 120))
    fn = CORRUPTIONS[name]
    a = fn(img, 3, np.random.default_rng(42))
    b = fn(img, 3, np.random.default_rng(42))
    assert a.size == img.size and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(fn(img, 5, np.random.default_rng(1))), np.asarray(img))


def test_leaf_mask_separates_leaf_from_paper():
    # green ellipse with a dark lesion on pale blue paper
    a = np.full((120, 160, 3), (170, 190, 230), np.uint8)
    yy, xx = np.mgrid[:120, :160]
    leaf = ((yy - 60) / 35) ** 2 + ((xx - 80) / 60) ** 2 <= 1
    a[leaf] = (40, 110, 40)
    a[(yy - 60) ** 2 + (xx - 80) ** 2 <= 36] = (60, 35, 20)  # lesion inside the leaf
    img = Image.fromarray(a)
    mask = leaf_mask(img)
    assert (mask == leaf).mean() > 0.97
    bg_only = np.asarray(apply_mask(img, mask, keep=False))
    assert (bg_only[leaf] == 128).all(axis=1).mean() > 0.97  # edges may be shaved


def test_severity_proxy_zero_for_clean_leaf_and_grows_with_lesions():
    from coffeeguard.evaluation.severity import lesion_fraction

    a = np.full((120, 160, 3), (225, 225, 230), np.uint8)
    yy, xx = np.mgrid[:120, :160]
    leaf = ((yy - 60) / 35) ** 2 + ((xx - 80) / 60) ** 2 <= 1
    a[leaf] = (50, 120, 45)
    clean, leaf_share = lesion_fraction(Image.fromarray(a))
    assert clean < 0.01 and 0.2 < leaf_share < 0.5
    spotted = a.copy()
    for cy, cx in ((50, 60), (65, 95), (58, 110)):  # orange rust-like pustules
        spotted[(yy - cy) ** 2 + (xx - cx) ** 2 <= 25] = (230, 140, 30)
    few, _ = lesion_fraction(Image.fromarray(spotted))
    assert few > 0.01
