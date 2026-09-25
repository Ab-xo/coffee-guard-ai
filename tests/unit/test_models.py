from __future__ import annotations

import math

import pytest
import torch

from coffeeguard.models.factory import build_model, set_trainable

pytestmark = pytest.mark.ml


@pytest.mark.parametrize("name", ["mobilenetv3_small_050", "efficientnet_b0"])
def test_new_classifier_starts_at_uniform_prediction(name):
    # timm's EfficientNet-family init scales Linear weights by 1/sqrt(num_classes); with 4
    # classes that produced logits with std ~6 and a linear probe stuck at chance.
    model = build_model(name, num_classes=4, pretrained=False).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))
    assert torch.count_nonzero(logits) == 0
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3]))
    assert loss.item() == pytest.approx(math.log(4))


def test_head_mode_trains_only_the_classifier():
    model = build_model("mobilenetv3_small_050", num_classes=4, pretrained=False)
    trainable, total = set_trainable(model, "head")
    clf = model.get_classifier()
    assert trainable == clf.weight.numel() + clf.bias.numel() < total
