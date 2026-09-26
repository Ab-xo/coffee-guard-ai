"""Model construction and freezing helpers (timm backbones)."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

# Parameter-name prefixes that belong to the "head end" of timm CNNs
# (final 1x1 conv, its norm, and the classifier). Always trainable when fine-tuning.
_HEAD_PREFIXES = ("conv_head", "bn2", "norm_head", "head", "classifier", "fc")


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.2,
    drop_path_rate: float = 0.1,
) -> nn.Module:
    import timm

    kwargs = {"drop_rate": drop_rate}
    if drop_path_rate:
        kwargs["drop_path_rate"] = drop_path_rate
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)
    zero_init_classifier(model)
    return model


@torch.no_grad()
def zero_init_classifier(model: nn.Module) -> None:
    """Start the new classifier at zero (uniform prediction, loss = ln C).

    timm's EfficientNet/MobileNet init draws Linear weights from U(±1/sqrt(fan_out)) with
    fan_out = num_classes: tiny for ImageNet's 1000 classes, but ±0.5 for our 4, which
    gave initial logits with std ≈ 6 and a linear-probe stage stuck at chance. A zero head
    is the usual LP-FT start; gradients are still non-zero because features differ.
    """
    clf = model.get_classifier()
    if isinstance(clf, nn.Linear):
        nn.init.zeros_(clf.weight)
        nn.init.zeros_(clf.bias)


def model_data_config(model: nn.Module) -> dict:
    """Input size / mean / std the pretrained weights expect."""
    import timm

    return timm.data.resolve_data_config({}, model=model)


def set_trainable(
    model: nn.Module, mode: Literal["head", "last_n", "all"], last_n_stages: int = 0
) -> tuple[int, int]:
    """Freeze/unfreeze parameters. Returns (trainable, total) parameter counts.

    - ``head``: only the classifier (linear probe).
    - ``last_n``: the last ``last_n_stages`` entries of ``model.blocks`` + head end.
    - ``all``: everything.
    """
    classifier = model.get_classifier()
    head_ids = {id(p) for p in classifier.parameters()}
    for name, p in model.named_parameters():
        if mode == "all":
            p.requires_grad = True
        elif mode == "head":
            p.requires_grad = id(p) in head_ids
        else:
            p.requires_grad = id(p) in head_ids or name.startswith(_HEAD_PREFIXES)
    if mode == "last_n":
        if not hasattr(model, "blocks"):
            raise ValueError(f"{type(model).__name__} has no .blocks; use 'head' or 'all'")
        n = max(0, min(last_n_stages, len(model.blocks)))
        for block in list(model.blocks)[len(model.blocks) - n :]:
            for p in block.parameters():
                p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def freeze_frozen_batchnorm(model: nn.Module) -> None:
    """Put BatchNorm layers whose parameters are frozen into eval mode.

    Call after ``model.train()``: otherwise frozen layers would still update their
    running statistics, silently changing the "frozen" backbone.
    """
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            params = list(m.parameters(recurse=False))
            if params and not any(p.requires_grad for p in params):
                m.eval()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def head_weights(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Classifier weight ``(C, F)`` and bias ``(C,)`` (used for CAM)."""
    clf = model.get_classifier()
    return clf.weight.detach().clone(), clf.bias.detach().clone()
