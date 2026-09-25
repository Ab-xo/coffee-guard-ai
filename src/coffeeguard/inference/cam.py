"""Class activation maps from one ONNX forward pass (NumPy + Pillow only).

For a global-average-pool → linear head (the EfficientNet family, ``cam_exact`` in
bundle.json), CAM_c = ReLU(Σ_k W_ck · A_k) is identical up to scale to Grad-CAM at the
last conv layer, because the gradient of logit c w.r.t. A_k is W_ck / (H·W) everywhere.
So the API needs no torch and no backward pass to explain a prediction.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def class_activation_map(
    feature_map: np.ndarray, weight: np.ndarray, class_id: int, size: int
) -> np.ndarray:
    """``(C, h, w)`` features, ``(K, C)`` classifier weight → ``(size, size)`` map in [0, 1]."""
    cam = np.maximum(np.tensordot(weight[class_id], feature_map, axes=(0, 0)), 0.0)
    cam = np.asarray(
        Image.fromarray(cam.astype(np.float32), mode="F").resize(
            (size, size), Image.Resampling.BILINEAR
        )
    )
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def _turbo_like(x: np.ndarray) -> np.ndarray:
    """Blue → green → yellow → red colour ramp for values in [0, 1] (no matplotlib)."""
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a heat map over the image the model saw (same size as ``cam``)."""
    base = np.asarray(img.convert("RGB").resize(cam.shape[::-1]), dtype=np.float32)
    heat = _turbo_like(cam).astype(np.float32)
    a = alpha * cam[..., None]  # transparent where the map is cold
    return Image.fromarray((base * (1 - a) + heat * a).astype(np.uint8))
