"""Colour-based leaf segmentation for the shortcut test and the CAM leaf-focus score.

Backgrounds in this dataset are mostly white or blue paper (low saturation or blue hue)
and, for field photos, soil and foliage. A pixel counts as plant when it is saturated
and not blue; the largest connected component (holes filled, so lesions stay inside)
is the leaf. On field photos surrounding foliage can join the mask - a known limit,
reported with the results.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def leaf_mask(img: Image.Image) -> np.ndarray:
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)  # H in [0, 180)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # saturated non-blue pixels (green/yellow/brown leaf and lesions) + dark pixels (the
    # very dark Cercospora leaves are barely saturated; paper backgrounds are bright)
    plant = ((s > 45) & (v > 25) & ((h < 95) | (h > 165))) | (v < 90)
    plant = plant.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    plant = cv2.morphologyEx(plant, cv2.MORPH_CLOSE, k, iterations=2)
    plant = cv2.morphologyEx(plant, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(plant, connectivity=8)
    if n <= 1:
        return np.zeros(plant.shape, bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == biggest).astype(np.uint8)
    # fill holes: everything not reachable from the border through background is leaf
    flood = mask.copy()
    ff = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    for y, x in ((0, 0), (0, mask.shape[1] - 1), (mask.shape[0] - 1, 0)):
        if flood[y, x] == 0:
            cv2.floodFill(flood, ff, (x, y), 1)
    return (mask | (1 - flood)).astype(bool)


def apply_mask(img: Image.Image, mask: np.ndarray, keep: bool, fill: int = 128) -> Image.Image:
    """``keep=True`` keeps the masked region (leaf-only); False removes it (background-only)."""
    a = np.asarray(img.convert("RGB")).copy()
    region = ~mask if keep else mask
    a[region] = fill
    return Image.fromarray(a)
