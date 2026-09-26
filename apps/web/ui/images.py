"""Image handling in the UI: orientation, small uploads to the API, heat-map blending."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageOps

# The model works on 384 px (then 224 px); sending more only costs upload time.
UPLOAD_SIDE = 1024


def open_oriented(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def shrink_for_upload(img: Image.Image, side: int = UPLOAD_SIDE) -> bytes:
    """JPEG of at most ``side`` px on the long side (orientation already applied)."""
    img = img.copy()
    img.thumbnail((side, side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def model_view(img: Image.Image, size: int) -> Image.Image:
    """The square image the model sees (same squash as the server's preprocessing)."""
    return img.resize((size, size), Image.Resampling.BICUBIC)


def blend_heatmap(img: Image.Image, cam_png_base64: str, strength: float) -> Image.Image:
    """Mix the plain model view with the server's CAM overlay; 0 = photo only."""
    overlay = Image.open(io.BytesIO(base64.b64decode(cam_png_base64))).convert("RGB")
    return Image.blend(model_view(img, overlay.size[0]), overlay, max(0.0, min(1.0, strength)))
