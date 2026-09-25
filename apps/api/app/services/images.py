"""Turn uploaded bytes into a safe, decoded PIL image, or raise an ``ApiError``."""

from __future__ import annotations

import io
import warnings

from PIL import Image


class ApiError(Exception):
    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(detail)
        self.status_code, self.error, self.detail = status_code, error, detail


# Checked on the bytes themselves; the client's Content-Type header is not trusted.
_MAGIC = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
}


def _sniff(data: bytes) -> str | None:
    for magic, fmt in _MAGIC.items():
        if data.startswith(magic):
            return fmt
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def decode_image(data: bytes, max_bytes: int, max_pixels: int) -> Image.Image:
    if not data:
        raise ApiError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise ApiError(
            413,
            "file_too_large",
            f"File is {len(data) / 1e6:.1f} MB; limit is {max_bytes / 1e6:.0f} MB.",
        )
    if _sniff(data) is None:
        raise ApiError(415, "unsupported_media_type", "Upload a JPEG, PNG or WebP image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            if (mp := img.width * img.height / 1e6) > max_pixels / 1e6:
                raise ApiError(
                    413,
                    "image_too_large",
                    f"Image has {mp:.0f} MP; limit is {max_pixels / 1e6:.0f} MP.",
                )
            img.load()  # full decode: catches truncated files
    except ApiError:
        raise
    except Exception as exc:  # PIL raises many types for corrupt data
        raise ApiError(400, "invalid_image", f"Could not decode the image: {exc}") from exc
    return img
