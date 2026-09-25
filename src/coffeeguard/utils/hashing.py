"""Content hashing for files and datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(items: Iterable[str]) -> str:
    """Order-independent fingerprint of a collection of strings (e.g. file hashes).

    Returns the first 16 hex characters, which is plenty to tell dataset versions apart.
    """
    h = hashlib.sha256()
    for item in sorted(items):
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]
