"""Small JSON helpers that understand NumPy and Path values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_default, ensure_ascii=False), "utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))
