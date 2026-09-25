from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.synthetic import make_raw_dataset


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> Path:
    """A small Kaggle-shaped raw dataset (4 classes × 6 images) in a temp dir."""
    root = tmp_path / "raw"
    make_raw_dataset(root)
    return root
