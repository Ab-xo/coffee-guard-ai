from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from coffeeguard.config import DataConfig, SplitConfig, dump_config, load_config


def test_repo_data_config_is_valid():
    cfg = load_config(DataConfig, "configs/data.yaml")
    assert cfg.classes == ["Healthy", "Cercospora", "Leaf Rust", "Phoma"]
    assert cfg.class_to_id["Phoma"] == 3
    assert cfg.raw_dir.is_absolute()


def test_canonical_label_handles_typo_and_case():
    cfg = DataConfig(class_aliases={"Cerscospora": "Cercospora"})
    assert cfg.canonical_label("Cerscospora") == "Cercospora"
    assert cfg.canonical_label("  leaf RUST ") == "Leaf Rust"
    assert cfg.canonical_label("maize") is None


def test_alias_to_unknown_class_is_rejected():
    with pytest.raises(ValidationError):
        DataConfig(class_aliases={"x": "NotAClass"})


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        DataConfig.model_validate({"min_sidee": 50})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train": 0.6, "val": 0.15, "test": 0.15},  # does not sum to 1
        {"train": 0.72, "val": 0.14, "test": 0.14},  # 0.14 × 20 folds is not whole
    ],
)
def test_bad_split_fractions(kwargs):
    with pytest.raises(ValidationError):
        SplitConfig(**kwargs)


def test_overrides_are_parsed_as_yaml():
    cfg = load_config(
        DataConfig,
        "configs/data.yaml",
        ["min_side=64", "split.seed=7", "aspect_warn=[0.4, 2.5]"],
    )
    assert cfg.min_side == 64
    assert cfg.split.seed == 7
    assert cfg.aspect_warn == (0.4, 2.5)


def test_dump_roundtrip(tmp_path: Path):
    cfg = load_config(DataConfig, "configs/data.yaml", ["min_side=77"])
    out = dump_config(cfg, tmp_path / "cfg.yaml")
    again = load_config(DataConfig, out)
    assert again.model_dump() == cfg.model_dump()
