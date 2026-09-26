"""Typed configuration.

Configs are YAML files validated by pydantic models, so a typo or wrong type fails
immediately instead of silently producing a different experiment. Relative paths are
resolved against the project root. Values can be overridden from the CLI with
``--set key.sub=value`` (the value is parsed as YAML, so numbers and lists work).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from coffeeguard.utils.paths import portable_path, resolve

T = TypeVar("T", bound=BaseModel)


class StrictModel(BaseModel):
    """Base model that rejects unknown keys (catches typos in YAML)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- data


class SplitConfig(StrictModel):
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    n_folds: int = 20
    seed: int = 42

    @model_validator(mode="after")
    def _fractions(self) -> SplitConfig:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")
        for name in ("val", "test"):
            folds = getattr(self, name) * self.n_folds
            if abs(folds - round(folds)) > 1e-6:
                raise ValueError(f"{name} fraction × n_folds must be a whole number of folds")
        return self


class Exclusion(StrictModel):
    sha256: str
    reason: str


_PATH_FIELDS = (
    "raw_dir",
    "processed_dir",
    "manifest_path",
    "splits_dir",
    "embeddings_path",
    "reports_dir",
    "figures_dir",
)


class DataConfig(StrictModel):
    # Canonical class order defines class_id (0..3). Never reorder after training.
    classes: list[str] = ["Healthy", "Cercospora", "Leaf Rust", "Phoma"]
    # Raw folder name (case-insensitive) -> canonical class.
    class_aliases: dict[str, str] = Field(default_factory=dict)

    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    manifest_path: Path = Path("data/manifests/manifest.parquet")
    splits_dir: Path = Path("data/splits")
    embeddings_path: Path = Path("data/manifests/embeddings.npz")
    reports_dir: Path = Path("artifacts/reports")
    figures_dir: Path = Path("artifacts/figures")

    kaggle_dataset: str = "biniyamyoseph/ethiopian-coffee-leaf-disease"

    allowed_formats: list[str] = ["JPEG", "PNG"]
    min_side: int = 100
    aspect_warn: tuple[float, float] = (0.5, 2.0)

    cache_long_side: int = 384
    cache_quality: int = 95

    phash_max_distance: int = 5
    embed_similarity: float = 0.95
    embed_model: str = "vit_small_patch14_dinov2.lvd142m"

    split: SplitConfig = SplitConfig()
    exclude: list[Exclusion] = Field(default_factory=list)
    # Regexes on the file name; matches get status "excluded_pattern" (e.g. the
    # dataset author's pre-generated augmentations, which are not real photos).
    exclude_name_patterns: list[str] = Field(default_factory=list)

    @field_validator(*_PATH_FIELDS)
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return resolve(v)

    @field_serializer(*_PATH_FIELDS)
    def _relative(self, v: Path) -> str:
        return portable_path(v)

    @model_validator(mode="after")
    def _aliases_target_known_classes(self) -> DataConfig:
        unknown = set(self.class_aliases.values()) - set(self.classes)
        if unknown:
            raise ValueError(f"class_aliases map to unknown classes: {sorted(unknown)}")
        return self

    def canonical_label(self, folder_name: str) -> str | None:
        """Map a raw folder name to a canonical class, or None if unknown."""
        key = folder_name.strip().casefold()
        for alias, cls in self.class_aliases.items():
            if alias.casefold() == key:
                return cls
        for cls in self.classes:
            if cls.casefold() == key:
                return cls
        return None

    @property
    def class_to_id(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}


# -------------------------------------------------------------------------- train


class AugmentConfig(StrictModel):
    """Training-only augmentation. No hue shift: lesion colour is diagnostic."""

    scale_min: float = 0.6  # RandomResizedCrop area range (scale_min, 1.0)
    hflip: float = 0.5
    vflip: float = 0.5
    rotation: float = 20.0  # degrees
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.1
    blur_p: float = 0.1
    # Quality equalisation: image resolution/sharpness/compression differ by class in this
    # dataset (EDA), so randomly degrade clean photos until quality stops predicting class.
    downscale_p: float = 0.5  # downscale then upscale back (loses fine detail)
    downscale_min: float = 0.35  # smallest scale factor of the long side
    jpeg_p: float = 0.5  # re-encode as JPEG
    jpeg_quality_min: int = 30  # quality drawn from [jpeg_quality_min, 95]
    # Background swap (Phase 5 finding: backgrounds carry class information): with this
    # probability paste the leaf onto another photo's background or a plain one.
    bg_swap_p: float = 0.0


class StageConfig(StrictModel):
    """One training stage. LP-FT = a 'head' stage followed by a fine-tuning stage."""

    name: str
    epochs: int
    lr: float
    trainable: Literal["head", "last_n", "all"] = "all"
    last_n_stages: int = 0  # used when trainable == "last_n"
    layer_decay: float | None = None  # layer-wise LR decay (e.g. 0.75) for "all"
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    min_lr_ratio: float = 0.01  # cosine floor as a fraction of lr
    patience: int = 5  # early stopping on val macro-F1


class TrainConfig(StrictModel):
    name: str
    model: str  # timm model name, e.g. tf_efficientnetv2_b0.in1k
    pretrained: bool = True
    img_size: int = 224
    drop_rate: float = 0.2
    drop_path_rate: float = 0.1

    data_config: Path = Path("configs/data.yaml")
    batch_size: int = 32
    num_workers: int = 4
    preload: bool = False  # decode all cached images into RAM once (Kaggle)

    label_smoothing: float = 0.1
    class_weights: bool = False  # enable only if EDA shows imbalance > 3:1
    ema_decay: float | None = 0.999
    amp: bool = True  # mixed precision (CUDA only)
    channels_last: bool = True
    grad_clip: float | None = 1.0
    seed: int = 0
    device: str = "auto"  # auto | cuda | cpu

    augment: AugmentConfig = AugmentConfig()
    stages: list[StageConfig]

    runs_dir: Path = Path("runs")
    max_train_batches: int | None = None  # smoke tests only
    max_eval_batches: int | None = None

    @field_validator("data_config", "runs_dir")
    @classmethod
    def _resolve(cls, v: Path) -> Path:
        return resolve(v)

    @field_serializer("data_config", "runs_dir")
    def _relative(self, v: Path) -> str:
        return portable_path(v)


# ------------------------------------------------------------------------ loading


def _apply_override(data: dict[str, Any], dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"override must look like key.sub=value, got {dotted!r}")
    key, raw = dotted.split("=", 1)
    parts = key.strip().split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot override {key!r}: {part!r} is not a mapping")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(
    model: type[T], path: str | Path | None = None, overrides: list[str] | None = None
) -> T:
    """Load ``path`` (YAML) into ``model``, applying ``key=value`` overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(resolve(path).read_text("utf-8"))
        data = loaded or {}
    for item in overrides or []:
        _apply_override(data, item)
    return model.model_validate(data)


def dump_config(cfg: BaseModel, path: str | Path) -> Path:
    """Write a config back to YAML (e.g. into a run directory)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump(mode="json")
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), "utf-8")
    return path
