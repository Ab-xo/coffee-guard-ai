"""CPU smoke test: prepare synthetic data, train a tiny model through both LP-FT stages."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from coffeeguard.config import DataConfig, SplitConfig, TrainConfig, dump_config, load_config
from coffeeguard.data.prepare import prepare
from coffeeguard.utils.io import read_json

pytestmark = [pytest.mark.slow, pytest.mark.ml]


def test_train_smoke(synthetic_raw: Path, tmp_path: Path):
    data_cfg = DataConfig(
        class_aliases={"Cerscospora": "Cercospora", "Leaf rust": "Leaf Rust"},
        raw_dir=synthetic_raw,
        processed_dir=tmp_path / "processed",
        manifest_path=tmp_path / "manifests" / "manifest.parquet",
        splits_dir=tmp_path / "splits",
        embeddings_path=tmp_path / "manifests" / "embeddings.npz",
        split=SplitConfig(n_folds=4, train=0.5, val=0.25, test=0.25),
    )
    prepare(data_cfg, stages=("scan", "group", "split"), workers=1)
    data_yaml = dump_config(data_cfg, tmp_path / "data.yaml")

    cfg = load_config(
        TrainConfig,
        "configs/train/smoke.yaml",
        [f"data_config={data_yaml.as_posix()}", f"runs_dir={(tmp_path / 'runs').as_posix()}"],
    )
    from coffeeguard.training.trainer import train

    run_dir = train(cfg)

    assert (run_dir / "best.pt").exists()
    assert (run_dir / "config.yaml").exists()
    history = (run_dir / "history.jsonl").read_text().strip().splitlines()
    assert len(history) == 3  # 1 head epoch + 2 finetune epochs
    summary = read_json(run_dir / "summary.json")
    assert 0.0 <= summary["best_val_macro_f1"] <= 1.0
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["classes"] == ["Healthy", "Cercospora", "Leaf Rust", "Phoma"]
    assert ckpt["data_fingerprint"] is not None

    from coffeeguard.training.curves import plot_curves

    assert plot_curves(run_dir).stat().st_size > 0

    # --- export to ONNX and predict through the slim runtime
    from PIL import Image

    from coffeeguard.export.onnx_export import export_bundle
    from coffeeguard.inference.predictor import Predictor

    bundle = export_bundle(run_dir / "best.pt", tmp_path / "bundle")
    meta = read_json(bundle / "bundle.json")
    assert meta["parity"]["argmax_agreement"] == 1.0
    predictor = Predictor(bundle)
    img_path = next((tmp_path / "processed").rglob("*.jpg"))
    with Image.open(img_path) as im:
        pred = predictor.predict(im)
    assert pred.label in meta["classes"]
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-5
    assert pred.feature_map.ndim == 3

    # --- Phase 4 evaluation on the same bundle (val + test, calibration, conformal, figures)
    from coffeeguard.evaluation.evaluate import evaluate_bundle

    res = evaluate_bundle(bundle, data_cfg, tmp_path / "eval")
    assert res["temperature"] > 0 and 0 <= res["conformal_qhat"] <= 1
    assert 0 <= res["test"]["macro_f1"] <= 1
    assert (tmp_path / "eval" / "bundle" / "figures" / "reliability.png").exists()
    assert read_json(bundle / "bundle.json")["temperature"] == res["temperature"]
