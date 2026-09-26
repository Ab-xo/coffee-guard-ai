from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from coffeeguard.evaluation.compare import gather, markdown
from coffeeguard.export.benchmark import benchmark_bundle
from coffeeguard.utils.io import write_json
from tests.fixtures.bundle import make_fake_bundle

pytestmark = pytest.mark.ml


def test_benchmark_reports_latency_and_size(tmp_path: Path):
    bundle = make_fake_bundle(tmp_path / "fake")
    photo = tmp_path / "leaf.jpg"
    Image.new("RGB", (640, 480), (30, 140, 40)).save(photo)
    res = benchmark_bundle(bundle, photo, runs=5, warmup=1)
    assert res["params_m"] > 0 and res["onnx_mb"] > 0
    for key in ("model_only", "predict", "end_to_end"):
        assert 0 < res[key]["p50_ms"] <= res[key]["p95_ms"]


def test_compare_gathers_available_results_and_renders_table(tmp_path: Path):
    write_json(
        tmp_path / "eval" / "m1" / "metrics.json",
        {
            "model": "net",
            "val": {"macro_f1": 0.98},
            "test": {
                "macro_f1": 0.97,
                "errors": 9,
                "bootstrap_95ci": {"macro_f1": {"lo": 0.95, "hi": 0.99}},
                "calibration": {"ece_after": 0.01},
            },
        },
    )
    rows = gather(["m1", "missing"], tmp_path)
    assert rows[0]["test_macro_f1"] == 0.97 and rows[1] == {"bundle": "missing"}
    table = markdown(rows)
    assert "0.970 [0.950, 0.990]" in table and table.count("\n") == 4  # header, rule, 2 rows
