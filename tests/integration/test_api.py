"""FastAPI service tests against a tiny hand-built ONNX bundle (no trained model needed)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from tests.fixtures.bundle import CLASSES, make_fake_bundle

pytest.importorskip("fastapi")
pytestmark = pytest.mark.ml


@pytest.fixture
def client(tmp_path: Path):
    from app.main import create_app
    from app.settings import Settings
    from fastapi.testclient import TestClient

    bundle = make_fake_bundle(tmp_path / "bundle")
    settings = Settings(model_bundle=bundle, max_upload_mb=0.5, max_pixels=4_000_000)
    with TestClient(create_app(settings)) as c:
        yield c


def _png(color: tuple[int, int, int], size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _jpeg(color: tuple[int, int, int], size: tuple[int, int] = (64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_name": "fake-bundle"}


@pytest.mark.parametrize(
    ("data", "expected"),
    [(_png((20, 200, 30)), "Healthy"), (_jpeg((220, 30, 20)), "Cercospora")],
)
def test_predict_returns_class_and_probabilities(client, data, expected):
    r = client.post("/predict", files={"file": ("leaf.img", data, "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["label"] == expected
    assert set(body["probabilities"]) == set(CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-5
    assert body["confidence"] == pytest.approx(body["probabilities"][expected])
    assert body["latency_ms"] >= 0


def test_rejects_non_image_even_with_image_content_type(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image at all", "image/jpeg")})
    assert r.status_code == 415
    assert r.json()["error"] == "unsupported_media_type"


def test_rejects_truncated_jpeg(client):
    data = _jpeg((10, 150, 10), size=(300, 300))
    r = client.post("/predict", files={"file": ("x.jpg", data[: len(data) // 2], "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_image"


def test_rejects_empty_file(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"", "image/jpeg")})
    assert r.status_code == 400
    assert r.json()["error"] == "empty_file"


def test_rejects_oversized_file(client):
    data = _png((1, 2, 3)) + b"\0" * 600_000  # limit in this fixture is 0.5 MB
    r = client.post("/predict", files={"file": ("x.png", data, "image/png")})
    assert r.status_code == 413
    assert r.json()["error"] == "file_too_large"


def test_rejects_too_many_pixels(client):
    r = client.post(
        "/predict", files={"file": ("x.png", _png((0, 0, 0), (2500, 2000)), "image/png")}
    )
    assert r.status_code == 413
    assert r.json()["error"] == "image_too_large"


def test_missing_file_field_gives_consistent_error(client):
    r = client.post("/predict")
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"
