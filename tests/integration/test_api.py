"""FastAPI service tests against a tiny hand-built ONNX bundle (no trained model needed)."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path

import pytest
from PIL import Image

from tests.fixtures.bundle import CLASSES, make_fake_bundle, textured

pytest.importorskip("fastapi")
pytestmark = pytest.mark.ml

GREEN, RED, BLUE = (25, 180, 25), (205, 40, 25), (25, 25, 205)


@pytest.fixture
def client(tmp_path: Path):
    from app.main import create_app
    from app.settings import Settings
    from fastapi.testclient import TestClient

    bundle = make_fake_bundle(tmp_path / "bundle")
    settings = Settings(model_bundle=bundle, max_upload_mb=0.5, max_pixels=4_000_000)
    with TestClient(create_app(settings)) as c:
        yield c


def _bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def _post(client, path: str, data: bytes, name: str = "leaf.png", ctype: str = "image/png"):
    return client.post(path, files={"file": (name, data, ctype)})


# ------------------------------------------------------------------ metadata endpoints


def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "fake-bundle",
        "model_version": "0.0.1",
    }


def test_model_info_exposes_classes_and_thresholds(client):
    body = client.get("/model-info").json()
    assert body["classes"] == CLASSES and body["version"] == "0.0.1"
    assert body["thresholds"]["ood_scorer"] == "knn" and body["thresholds"]["tau_conf"] == 0.6
    assert set(body["thresholds"]["quality"]) >= {"min_brightness", "min_sharpness"}


def test_openapi_documents_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/model-info", "/predict", "/analyze"} <= set(paths)


# ------------------------------------------------------------------ decisions


@pytest.mark.parametrize(("color", "label"), [(GREEN, "Healthy"), (RED, "Cercospora")])
def test_clear_leaf_colours_are_accepted(client, color, label):
    r = _post(client, "/predict", _bytes(textured(color)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["status"], body["reason"], body["label"]) == ("accepted", None, label)
    assert body["prediction_set"] == [label] and body["confidence"] > 0.9
    assert body["model_version"] == "0.0.1" and body["latency_ms"] >= 0


def test_jpeg_upload_works_too(client):
    r = _post(client, "/predict", _bytes(textured(GREEN), "JPEG"), "leaf.jpg", "image/jpeg")
    assert r.json()["label"] == "Healthy"


def test_unknown_colour_is_rejected_as_out_of_distribution(client):
    body = _post(client, "/predict", _bytes(textured(BLUE))).json()
    assert (body["status"], body["reason"], body["label"]) == ("rejected", "ood", None)
    assert body["advice"]


def test_blank_image_is_rejected_as_low_quality(client):
    body = _post(client, "/predict", _bytes(Image.new("RGB", (96, 72), (128, 128, 128)))).json()
    assert (body["status"], body["reason"]) == ("rejected", "low_quality")
    assert "too_blurry" in body["issues"] and body["advice"]


def test_dark_image_is_rejected_as_too_dark(client):
    dark = Image.eval(textured(GREEN), lambda v: v // 12)
    body = _post(client, "/predict", _bytes(dark)).json()
    assert body["reason"] == "low_quality" and "too_dark" in body["issues"]


def test_analyze_adds_probabilities_quality_ood_and_cam(client):
    r = _post(client, "/analyze", _bytes(textured(GREEN)))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "accepted"
    assert set(body["probabilities"]) == set(CLASSES)
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-5
    assert body["ood_score"] <= body["ood_threshold"]
    assert {"brightness", "contrast", "sharpness", "green_frac"} <= set(body["quality"])
    cam = Image.open(io.BytesIO(base64.b64decode(body["cam_png_base64"])))
    assert cam.size == (32, 32) and cam.mode == "RGB"
    assert {"quality", "model", "cam"} <= set(body["timings_ms"])


def test_analyze_of_rejected_photo_has_no_cam_or_probabilities(client):
    body = _post(client, "/analyze", _bytes(Image.new("RGB", (96, 72), (128, 128, 128)))).json()
    assert body["status"] == "rejected" and body["cam_png_base64"] is None
    assert body["probabilities"] is None and body["quality"]["sharpness"] == 0


# ------------------------------------------------------------------ hardening


def test_request_id_is_generated_or_echoed(client):
    assert client.get("/health").headers["X-Request-ID"]
    r = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"


def test_requests_are_logged_as_json(client, caplog):
    with caplog.at_level(logging.INFO, logger="coffeeguard.api"):
        _post(client, "/predict", _bytes(textured(GREEN)))
    rec = [r for r in caplog.records if r.getMessage() == "request"][-1]
    assert rec.fields["path"] == "/predict" and rec.fields["status"] == "accepted"
    from app.main import JsonFormatter

    line = json.loads(JsonFormatter().format(rec))
    assert line["status_code"] == 200 and line["label"] == "Healthy" and line["request_id"]


def test_rejects_non_image_even_with_image_content_type(client):
    r = _post(client, "/predict", b"not an image at all", "x.jpg", "image/jpeg")
    assert r.status_code == 415
    assert r.json()["error"] == "unsupported_media_type"


def test_rejects_truncated_jpeg(client):
    data = _bytes(textured(GREEN, (300, 300)), "JPEG")
    r = _post(client, "/predict", data[: len(data) // 2], "x.jpg", "image/jpeg")
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_image"


def test_rejects_empty_file(client):
    r = _post(client, "/predict", b"", "x.jpg", "image/jpeg")
    assert r.status_code == 400
    assert r.json()["error"] == "empty_file"


def test_rejects_oversized_file(client):
    data = _bytes(textured(GREEN)) + b"\0" * 600_000  # limit in this fixture is 0.5 MB
    r = _post(client, "/analyze", data)
    assert r.status_code == 413
    assert r.json()["error"] == "file_too_large"


def test_rejects_too_many_pixels(client):
    r = _post(client, "/predict", _bytes(Image.new("RGB", (2500, 2000))))
    assert r.status_code == 413
    assert r.json()["error"] == "image_too_large"


def test_missing_file_field_gives_consistent_error(client):
    r = client.post("/predict")
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_request"


def test_exif_rotation_is_applied_before_prediction(client):
    img = textured(GREEN, (96, 48))
    exif = img.getexif()
    exif[0x0112] = 6  # "rotate 90° CW to display"
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    r = _post(client, "/analyze", buf.getvalue(), "x.jpg", "image/jpeg")
    assert r.json()["label"] == "Healthy"


def test_bundle_without_gates_fails_at_startup(tmp_path: Path):
    from app.main import create_app
    from app.settings import Settings
    from fastapi.testclient import TestClient

    bundle = make_fake_bundle(tmp_path / "bundle")
    meta = json.loads((bundle / "bundle.json").read_text())
    del meta["ood"]
    (bundle / "bundle.json").write_text(json.dumps(meta))
    with (
        pytest.raises(ValueError, match="ood fit"),
        TestClient(create_app(Settings(model_bundle=bundle))),
    ):
        pass
