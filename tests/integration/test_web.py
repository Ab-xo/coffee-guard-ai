"""Streamlit app rendered headlessly with ``AppTest`` (no browser; the API client is mocked).

The report pages (Home, Model Comparison, EDA, Model Analysis, About Team) read the
committed files in ``artifacts/`` and must render without errors.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

WEB = Path(__file__).resolve().parents[2] / "apps" / "web"
VIEWS = WEB / "views"
DIAGNOSE, ENTRY = str(VIEWS / "diagnose.py"), str(WEB / "streamlit_app.py")
PROBS = {"Healthy": 0.01, "Cercospora": 0.02, "Leaf Rust": 0.96, "Phoma": 0.01}


def _cam_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (200, 60, 30)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _response(status="accepted", **kw) -> dict:
    base = {
        "status": status,
        "reason": None,
        "label": "Leaf Rust",
        "confidence": 0.96,
        "prediction_set": ["Leaf Rust"],
        "issues": [],
        "advice": [],
        "model_version": "1.0.0",
        "latency_ms": 40.0,
        "probabilities": PROBS,
        "ood_score": 0.3,
        "ood_threshold": 0.55,
        "quality": {"brightness": 150, "contrast": 40, "sharpness": 300, "green_frac": 0.4},
        "cam_png_base64": _cam_b64(),
        "timings_ms": {"quality": 5, "model": 20, "cam": 10},
    }
    return base | kw


REJECTED_DARK = {
    "reason": "low_quality",
    "label": None,
    "confidence": None,
    "prediction_set": [],
    "issues": ["too_dark"],
    "advice": ["The photo is too dark. Take it in daylight or open shade."],
    "probabilities": None,
    "ood_score": None,
    "cam_png_base64": None,
}


@pytest.fixture
def api(monkeypatch):
    import streamlit as st
    from ui import api as client

    st.cache_data.clear()  # the page caches API answers per photo; tests reuse the same sample
    state = {"response": _response(), "sent": []}
    monkeypatch.setattr(client, "health", lambda: {"status": "ok", "model_loaded": True})

    def fake_analyze(data, filename="leaf.jpg"):
        state["sent"].append(data)
        return state["response"]

    monkeypatch.setattr(client, "analyze", fake_analyze)
    return state


def _html(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def _pick_sample(state, response, stem: str = "leaf_rust") -> AppTest:
    state["response"] = response
    at = AppTest.from_file(DIAGNOSE, default_timeout=30).run()
    at.radio[0].set_value("Try a sample").run()
    at.button(key=f"pick_{stem}").click().run()
    assert not at.exception, at.exception
    return at


# ---------------------------------------------------------------- report pages


@pytest.mark.parametrize("page", ["home.py", "comparison.py", "eda.py", "analysis.py", "team.py"])
def test_report_pages_render_from_committed_artifacts(page):
    at = AppTest.from_file(str(VIEWS / page), default_timeout=60).run()
    assert not at.exception, at.exception
    assert not at.error


def test_home_shows_headline_numbers_and_limits():
    at = AppTest.from_file(str(VIEWS / "home.py"), default_timeout=60).run()
    html = _html(at)
    assert "Spot coffee leaf disease" in html and "macro-F1 on 379 unseen test photos" in html
    assert "not yet been tested on new farm photos" in html


def test_analysis_page_has_the_four_sections():
    at = AppTest.from_file(str(VIEWS / "analysis.py"), default_timeout=60).run()
    labels = {t.label for t in at.tabs}
    assert {
        "Feature importance",
        "Learning curves",
        "Residuals & predictions",
        "Cross-validation",
    } <= labels


def test_entry_app_builds_navigation(api):
    at = AppTest.from_file(ENTRY, default_timeout=60).run()
    assert not at.exception, at.exception


# ---------------------------------------------------------------- diagnose


def test_api_down_shows_clear_error_and_nothing_else(monkeypatch):
    from ui import api as client

    monkeypatch.setattr(client, "health", lambda: None)
    at = AppTest.from_file(DIAGNOSE, default_timeout=30).run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)
    assert not at.radio


def test_no_photo_yet_shows_input_choice_and_tips(api):
    at = AppTest.from_file(DIAGNOSE, default_timeout=30).run()
    assert at.radio[0].options == ["Upload", "Camera", "Try a sample"]
    assert "Add a photo to get a diagnosis" in _html(at)


def test_accepted_answer_shows_card_heatmap_slider_and_chart(api):
    at = _pick_sample(api, _response())
    html = _html(at)
    assert "Confident answer" in html and "Leaf Rust" in html and "96% confident" in html
    assert at.slider and at.slider[0].value == 1.0  # heat-map strength
    assert len(at.get("vega_lite_chart")) == 1


def test_uncertain_answer_names_the_candidates(api):
    probs = {"Healthy": 0.02, "Cercospora": 0.55, "Leaf Rust": 0.41, "Phoma": 0.02}
    res = _response(
        "uncertain",
        reason="low_confidence",
        label="Cercospora",
        confidence=0.55,
        prediction_set=["Cercospora"],
        probabilities=probs,
        advice=["The model isn't sure. Retake the photo."],
    )
    html = _html(_pick_sample(api, res))
    assert "Not certain" in html and "Cercospora or Leaf Rust" in html and "Retake" in html


def test_rejected_answer_shows_reason_and_advice_without_heatmap(api):
    at = _pick_sample(api, _response("rejected", **REJECTED_DARK), stem="too_dark")
    html = _html(at)
    assert "No diagnosis" in html and "too poor" in html and "Too dark" in html
    assert "daylight" in html
    assert not at.slider and not at.get("vega_lite_chart")


def test_rejected_ood_hides_the_probability_chart(api):
    res = _response(
        "rejected",
        reason="ood",
        label=None,
        confidence=None,
        prediction_set=[],
        advice=["This doesn't look like a coffee leaf the model knows."],
        cam_png_base64=None,
    )  # the API still reports probabilities; the page must not show them
    at = _pick_sample(api, res, stem="not_coffee_bean_leaf")
    assert "coffee leaf" in _html(at) and not at.get("vega_lite_chart")


def test_small_photos_are_sent_unchanged(api):
    _pick_sample(api, _response(), stem="healthy")
    assert api["sent"][-1] == (WEB / "samples" / "healthy.jpg").read_bytes()


def test_sample_query_parameter_opens_that_sample(api):
    at = AppTest.from_file(DIAGNOSE, default_timeout=30)
    at.query_params["sample"] = "phoma"
    at.run()
    assert at.radio[0].value == "Try a sample"
    assert api["sent"][-1] == (WEB / "samples" / "phoma.jpg").read_bytes()


# ---------------------------------------------------------------- helpers


def test_shrink_for_upload_limits_size_and_applies_exif():
    from ui.images import open_oriented, shrink_for_upload

    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (1500, 3000, 3), np.uint8))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90° for display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    oriented = open_oriented(buf.getvalue())
    assert oriented.size == (1500, 3000)  # portrait after EXIF
    small = Image.open(io.BytesIO(shrink_for_upload(oriented)))
    assert max(small.size) == 1024 and small.size[0] < small.size[1]


def test_candidates_prefers_conformal_set_then_probabilities():
    from ui.components import candidates

    res = {"prediction_set": ["A", "B"], "probabilities": {"A": 0.6, "B": 0.4}}
    assert [c for c, _ in candidates(res)] == ["A", "B"]
    probs = {"A": 0.7, "B": 0.2, "C": 0.1}
    picked = [c for c, _ in candidates({"prediction_set": ["A"], "probabilities": probs})]
    assert picked == ["A", "B"]


def test_percentages_never_overclaim():
    from ui.components import pct

    assert (pct(0.9996), pct(0.994), pct(0.004), pct(0.4)) == (">99%", "99%", "<1%", "40%")


def test_probability_chart_highlights_the_predicted_class():
    from ui.chart import BLUE, probability_chart

    spec = probability_chart(PROBS, "Leaf Rust").to_dict()
    assert BLUE in str(spec) and len(spec["datasets"][next(iter(spec["datasets"]))]) == 4


def test_cross_validation_results_are_shown():
    at = AppTest.from_file(str(VIEWS / "analysis.py"), default_timeout=60).run()
    html = _html(at)
    assert "validation macro-F1 over 5 folds" in html and " ± " in html
