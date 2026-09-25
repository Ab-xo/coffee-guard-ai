"""Streamlit UI rendered headlessly with ``AppTest`` (no browser, no real API)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "apps" / "web" / "streamlit_app.py")


def test_shows_clear_error_when_api_is_down(monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("not reachable" in e.value for e in at.error)
    assert not at.radio  # nothing else is rendered


def test_shows_model_and_input_choice_when_api_is_up(monkeypatch):
    def fake_get(url, **_):
        assert url.endswith("/health")
        body = {"status": "ok", "model_loaded": True, "model_name": "fake-bundle"}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and not at.error
    assert "fake-bundle" in at.sidebar.markdown[0].value
    assert at.radio[0].options == ["Upload a photo", "Use the camera"]
