"""Thin client for the CoffeeGuard API (the UI never loads the model itself)."""

from __future__ import annotations

import os

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


class ApiUnavailable(Exception):
    """The API could not be reached or answered with an error page."""


def health() -> dict | None:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


def model_info() -> dict:
    try:
        r = httpx.get(f"{API_URL}/model-info", timeout=5)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as exc:
        raise ApiUnavailable(str(exc)) from exc


def analyze(data: bytes, filename: str = "leaf.jpg") -> dict:
    """POST /analyze. Returns the JSON body; API errors come back as {"error", "detail"}."""
    try:
        r = httpx.post(
            f"{API_URL}/analyze", files={"file": (filename, data, "image/jpeg")}, timeout=30
        )
    except httpx.HTTPError as exc:
        raise ApiUnavailable(str(exc)) from exc
    if r.headers.get("content-type", "").startswith("application/json"):
        body = r.json()
        body["_request_id"] = r.headers.get("X-Request-ID")
        return body
    raise ApiUnavailable(f"HTTP {r.status_code}")
