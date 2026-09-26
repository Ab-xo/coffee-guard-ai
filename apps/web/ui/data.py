"""Read-only access to the committed evaluation artifacts (the analysis pages never call the API).

``ARTIFACTS_DIR`` overrides the location (Docker); by default it is the repo's
``artifacts/`` folder next to ``apps/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = Path(os.environ.get("ARTIFACTS_DIR", ROOT / "artifacts"))
MAIN = "cand-effv2b0-bgswap"  # evaluation name of the deployed model
RELEASE = "coffeeguard-effv2b0-v1"  # its release bundle
CLASSES = ["Healthy", "Cercospora", "Leaf Rust", "Phoma"]
MODEL_NAMES = {
    "cand-effv2b0-bgswap": "EfficientNetV2-B0 + bg swap",
    "cand-effv2b0": "EfficientNetV2-B0",
    "cand-effb0": "EfficientNet-B0",
    "cand-mnv3s": "MobileNetV3-Small",
    "cand-mnv3l": "MobileNetV3-Large",
    "coffeeguard-mnv3s-baseline": "MobileNetV3-S (Phase 2, CPU)",
}


def path(*parts: str) -> Path:
    return ARTIFACTS.joinpath(*parts)


@st.cache_data(show_spinner=False)
def load_json(*parts: str) -> dict | list | None:
    p = path(*parts)
    return json.loads(p.read_text("utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_csv(*parts: str) -> pd.DataFrame | None:
    p = path(*parts)
    return pd.read_csv(p) if p.exists() else None


def figure(*parts: str) -> Path | None:
    p = path(*parts)
    return p if p.exists() else None


def release_meta() -> dict:
    return load_json("models", RELEASE, "bundle.json") or {}
