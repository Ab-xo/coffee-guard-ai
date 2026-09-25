"""CoffeeGuard AI - Streamlit UI (talks to the FastAPI service).

Run from the repo root:
    uv run streamlit run apps/web/streamlit_app.py
The API address comes from API_URL (default http://localhost:8000).
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="CoffeeGuard AI", page_icon="🍃", layout="centered")
st.title("CoffeeGuard AI")
st.caption("Coffee leaf disease check: Healthy, Cercospora, Leaf Rust or Phoma.")


def api_health() -> dict | None:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


health = api_health()
if health is None:
    st.error(f"The prediction service at {API_URL} is not reachable. Start the API and reload.")
    st.stop()
st.sidebar.markdown(f"**Model:** `{health.get('model_name')}`  \n**API:** {API_URL}")

source = st.radio("Image source", ["Upload a photo", "Use the camera"], horizontal=True)
image = (
    st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "webp"])
    if source == "Upload a photo"
    else st.camera_input("Take a photo of one leaf")
)

if image is not None:
    col_img, col_res = st.columns([1, 1])
    col_img.image(image, caption="Your photo", width="stretch")
    with st.spinner("Analysing..."):
        try:
            r = httpx.post(
                f"{API_URL}/predict",
                files={"file": (image.name, image.getvalue(), image.type)},
                timeout=30,
            )
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
            st.stop()
    if r.status_code != 200:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        st.error(body.get("detail", f"The service returned HTTP {r.status_code}."))
        st.stop()

    res = r.json()
    with col_res:
        st.metric(
            "Prediction", res["label"], f"{res['confidence']:.0%} confidence", delta_color="off"
        )
        probs = pd.DataFrame(
            {
                "class": list(res["probabilities"]),
                "probability": list(res["probabilities"].values()),
            }
        ).set_index("class")
        st.bar_chart(probs, horizontal=True, height=200)
        st.caption(f"Model latency: {res['latency_ms']:.0f} ms")
