"""Diagnose page: photo in → decision, evidence (heat map) and probabilities out."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st
from ui import api
from ui.chart import probability_chart
from ui.components import status_banner
from ui.images import (
    UPLOAD_SIDE,
    blend_heatmap,
    model_view,
    open_oriented,
    shrink_for_upload,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"

st.title("CoffeeGuard")
st.caption("Photograph one coffee leaf — get the likely disease, how sure the model is, and why.")

if api.health() is None:
    st.error(
        f"The prediction service at {api.API_URL} is not reachable. Start the API and reload.",
        icon=":material/cloud_off:",
    )
    st.stop()


@st.cache_data(show_spinner=False, max_entries=32)
def _analyze(digest: str, data: bytes) -> dict:  # digest keys the cache cheaply
    return api.analyze(data)


# ---------------------------------------------------------------- input
samples = json.loads((SAMPLES / "samples.json").read_text("utf-8"))
# ?sample=<file stem> opens the page on a sample (shareable demo links)
wanted = st.query_params.get("sample")
sample_idx = next((i for i, s in enumerate(samples) if Path(s["file"]).stem == wanted), None)
source = st.radio(
    "Photo",
    ["Upload", "Camera", "Try a sample"],
    index=2 if sample_idx is not None else 0,
    horizontal=True,
    label_visibility="collapsed",
)
raw: bytes | None = None
truth: str | None = None
if source == "Upload":
    f = st.file_uploader(
        "Leaf photo (JPEG, PNG or WebP, up to 10 MB)", type=["jpg", "jpeg", "png", "webp"]
    )
    raw = f.getvalue() if f else None
elif source == "Camera":
    shot = st.camera_input("Fill the frame with one leaf, in daylight or open shade")
    raw = shot.getvalue() if shot else None
else:
    pick = st.selectbox(
        "Sample photo",
        samples,
        index=sample_idx or 0,
        format_func=lambda s: f"{s['title']} — shows {s['shows']}",
    )
    raw = (SAMPLES / pick["file"]).read_bytes()
    truth = pick.get("true_label")
    st.caption(f"Source: {pick['source']}" + (f" · true class: {truth}" if truth else ""))

if raw is None:
    with st.expander("Tips for a good photo", icon=":material/lightbulb:"):
        st.markdown(
            "- One leaf, filling most of the frame\n"
            "- Daylight or open shade — no direct sun, no flash glare\n"
            "- Hold still and tap the leaf to focus\n"
            "- Both sides of the leaf can matter: photograph the side with spots"
        )
    st.stop()

# ---------------------------------------------------------------- analysis
try:
    img = open_oriented(raw)
except Exception:
    st.error("This file could not be read as an image.", icon=":material/broken_image:")
    st.stop()
# The model needs ≤ 384 px: shrink big photos (fast uploads), send small ones untouched
# (re-compressing would change the pixels the model sees).
small = shrink_for_upload(img) if max(img.size) > UPLOAD_SIDE else raw
with st.spinner("Analysing…"):
    try:
        res = _analyze(hashlib.sha256(small).hexdigest(), small)
    except api.ApiUnavailable as exc:
        st.error(f"The request failed: {exc}", icon=":material/cloud_off:")
        st.stop()
if "error" in res:
    st.error(res.get("detail", res["error"]), icon=":material/error:")
    st.stop()

status_banner(res)

left, right = st.columns(2, gap="medium")
size = 224
with left:
    st.markdown("**Your photo**")
    st.image(model_view(img, size), width="stretch")
with right:
    if res.get("cam_png_base64"):
        st.markdown("**Where the model looked**")
        slot = st.empty()  # image first, slider underneath, so both photos line up
        strength = st.slider("Heat-map strength", 0.0, 1.0, 1.0, 0.05)
        slot.image(blend_heatmap(img, res["cam_png_base64"], strength), width="stretch")
        st.caption("Warm colours mark the regions that drove the prediction.")
    else:
        st.markdown("**No explanation**")
        st.caption("The photo was not analysed further, so there is nothing to explain.")

if res.get("probabilities") and res["status"] != "rejected":  # no numbers we refused to give
    st.markdown("**Probability per class** (calibrated)")
    st.altair_chart(probability_chart(res["probabilities"], res.get("label")), width="stretch")

with st.expander("Details", icon=":material/info:"):
    q = res.get("quality", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Brightness", f"{q.get('brightness', 0):.0f}")
    c2.metric("Contrast", f"{q.get('contrast', 0):.0f}")
    c3.metric("Sharpness", f"{q.get('sharpness', 0):.0f}")
    if res.get("ood_score") is not None:
        st.write(
            f"Similarity check: score {res['ood_score']:.3f} "
            f"(photos above {res['ood_threshold']:.3f} are turned away)"
        )
    st.caption(
        f"Model v{res['model_version']} · {res['latency_ms']:.0f} ms on the server · "
        f"sent {len(small) / 1024:.0f} KB (original {len(raw) / 1024:.0f} KB) · "
        f"request {res.get('_request_id', '—')}"
    )
