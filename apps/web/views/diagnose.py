"""Diagnose: photo in → decision card, heat map, calibrated probabilities out."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st
from PIL import Image
from ui import api, theme
from ui.chart import probability_chart
from ui.components import result_card
from ui.images import (
    UPLOAD_SIDE,
    blend_heatmap,
    open_oriented,
    shrink_for_upload,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
theme.inject()
st.markdown(
    '<div class="cg-kicker">Diagnose</div><h1 style="margin-top:0">Check a coffee leaf</h1>'
    '<p class="cg-lead">Photograph <b>one leaf</b> in daylight, filling most of the frame. '
    "You get the likely condition, how sure the model is, and where it looked.</p>",
    unsafe_allow_html=True,
)

if api.health() is None:
    st.error(
        f"The prediction service at {api.API_URL} is not reachable. Start the API and reload.",
        icon=":material/cloud_off:",
    )
    st.stop()


@st.cache_data(show_spinner=False)
def _thumb(path: str) -> Image.Image:
    """Square centre crop, so the gallery is an even grid."""
    with Image.open(path) as im:
        im = im.convert("RGB")
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((left, top, left + s, top + s)).resize((200, 200), Image.Resampling.LANCZOS)


@st.cache_data(show_spinner=False, max_entries=32)
def _analyze(digest: str, data: bytes) -> dict:  # digest keys the cache cheaply
    return api.analyze(data)


samples = json.loads((SAMPLES / "samples.json").read_text("utf-8"))
stems = [Path(s["file"]).stem for s in samples]
if "sample" not in st.session_state:  # ?sample=<name> opens the page on a sample
    wanted = st.query_params.get("sample")
    st.session_state.sample = wanted if wanted in stems else None

left, right = st.columns([1, 1.15], gap="large")
raw: bytes | None = None
with left:
    st.markdown("#### 1 · Photo")
    source = st.radio(
        "Photo source",
        ["Upload", "Camera", "Try a sample"],
        index=2 if st.session_state.sample else 0,
        horizontal=True,
        label_visibility="collapsed",
    )
    if source == "Upload":
        f = st.file_uploader(
            "Leaf photo (JPEG, PNG or WebP, up to 10 MB)", type=["jpg", "jpeg", "png", "webp"]
        )
        raw = f.getvalue() if f else None
    elif source == "Camera":
        shot = st.camera_input("Fill the frame with one leaf, in daylight or open shade")
        raw = shot.getvalue() if shot else None
    else:
        st.caption("Pick a sample — each shows a different behaviour.")
        with st.container(key="samples"):
            for row in range(0, len(samples), 4):
                for col, s in zip(st.columns(4), samples[row : row + 4], strict=False):
                    stem = Path(s["file"]).stem
                    col.image(_thumb(str(SAMPLES / s["file"])), width="stretch")
                    picked = st.session_state.sample == stem
                    if col.button(
                        s["short"],
                        key=f"pick_{stem}",
                        help=f"{s['title']} — shows {s['shows']}",
                        type="primary" if picked else "secondary",
                        width="stretch",
                    ):
                        st.session_state.sample = stem
                        st.rerun()
        if st.session_state.sample:
            pick = samples[stems.index(st.session_state.sample)]
            raw = (SAMPLES / pick["file"]).read_bytes()
            truth = pick.get("true_label")
            st.caption(
                f"Selected: **{pick['title']}** — shows {pick['shows']}. Source: {pick['source']}"
                + (f" · true class: {truth}" if truth else "")
            )
    if raw is not None:
        try:
            img = open_oriented(raw)
        except Exception:
            st.error("This file could not be read as an image.", icon=":material/broken_image:")
            st.stop()
        # wide photos fill the column; tall phone photos get a capped height instead
        tall = img.height > img.width
        st.image(
            img,
            caption="Your photo",
            width=int(380 * img.width / img.height) if tall else "stretch",
        )

with right:
    st.markdown("#### 2 · Result")
    if raw is None:
        st.markdown(
            '<div class="cg-result"><span class="cg-pill ok">Ready</span>'
            '<div class="title" style="font-size:1.4rem">Add a photo to get a diagnosis</div>'
            '<ul class="cg-advice"><li>One leaf, filling most of the frame</li>'
            "<li>Daylight or open shade — no direct sun or flash glare</li>"
            "<li>Hold still and tap the leaf to focus</li>"
            "<li>Photograph the side with spots</li></ul></div>",
            unsafe_allow_html=True,
        )
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

    st.markdown(result_card(res), unsafe_allow_html=True)

    if res.get("cam_png_base64"):
        st.markdown("#### 3 · Where the model looked")
        slot = st.empty()  # image first, slider underneath
        strength = st.slider("Heat-map strength", 0.0, 1.0, 1.0, 0.05)
        slot.image(blend_heatmap(img, res["cam_png_base64"], strength), width="stretch")
        st.caption(
            "Warm colours mark the regions that drove the prediction. Slide to 0 to "
            "compare with the plain photo."
        )

    if res.get("probabilities") and res["status"] != "rejected":  # no numbers we refused
        st.markdown("#### 4 · Probability per class")
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
