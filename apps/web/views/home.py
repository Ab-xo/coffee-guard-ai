"""Home: what CoffeeGuard does, the problem it addresses, and how far to trust it."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from ui import data, theme

ASSETS = Path(__file__).resolve().parents[1] / "assets"
theme.inject()
meta = data.release_meta()
m = meta.get("metrics", {})
CLASSES = [
    ("Healthy", "hero_healthy.jpg"),
    ("Cercospora", "hero_cercospora.jpg"),
    ("Leaf Rust", "hero_leaf_rust.jpg"),
    ("Phoma", "hero_phoma.jpg"),
]

# ---------------------------------------------------------------- hero
tiles = "".join(
    f'<figure><img src="{theme.img_uri(ASSETS / f)}" alt="{name} coffee leaf">'
    f"<figcaption>{name}</figcaption></figure>"
    for name, f in CLASSES
)
st.markdown(
    f"""
<div class="cg-hero">
  <div>
    <div class="cg-kicker">AI for Ethiopian coffee</div>
    <h1>Spot coffee leaf disease from a single photo.</h1>
    <p>CoffeeGuard checks the photo, says whether the leaf looks <b>healthy</b> or shows
    <b>Cercospora</b>, <b>Leaf Rust</b> or <b>Phoma</b>, tells you how sure it is, shows which
    part of the leaf it looked at — and says so when it can't judge.</p>
    <a class="cg-cta" href="diagnose" target="_self">Diagnose a leaf →</a>
    <a class="cg-cta-ghost" href="model_analysis" target="_self">How well it works</a>
  </div>
  <div class="cg-collage">{tiles}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")
theme.stats_row(
    [
        (f"{m.get('test_macro_f1', 0):.1%}", "macro-F1 on 379 unseen test photos"),
        (f"{m.get('accepted_accuracy_test', 0):.1%}", "of direct answers are correct"),
        (f"{m.get('ood_auroc_near', 0):.3f}", "AUROC telling coffee leaves from other plants"),
        ("63 ms", "per photo on an ordinary CPU server"),
    ]
)

# ---------------------------------------------------------------- problem
variation = [
    ("var_tree.jpg", "On the tree"),
    ("var_small.jpg", "Small in the frame"),
    ("var_dark.jpg", "Poor light"),
    ("var_blur.jpg", "Out of focus"),
    ("var_mild.jpg", "Mild symptoms"),
    ("var_severe.jpg", "Severe symptoms"),
]
grid = "".join(
    f'<figure><img src="{theme.img_uri(ASSETS / f)}" alt="{c}"><figcaption>{c}</figcaption>'
    "</figure>"
    for f, c in variation
)
text_col, img_col = st.columns([1, 1.05], gap="large")
with text_col:
    theme.section(
        "The problem",
        "Coffee leaf diseases can be difficult to identify quickly from photographs, "
        "especially when images vary in <b>lighting, blur, background, framing and disease "
        "severity</b>.<br><br>Ethiopia is Africa's largest coffee producer and the home of "
        "Arabica coffee; most of it is grown by smallholder farmers, for whom an early, "
        "reliable first opinion on a sick leaf matters.<br><br>CoffeeGuard analyses a leaf "
        "photo and predicts its condition while exposing its <b>confidence</b>, an "
        "<b>explanation</b>, its <b>robustness</b> and when it <b>refuses to answer</b>.",
        kicker="Why",
    )
with img_col:
    st.markdown(
        f'<div class="cg-section"><div class="cg-grid3">{grid}</div>'
        '<p style="font-size:.85rem;margin-top:8px">Real photos from the dataset — the same '
        "diseases look very different depending on how they are photographed.</p></div>",
        unsafe_allow_html=True,
    )

theme.section("Four conditions it recognises", kicker="What")
conditions = [
    ("Healthy", "Even, glossy green leaf without spots or dead tissue."),
    (
        "Cercospora (brown eye spot)",
        "Round brown spots, often with a pale centre and a yellow halo.",
    ),
    ("Leaf Rust", "Yellow to orange powdery patches; severe cases make the tree shed its leaves."),
    ("Phoma", "Dark brown to black dead patches, often from the edge or tip."),
]
for col, (name, text), (_, img) in zip(st.columns(4), conditions, CLASSES, strict=True):
    col.markdown(theme.card(name, text, ASSETS / img), unsafe_allow_html=True)

# ---------------------------------------------------------------- how it works
theme.section(
    "How it decides",
    "Every photo passes four steps. The model only answers when it should.",
    kicker="How",
)
q = meta.get("quality_thresholds", {})
steps = [
    (
        "Photo check",
        f"Too dark, washed-out, over-exposed or blurred photos get a retake request with the "
        f"reason (e.g. brightness below {q.get('min_brightness', 41):.0f}).",
    ),
    (
        "Is it a coffee leaf?",
        "The photo is compared with the 1,764 training photos; anything too unlike them — other "
        "plants, rooms, screenshots — is turned away.",
    ),
    (
        "Diagnosis",
        f"Calibrated probabilities. A direct answer needs one clear class and at least "
        f"{meta.get('tau_conf', 0.947):.0%} confidence; otherwise the likely candidates are shown.",
    ),
    (
        "Explanation",
        "A heat map shows which part of the leaf drove the answer — checked to be faithful to "
        "the model.",
    ),
]
for col, (i, (title, text)) in zip(st.columns(4), enumerate(steps, 1), strict=True):
    col.markdown(
        f'<div class="cg-card"><div class="cg-num">{i}</div><h4>{theme.esc(title)}</h4>'
        f"<p>{text}</p></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- trust
theme.section("How far to trust it", kicker="Limits")
theme.note(
    "<b>A first opinion, not a diagnosis.</b> Trained and tested on 2,520 photos from one "
    "public dataset in which each disease was photographed in its own setup; it has "
    "<b>not yet been tested on new farm photos</b>. Very early infection, several leaves in one "
    "photo, heavy noise or compression, and diseases other than these four are outside what it "
    "has learned. Confirm any disease finding with an agronomist before treating.",
    warn=True,
)
st.caption(
    "Data: Ethiopian Coffee Leaf Disease dataset by Biniyam Yoseph (Kaggle, CC0). "
    f"Model: {meta.get('model', '')} · version {meta.get('version', '')}."
)
