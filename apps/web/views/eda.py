"""EDA: what the raw dataset really contained, how it was cleaned, and the confounds found."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import data, theme, viz

theme.inject()
st.markdown(
    '<div class="cg-kicker">Exploratory data analysis</div><h1 style="margin-top:0">The data, '
    'honestly</h1><p class="cg-lead">The public dataset looked like 12,000 photos. Most of '
    "them turned out to be generated copies or duplicates — and each disease was photographed "
    "in its own setup. This page shows what was found and how it shaped the model.</p>",
    unsafe_allow_html=True,
)

e = data.load_json("reports", "eda_summary.json")
if not e:
    st.info("No EDA summary found (run `coffeeguard data report`).")
    st.stop()
d = e["dedup"]

theme.stats_row(
    [
        (f"{e['files']:,}", "files in the Kaggle download"),
        (f"{d['excluded_by_name_pattern']:,}", "author-generated copies set aside"),
        (
            f"{e['status_counts']['duplicate'] + e['status_counts']['conflict']:,}",
            "exact copies and cross-label conflicts removed",
        ),
        (f"{e['status_counts']['ok']:,}", "clean, unique photos used"),
    ]
)

# ---------------------------------------------------------------- funnel
theme.section("What happened to every file", kicker="Cleaning")
status = pd.DataFrame(
    [
        {
            "step": "Generated copies (aug_*) set aside",
            "files": e["status_counts"]["excluded_pattern"],
        },
        {"step": "Clean photos kept", "files": e["status_counts"]["ok"]},
        {"step": "Byte-identical duplicates removed", "files": e["status_counts"]["duplicate"]},
        {"step": "Same photo under two labels removed", "files": e["status_counts"]["conflict"]},
    ]
)
c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    st.altair_chart(
        viz.bars(status, "step", "files", "files", sort="-x", height=190), width="stretch"
    )
with c2:
    theme.note(
        "<b>Why this matters.</b> The author's own test set overlapped their training set, and "
        "70–88% of the training folder were generated copies. Scores on the original split "
        "would be inflated, so the data was re-split from scratch with duplicate grouping: "
        f"{d['groups']:,} groups of the same leaf (incl. {d['group_size_histogram'].get('4', 0)} "
        "photos stored in 4 rotations) never cross train / validation / test."
    )

# ---------------------------------------------------------------- balance
theme.section(
    "Classes and splits",
    "Group-aware stratified split, close to 70 / 15 / 15 per "
    f"class. Imbalance ratio {e['imbalance_ratio']:.2f} : 1 — mild, so no class "
    "weights; macro-F1 is the main metric.",
    kicker="Balance",
)
bal = pd.DataFrame(
    [{"class": c, **{k: v for k, v in e["balance"][c].items()}} for c in viz.CLASSES]
)
b1, b2 = st.columns([1.2, 1], gap="large")
with b1:
    st.altair_chart(viz.class_bars(bal, "total", "clean photos", fmt=",.0f"), width="stretch")
with b2:
    st.dataframe(bal.rename(columns=str.capitalize), hide_index=True, width="stretch")

# ---------------------------------------------------------------- confound
theme.section(
    "The photo-setup confound",
    "The classes were photographed differently: resolution, compression, sharpness and "
    "background all differ by class. A model could learn <i>how</i> the photo was taken instead "
    "of the disease — so training randomly degrades photo quality and swaps backgrounds, and "
    "the shortcut is measured on the Model Analysis page.",
    kicker="Key finding",
)
res = e["resolution_by_class"]
q = e["quality_by_class"]
conf = pd.DataFrame(
    [
        {
            "class": c,
            "long side (px)": res[c]["long_side"]["median"],
            "file size (KB)": res[c]["kb"]["median"],
            "sharpness": q["sharpness"][c]["median"],
        }
        for c in viz.CLASSES
    ]
)
k1, k2, k3 = st.columns(3, gap="medium")
k1.markdown("**Original resolution** (median long side, px)")
k1.altair_chart(
    viz.class_bars(conf, "long side (px)", "px", fmt=",.0f", height=170), width="stretch"
)
k2.markdown("**File size** (median KB — compression)")
k2.altair_chart(
    viz.class_bars(conf, "file size (KB)", "KB", fmt=",.0f", height=170), width="stretch"
)
k3.markdown("**Sharpness** (median Laplacian variance)")
k3.altair_chart(
    viz.class_bars(conf, "sharpness", "sharpness", fmt=",.0f", height=170), width="stretch"
)
patterns = pd.DataFrame(e["name_pattern_by_class"]).T.reindex(columns=viz.CLASSES)
patterns.index.name = "file-name pattern (source)"
st.markdown("**Photo sources by class** (file-name pattern → camera / photographer)")
st.dataframe(patterns, width="stretch")

# ---------------------------------------------------------------- samples
theme.section("What the classes look like", "Random clean photos per class.", kicker="Samples")
tabs = st.tabs(viz.CLASSES)
for tab, c in zip(tabs, viz.CLASSES, strict=True):
    fig = data.figure("figures", "eda", f"samples_{c.lower().replace(' ', '_')}.png")
    if fig:
        tab.image(str(fig), width=900)

# ---------------------------------------------------------------- duplicates & labels
theme.section("Duplicates and suspicious labels", kicker="Quality")
t1, t2 = st.tabs(["Duplicate groups", "Possible mislabels"])
with t1:
    hist = pd.DataFrame(
        [
            {"group size": f"{k} photo" + ("s" if k != "1" else ""), "groups": v, "k": int(k)}
            for k, v in d["group_size_histogram"].items()
        ]
    ).sort_values("k")
    st.altair_chart(
        viz.bars(
            hist,
            "group size",
            "groups",
            "number of groups",
            sort=list(hist["group size"]),
            height=220,
        ),
        width="stretch",
    )
    fig = data.figure("figures", "eda", "duplicate_groups.png")
    if fig:
        st.image(str(fig), caption="Examples of photos grouped as the same leaf", width=900)
with t2:
    issues = data.load_csv("reports", "label_issues.csv")
    n = 0 if issues is None else len(issues)
    theme.note(
        f"An automatic audit (DINOv2 features + cleanlab, out-of-fold) flagged <b>{n} photos "
        f"({n / e['status_counts']['ok']:.2%})</b> whose label looks inconsistent — mostly "
        "Cercospora ↔ Leaf Rust, the most similar pair. None were removed: that needs an "
        "agronomist. Half of the final model's test errors are among these photos.",
        warn=True,
    )
    fig = data.figure("figures", "eda", "label_issues.png")
    if fig:
        st.image(str(fig), width=900)

# ---------------------------------------------------------------- preprocessing
theme.section(
    "Training augmentation",
    "Rotation, crops, flips, mild colour jitter (no hue "
    "shift — lesion colour is diagnostic), quality degradation, and background swap.",
    kicker="Preprocessing",
)
a1, a2 = st.tabs(["Standard augmentation", "With background swap"])
for tab, name in ((a1, "augmentation_preview.png"), (a2, "augmentation_preview_bgswap.png")):
    fig = data.figure("figures", name)
    if fig:
        tab.image(str(fig), width=900)
