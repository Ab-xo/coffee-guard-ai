"""Model page: what the deployed model does, how well, how it decides, and its limits."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from ui import api

ROOT = Path(__file__).resolve().parents[3]
FIGURES = {
    "Confusion matrix (test)": "artifacts/eval/cand-effv2b0-bgswap/figures/confusion_test.png",
    "Calibration": "artifacts/eval/cand-effv2b0-bgswap/figures/reliability.png",
    "Robustness to damaged photos": (
        "artifacts/robustness/cand-effv2b0-bgswap/figures/degradation.png"
    ),
}

st.title("The model")
try:
    info = api.model_info()
except api.ApiUnavailable:
    st.error(
        f"The prediction service at {api.API_URL} is not reachable.", icon=":material/cloud_off:"
    )
    st.stop()

m, t = info["metrics"], info["thresholds"]
st.caption(
    f"{info['architecture']} · version {info['version']} · classes: {', '.join(info['classes'])} · "
    f"data fingerprint {info.get('data_fingerprint')}"
)

st.subheader("How well it works")
st.caption("On 379 held-out test photos that were never used for training or tuning.")
c1, c2, c3 = st.columns(3)
lo, hi = m.get("test_macro_f1_ci95", [None, None])
c1.metric("Macro-F1", f"{m['test_macro_f1']:.3f}", help=f"95% interval {lo:.3f}–{hi:.3f}")
c2.metric("Answered directly", f"{m['accepted_share_test']:.0%}")
c3.metric("…and correct", f"{m['accepted_accuracy_test']:.1%}")
c4, c5, c6 = st.columns(3)
c4.metric("Calibration error", f"{m['test_ece_after_temperature']:.3f}", help="ECE; 0 = perfect")
c5.metric(
    "Kept under damage",
    f"{m['relative_robustness_sev1_3']:.0%}",
    help="Share of the score kept under moderate blur, noise, darkness, JPEG, …",
)
c6.metric(
    "Spots non-coffee images",
    f"{m['ood_auroc_far']:.3f}",
    help=f"AUROC; other plant leaves: {m['ood_auroc_near']:.3f}",
)

st.subheader("How it decides")
q = t["quality"]
st.markdown(
    f"1. **Photo check** — rejects photos that are too dark "
    f"(brightness < {q['min_brightness']:.0f}), "
    f"over-exposed (> {q['max_brightness']:.0f}), washed out (contrast < {q['min_contrast']:.1f}) "
    f"or blurred, and asks for a retake.\n"
    f"2. **Is it a coffee leaf?** — compares the photo with the training photos; turns it away if "
    f"it is too unlike all of them.\n"
    f"3. **Diagnosis** — calibrated probabilities; answered directly only when one class is "
    f"clearly ahead and confidence ≥ {t['tau_conf']:.0%}, otherwise *uncertain* with the likely "
    f"candidates.\n"
    f"4. **Explanation** — a heat map of the leaf regions that drove the answer."
)

st.subheader("Limits — read before relying on it")
st.markdown(
    "- Trained on **2,520 photos from one dataset**; each class was photographed in its own setup "
    "(blue or white paper, some field photos). **Not yet tested on new farm photos.**\n"
    "- **Early infection** with barely visible spots is not in the data — unknown performance.\n"
    "- **One leaf per photo.** Other leaves in the frame can mislead it.\n"
    "- Heavy **noise or compression** isn't detected and tends to produce *Healthy*.\n"
    "- Only **four classes**: other diseases, pests or deficiencies will be forced into one of "
    "them unless the similarity check turns the photo away.\n"
    "- A first opinion, **not a diagnosis** — confirm disease findings with an agronomist."
)

shown = [(title, ROOT / rel) for title, rel in FIGURES.items() if (ROOT / rel).exists()]
if shown:
    st.subheader("Evaluation figures")
    for tab, (_title, path) in zip(st.tabs([s[0] for s in shown]), shown, strict=True):
        tab.image(str(path), width="stretch")
st.caption("Full details: docs/MODEL_CARD.md and docs/decisions/001-deployment-model.md.")
