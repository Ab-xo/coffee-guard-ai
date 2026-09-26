"""Model comparison: five models, same recipe, same test photos — and why one was deployed."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from ui import data, theme, viz

theme.inject()
st.markdown(
    '<div class="cg-kicker">Model comparison</div><h1 style="margin-top:0">Which model, and why'
    '</h1><p class="cg-lead">Five image models were trained with the same recipe on the same '
    "photos and judged on the same 379 held-out test photos — on accuracy, but also on how much "
    "they rely on the leaf, robustness, spotting non-coffee images, speed and size.</p>",
    unsafe_allow_html=True,
)

rows = data.load_json("metrics", "model_comparison.json")
if not rows:
    st.info("No comparison results found (run `coffeeguard compare`).")
    st.stop()
df = pd.DataFrame(rows)
df["name"] = df["bundle"].map(data.MODEL_NAMES)
df["ci_lo"] = df["test_ci"].map(lambda c: c[0])
df["ci_hi"] = df["test_ci"].map(lambda c: c[1])
chosen = data.MODEL_NAMES[data.MAIN]

theme.note(
    f"<b>Deployed: {chosen}.</b> The three EfficientNets tie on test accuracy (their 95% "
    "intervals overlap and paired tests find no real difference). The background-swap model "
    "was chosen because it relies most on the leaf itself rather than the photo setup — what "
    "matters for farm photos — while staying fast (≈ 15 ms per photo)."
)

theme.section(
    "Accuracy on unseen test photos", "Macro-F1 with 95% bootstrap interval. Blue = deployed."
)
st.altair_chart(
    viz.model_dots(
        df, "test_macro_f1", "ci_lo", "ci_hi", "test macro-F1", chosen, domain=[0.92, 1.0]
    ),
    width="stretch",
)

theme.section("Beyond accuracy", "Blue = deployed model; higher is better on every panel.")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("**Accuracy with only the leaf visible** (background removed)")
    st.altair_chart(
        viz.model_bars(df, "leaf_only_acc", "accuracy", chosen, domain=[0, 1]), width="stretch"
    )
    st.markdown("**Score kept on damaged photos** (blur, noise, dark, JPEG…, moderate levels)")
    st.altair_chart(
        viz.model_bars(
            df, "relative_robustness_sev1_3", "relative robustness", chosen, domain=[0, 1]
        ),
        width="stretch",
    )
with c2:
    st.markdown("**Telling coffee leaves from other plant leaves** (AUROC)")
    st.altair_chart(
        viz.model_bars(df, "ood_auroc_near", "AUROC", chosen, fmt=".3f", domain=[0, 1]),
        width="stretch",
    )
    st.markdown("**Photos answered directly** (the rest: uncertain or retake)")
    st.altair_chart(
        viz.model_bars(df, "accepted_share", "share of genuine photos", chosen, domain=[0, 1]),
        width="stretch",
    )

theme.section(
    "Speed and size",
    "Server CPU time per photo (ONNX Runtime, batch 1) vs. accuracy. "
    "All are far below the 100 ms target, so speed did not decide.",
)
sc1, sc2 = st.columns([1.4, 1], gap="large")
with sc1:
    st.altair_chart(
        viz.model_dots(
            df,
            "latency_model_p50_ms",
            None,
            None,
            "ms per photo (p50)",
            chosen,
            fmt=".1f",
            domain=[0, 25],
        ),
        width="stretch",
    )
with sc2:
    st.altair_chart(
        viz.model_bars(df, "onnx_mb", "model file (MB)", chosen, fmt=".1f"), width="stretch"
    )

theme.section("All numbers")
table = df[
    [
        "name",
        "test_macro_f1",
        "ci_lo",
        "ci_hi",
        "test_errors",
        "ece_after",
        "leaf_only_acc",
        "relative_robustness_sev1_3",
        "ood_auroc_near",
        "ood_auroc_far",
        "accepted_share",
        "accepted_accuracy",
        "params_m",
        "onnx_mb",
        "latency_model_p50_ms",
    ]
].sort_values("test_macro_f1", ascending=False)
pc = st.column_config.NumberColumn
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        "name": "Model",
        "test_macro_f1": pc("Macro-F1", format="%.3f"),
        "ci_lo": pc("CI low", format="%.3f"),
        "ci_hi": pc("CI high", format="%.3f"),
        "test_errors": pc("Errors /379"),
        "ece_after": pc("ECE", format="%.3f"),
        "leaf_only_acc": pc("Leaf-only acc", format="%.3f"),
        "relative_robustness_sev1_3": pc("Robustness", format="%.3f"),
        "ood_auroc_near": pc("AUROC near", format="%.3f"),
        "ood_auroc_far": pc("AUROC far", format="%.3f"),
        "accepted_share": pc("Answered", format="%.3f"),
        "accepted_accuracy": pc("Answer acc", format="%.3f"),
        "params_m": pc("Params (M)", format="%.2f"),
        "onnx_mb": pc("MB", format="%.1f"),
        "latency_model_p50_ms": pc("ms", format="%.1f"),
    },
)

paired = (data.load_json("metrics", "test_metrics.json") or {}).get("paired_bootstrap_vs_main", {})
if paired.get("comparisons"):
    theme.section(
        "Is the difference real?",
        "Paired bootstrap on the same test photos: the "
        "deployed model's macro-F1 minus each alternative. An interval containing 0 "
        "means no clear difference.",
    )
    pr = pd.DataFrame(
        [
            {
                "Compared with": data.MODEL_NAMES.get(k, k),
                "Difference": v["diff"],
                "95% CI low": v["lo"],
                "95% CI high": v["hi"],
                "Verdict": "no clear difference"
                if v["lo"] < 0 < v["hi"]
                else ("deployed better" if v["lo"] > 0 else "deployed worse"),
            }
            for k, v in paired["comparisons"].items()
        ]
    )
    st.dataframe(
        pr,
        hide_index=True,
        width="stretch",
        column_config={
            "Difference": pc(format="%+.3f"),
            "95% CI low": pc(format="%+.3f"),
            "95% CI high": pc(format="%+.3f"),
        },
    )

theme.section("How the training recipe was chosen", kicker="Ablation")
st.markdown(
    "- **Fine-tune the last 3 stages (A) vs. all layers (B)** of EfficientNetV2-B0, same seed: "
    "same validation accuracy (6 errors of 377 each); A was kept for its flatter validation "
    "curve. The last 3 stages hold 97% of the weights, so A is not a much smaller model.\n"
    "- **Shorter schedule** (5 probe + ≤ 15 fine-tune epochs with early stopping) lost nothing.\n"
    "- **Background swap** (leaves pasted onto other photos' backgrounds during training) raised "
    "leaf-only accuracy from 0.926 to 0.963 — the reason it was deployed.\n"
    "- Full decision record: `docs/decisions/001-deployment-model.md`."
)
