"""Model analysis of the deployed model: feature importance, learning curves, residuals &
predictions, cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from ui import data, theme, viz

theme.inject()
st.markdown(
    '<div class="cg-kicker">Model analysis</div><h1 style="margin-top:0">Inside the deployed '
    f'model</h1><p class="cg-lead">{data.MODEL_NAMES[data.MAIN]} — what it looks at, how it '
    "learned, where it goes wrong, and how stable its score is across different data splits.</p>",
    unsafe_allow_html=True,
)
tab_fi, tab_lc, tab_res, tab_cv = st.tabs(
    ["Feature importance", "Learning curves", "Residuals & predictions", "Cross-validation"]
)

# ======================================================================= feature importance
with tab_fi:
    ex = data.load_json("explain", data.MAIN, "explain.json")
    rb = data.load_json("robustness", data.MAIN, "robustness.json")
    theme.note(
        "For an image model, feature importance is <b>which pixels drove the answer</b>. "
        "CoffeeGuard uses class activation maps (CAM) — proven equal to Grad-CAM for this "
        "network by a unit test — and checks that they are <b>faithful</b>: hiding the regions "
        "the map marks should hurt the prediction far more than hiding random regions."
    )
    if ex:
        dl = ex["deletion"]
        curve = pd.DataFrame(
            {
                "share hidden": dl["fractions"] * 2,
                "probability": dl["cam_order_curve"] + dl["random_order_curve"],
                "order": ["hottest regions first (CAM)"] * len(dl["fractions"])
                + ["random regions"] * len(dl["fractions"]),
            }
        )
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            st.markdown("**Deletion test** — confidence as image patches are hidden")
            st.altair_chart(
                viz.lines(
                    curve,
                    "share hidden",
                    "probability",
                    "order",
                    {"hottest regions first (CAM)": viz.BLUE, "random regions": viz.ORANGE},
                    "share of image hidden",
                    "probability of predicted class",
                ),
                width="stretch",
            )
        with c2:
            theme.stats_row(
                [
                    (
                        f"{dl['share_images_cam_faster']:.0%}",
                        "of test photos: CAM order hurts faster than random",
                    )
                ]
            )
            st.write("")
            lf = ex["leaf_focus"]
            theme.stats_row(
                [
                    (
                        f"{lf['mean']:.0%}",
                        f"of the heat lands on the leaf, which covers "
                        f"only {lf['mean_leaf_area']:.0%} of the image",
                    )
                ]
            )
        focus = pd.DataFrame([{"class": c, "leaf focus": v} for c, v in lf["per_class"].items()])
        st.markdown("**Share of heat-map mass on the leaf**, per class")
        st.altair_chart(
            viz.class_bars(focus, "leaf focus", "share of CAM mass", fmt=".0%", height=170),
            width="stretch",
        )
    st.markdown(
        "**Heat maps on test photos** — top row: confident & correct; bottom row: "
        "errors or least confident"
    )
    ctabs = st.tabs(viz.CLASSES)
    for tab, c in zip(ctabs, viz.CLASSES, strict=True):
        fig = data.figure("explain", data.MAIN, "figures", f"cam_{c.lower().replace(' ', '_')}.png")
        if fig:
            tab.image(str(fig), width=900)
    if rb:
        s = rb["shortcut"]
        st.markdown("**Shortcut test** — photos with the background or the leaf blanked out")
        theme.stats_row(
            [
                (f"{s['leaf_only']['accuracy']:.1%}", "accuracy with <b>only the leaf</b>"),
                (
                    f"{s['background_only']['accuracy']:.1%}",
                    f"accuracy with <b>only the background</b> (majority class "
                    f"{s['majority_class_rate']:.0%})",
                ),
                (
                    f"{s['background_only']['mean_confidence']:.0%}",
                    "average confidence without a leaf — low is good",
                ),
            ]
        )

# ======================================================================= learning curves
with tab_lc:
    hist = data.load_json("app", "histories.json") or {}
    models = [k for k in hist if not k.startswith("cv")]
    if not models:
        st.info("No training histories found (run `coffeeguard app-data`).")
    else:
        pick = st.selectbox("Model", models, format_func=lambda k: data.MODEL_NAMES.get(k, k))
        rows = pd.DataFrame(hist[pick]["rows"])
        rows["epoch_all"] = np.arange(1, len(rows) + 1)
        ft_start = rows.index[rows["stage"] != rows["stage"].iloc[0]]
        rule = [float(rows.loc[ft_start[0], "epoch_all"]) - 0.5] if len(ft_start) else []
        st.caption(
            "Stage 1 (left of the dashed line): only the new classifier head learns, the "
            "pretrained network is frozen. Stage 2: the last 3 stages are fine-tuned. Training "
            "stops early when validation macro-F1 stops improving; the best epoch is kept."
        )
        loss = rows.melt("epoch_all", ["train_loss", "val_loss"], "series", "loss")
        loss["series"] = loss["series"].map({"train_loss": "training", "val_loss": "validation"})
        f1 = rows.melt("epoch_all", ["val_macro_f1", "ema_val_macro_f1"], "series", "macro-F1")
        f1["series"] = f1["series"].map(
            {"val_macro_f1": "raw weights", "ema_val_macro_f1": "EMA weights"}
        )
        l1, l2 = st.columns(2, gap="large")
        with l1:
            st.markdown("**Loss**")
            st.altair_chart(
                viz.lines(
                    loss,
                    "epoch_all",
                    "loss",
                    "series",
                    {"training": viz.BLUE, "validation": viz.ORANGE},
                    "epoch (both stages)",
                    "loss",
                    rules=rule,
                ),
                width="stretch",
            )
        with l2:
            st.markdown("**Validation macro-F1**")
            st.altair_chart(
                viz.lines(
                    f1,
                    "epoch_all",
                    "macro-F1",
                    "series",
                    {"raw weights": viz.BLUE, "EMA weights": viz.ORANGE},
                    "epoch (both stages)",
                    "macro-F1",
                    rules=rule,
                ),
                width="stretch",
            )
        st.caption(
            "Training loss stays above validation loss because training uses label smoothing "
            "and heavy augmentation (harder images) — not a sign of a problem."
        )
        folds = {k: v for k, v in hist.items() if k.startswith("cv")}
        if folds:
            st.markdown("**The 5 cross-validation folds** (same recipe, different validation data)")
            cvr = []
            for k, v in sorted(folds.items()):
                r = pd.DataFrame(v["rows"])
                r["epoch_all"] = np.arange(1, len(r) + 1)
                r["fold"] = f"fold {k[2:]}"
                cvr.append(r[["epoch_all", "val_macro_f1", "fold"]])
            cvdf = pd.concat(cvr)
            colors = dict(
                zip(
                    sorted(cvdf["fold"].unique()),
                    [viz.BLUE, viz.ORANGE, viz.AQUA, viz.YELLOW, "#e87ba4"],
                    strict=False,
                )
            )
            st.altair_chart(
                viz.lines(
                    cvdf,
                    "epoch_all",
                    "val_macro_f1",
                    "fold",
                    colors,
                    "epoch (both stages)",
                    "validation macro-F1",
                ),
                width="stretch",
            )

# ======================================================================= residuals
with tab_res:
    ev = data.load_json("eval", data.MAIN, "metrics.json")
    pred = data.load_csv("app", "predictions_test.csv")
    if not ev:
        st.info("No evaluation results found (run `coffeeguard evaluate`).")
    else:
        t = ev["test"]
        theme.stats_row(
            [
                (
                    f"{t['macro_f1']:.3f}",
                    f"test macro-F1 (95% CI {t['bootstrap_95ci']['macro_f1']['lo']:.3f}"
                    f"–{t['bootstrap_95ci']['macro_f1']['hi']:.3f})",
                ),
                (f"{t['errors']}", f"errors on {t['n']} test photos"),
                (f"{t['roc_auc_ovr_macro']:.3f}", "ROC-AUC (one-vs-rest, macro)"),
                (
                    f"{t['calibration']['ece_after']:.3f}",
                    "calibration error after temperature scaling "
                    f"(was {t['calibration']['ece_before']:.3f})",
                ),
            ]
        )
        r1, r2 = st.columns([1.1, 1], gap="large")
        with r1:
            st.markdown("**Confusion matrix** (test; rows = true class)")
            st.altair_chart(viz.confusion(t["confusion_matrix"], ev["classes"]), width="stretch")
        with r2:
            pcs = pd.DataFrame(
                [
                    {
                        "class": c,
                        "precision": t["per_class"][c]["precision"],
                        "recall": t["per_class"][c]["recall"],
                        "F1": t["per_class"][c]["f1-score"],
                        "photos": int(t["per_class"][c]["support"]),
                    }
                    for c in ev["classes"]
                ]
            )
            st.markdown("**F1 per class**")
            st.altair_chart(viz.class_bars(pcs, "F1", "F1", fmt=".3f", height=170), width="stretch")
            st.dataframe(
                pcs,
                hide_index=True,
                width="stretch",
                column_config={
                    c: st.column_config.NumberColumn(format="%.3f")
                    for c in ("precision", "recall", "F1")
                },
            )

        if pred is not None:
            pred["outcome"] = np.where(pred["correct"], "correct", "wrong")
            p1, p2 = st.columns(2, gap="large")
            with p1:
                st.markdown("**How confident, when right vs. wrong**")
                bins = pd.cut(
                    pred["confidence"],
                    [0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0001],
                    labels=["0.4–0.6", "0.6–0.8", "0.8–0.9", "0.9–0.95", "0.95–0.99", "≥0.99"],
                )
                h = pred.assign(bin=bins).groupby(["bin", "outcome"], observed=False).size()
                h = h.reset_index(name="photos")
                import altair as alt

                chart = (
                    alt.Chart(h)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("bin:N", title="calibrated confidence", sort=None),
                        xOffset="outcome:N",
                        y=alt.Y("photos:Q", title="test photos", scale=alt.Scale(type="symlog")),
                        color=alt.Color(
                            "outcome:N",
                            scale=alt.Scale(
                                domain=["correct", "wrong"], range=[viz.BLUE, viz.ORANGE]
                            ),
                            legend=alt.Legend(orient="top", title=None),
                        ),
                        tooltip=["bin:N", "outcome:N", "photos:Q"],
                    )
                    .properties(height=250)
                    .configure_view(strokeWidth=0)
                )
                st.altair_chart(chart, width="stretch")
                st.caption(
                    "Log scale. Wrong answers cluster at lower confidence — which is what "
                    "lets the app answer 'uncertain' instead."
                )
            with p2:
                st.markdown("**Reliability** — does 90% confident mean 90% right?")
                edges = [0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
                rel = (
                    pred.assign(bin=pd.cut(pred["confidence"], edges))
                    .groupby("bin", observed=True)
                    .agg(
                        confidence=("confidence", "mean"),
                        accuracy=("correct", "mean"),
                        photos=("correct", "size"),
                    )
                    .reset_index(drop=True)
                )
                rel["series"] = "model"
                diag = pd.DataFrame(
                    {
                        "confidence": [0.4, 1.0],
                        "accuracy": [0.4, 1.0],
                        "series": "perfect calibration",
                    }
                )
                st.altair_chart(
                    viz.lines(
                        pd.concat([rel, diag]),
                        "confidence",
                        "accuracy",
                        "series",
                        {"model": viz.BLUE, "perfect calibration": viz.GREY},
                        "average confidence",
                        "accuracy",
                        y_domain=[0.3, 1.02],
                    ),
                    width="stretch",
                )
                st.caption(
                    "Few test photos have confidence below 0.8, so those points rest on a "
                    "handful of photos each."
                )

        fig = data.figure("eval", data.MAIN, "figures", "errors_test.png")
        if fig:
            st.markdown("**Every test error** (most confident first)")
            st.image(str(fig), width=1000)

        rb = data.load_json("robustness", data.MAIN, "robustness.json")
        if rb:
            st.markdown("**Predictions on damaged photos** — test macro-F1 by severity (0 = clean)")
            rrows = [
                {
                    "corruption": c.replace("_", " "),
                    "severity": 0,
                    "macro-F1": rb["clean"]["macro_f1"],
                }
                for c in rb["corruptions"]
            ]
            rrows += [
                {"corruption": c.replace("_", " "), "severity": int(s), "macro-F1": v["macro_f1"]}
                for c, sv in rb["corruptions"].items()
                for s, v in sv.items()
            ]
            import altair as alt

            rdf = pd.DataFrame(rrows)
            base = alt.Chart(rdf).encode(
                x=alt.X("severity:Q", title="severity", axis=alt.Axis(tickCount=6)),
                y=alt.Y("macro-F1:Q", scale=alt.Scale(domain=[0.3, 1]), title="macro-F1"),
            )
            small = (
                alt.layer(
                    base.mark_line(color=viz.BLUE, strokeWidth=2),
                    base.mark_circle(color=viz.BLUE, size=35).encode(
                        tooltip=[
                            "corruption:N",
                            "severity:Q",
                            alt.Tooltip("macro-F1:Q", format=".3f"),
                        ]
                    ),
                )
                .properties(width=200, height=120)
                .facet(facet=alt.Facet("corruption:N", title=None), columns=3)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(small, width="content")
            st.caption(
                f"Keeps {rb['relative_robustness']['severity_1_3']:.1%} of its score at severities "
                "1–3. Heavy noise, occlusion and JPEG hurt most — and push answers towards "
                "'Healthy', which is why the photo check asks for a retake on poor photos."
            )

# ======================================================================= cross-validation
with tab_cv:
    cv = data.load_json("app", "cv.json")
    theme.note(
        "<b>Cross-validation</b> checks how much the score depends on which photos happen to "
        "be in the validation set. The 2,141 training + validation photos were split into 5 "
        "folds <b>by leaf group</b> (copies of the same leaf never span two folds) and "
        "stratified by class; the deployed recipe was trained 5 times, each time validated on "
        "a different fold. The 379 test photos were never used."
    )
    if not cv:
        st.info(
            "Cross-validation results are not available yet (run the 5-fold job, then "
            "`coffeeguard app-data`)."
        )
    else:
        f = pd.DataFrame(cv["folds"])
        f["fold"] = "fold " + f["fold"].astype(str)
        test_f1 = (data.load_json("eval", data.MAIN, "metrics.json") or {}).get("test", {})
        theme.stats_row(
            [
                (
                    f"{cv['macro_f1_mean']:.3f} ± {cv['macro_f1_std']:.3f}",
                    f"validation macro-F1 over {cv['k']} folds (mean ± std)",
                ),
                (
                    f"{f['val_macro_f1'].min():.3f}–{f['val_macro_f1'].max():.3f}",
                    "range across folds",
                ),
                (
                    f"{test_f1.get('macro_f1', 0):.3f}",
                    "held-out test macro-F1 of the deployed model",
                ),
            ]
        )
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            st.markdown("**Validation macro-F1 per fold**")
            st.altair_chart(
                viz.bars(
                    f,
                    "fold",
                    "val_macro_f1",
                    "macro-F1",
                    fmt=".3f",
                    sort=list(f["fold"]),
                    height=200,
                ),
                width="stretch",
            )
        with c2:
            st.dataframe(
                f[["fold", "val_macro_f1", "val_accuracy", "n_val", "best", "minutes"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "val_macro_f1": st.column_config.NumberColumn("macro-F1", format="%.3f"),
                    "val_accuracy": st.column_config.NumberColumn("accuracy", format="%.3f"),
                    "n_val": "photos",
                    "minutes": st.column_config.NumberColumn(format="%.1f"),
                },
            )
        st.caption("Learning curves of every fold are on the 'Learning curves' tab.")
