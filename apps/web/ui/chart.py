"""Class-probability bar chart (Altair).

One series (probability per class) → one hue, no legend. The predicted class carries the
primary blue, the others a neutral grey; value labels sit at the bar tip and keep the
Streamlit theme's text colour (never the bar colour), so the chart works in light and dark
mode; bars ≤ 24 px thick with 4 px rounded data-ends; hover tooltip.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

BLUE = "#2a78d6"  # categorical slot 1 of the validated reference palette
MUTED = "#9e9d97"  # neutral mid-grey: reads as "not chosen" on light and dark backgrounds


def probability_chart(probs: dict[str, float], highlight: str | None):
    from ui.components import pct

    df = pd.DataFrame({"class": list(probs), "p": list(probs.values())})
    df["emphasis"] = df["class"].eq(highlight)
    df["label"] = df["p"].map(pct)
    order = df.sort_values("p", ascending=False)["class"].tolist()
    y = alt.Y("class:N", sort=order, title=None, axis=alt.Axis(labelFontSize=13, labelPadding=8))
    x = alt.X("p:Q", scale=alt.Scale(domain=[0, 1.12]), axis=None)
    bars = (
        alt.Chart(df)
        .mark_bar(size=22, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(
            x=x,
            y=y,
            color=alt.condition(alt.datum.emphasis, alt.value(BLUE), alt.value(MUTED)),
            tooltip=[
                alt.Tooltip("class:N", title="Class"),
                alt.Tooltip("p:Q", title="Probability", format=".1%"),
            ],
        )
    )
    labels = (
        alt.Chart(df).mark_text(align="left", dx=6, fontSize=13).encode(x=x, y=y, text="label:N")
    )
    return (bars + labels).properties(height=4 * 38).configure_view(strokeWidth=0)
