"""Altair charts for the analysis pages, following the dataviz rules.

- Classes use the reference palette's first four categorical slots in fixed order
  (validated: all checks pass; aqua/yellow are < 3:1 on the surface, so every class chart
  carries visible labels).
- One series → one hue; comparisons highlight the deployed model in slot 1, others grey.
- Thin marks, ≤ 24 px bars with 4 px rounded data-ends, 2 px lines, hover tooltips,
  recessive axes; text keeps the theme's text colour.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#9e9d97"
CLASS_COLORS = {"Healthy": BLUE, "Cercospora": ORANGE, "Leaf Rust": AQUA, "Phoma": YELLOW}
CLASSES = list(CLASS_COLORS)
YAXIS = alt.Axis(labelLimit=320, labelFontSize=12)  # full names (Streamlit's theme cuts at ~180 px)


def _finish(chart: alt.Chart, height: int) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            grid=True,
            gridColor="#e4e3df",
            gridWidth=1,
            domain=False,
            tickColor="#e4e3df",
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight="normal",
            labelLimit=320,  # full model names, never "EfficientNetV2-B0…"
        )
    )


def class_bars(df: pd.DataFrame, value: str, title: str, fmt: str = ".0f", height: int = 180):
    """Horizontal bars, one per class (class colours), value label at the tip."""
    y = alt.Y("class:N", sort=CLASSES, title=None, axis=YAXIS)
    x = alt.X(f"{value}:Q", title=title, axis=alt.Axis(grid=True, tickCount=4))
    color = alt.Color(
        "class:N", scale=alt.Scale(domain=CLASSES, range=list(CLASS_COLORS.values())), legend=None
    )
    bars = (
        alt.Chart(df)
        .mark_bar(size=20, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(x=x, y=y, color=color, tooltip=["class:N", alt.Tooltip(f"{value}:Q", format=fmt)])
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=6, fontSize=12)
        .encode(x=x, y=y, text=alt.Text(f"{value}:Q", format=fmt))
    )
    return _finish(bars + labels, height)


def model_dots(
    df: pd.DataFrame,
    value: str,
    lo: str | None,
    hi: str | None,
    title: str,
    highlight: str,
    fmt: str = ".3f",
    height: int = 220,
    domain=None,
):
    """Dot (+ optional interval) per model, deployed model highlighted, value labels."""
    y = alt.Y(
        "name:N", sort=alt.EncodingSortField(value, order="descending"), title=None, axis=YAXIS
    )
    scale = alt.Scale(domain=domain, zero=False) if domain else alt.Scale(zero=False)
    x = alt.X(f"{value}:Q", title=title, scale=scale)
    color = alt.condition(alt.datum.name == highlight, alt.value(BLUE), alt.value(GREY))
    layers = []
    if lo and hi:
        layers.append(
            alt.Chart(df)
            .mark_rule(strokeWidth=2)
            .encode(x=alt.X(f"{lo}:Q", scale=scale, title=title), x2=f"{hi}:Q", y=y, color=color)
        )
    layers.append(
        alt.Chart(df)
        .mark_circle(size=140, opacity=1, stroke="#fcfcfb", strokeWidth=2)
        .encode(
            x=x,
            y=y,
            color=color,
            tooltip=["name:N", alt.Tooltip(f"{value}:Q", format=fmt)]
            + (
                [
                    alt.Tooltip(f"{lo}:Q", format=fmt, title="95% CI low"),
                    alt.Tooltip(f"{hi}:Q", format=fmt, title="95% CI high"),
                ]
                if lo
                else []
            ),
        )
    )
    layers.append(
        alt.Chart(df)
        .mark_text(dy=-14, fontSize=12)
        .encode(x=x, y=y, text=alt.Text(f"{value}:Q", format=fmt))
    )
    return _finish(alt.layer(*layers), height)


def model_bars(
    df: pd.DataFrame,
    value: str,
    title: str,
    highlight: str,
    fmt: str = ".0%",
    height: int = 190,
    domain=None,
):
    y = alt.Y(
        "name:N", sort=alt.EncodingSortField(value, order="descending"), title=None, axis=YAXIS
    )
    x = alt.X(
        f"{value}:Q",
        title=title,
        scale=alt.Scale(domain=domain) if domain else alt.Scale(),
        axis=alt.Axis(format=fmt),
    )
    color = alt.condition(alt.datum.name == highlight, alt.value(BLUE), alt.value(GREY))
    bars = (
        alt.Chart(df)
        .mark_bar(size=18, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(x=x, y=y, color=color, tooltip=["name:N", alt.Tooltip(f"{value}:Q", format=fmt)])
    )
    labels = (
        alt.Chart(df)
        .mark_text(align="left", dx=6, fontSize=12)
        .encode(x=x, y=y, text=alt.Text(f"{value}:Q", format=fmt))
    )
    return _finish(bars + labels, height)


def lines(
    df: pd.DataFrame,
    x: str,
    y: str,
    series: str,
    colors: dict[str, str],
    x_title: str,
    y_title: str,
    height: int = 260,
    fmt: str = ".3f",
    rules: list[float] | None = None,
    y_domain=None,
):
    """Multi-series lines (2 px) with points, a legend and a nearest-point tooltip."""
    scale = alt.Color(
        f"{series}:N",
        scale=alt.Scale(domain=list(colors), range=list(colors.values())),
        legend=alt.Legend(orient="top", title=None, labelFontSize=12),
    )
    ys = alt.Scale(domain=y_domain, zero=False) if y_domain else alt.Scale(zero=False)
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:Q", title=x_title), y=alt.Y(f"{y}:Q", title=y_title, scale=ys), color=scale
    )
    layers = [
        base.mark_line(strokeWidth=2),
        base.mark_circle(size=40).encode(
            tooltip=[f"{series}:N", alt.Tooltip(f"{x}:Q"), alt.Tooltip(f"{y}:Q", format=fmt)]
        ),
    ]
    for r in rules or []:
        layers.append(
            alt.Chart(pd.DataFrame({"x": [r]}))
            .mark_rule(strokeDash=[4, 4], color=GREY)
            .encode(x="x:Q")
        )
    return _finish(alt.layer(*layers), height)


def confusion(cm: list[list[int]], classes: list[str], height: int = 300):
    rows = [
        {"true": t, "pred": p, "n": cm[i][j], "share": cm[i][j] / max(sum(cm[i]), 1)}
        for i, t in enumerate(classes)
        for j, p in enumerate(classes)
    ]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("pred:N", sort=classes, title="predicted"),
        y=alt.Y("true:N", sort=classes, title="true class"),
    )
    rect = base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("share:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), legend=None),
        tooltip=["true:N", "pred:N", "n:Q", alt.Tooltip("share:Q", format=".1%")],
    )
    text = base.mark_text(fontSize=13).encode(
        text=alt.Text("n:Q"),
        color=alt.condition(alt.datum.share > 0.5, alt.value("#ffffff"), alt.value("#0b0b0b")),
    )
    return _finish(rect + text, height).configure_axis(grid=False, labelAngle=0)


def bars(
    df: pd.DataFrame,
    cat: str,
    value: str,
    title: str,
    fmt: str = ",.0f",
    height: int = 200,
    sort: list[str] | str = "-x",
):
    """One series → one hue (slot 1): horizontal bars with the value at the tip."""
    y = alt.Y(f"{cat}:N", sort=sort, title=None, axis=YAXIS)
    x = alt.X(f"{value}:Q", title=title, axis=alt.Axis(tickCount=4))
    b = (
        alt.Chart(df)
        .mark_bar(size=20, color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
        .encode(x=x, y=y, tooltip=[f"{cat}:N", alt.Tooltip(f"{value}:Q", format=fmt)])
    )
    t = (
        alt.Chart(df)
        .mark_text(align="left", dx=6, fontSize=12)
        .encode(x=x, y=y, text=alt.Text(f"{value}:Q", format=fmt))
    )
    return _finish(b + t, height)


def before_after(df: pd.DataFrame, cat: str, before: str, after: str, height: int = 280):
    """Dumbbell per category: grey dot = before, blue dot = after (labelled), on 0-100%.

    ``before`` / ``after`` are column names shown in the legend; they may contain dots, so
    the chart uses plain internal field names (Vega-Lite reads "a.b" as a nested field).
    """
    d = pd.DataFrame({"cat": df[cat], "b": df[before], "a": df[after]})
    d["lab"] = d[["a", "b"]].max(axis=1)  # label right of both dots
    long = d.melt(id_vars=["cat"], value_vars=["b", "a"], var_name="k", value_name="v")
    long["gate"] = long["k"].map({"b": before, "a": after})
    y = alt.Y(
        "cat:N",
        sort=list(d["cat"]),
        title=None,
        axis=alt.Axis(labelLimit=320, labelFontSize=12, labelOverlap=False),
    )
    scale = alt.Scale(domain=[0, 1.12])
    axis = alt.Axis(format="%", values=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    rule = (
        alt.Chart(d)
        .mark_rule(color=GREY, strokeWidth=2)
        .encode(x=alt.X("b:Q", scale=scale, axis=axis, title="share passed"), x2="a:Q", y=y)
    )
    dots = (
        alt.Chart(long)
        .mark_circle(size=110, opacity=1, stroke="#ffffff", strokeWidth=2)
        .encode(
            x=alt.X("v:Q", scale=scale, axis=axis, title="share passed"),
            y=y,
            color=alt.Color(
                "gate:N",
                scale=alt.Scale(domain=[before, after], range=[GREY, BLUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip("cat:N", title=cat), "gate:N", alt.Tooltip("v:Q", format=".0%")],
        )
    )
    labels = (
        alt.Chart(d)
        .mark_text(align="left", dx=10, fontSize=12)
        .encode(x=alt.X("lab:Q", scale=scale), y=y, text=alt.Text("a:Q", format=".0%"))
    )
    return _finish(rule + dots + labels, height)
