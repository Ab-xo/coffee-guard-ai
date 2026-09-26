"""Status banner: colour + icon + words, never colour alone."""

from __future__ import annotations

import streamlit as st


def pct(p: float) -> str:
    """Percent without over-claiming: 0.9996 is ">99%", not "100%"."""
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p:.0%}"


REASON_TITLE = {
    "low_quality": "Photo too poor to judge",
    "ood": "This doesn't look like a coffee leaf",
}
ISSUE_WORDS = {
    "too_dark": "too dark",
    "too_bright": "over-exposed",
    "too_blurry": "blurry",
    "low_contrast": "washed out",
    "no_leaf": "no leaf visible",
}


def candidates(res: dict, min_p: float = 0.15) -> list[tuple[str, float]]:
    """Classes worth naming when uncertain: the conformal set, else the likely ones."""
    probs = res.get("probabilities") or {}
    names = res.get("prediction_set") or []
    if len(names) < 2:
        names = [c for c, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= min_p]
    return [(c, probs.get(c, 0.0)) for c in names[:3]]


def status_banner(res: dict) -> None:
    status = res["status"]
    if status == "accepted":
        st.success(
            f"**{res['label']}** · {pct(res['confidence'])} confident",
            icon=":material/check_circle:",
        )
    elif status == "uncertain":
        names = candidates(res)
        text = (
            " or ".join(f"**{c}** ({pct(p)})" for c, p in names)
            if len(names) > 1
            else (f"Probably **{res['label']}** ({pct(res['confidence'])}), but not sure enough")
        )
        st.warning(f"Not certain — {text}", icon=":material/help:")
    else:
        title = REASON_TITLE.get(res.get("reason") or "", "No diagnosis for this photo")
        issues = ", ".join(ISSUE_WORDS.get(i, i) for i in res.get("issues", []))
        st.error(f"**{title}**" + (f" — {issues}" if issues else ""), icon=":material/block:")
    for tip in res.get("advice", []):
        st.markdown(f"- {tip}")
