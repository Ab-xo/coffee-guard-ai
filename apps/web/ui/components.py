"""Diagnosis result card: status pill (icon + words), headline, confidence meter, advice."""

from __future__ import annotations

from ui.theme import esc

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


def pct(p: float) -> str:
    """Percent without over-claiming: 0.9996 is ">99%", not "100%"."""
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p:.0%}"


def candidates(res: dict, min_p: float = 0.15) -> list[tuple[str, float]]:
    """Classes worth naming when uncertain: the conformal set, else the likely ones."""
    probs = res.get("probabilities") or {}
    names = res.get("prediction_set") or []
    if len(names) < 2:
        names = [c for c, p in sorted(probs.items(), key=lambda kv: -kv[1]) if p >= min_p]
    return [(c, probs.get(c, 0.0)) for c in names[:3]]


def _meter(p: float, warn: bool = False) -> str:
    return (
        f'<div class="cg-meter{" warn" if warn else ""}" role="meter" aria-valuenow="{p:.2f}">'
        f'<div style="width:{max(p, 0.02) * 100:.1f}%"></div></div>'
    )


def result_card(res: dict) -> str:
    status = res["status"]
    advice = "".join(f"<li>{esc(a)}</li>" for a in res.get("advice", []))
    advice = f'<ul class="cg-advice">{advice}</ul>' if advice else ""
    if status == "accepted":
        return (
            '<div class="cg-result"><span class="cg-pill ok">✓ Confident answer</span>'
            f'<div class="title">{esc(res["label"])}</div>'
            f'<div class="sub">{pct(res["confidence"])} confident (calibrated)</div>'
            f"{_meter(res['confidence'])}"
            '<ul class="cg-advice"><li>A first opinion — confirm with an agronomist before '
            "treating.</li></ul></div>"
        )
    if status == "uncertain":
        names = candidates(res)
        if len(names) > 1:
            title = " or ".join(esc(c) for c, _ in names)
            sub = " · ".join(f"{esc(c)} {pct(p)}" for c, p in names)
        else:
            title = f"Probably {esc(res['label'])}"
            sub = f"{pct(res['confidence'])} — not sure enough for a direct answer"
        return (
            '<div class="cg-result"><span class="cg-pill warn">? Not certain</span>'
            f'<div class="title">{title}</div><div class="sub">{sub}</div>'
            f"{_meter(res['confidence'], warn=True)}{advice}</div>"
        )
    title = REASON_TITLE.get(res.get("reason") or "", "No diagnosis for this photo")
    issues = ", ".join(ISSUE_WORDS.get(i, i) for i in res.get("issues", []))
    sub = f'<div class="sub">{esc(issues.capitalize())}</div>' if issues else ""
    return (
        '<div class="cg-result"><span class="cg-pill bad">⛔ No diagnosis</span>'
        f'<div class="title">{esc(title)}</div>{sub}{advice}</div>'
    )
