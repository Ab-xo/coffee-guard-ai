"""The serving decision: quality gate → OOD gate → conformal set + confidence (NumPy only).

``accepted``  - one plausible class and confident enough;
``uncertain`` - a leaf, but the model can't commit (conformal set > 1 class or low
                confidence): the top candidates are returned with retake advice;
``rejected``  - ``low_quality`` (photo too dark / blurry / flat / bright / no leaf
                colours) or ``ood`` (doesn't look like the coffee leaves it knows).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ADVICE = {
    "too_dark": "The photo is too dark. Take it in daylight or open shade.",
    "too_bright": "The photo is over-exposed. Avoid direct sun on the leaf.",
    "too_blurry": "The photo is blurry. Hold the phone steady and tap to focus on the leaf.",
    "low_contrast": "The photo is washed out. Move closer and avoid glare.",
    "no_leaf": "No leaf colours found. Fill most of the frame with a single leaf.",
    "ood": "This doesn't look like a coffee leaf the model knows. Photograph one coffee leaf, "
    "close up.",
    "low_confidence": "The model isn't sure. Retake the photo of one leaf, close up and in "
    "focus, or ask an agronomist.",
}


@dataclass
class QualityThresholds:
    """Set from the training photos (low percentiles), stored in bundle.json."""

    min_brightness: float
    max_brightness: float
    min_contrast: float
    min_sharpness: float
    min_green_frac: float

    def check(self, q: dict[str, float]) -> list[str]:
        issues = []
        if q["brightness"] < self.min_brightness:
            issues.append("too_dark")
        if q["brightness"] > self.max_brightness:
            issues.append("too_bright")
        if q["sharpness"] < self.min_sharpness:
            issues.append("too_blurry")
        if q["contrast"] < self.min_contrast:
            issues.append("low_contrast")
        if q["green_frac"] < self.min_green_frac:
            issues.append("no_leaf")
        return issues


@dataclass
class Decision:
    status: str  # accepted | uncertain | rejected
    reason: str | None  # None | low_quality | ood | low_confidence
    label: str | None
    confidence: float | None
    prediction_set: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)


def decide(
    classes: list[str],
    probs: np.ndarray,
    ood_score: float,
    quality_issues: list[str],
    tau_ood: float,
    qhat: float,
    tau_conf: float,
) -> Decision:
    if quality_issues:
        return Decision(
            "rejected",
            "low_quality",
            None,
            None,
            [],
            quality_issues,
            [ADVICE[i] for i in quality_issues],
        )
    if ood_score > tau_ood:
        return Decision("rejected", "ood", None, None, [], [], [ADVICE["ood"]])
    k = int(probs.argmax())
    members = np.flatnonzero(probs >= 1.0 - qhat).tolist()
    if k not in members:
        members.append(k)
    pred_set = [classes[i] for i in sorted(members, key=lambda i: -probs[i])]
    if len(pred_set) == 1 and probs[k] >= tau_conf:
        return Decision("accepted", None, classes[k], float(probs[k]), pred_set)
    return Decision(
        "uncertain",
        "low_confidence",
        classes[k],
        float(probs[k]),
        pred_set,
        [],
        [ADVICE["low_confidence"]],
    )
