"""The full serving pipeline for one photo (NumPy + Pillow + ONNX Runtime only).

EXIF-orient → quality gate → one ONNX forward pass (logits, embedding, feature map) →
OOD score → calibrated probabilities + conformal set → decision → optional CAM overlay.
Everything comes from the bundle (``bundle.json`` + files), so the API is a thin layer.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from coffeeguard.inference.cam import class_activation_map, overlay
from coffeeguard.inference.decision import Decision, QualityThresholds, decide
from coffeeguard.inference.ood import load_state, score
from coffeeguard.inference.predictor import Predictor, softmax
from coffeeguard.inference.quality import quality_metrics, to_rgb


@dataclass
class Result:
    decision: Decision
    quality: dict[str, float]
    probabilities: dict[str, float] | None = None
    ood_score: float | None = None
    cam_png_base64: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class Pipeline:
    def __init__(self, bundle_dir: str | Path, threads: int | None = None) -> None:
        self.predictor = Predictor(bundle_dir, threads=threads)
        meta = self.predictor.meta
        missing = [k for k in ("ood", "quality_thresholds", "tau_conf") if k not in meta]
        if missing:
            raise ValueError(
                f"bundle {bundle_dir} has no {missing}: run `coffeeguard ood fit` on it first"
            )
        self.meta = meta
        self.classes = self.predictor.classes
        self.quality = QualityThresholds(**meta["quality_thresholds"])
        self.ood_scorer = meta["ood"]["scorer"]
        self.tau_ood = float(meta["ood"]["tau"])
        self.ood_state = load_state(bundle_dir, meta)
        self.qhat = float(meta.get("conformal_qhat") or 0.0)
        self.tau_conf = float(meta["tau_conf"])
        self.cam_exact = bool(meta.get("cam_exact"))

    @property
    def version(self) -> str:
        return str(self.meta.get("version", "0"))

    def run(self, img: Image.Image, with_cam: bool = False) -> Result:
        t0 = time.perf_counter()
        img = to_rgb(img)
        q = quality_metrics(img)
        issues = self.quality.check(q)
        timings = {"quality": (time.perf_counter() - t0) * 1000}
        if issues:  # no point running the model on a photo we will reject anyway
            d = decide(self.classes, np.zeros(len(self.classes)), 0.0, issues, 0, 0, 1)
            return Result(d, q, timings_ms=timings)

        t1 = time.perf_counter()
        x, shown = self.predictor.prepare(img)
        logits, emb, fmap = self.predictor.run(x[None])
        probs = softmax(logits, self.predictor.temperature)[0]
        ood = float(score(self.ood_scorer, logits, emb, self.ood_state)[0])
        timings["model"] = (time.perf_counter() - t1) * 1000
        d = decide(self.classes, probs, ood, [], self.tau_ood, self.qhat, self.tau_conf)
        res = Result(
            d,
            q,
            probabilities={c: float(p) for c, p in zip(self.classes, probs, strict=True)},
            ood_score=ood,
            timings_ms=timings,
        )
        if with_cam and d.status != "rejected":
            t2 = time.perf_counter()
            k = self.classes.index(d.label) if d.label else int(probs.argmax())
            cam = class_activation_map(fmap[0], self.predictor.classifier_weight, k, shown.size[0])
            buf = io.BytesIO()
            overlay(shown, cam).save(buf, "PNG")
            res.cam_png_base64 = base64.b64encode(buf.getvalue()).decode("ascii")
            timings["cam"] = (time.perf_counter() - t2) * 1000
        return res
