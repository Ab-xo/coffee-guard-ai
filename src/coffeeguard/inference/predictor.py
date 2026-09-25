"""Slim ONNX Runtime predictor used by the API, the UI and offline evaluation.

Depends only on numpy, pillow and onnxruntime.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from coffeeguard.inference.preprocess import normalize, resize_for_model


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class Prediction:
    label: str
    class_id: int
    confidence: float
    probabilities: dict[str, float]
    logits: np.ndarray = field(repr=False)
    embedding: np.ndarray = field(repr=False)
    feature_map: np.ndarray = field(repr=False)
    model_input: Image.Image = field(repr=False)  # the resized RGB image the model saw


class Predictor:
    def __init__(self, bundle_dir: str | Path, threads: int | None = None) -> None:
        import onnxruntime as ort

        self.bundle_dir = Path(bundle_dir)
        self.meta: dict = json.loads((self.bundle_dir / "bundle.json").read_text("utf-8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2) // 2)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.bundle_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.classes: list[str] = self.meta["classes"]
        self.img_size: int = self.meta["img_size"]
        self.temperature: float = float(self.meta.get("temperature") or 1.0)
        self.classifier_weight = np.load(self.bundle_dir / "classifier_weight.npy")

    def run(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Raw forward pass on a preprocessed ``(B, 3, H, W)`` float32 batch."""
        logits, emb, fmap = self.session.run(None, {"input": batch.astype(np.float32)})
        return logits, emb, fmap

    def prepare(self, img: Image.Image) -> tuple[np.ndarray, Image.Image]:
        shown = resize_for_model(img, self.img_size)
        return normalize(shown, self.meta["mean"], self.meta["std"]), shown

    def predict(self, img: Image.Image) -> Prediction:
        x, shown = self.prepare(img)
        logits, emb, fmap = self.run(x[None])
        probs = softmax(logits, self.temperature)[0]
        k = int(probs.argmax())
        return Prediction(
            label=self.classes[k],
            class_id=k,
            confidence=float(probs[k]),
            probabilities={c: float(p) for c, p in zip(self.classes, probs, strict=True)},
            logits=logits[0],
            embedding=emb[0],
            feature_map=fmap[0],
            model_input=shown,
        )
