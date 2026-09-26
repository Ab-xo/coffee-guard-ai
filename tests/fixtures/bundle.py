"""A tiny hand-built ONNX model bundle, so API tests never need a trained model.

The graph has the same interface as a real export (input ``(B, 3, H, W)`` → ``logits``,
``embedding``, ``feature_map``): feature map = the input, embedding = its global average
(per-channel mean colour), logits = a fixed linear layer on that embedding. Class 0 wins
for green-dominant images, class 1 for red-dominant ones, which makes predictions testable.
The bundle also carries the Phase 6 gates (permissive quality thresholds, a KNN bank of
green / red / leaf-brown colours, τ values), so blue images are out of distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLASSES = ["Healthy", "Cercospora", "Leaf Rust", "Phoma"]


def make_fake_bundle(out_dir: Path, img_size: int = 32) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight = 10 * np.array(
        [[0.0, 4.0, 0.0], [4.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )  # (classes, channels); x10 so clear colours give confident predictions
    bias = np.zeros(len(CLASSES), dtype=np.float32)
    nodes = [
        helper.make_node("Identity", ["input"], ["feature_map"]),
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["embedding"], axis=1),
        helper.make_node("Gemm", ["embedding", "W", "b"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes,
        "fake_coffeeguard",
        [
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["batch", 3, img_size, img_size]
            )
        ],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(CLASSES)]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 3]),
            helper.make_tensor_value_info(
                "feature_map", TensorProto.FLOAT, ["batch", 3, img_size, img_size]
            ),
        ],
        initializer=[numpy_helper.from_array(weight, "W"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_dir / "model.onnx")
    np.save(out_dir / "classifier_weight.npy", weight)
    bank = np.repeat(np.array([[0.1, 0.7, 0.1], [0.8, 0.15, 0.1], [0.45, 0.5, 0.3]]), 4, axis=0)
    np.save(
        out_dir / "ood_knn_bank.npy",
        (bank / np.linalg.norm(bank, axis=1, keepdims=True)).astype(np.float16),
    )
    meta = {
        "name": "fake-bundle",
        "version": "0.0.1",
        "model": "fake",
        "classes": CLASSES,
        "img_size": img_size,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "temperature": 1.0,
        "cam_exact": True,
        "conformal_qhat": 0.5,
        "tau_conf": 0.6,
        "ood": {"scorer": "knn", "tau": 0.05, "knn_k": 3},
        "quality_thresholds": {
            "min_brightness": 20.0,
            "max_brightness": 250.0,
            "min_contrast": 5.0,
            "min_sharpness": 5.0,
            "min_green_frac": 0.0,
        },
    }
    (out_dir / "bundle.json").write_text(json.dumps(meta, indent=2), "utf-8")
    return out_dir


def textured(color: tuple[int, int, int], size: tuple[int, int] = (96, 72), seed: int = 0):
    """An RGB image of ``color`` with fine noise, so it passes the quality gate."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    a = np.asarray(color, np.float32) + rng.normal(0, 25, (size[1], size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
