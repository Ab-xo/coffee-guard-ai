"""Export a training checkpoint to an ONNX model bundle and verify parity.

The exported graph has three outputs from one forward pass:

- ``logits``      ``(B, C)``      class scores
- ``embedding``   ``(B, F)``      pooled pre-classifier features (OOD scoring)
- ``feature_map`` ``(B, F, h, w)`` last conv features (CAM heatmaps)

For CNNs whose head is global-average-pool → linear (EfficientNet family),
``embedding == mean(feature_map)``; the class activation map ``Σ_k w_ck · A_k`` then
equals Grad-CAM at that layer up to a positive scale, so the API can draw heatmaps
without torch or a backward pass. ``bundle.json`` records whether that holds
(``cam_exact``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from coffeeguard.models.factory import build_model
from coffeeguard.utils.hashing import sha256_file
from coffeeguard.utils.io import read_json, write_json
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fmap = self.model.forward_features(x)
        pooled = self.model.forward_head(fmap, pre_logits=True)
        logits = self.model.get_classifier()(pooled)
        return logits, pooled, fmap


def load_checkpoint_model(ckpt_path: Path) -> tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(
        ckpt["model"], len(ckpt["classes"]), pretrained=False, drop_rate=0.0, drop_path_rate=0.0
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def export_bundle(
    ckpt_path: Path, out_dir: Path, opset: int = 17, parity_images: np.ndarray | None = None
) -> Path:
    """Write ``model.onnx`` + ``bundle.json`` into ``out_dir``; return ``out_dir``."""
    import onnxruntime as ort

    model, ckpt = load_checkpoint_model(ckpt_path)
    wrapper = ExportWrapper(model).eval()
    size = ckpt["img_size"]
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    dummy = torch.randn(2, 3, size, size)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["input"],
            output_names=["logits", "embedding", "feature_map"],
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
                "embedding": {0: "batch"},
                "feature_map": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )

    # --- parity: torch vs onnxruntime on real or random inputs
    x = (
        parity_images
        if parity_images is not None
        else np.random.default_rng(0).normal(size=(8, 3, size, size)).astype(np.float32)
    )
    with torch.no_grad():
        t_logits, t_emb, t_fmap = (o.numpy() for o in wrapper(torch.from_numpy(x)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    o_logits, _, _ = sess.run(None, {"input": x})
    max_diff = float(np.abs(t_logits - o_logits).max())
    argmax_agree = float((t_logits.argmax(1) == o_logits.argmax(1)).mean())
    # True only when the head is GAP -> linear (e.g. EfficientNet). MobileNetV3 applies a
    # 1x1 conv + activation *after* pooling, so its embedding is not the pooled map.
    gap = t_fmap.mean(axis=(2, 3))
    cam_exact = gap.shape == t_emb.shape and bool(np.allclose(gap, t_emb, atol=1e-4))
    log.info(
        "ONNX parity: max |Δlogit| %.2e, argmax agreement %.3f, cam_exact %s",
        max_diff,
        argmax_agree,
        cam_exact,
    )
    if max_diff > 1e-3 or argmax_agree < 1.0:
        raise AssertionError(f"ONNX parity failed: max diff {max_diff}, agree {argmax_agree}")

    weight, bias = (
        p.detach().numpy() for p in (model.get_classifier().weight, model.get_classifier().bias)
    )
    np.save(out_dir / "classifier_weight.npy", weight.astype(np.float32))
    np.save(out_dir / "classifier_bias.npy", bias.astype(np.float32))

    run_dir = ckpt_path.parent
    summary = read_json(run_dir / "summary.json") if (run_dir / "summary.json").exists() else {}
    meta = {
        "name": out_dir.name,
        "model": ckpt["model"],
        "classes": ckpt["classes"],
        "img_size": size,
        "mean": list(ckpt["mean"]),
        "std": list(ckpt["std"]),
        "data_fingerprint": ckpt.get("data_fingerprint"),
        "source_run": run_dir.name,
        "weights": ckpt.get("weights"),
        "val": ckpt.get("val") or summary.get("val"),
        "cam_exact": cam_exact,
        "feature_dim": int(t_emb.shape[1]),
        "feature_map_size": list(t_fmap.shape[2:]),
        "onnx_opset": opset,
        "parity": {"max_abs_logit_diff": max_diff, "argmax_agreement": argmax_agree},
        # filled in by calibration / OOD fitting (Phases 4 and 6)
        "temperature": 1.0,
        "conformal_qhat": None,
        "thresholds": {},
        "files": {},
    }
    for f in ("model.onnx", "classifier_weight.npy", "classifier_bias.npy"):
        meta["files"][f] = sha256_file(out_dir / f)
    write_json(out_dir / "bundle.json", meta)
    log.info("Bundle written to %s", out_dir)
    return out_dir
