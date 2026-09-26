"""Robustness sweep (9 corruptions x 5 severities) and shortcut test on the test split.

Runs through the ONNX bundle with its calibrated temperature. Per corruption/severity:
accuracy, macro-F1, mean confidence, ECE and where the errors go (predicted-class
counts), which shows whether low image quality pushes predictions to one class - the
confound found in EDA. Shortcut test: leaf-only and background-only versions of test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix

from coffeeguard.config import DataConfig
from coffeeguard.data.dataset import read_split
from coffeeguard.evaluation.calibration import ece
from coffeeguard.evaluation.metrics import core_metrics
from coffeeguard.inference.predictor import Predictor, softmax
from coffeeguard.robustness.corruptions import CORRUPTIONS
from coffeeguard.robustness.leafmask import apply_mask, leaf_mask
from coffeeguard.utils.io import write_json
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)
SEVERITIES = (1, 2, 3, 4, 5)
MAX_LEAF_FRACTION = 0.8  # above this the mask is the whole scene (field photo)


def _seed(sha: str, *extra: int) -> np.random.Generator:
    return np.random.default_rng([int(sha[:12], 16), *extra])


def _probs(predictor: Predictor, images: list[Image.Image], batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(images), batch):
        x = np.stack([predictor.prepare(im)[0] for im in images[i : i + batch]])
        out.append(softmax(predictor.run(x)[0], predictor.temperature))
    return np.concatenate(out)


def _score(probs: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    pred = probs.argmax(1)
    m = core_metrics(y, pred)
    m["mean_confidence"] = float(probs.max(1).mean())
    m["ece"] = ece(probs, y)
    m["predicted_counts"] = {c: int((pred == k).sum()) for k, c in enumerate(classes)}
    m["errors_to"] = {c: int(((pred == k) & (pred != y)).sum()) for k, c in enumerate(classes)}
    m["confusion_matrix"] = confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist()
    return m


def run_robustness(bundle_dir: Path, cfg: DataConfig, out_root: Path) -> dict:
    predictor = Predictor(bundle_dir)
    classes, name = predictor.classes, bundle_dir.name
    test = read_split(cfg.splits_dir, "test")
    y = test["class_id"].to_numpy()
    images = []
    for rel in test["image"]:
        with Image.open(cfg.processed_dir / rel) as im:
            images.append(im.convert("RGB"))

    res: dict = {
        "bundle": name,
        "n": len(y),
        "clean": _score(_probs(predictor, images), y, classes),
    }
    clean_f1 = res["clean"]["macro_f1"]
    res["corruptions"] = {}
    for c_i, (cname, fn) in enumerate(CORRUPTIONS.items()):
        res["corruptions"][cname] = {}
        for s in SEVERITIES:
            corrupted = [
                fn(im, s, _seed(sha, c_i, s))
                for im, sha in zip(images, test["sha256"], strict=True)
            ]
            res["corruptions"][cname][str(s)] = _score(_probs(predictor, corrupted), y, classes)
        f1s = [res["corruptions"][cname][str(s)]["macro_f1"] for s in SEVERITIES]
        log.info("%s %-15s macro-F1 by severity %s", name, cname, " ".join(f"{f:.3f}" for f in f1s))

    def rel(max_sev: int) -> float:
        vals = [
            res["corruptions"][c][str(s)]["macro_f1"]
            for c in CORRUPTIONS
            for s in SEVERITIES
            if s <= max_sev
        ]
        return float(np.mean(vals) / clean_f1)

    res["relative_robustness"] = {"severity_1_3": rel(3), "severity_1_5": rel(5)}

    # ---- shortcut test, on photos whose background can be separated: in field photos the
    # colour mask covers the whole frame (surrounding foliage), so they are left out
    masks = [leaf_mask(im) for im in images]
    frac = np.array([m.mean() for m in masks])
    keep = np.flatnonzero(frac <= MAX_LEAF_FRACTION)
    ys = y[keep]
    leaf_only = [apply_mask(images[i], masks[i], keep=True) for i in keep]
    bg_only = [apply_mask(images[i], masks[i], keep=False) for i in keep]
    counts = np.bincount(ys, minlength=len(classes))
    res["shortcut"] = {
        "n_used": len(keep),
        "excluded_field_photos": {
            c: int(((frac > MAX_LEAF_FRACTION) & (y == k)).sum()) for k, c in enumerate(classes)
        },
        "majority_class_rate": float(counts.max() / counts.sum()),
        "leaf_mask_fraction_mean": float(frac[keep].mean()),
        "original": _score(_probs(predictor, [images[i] for i in keep]), ys, classes),
        "leaf_only": _score(_probs(predictor, leaf_only), ys, classes),
        "background_only": _score(_probs(predictor, bg_only), ys, classes),
    }
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "robustness.json", res)
    from coffeeguard.robustness.figures import fig_degradation, fig_shortcut_examples

    fig_degradation(res, out_dir / "figures" / "degradation.png")
    fig_shortcut_examples(images, masks, test, out_dir / "figures" / "shortcut_examples.png")
    log.info(
        "%s: relative robustness (sev<=3) %.3f | leaf-only acc %.3f | background-only acc %.3f",
        name,
        res["relative_robustness"]["severity_1_3"],
        res["shortcut"]["leaf_only"]["accuracy"],
        res["shortcut"]["background_only"]["accuracy"],
    )
    return res


def summary_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "bundle": r["bundle"],
            "clean_macro_f1": r["clean"]["macro_f1"],
            "relative_robustness_sev1_3": r["relative_robustness"]["severity_1_3"],
            "relative_robustness_sev1_5": r["relative_robustness"]["severity_1_5"],
            "leaf_only_acc": r["shortcut"]["leaf_only"]["accuracy"],
            "background_only_acc": r["shortcut"]["background_only"]["accuracy"],
        }
        for c in CORRUPTIONS:
            row[f"{c}_sev3_f1"] = r["corruptions"][c]["3"]["macro_f1"]
        rows.append(row)
    return pd.DataFrame(rows)
