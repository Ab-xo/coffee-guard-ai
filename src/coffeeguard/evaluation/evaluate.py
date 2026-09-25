"""Phase 4 evaluation of exported ONNX bundles (exactly what the API serves).

For each bundle: predict val + test → fit temperature and conformal q̂ on **val only** →
report test metrics with bootstrap CIs, calibration, conformal coverage, confidence
buckets, selective accuracy and error rate by photo quality → figures. The fitted
temperature and q̂ are written back into ``bundle.json`` so serving uses them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from coffeeguard.config import DataConfig
from coffeeguard.data.dataset import read_split
from coffeeguard.evaluation.calibration import ece, fit_temperature, nll
from coffeeguard.evaluation.conformal import fit_qhat, prediction_sets, set_metrics
from coffeeguard.evaluation.metrics import bootstrap_ci, full_metrics, paired_bootstrap_diff
from coffeeguard.inference.predictor import Predictor, softmax
from coffeeguard.utils.io import write_json
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)

# Buckets from the specification (docs/PROJECT_SPECIFICATION.md, "Confidence Buckets").
BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
QUALITY_COLS = ["sharpness", "brightness", "contrast", "green_frac", "bytes"]


def predict_split(
    predictor: Predictor, frame: pd.DataFrame, root: Path, batch: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Logits ``(N, C)`` and embeddings ``(N, D)`` for every image of a split."""
    logits, embs = [], []
    for start in range(0, len(frame), batch):
        xs = []
        for rel in frame["image"].iloc[start : start + batch]:
            with Image.open(root / rel) as im:
                xs.append(predictor.prepare(im)[0])
        lg, emb, _ = predictor.run(np.stack(xs))
        logits.append(lg)
        embs.append(emb)
    return np.concatenate(logits), np.concatenate(embs)


def _bucket_table(probs: np.ndarray, labels: np.ndarray) -> list[dict]:
    conf, correct = probs.max(1), probs.argmax(1) == labels
    n_err = max(int((~correct).sum()), 1)
    rows = []
    for lo, hi in BUCKETS:
        m = (conf >= lo) & (conf < hi)
        rows.append(
            {
                "bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "count": int(m.sum()),
                "accuracy": float(correct[m].mean()) if m.any() else None,
                "share_of_errors": float((~correct[m]).sum() / n_err),
            }
        )
    return rows


def _selective(probs: np.ndarray, labels: np.ndarray) -> dict[str, list[float]]:
    """Accuracy on the kept images when rejecting everything below a confidence threshold."""
    conf, correct = probs.max(1), (probs.argmax(1) == labels)
    order = np.argsort(-conf)
    kept = np.arange(1, len(conf) + 1)
    return {
        "coverage": (kept / len(conf)).tolist(),
        "accuracy": (np.cumsum(correct[order]) / kept).tolist(),
        "threshold": conf[order].tolist(),
    }


def _quality_errors(df: pd.DataFrame) -> dict:
    """Error rate per quartile of each photo-quality measure (EDA confound check)."""
    out = {}
    for col in QUALITY_COLS:
        if col not in df:
            continue
        q = pd.qcut(df[col].rank(method="first"), 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        g = df.groupby(q, observed=True)["correct"]
        out[col] = {str(k): {"n": int(v.size), "error_rate": float(1 - v.mean())} for k, v in g}
    return out


def evaluate_bundle(
    bundle_dir: Path, data_cfg: DataConfig, out_root: Path, alpha: float = 0.02
) -> dict:
    predictor = Predictor(bundle_dir)
    classes = predictor.classes
    name = bundle_dir.name
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(data_cfg.manifest_path, columns=["sha256", *QUALITY_COLS])
    manifest = manifest.drop_duplicates("sha256")

    res: dict = {"bundle": name, "model": predictor.meta.get("model"), "classes": classes}
    data: dict[str, dict] = {}
    for split in ("val", "test"):
        frame = read_split(data_cfg.splits_dir, split)
        logits, emb = predict_split(predictor, frame, data_cfg.processed_dir)
        labels = frame["class_id"].to_numpy()
        data[split] = {"frame": frame, "logits": logits, "labels": labels}
        np.save(out_dir / f"embeddings_{split}.npy", emb.astype(np.float16))
        log.info("%s: predicted %d %s images", name, len(frame), split)

    # ---- fitted on val only
    v, t = data["val"], data["test"]
    temp = fit_temperature(v["logits"], v["labels"])
    pv, pt = softmax(v["logits"], temp), softmax(t["logits"], temp)
    qhat = fit_qhat(pv, v["labels"], alpha)

    for split, d, p in (("val", v, pv), ("test", t, pt)):
        y, raw = d["labels"], softmax(d["logits"])
        pred = raw.argmax(1)  # temperature never changes the argmax
        m = full_metrics(y, pred, classes, raw)
        m["n"] = len(y)
        m["errors"] = int((pred != y).sum())
        m["bootstrap_95ci"] = bootstrap_ci(y, pred)
        m["calibration"] = {
            "ece_before": ece(raw, y),
            "ece_after": ece(p, y),
            "nll_before": nll(d["logits"], y),
            "nll_after": nll(d["logits"], y, temp),
            "mean_conf_before": float(raw.max(1).mean()),
            "mean_conf_after": float(p.max(1).mean()),
        }
        m["conformal"] = set_metrics(prediction_sets(p, qhat), y, classes)
        m["confidence_buckets"] = _bucket_table(p, y)
        res[split] = m

        frame = d["frame"].copy()
        frame["pred"] = pred
        frame["pred_label"] = [classes[k] for k in pred]
        frame["confidence"] = p.max(1)
        frame["correct"] = pred == y
        frame["set_size"] = prediction_sets(p, qhat).sum(1)
        for k, c in enumerate(classes):
            frame[f"p_{c}"] = p[:, k]
        frame = frame.merge(manifest, on="sha256", how="left")
        frame.to_parquet(out_dir / f"predictions_{split}.parquet", index=False)
        d["frame"], d["probs"] = frame, p

    res["temperature"], res["conformal_alpha"], res["conformal_qhat"] = temp, alpha, qhat
    res["test"]["selective"] = _selective(pt, t["labels"])
    res["test"]["quality_errors"] = _quality_errors(
        pd.concat([v["frame"], t["frame"]], ignore_index=True)
    )
    write_json(out_dir / "metrics.json", res)

    # ---- the bundle serves calibrated probabilities and the conformal threshold
    meta_path = bundle_dir / "bundle.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta.update(temperature=temp, conformal_alpha=alpha, conformal_qhat=qhat)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", "utf-8")

    from coffeeguard.evaluation.figures import make_figures

    make_figures(name, classes, data, res, out_dir / "figures", data_cfg)
    log.info(
        "%s test: macro-F1 %.4f [%.4f, %.4f], ECE %.3f -> %.3f (T=%.2f), coverage %.3f",
        name,
        res["test"]["macro_f1"],
        res["test"]["bootstrap_95ci"]["macro_f1"]["lo"],
        res["test"]["bootstrap_95ci"]["macro_f1"]["hi"],
        res["test"]["calibration"]["ece_before"],
        res["test"]["calibration"]["ece_after"],
        temp,
        res["test"]["conformal"]["coverage"],
    )
    return res


def compare(results: list[dict], out_root: Path) -> dict:
    """Summary table over bundles + paired bootstrap of the main model vs. each other."""
    table = []
    for r in results:
        t = r["test"]
        table.append(
            {
                "bundle": r["bundle"],
                "model": r["model"],
                "val_macro_f1": r["val"]["macro_f1"],
                "test_macro_f1": t["macro_f1"],
                "test_macro_f1_ci": [
                    t["bootstrap_95ci"]["macro_f1"]["lo"],
                    t["bootstrap_95ci"]["macro_f1"]["hi"],
                ],
                "test_accuracy": t["accuracy"],
                "test_errors": t["errors"],
                "test_ece_after_ts": t["calibration"]["ece_after"],
                "temperature": r["temperature"],
                "conformal_coverage": t["conformal"]["coverage"],
                "avg_set_size": t["conformal"]["avg_set_size"],
            }
        )
    out: dict = {"split": "test", "n": results[0]["test"]["n"], "models": table}
    # Paired bootstrap: the main model (first bundle, fixed in advance by the spec) vs.
    # each alternative on the same test images. Nothing is selected on test here.
    ref = results[0]
    pa = pd.read_parquet(out_root / ref["bundle"] / "predictions_test.parquet")
    out["paired_bootstrap_vs_main"] = {"main": ref["bundle"], "comparisons": {}}
    for r in results[1:]:
        pb = pd.read_parquet(out_root / r["bundle"] / "predictions_test.parquet")
        assert (pa["image"].to_numpy() == pb["image"].to_numpy()).all()
        out["paired_bootstrap_vs_main"]["comparisons"][r["bundle"]] = paired_bootstrap_diff(
            pa["class_id"].to_numpy(), pa["pred"].to_numpy(), pb["pred"].to_numpy()
        )
    return out
