"""``coffeeguard embed audit``: label-issue audit + DINOv2 linear-probe baseline.

1. **Label audit.** 5-fold group-aware cross-validated logistic regression on frozen
   DINOv2 embeddings gives out-of-fold class probabilities for every clean image.
   ``cleanlab.filter.find_label_issues`` ranks images whose given label disagrees
   with what the rest of the data implies. Nothing is removed automatically: the
   gallery is reviewed and confirmed issues go into ``configs/data.yaml: exclude``.
2. **Probe baseline.** The same classifier trained on *train* and scored on *val*
   (test stays sealed) — a strong reference point from a frozen foundation model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from coffeeguard.config import DataConfig
from coffeeguard.embeddings.extract import aligned_embeddings, load_store
from coffeeguard.utils.io import write_json
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)


def _probe() -> object:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)


def run_audit(cfg: DataConfig, n_folds: int = 5, max_gallery: int = 40) -> dict:
    from cleanlab.filter import find_label_issues

    m = pd.read_parquet(cfg.manifest_path)
    ok = m[m["status"] == "ok"].reset_index(drop=True)
    store = load_store(cfg.embeddings_path.with_suffix(".npz"))
    X = aligned_embeddings(ok, store)
    if X is None or not np.abs(X).sum(axis=1).all():
        raise RuntimeError(
            "Missing embeddings: run `coffeeguard data prepare` with the embed stage"
        )
    X = _l2(X)
    y = ok["class_id"].to_numpy()
    groups = ok["group_id"].to_numpy()

    # -- 1. out-of-fold probabilities for the whole clean set
    oof = np.zeros((len(ok), len(cfg.classes)))
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg.split.seed)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        clf = _probe().fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])
        log.info("audit fold %d/%d done", fold + 1, n_folds)

    issue_idx = find_label_issues(
        labels=y, pred_probs=oof, return_indices_ranked_by="self_confidence"
    )
    issues = ok.loc[issue_idx, ["path", "cache_path", "sha256", "label", "split"]].copy()
    issues["predicted"] = [cfg.classes[i] for i in oof[issue_idx].argmax(1)]
    issues["given_label_prob"] = oof[issue_idx, y[issue_idx]]
    issues["predicted_prob"] = oof[issue_idx].max(1)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    issues.to_csv(cfg.reports_dir / "label_issues.csv", index=False)

    oof_pred = oof.argmax(1)
    audit = {
        "n_images": len(ok),
        "n_label_issues": len(issues),
        "issue_rate": len(issues) / len(ok),
        "issues_by_given_label": issues["label"].value_counts().to_dict(),
        "issues_by_pair": issues.groupby(["label", "predicted"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict(orient="records"),
        "oof_accuracy": float(accuracy_score(y, oof_pred)),
        "oof_macro_f1": float(f1_score(y, oof_pred, average="macro")),
    }

    # -- 2. probe baseline: train -> val
    tr = (ok["split"] == "train").to_numpy()
    va = (ok["split"] == "val").to_numpy()
    clf = _probe().fit(X[tr], y[tr])
    val_pred = clf.predict(X[va])
    probe = {
        "model": f"{cfg.embed_model} (frozen) + logistic regression",
        "split": "val",
        "accuracy": float(accuracy_score(y[va], val_pred)),
        "macro_f1": float(f1_score(y[va], val_pred, average="macro")),
        "per_class": classification_report(
            y[va], val_pred, target_names=cfg.classes, output_dict=True, zero_division=0
        ),
    }
    write_json(cfg.reports_dir / "label_audit.json", audit)
    write_json(cfg.reports_dir.parent / "metrics" / "baseline_dinov2_probe_val.json", probe)

    if len(issues):
        _gallery(issues.head(max_gallery), cfg)
    log.info(
        "Label audit: %d possible issues (%.1f%%); probe val macro-F1 %.4f",
        len(issues),
        100 * audit["issue_rate"],
        probe["macro_f1"],
    )
    return {"audit": audit, "probe": {k: v for k, v in probe.items() if k != "per_class"}}


def _gallery(issues: pd.DataFrame, cfg: DataConfig, cols: int = 8) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    from coffeeguard.utils.plotting import TEXT, TEXT_MUTED, apply_style, save

    apply_style()
    rows = int(np.ceil(len(issues) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.0))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), issues.iterrows(), strict=False):
        with Image.open(cfg.processed_dir / r["cache_path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(
            f"given: {r['label']}\npred: {r['predicted']} {r['predicted_prob']:.2f}",
            fontsize=6.5,
            color=TEXT_MUTED,
        )
    fig.suptitle(
        "Possible label issues (ranked, most suspicious first) — review before excluding",
        x=0.01,
        ha="left",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    save(fig, cfg.figures_dir / "eda" / "label_issues.png")
