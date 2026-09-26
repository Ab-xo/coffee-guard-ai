"""Phase 5 explainability on the test split: CAM galleries, leaf focus, faithfulness.

- **Galleries**: per class, the most confident correct predictions and the errors /
  least confident ones, with the CAM of the predicted class.
- **Leaf focus**: share of CAM mass inside the colour leaf mask (photos on paper only,
  where the mask is reliable). High = the model looks at the leaf, not the background.
- **Deletion faithfulness**: hide 16 px patches in CAM order (hottest first) vs. random
  order and track the calibrated probability of the originally predicted class. If the
  CAM points at what the model uses, probability falls faster under CAM order (lower
  area under the curve).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from coffeeguard.config import DataConfig
from coffeeguard.data.dataset import read_split
from coffeeguard.inference.cam import class_activation_map, overlay
from coffeeguard.inference.predictor import Predictor, softmax
from coffeeguard.robustness.leafmask import leaf_mask
from coffeeguard.robustness.sweep import MAX_LEAF_FRACTION
from coffeeguard.utils.io import write_json
from coffeeguard.utils.log import get_logger
from coffeeguard.utils.plotting import SERIES, TEXT_MUTED, apply_style, save

log = get_logger(__name__)
PATCH = 16
DELETE_FRACTIONS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])


def _deletion_curve(
    predictor: Predictor, x: np.ndarray, order: np.ndarray, k: int, grid: int
) -> np.ndarray:
    """Prob. of class ``k`` after zeroing (= dataset-mean colour) the first patches in ``order``."""
    batch = []
    for f in DELETE_FRACTIONS:
        xi = x.copy()
        for p in order[: int(f * grid * grid + 0.5)]:
            r, c = divmod(int(p), grid)
            xi[:, r * PATCH : (r + 1) * PATCH, c * PATCH : (c + 1) * PATCH] = 0.0
        batch.append(xi)
    logits = predictor.run(np.stack(batch))[0]
    return softmax(logits, predictor.temperature)[:, k]


def _auc(curve: np.ndarray) -> float:
    return float(np.trapezoid(curve, DELETE_FRACTIONS) / DELETE_FRACTIONS[-1])


def run_explain(bundle_dir: Path, cfg: DataConfig, out_root: Path, per_class: int = 4) -> dict:
    predictor = Predictor(bundle_dir)
    name, classes, size = bundle_dir.name, predictor.classes, predictor.img_size
    if not predictor.meta.get("cam_exact"):
        log.warning("%s: head is not pool→linear; CAM is only an approximation", name)
    weight = predictor.classifier_weight
    test = read_split(cfg.splits_dir, "test")
    grid = size // PATCH
    rng = np.random.default_rng(0)

    rows = []
    for rel, y in zip(test["image"], test["class_id"], strict=True):
        with Image.open(cfg.processed_dir / rel) as im:
            img = im.convert("RGB")
        x, shown = predictor.prepare(img)
        logits, _, fmap = predictor.run(x[None])
        p = softmax(logits, predictor.temperature)[0]
        k = int(p.argmax())
        cam = class_activation_map(fmap[0], weight, k, size)
        mask = leaf_mask(shown)
        patch_score = cam.reshape(grid, PATCH, grid, PATCH).mean((1, 3)).ravel()
        cam_curve = _deletion_curve(predictor, x, np.argsort(-patch_score), k, grid)
        rnd_curve = _deletion_curve(predictor, x, rng.permutation(grid * grid), k, grid)
        rows.append(
            {
                "image": rel,
                "y": int(y),
                "pred": k,
                "conf": float(p[k]),
                "cam": cam,
                "shown": shown,
                "mask_frac": float(mask.mean()),
                "leaf_focus": float((cam * mask).sum() / max(cam.sum(), 1e-9)),
                "del_cam": cam_curve,
                "del_rnd": rnd_curve,
            }
        )

    ok = [r for r in rows if r["mask_frac"] <= MAX_LEAF_FRACTION]
    correct = np.array([r["y"] == r["pred"] for r in rows])
    res = {
        "bundle": name,
        "n": len(rows),
        "leaf_focus": {
            "n_used": len(ok),
            "mean": float(np.mean([r["leaf_focus"] for r in ok])),
            "mean_leaf_area": float(np.mean([r["mask_frac"] for r in ok])),
            "per_class": {
                c: float(np.mean([r["leaf_focus"] for r in ok if r["y"] == k]))
                for k, c in enumerate(classes)
                if any(r["y"] == k for r in ok)
            },
            "correct": float(np.mean([r["leaf_focus"] for r in ok if r["y"] == r["pred"]])),
            "errors": float(np.mean([r["leaf_focus"] for r in ok if r["y"] != r["pred"]]))
            if any(r["y"] != r["pred"] for r in ok)
            else None,
        },
        "deletion": {
            "fractions": DELETE_FRACTIONS.tolist(),
            "cam_order_curve": np.mean([r["del_cam"] for r in rows], 0).tolist(),
            "random_order_curve": np.mean([r["del_rnd"] for r in rows], 0).tolist(),
            "auc_cam": float(np.mean([_auc(r["del_cam"]) for r in rows])),
            "auc_random": float(np.mean([_auc(r["del_rnd"]) for r in rows])),
            "share_images_cam_faster": float(
                np.mean([_auc(r["del_cam"]) < _auc(r["del_rnd"]) for r in rows])
            ),
        },
        "accuracy": float(correct.mean()),
    }
    out_dir = out_root / name
    write_json(out_dir / "explain.json", res)
    _figures(rows, classes, res, out_dir / "figures", per_class)
    log.info(
        "%s: leaf focus %.3f (leaf area %.3f) | deletion AUC cam %.3f vs random %.3f",
        name,
        res["leaf_focus"]["mean"],
        res["leaf_focus"]["mean_leaf_area"],
        res["deletion"]["auc_cam"],
        res["deletion"]["auc_random"],
    )
    return res


def _figures(rows: list[dict], classes: list[str], res: dict, out_dir: Path, n: int) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    # galleries: per class, n most confident correct + n errors/least confident
    for k, c in enumerate(classes):
        cls = [r for r in rows if r["y"] == k]
        good = sorted([r for r in cls if r["pred"] == k], key=lambda r: -r["conf"])[:n]
        bad = sorted(cls, key=lambda r: (r["pred"] == k, r["conf"]))[:n]
        fig, axes = plt.subplots(2, n, figsize=(n * 2.3, 5.2), squeeze=False)
        for row_i, (items, label) in enumerate(
            ((good, "confident & correct"), (bad, "errors / least confident"))
        ):
            for ax in axes[row_i]:
                ax.axis("off")
            for ax, r in zip(axes[row_i], items, strict=False):
                ax.imshow(overlay(r["shown"], r["cam"]))
                mark = "✓" if r["pred"] == r["y"] else "✗"
                ax.set_title(
                    f"{mark} {classes[r['pred']]} {r['conf']:.2f}",
                    fontsize=8,
                    color=TEXT_MUTED,
                    fontweight="normal",
                )
            axes[row_i, 0].text(
                -0.08,
                0.5,
                label,
                transform=axes[row_i, 0].transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=8,
                color=TEXT_MUTED,
            )
        fig.suptitle(
            f"{c} — CAM of the predicted class (test)",
            x=0.01,
            ha="left",
            fontsize=11,
            fontweight="bold",
        )
        fig.tight_layout()
        save(fig, out_dir / f"cam_{c.lower().replace(' ', '_')}.png")

    # deletion curves
    d = res["deletion"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(
        d["fractions"],
        d["cam_order_curve"],
        marker="o",
        ms=3,
        color=SERIES[0],
        label=f"CAM order (AUC {d['auc_cam']:.3f})",
    )
    ax.plot(
        d["fractions"],
        d["random_order_curve"],
        marker="o",
        ms=3,
        color=SERIES[1],
        label=f"random order (AUC {d['auc_random']:.3f})",
    )
    ax.set_xlabel("share of image patches hidden")
    ax.set_ylabel("prob. of predicted class")
    ax.set_title("Deletion test (test set)")
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir / "deletion.png")
