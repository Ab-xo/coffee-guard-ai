"""Fit and evaluate the rejection gates for one bundle (Phase 6).

Fitted on train / val / OOD-cal only; reported on test / OOD-test:

1. **Quality gate**: brightness, contrast and sharpness thresholds sit where the model's
   val accuracy on corrupted photos drops below 90%; over-exposure and leaf-colour
   fraction use the 0.5th / 99.5th percentile of the training photos.
2. **OOD scorer**: MSP, energy, Mahalanobis and KNN are compared by AUROC of val (ID)
   vs. OOD-cal, averaged over near- and far-OOD; the best one is kept.
3. **τ_ood** = the highest val percentile (95-99.5%) at which ≤ 1% of OOD-cal images pass.
4. **τ_conf** = lowest confidence at which val predictions that pass the gates are
   ≥ 99% correct (below it the answer is "uncertain").
Everything the API needs is written into the bundle (``bundle.json`` + ``ood_*.npy``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from coffeeguard.config import DataConfig
from coffeeguard.data.dataset import read_split
from coffeeguard.inference.decision import QualityThresholds, decide
from coffeeguard.inference.ood import SCORERS, OODState, _l2n, score
from coffeeguard.inference.predictor import Predictor, softmax
from coffeeguard.inference.quality import quality_metrics
from coffeeguard.robustness.corruptions import CORRUPTIONS
from coffeeguard.utils.hashing import sha256_file
from coffeeguard.utils.io import write_json
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)
QUALITY_PCT = 0.5  # percentile of training photos below which a photo is rejected
ID_ACCEPT = 0.95  # minimum share of genuine val photos the OOD gate accepts
MAX_OOD_CAL_ACCEPT = 0.01  # max share of OOD-cal images the gate may accept
TARGET_ACC = 0.99
MIN_GATE_ACC = 0.90  # reject photos of a quality at which val accuracy falls below this


def _run(predictor: Predictor, images: list[Image.Image], batch: int = 32):
    logits, embs = [], []
    for i in range(0, len(images), batch):
        x = np.stack([predictor.prepare(im)[0] for im in images[i : i + batch]])
        lg, emb, _ = predictor.run(x)
        logits.append(lg)
        embs.append(emb)
    return np.concatenate(logits), np.concatenate(embs)


def _load(paths: list[Path]) -> list[Image.Image]:
    out = []
    for p in paths:
        with Image.open(p) as im:
            out.append(im.convert("RGB"))
    return out


def _ood_frame(root: Path, split: str) -> pd.DataFrame:
    rows = [
        {"path": p, "kind": p.parent.parent.name, "source": p.parent.name}
        for p in sorted((root / split).glob("*/*/*.jpg"))
    ]
    if not rows:
        raise FileNotFoundError(f"no OOD images in {root / split}: run `coffeeguard ood collect`")
    return pd.DataFrame(rows)


def _fpr_at(tau: float, ood_scores: np.ndarray) -> float:
    """Share of OOD images that would be accepted (score ≤ τ)."""
    return float((ood_scores <= tau).mean())


def fit_gates(bundle_dir: Path, cfg: DataConfig, ood_root: Path, out_root: Path) -> dict:
    predictor = Predictor(bundle_dir)
    classes, name, temp = predictor.classes, bundle_dir.name, predictor.temperature
    meta = predictor.meta
    qhat = float(meta.get("conformal_qhat") or 0.0)

    # ---- in-distribution data
    splits = {s: read_split(cfg.splits_dir, s) for s in ("train", "val", "test")}
    imgs = {s: _load([cfg.processed_dir / r for r in f["image"]]) for s, f in splits.items()}
    out = {s: _run(predictor, imgs[s]) for s in imgs}
    ood_frames = {s: _ood_frame(ood_root, s) for s in ("cal", "test")}
    ood_imgs = {s: _load(list(f["path"])) for s, f in ood_frames.items()}
    ood_out = {s: _run(predictor, ood_imgs[s]) for s in ood_imgs}
    log.info(
        "Embedded train/val/test + %d / %d OOD images", len(ood_imgs["cal"]), len(ood_imgs["test"])
    )

    # ---- quality gate. Training-photo percentiles alone are far too strict (every training
    # photo is bright and sharp, but the model copes with much worse): for brightness,
    # contrast and sharpness the threshold is where the model's accuracy on corrupted val
    # photos drops below MIN_GATE_ACC; the percentile is used only if that is stricter
    # never happens. Over-exposure and leaf-colour fraction keep the percentile rule.
    manifest = pd.read_parquet(cfg.manifest_path).drop_duplicates("sha256").set_index("sha256")
    q_train = manifest.loc[splits["train"]["sha256"]]
    y_val = splits["val"]["class_id"].to_numpy()
    degr = {}
    for measure, cname in (
        ("brightness", "brightness"),
        ("contrast", "contrast"),
        ("sharpness", "gaussian_blur"),
    ):
        pct = float(np.percentile(q_train[measure], QUALITY_PCT))
        steps = []
        for sev in (1, 2, 3, 4, 5):
            cor = [
                CORRUPTIONS[cname](im, sev, np.random.default_rng(i))
                for i, im in enumerate(imgs["val"])
            ]
            acc = float((softmax(_run(predictor, cor)[0], temp).argmax(1) == y_val).mean())
            med = float(np.median([quality_metrics(im)[measure] for im in cor]))
            steps.append({"severity": sev, "accuracy": acc, "median_measure": med})
        bad = [st for st in steps if st["accuracy"] < MIN_GATE_ACC]
        if bad:
            first_bad = bad[0]
            ok = [st for st in steps if st["severity"] < first_bad["severity"]]
            last_ok = ok[-1]["median_measure"] if ok else pct
            thr = (last_ok + first_bad["median_measure"]) / 2
        else:  # never below target: reject only beyond the worst tested severity
            thr = steps[-1]["median_measure"]
        degr[measure] = {"steps": steps, "percentile_threshold": pct, "threshold": min(thr, pct)}
    qt = QualityThresholds(
        min_brightness=degr["brightness"]["threshold"],
        max_brightness=float(np.percentile(q_train["brightness"], 100 - QUALITY_PCT)),
        min_contrast=degr["contrast"]["threshold"],
        min_sharpness=degr["sharpness"]["threshold"],
        min_green_frac=float(np.percentile(q_train["green_frac"], QUALITY_PCT)),
    )
    qual = {s: [qt.check(quality_metrics(im)) for im in imgs[s]] for s in ("val", "test")}
    ood_qual = {s: [qt.check(quality_metrics(im)) for im in ood_imgs[s]] for s in ood_imgs}

    # ---- OOD scorers, fitted on train embeddings
    tr_emb, tr_y = out["train"][1].astype(np.float64), splits["train"]["class_id"].to_numpy()
    from sklearn.covariance import LedoitWolf

    means = np.stack([tr_emb[tr_y == k].mean(0) for k in range(len(classes))])
    lw = LedoitWolf().fit(tr_emb - means[tr_y])
    state = OODState(
        temperature=temp,
        knn_bank=_l2n(tr_emb).astype(np.float16),
        class_means=means.astype(np.float32),
        precision=lw.precision_.astype(np.float32),
    )
    val_lg, val_emb = out["val"]
    cal_kind = ood_frames["cal"]["kind"].to_numpy()
    comparison = {}
    for sc in SCORERS:
        s_val = score(sc, val_lg, val_emb, state)
        s_cal = score(sc, *ood_out["cal"], state)
        auc = {
            kind: float(
                roc_auc_score(
                    np.r_[np.zeros(len(s_val)), np.ones((cal_kind == kind).sum())],
                    np.r_[s_val, s_cal[cal_kind == kind]],
                )
            )
            for kind in ("near", "far")
        }
        comparison[sc] = {"cal_auroc": auc, "cal_auroc_mean": float(np.mean(list(auc.values())))}
    best = max(comparison, key=lambda k: comparison[k]["cal_auroc_mean"])
    s_val = score(best, val_lg, val_emb, state)
    # τ: the highest val-acceptance level (95-99.5%) that still lets at most
    # MAX_OOD_CAL_ACCEPT of the OOD-cal images through - decided on cal data only
    s_cal_best = score(best, *ood_out["cal"], state)
    id_accept = ID_ACCEPT
    for q in (0.95, 0.96, 0.97, 0.98, 0.99, 0.995):
        if (s_cal_best <= np.quantile(s_val, q)).mean() <= MAX_OOD_CAL_ACCEPT:
            id_accept = q
    tau_ood = float(np.quantile(s_val, id_accept))

    # ---- τ_conf on val predictions that pass both gates
    p_val = softmax(val_lg, temp)
    passed = (s_val <= tau_ood) & np.array([not q for q in qual["val"]])
    conf, correct = (
        p_val.max(1)[passed],
        (p_val.argmax(1) == splits["val"]["class_id"].to_numpy())[passed],
    )
    tau_conf = 0.5
    for t in np.unique(conf):
        if correct[conf >= t].mean() >= TARGET_ACC:
            tau_conf = float(t)
            break

    # ---- report on test / OOD-test
    test_lg, test_emb = out["test"]
    y_test = splits["test"]["class_id"].to_numpy()
    tf = ood_frames["test"]
    res: dict = {
        "bundle": name,
        "n": {
            "train": len(tr_y),
            "val": len(s_val),
            "test": len(y_test),
            "ood_cal": len(ood_frames["cal"]),
            "ood_test": len(tf),
        },
        "ood_sources": ood_frames["cal"].groupby(["kind", "source"]).size().to_dict()
        | tf.groupby(["kind", "source"]).size().to_dict(),
        "scorer_comparison": comparison,
        "scorer": best,
        "tau_ood": tau_ood,
        "tau_ood_val_acceptance": id_accept,
        "tau_conf": tau_conf,
        "quality_thresholds": qt.__dict__,
        "quality_threshold_fit": degr,
        "test": {},
    }
    res["ood_sources"] = {f"{k}/{s}": int(v) for (k, s), v in res["ood_sources"].items()}
    for sc in SCORERS:
        s_test, s_ood = score(sc, test_lg, test_emb, state), score(sc, *ood_out["test"], state)
        tau = float(np.quantile(score(sc, val_lg, val_emb, state), ID_ACCEPT))
        entry = {"id_acceptance": float((s_test <= tau).mean())}
        for kind in ("near", "far", "all"):
            m = np.ones(len(tf), bool) if kind == "all" else (tf["kind"] == kind).to_numpy()
            entry[kind] = {
                "auroc": float(
                    roc_auc_score(
                        np.r_[np.zeros(len(s_test)), np.ones(m.sum())], np.r_[s_test, s_ood[m]]
                    )
                ),
                "fpr_at_95_id": _fpr_at(tau, s_ood[m]),
            }
        res["test"][sc] = entry
    s_ood = score(best, *ood_out["test"], state)
    res["test_per_source_accepted_by_ood_gate"] = {
        src: float((s_ood[(tf["source"] == src).to_numpy()] <= tau_ood).mean())
        for src in tf["source"].unique()
    }

    # quality gate: genuine photos rejected, OOD rejected, corrupted photos rejected
    res["quality_gate"] = {
        "val_rejected": float(np.mean([bool(q) for q in qual["val"]])),
        "test_rejected": float(np.mean([bool(q) for q in qual["test"]])),
        "ood_test_rejected": float(np.mean([bool(q) for q in ood_qual["test"]])),
        "corrupted_test_rejected": {},
    }
    rng_seed = 0
    for cname in ("brightness", "contrast", "gaussian_blur", "gaussian_noise", "jpeg"):
        for sev in (3, 5):
            rej = [
                bool(
                    qt.check(
                        quality_metrics(
                            CORRUPTIONS[cname](im, sev, np.random.default_rng(rng_seed + i))
                        )
                    )
                )
                for i, im in enumerate(imgs["test"])
            ]
            res["quality_gate"]["corrupted_test_rejected"][f"{cname}_sev{sev}"] = float(
                np.mean(rej)
            )

    # ---- end-to-end decisions on test and OOD-test
    def statuses(lg, emb, quals, y=None):
        p, s = softmax(lg, temp), score(best, lg, emb, state)
        ds = [
            decide(classes, p[i], float(s[i]), quals[i], tau_ood, qhat, tau_conf)
            for i in range(len(p))
        ]
        counts = pd.Series([f"{d.status}:{d.reason}" for d in ds]).value_counts().to_dict()
        summary = {"counts": {k: int(v) for k, v in counts.items()}}
        if y is not None:
            acc = [
                d.label == classes[t] for d, t in zip(ds, y, strict=True) if d.status == "accepted"
            ]
            summary["accepted_share"] = float(np.mean([d.status == "accepted" for d in ds]))
            summary["accepted_accuracy"] = float(np.mean(acc)) if acc else None
        return summary

    res["decisions"] = {
        "test": statuses(test_lg, test_emb, qual["test"], y_test),
        "ood_test": statuses(*ood_out["test"], ood_qual["test"]),
    }

    # ---- write the gates into the bundle
    files = {}
    if best == "knn":
        np.save(bundle_dir / "ood_knn_bank.npy", state.knn_bank)
        files["ood_knn_bank.npy"] = None
    if best == "mahalanobis":
        np.save(bundle_dir / "ood_class_means.npy", state.class_means)
        np.save(bundle_dir / "ood_precision.npy", state.precision)
        files |= {"ood_class_means.npy": None, "ood_precision.npy": None}
    meta.update(
        ood={"scorer": best, "tau": tau_ood, "knn_k": state.knn_k, "id_acceptance": id_accept},
        tau_conf=tau_conf,
        quality_thresholds=qt.__dict__,
    )
    for f in files:
        meta.setdefault("files", {})[f] = sha256_file(bundle_dir / f)
    (bundle_dir / "bundle.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")

    out_dir = out_root / name
    write_json(out_dir / "ood.json", res)
    _figure(
        best,
        score(best, test_lg, test_emb, state),
        s_ood,
        tf,
        tau_ood,
        out_dir / "figures" / "ood_scores.png",
    )
    log.info(
        "%s: scorer %s, test AUROC near %.3f far %.3f, OOD accepted near %.2f far %.2f, "
        "ID accepted %.3f",
        name,
        best,
        res["test"][best]["near"]["auroc"],
        res["test"][best]["far"]["auroc"],
        res["test"][best]["near"]["fpr_at_95_id"],
        res["test"][best]["far"]["fpr_at_95_id"],
        res["test"][best]["id_acceptance"],
    )
    return res


def _figure(
    scorer: str, s_id: np.ndarray, s_ood: np.ndarray, tf: pd.DataFrame, tau: float, out: Path
):
    import matplotlib.pyplot as plt

    from coffeeguard.utils.plotting import SERIES, TEXT_MUTED, apply_style, save

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(min(s_id.min(), s_ood.min()), max(s_id.max(), s_ood.max()), 40)
    ax.hist(s_id, bins, alpha=0.7, color=SERIES[0], label="coffee leaves (test)")
    for kind, color in (("near", SERIES[1]), ("far", SERIES[2])):
        m = (tf["kind"] == kind).to_numpy()
        ax.hist(s_ood[m], bins, alpha=0.6, color=color, label=f"{kind}-OOD (test sources)")
    ax.axvline(tau, color=TEXT_MUTED, linestyle="--", linewidth=1.2)
    ax.annotate(
        "reject →",
        (tau, ax.get_ylim()[1] * 0.9),
        xytext=(4, 0),
        textcoords="offset points",
        color=TEXT_MUTED,
    )
    ax.set_xlabel(f"OOD score ({scorer}); higher = less like the training leaves")
    ax.set_ylabel("images")
    ax.legend()
    fig.tight_layout()
    save(fig, out)
