"""Small, committable data files for the Streamlit analysis pages (``artifacts/app/``).

Training histories live in git-ignored ``runs/`` and per-image predictions in git-ignored
parquet files; the web app (and its Docker image) only gets ``artifacts/``. This collects
what the pages plot:

- ``histories.json``: per-epoch training history of every compared model and CV fold;
- ``predictions_test.csv``: the release model's calibrated test predictions per image;
- ``cv.json``: k-fold cross-validation results of the deployed recipe (if run).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from coffeeguard.utils.io import read_json, write_json
from coffeeguard.utils.log import get_logger
from coffeeguard.utils.paths import resolve

log = get_logger(__name__)
KEEP = (
    "stage",
    "epoch",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_macro_f1",
    "val_accuracy",
    "ema_val_macro_f1",
    "selected",
    "seconds",
)


def _history(run_dir: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text("utf-8").splitlines()
        if line
    ]
    return [{k: r[k] for k in KEEP if k in r} for r in rows]


def build_app_data(bundles: list[str], main: str, out: Path | None = None) -> dict:
    out = out or resolve("artifacts/app")
    out.mkdir(parents=True, exist_ok=True)
    runs = resolve("runs")

    histories: dict = {}
    for b in bundles:
        meta = read_json(resolve("artifacts/models") / b / "bundle.json")
        run = runs / meta["source_run"]
        if (run / "history.jsonl").exists():
            histories[b] = {"model": meta["model"], "run": run.name, "rows": _history(run)}

    folds = []
    for run in sorted(runs.glob("*-cv[0-9]-s*")):
        if not (run / "summary.json").exists():
            continue
        s = read_json(run / "summary.json")
        fold = int(run.name.split("-cv")[1].split("-")[0])
        histories[f"cv{fold}"] = {"model": "cv", "run": run.name, "rows": _history(run)}
        env = read_json(run / "env.json")
        folds.append(
            {
                "fold": fold,
                "run": run.name,
                "val_macro_f1": s["best_val_macro_f1"],
                "val_accuracy": s["val"]["accuracy"],
                "best": f"{s['stage']} epoch {s['epoch']} ({s['weights']})",
                "minutes": s["minutes"],
                "gpu": env.get("gpu"),
                "n_val": read_json(resolve(f"data/splits_cv/fold{fold}/split_info.json"))["n_val"],
            }
        )
    write_json(out / "histories.json", histories)
    if folds:
        folds.sort(key=lambda f: f["fold"])
        f1 = pd.Series([f["val_macro_f1"] for f in folds])
        acc = pd.Series([f["val_accuracy"] for f in folds])
        write_json(
            out / "cv.json",
            {
                "recipe": "configs/train/effnetv2_b0_bgswap.yaml",
                "k": len(folds),
                "folds": folds,
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_std": float(f1.std(ddof=1)),
                "accuracy_mean": float(acc.mean()),
                "accuracy_std": float(acc.std(ddof=1)),
            },
        )

    pred = pd.read_parquet(resolve("artifacts/eval") / main / "predictions_test.parquet")
    cols = [
        "image",
        "label",
        "pred_label",
        "confidence",
        "correct",
        "set_size",
        *[c for c in pred.columns if c.startswith("p_")],
        "sharpness",
        "brightness",
        "contrast",
        "green_frac",
    ]
    pred[cols].round(5).to_csv(out / "predictions_test.csv", index=False)
    log.info(
        "app data: %d histories, %d CV folds, %d test predictions",
        len(histories),
        len(folds),
        len(pred),
    )
    return {"histories": list(histories), "cv_folds": len(folds), "predictions": len(pred)}
