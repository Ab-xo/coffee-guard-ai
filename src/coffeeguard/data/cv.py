"""K-fold group cross-validation splits over train + val (the test split stays sealed).

``StratifiedGroupKFold`` keeps every duplicate/re-shot group in one fold and keeps class
proportions per fold. Fold *i* uses fold *i* as validation and the rest for training.
Writes ``data/splits_cv/fold{i}/{train,val}.csv`` (+ ``split_info.json`` with the data
fingerprint) and ``configs/cv/data_fold{i}.yaml`` (the data config pointing at them), so
every fold is an ordinary training run: ``--set data_config=configs/cv/data_fold{i}.yaml``.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from coffeeguard.config import DataConfig, dump_config
from coffeeguard.data.dataset import read_split
from coffeeguard.utils.io import read_json, write_json
from coffeeguard.utils.paths import resolve

COLUMNS = ["image", "label", "class_id", "group_id", "sha256"]


def make_cv_folds(cfg: DataConfig, k: int = 5, seed: int = 42) -> list[dict]:
    pool = pd.concat([read_split(cfg.splits_dir, s) for s in ("train", "val")], ignore_index=True)
    pool = pool[COLUMNS]
    fingerprint = read_json(cfg.splits_dir / "split_info.json").get("fingerprint")
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out_root, cfg_root = resolve("data/splits_cv"), resolve("configs/cv")
    summary = []
    for i, (tr, va) in enumerate(skf.split(pool, pool["class_id"], pool["group_id"])):
        train, val = pool.iloc[tr], pool.iloc[va]
        assert not set(train["group_id"]) & set(val["group_id"]), "group leak between folds"
        assert not set(train["sha256"]) & set(val["sha256"]), "image leak between folds"
        d = out_root / f"fold{i}"
        d.mkdir(parents=True, exist_ok=True)
        train.to_csv(d / "train.csv", index=False)
        val.to_csv(d / "val.csv", index=False)
        info = {
            "fingerprint": fingerprint,
            "fold": i,
            "k": k,
            "seed": seed,
            "n_train": len(train),
            "n_val": len(val),
            "val_class_counts": val["label"].value_counts().to_dict(),
        }
        write_json(d / "split_info.json", info)
        fold_cfg = cfg.model_copy(update={"splits_dir": d})
        dump_config(fold_cfg, cfg_root / f"data_fold{i}.yaml")
        summary.append(info)
    return summary
