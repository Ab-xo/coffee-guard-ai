"""``coffeeguard data prepare``: scan → embed → group → split, each stage cached.

Outputs:
- ``data/manifests/scan.parquet``       per-image scan results (stage 1)
- ``data/manifests/embeddings.npz``     DINOv2 embeddings keyed by SHA-256 (stage 2)
- ``data/manifests/manifest.parquet``   full manifest with status, group, split
- ``data/splits/{train,val,test}.csv``  what training reads (committed to git)
- ``data/splits/split_info.json``       fingerprint, balance table, dedup summary
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from coffeeguard.config import DataConfig, dump_config
from coffeeguard.data.dedup import assign_groups, name_pattern_mask
from coffeeguard.data.scan import scan_dataset
from coffeeguard.data.split import assign_splits, balance_table, check_no_leakage, split_frames
from coffeeguard.utils.hashing import fingerprint
from coffeeguard.utils.io import write_json
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)

STAGES = ("scan", "embed", "group", "split")


def data_fingerprint(frames: dict[str, pd.DataFrame], classes: list[str]) -> str:
    items = [f"classes:{'|'.join(classes)}"]
    for name, frame in frames.items():
        items += [
            f"{name}:{s}:{c}" for s, c in zip(frame["sha256"], frame["class_id"], strict=True)
        ]
    return fingerprint(items)


def prepare(
    cfg: DataConfig,
    stages: Sequence[str] = STAGES,
    workers: int | None = None,
    force: bool = False,
) -> dict:
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {STAGES}")

    manifests = cfg.manifest_path.parent
    manifests.mkdir(parents=True, exist_ok=True)
    scan_path = manifests / "scan.parquet"
    emb_path = cfg.embeddings_path.with_suffix(".npz")

    # 1. scan
    if "scan" in stages and (force or not scan_path.exists()):
        scan = scan_dataset(cfg, workers=workers)
        scan.to_parquet(scan_path, index=False)
        log.info("Wrote %s (%d rows)", scan_path, len(scan))
    elif scan_path.exists():
        scan = pd.read_parquet(scan_path)
        log.info("Using cached scan: %s (%d rows)", scan_path.name, len(scan))
    else:
        raise FileNotFoundError(f"{scan_path} missing: run with the 'scan' stage")

    # 2. embed (valid, labelled images only)
    store: dict = {}
    if "embed" in stages or emb_path.exists():
        from coffeeguard.embeddings.extract import extract_embeddings, load_store

        if "embed" in stages:
            wanted = scan["is_valid"] & ~name_pattern_mask(scan, cfg.exclude_name_patterns)
            valid = scan[wanted].rename(columns={"cache_path": "image"})
            store = extract_embeddings(
                valid[["sha256", "image"]], cfg.processed_dir, emb_path, cfg.embed_model
            )
        else:
            store = load_store(emb_path)

    summary: dict = {"n_files": len(scan)}
    # 3. group
    if "group" in stages or "split" in stages:
        from coffeeguard.embeddings.extract import aligned_embeddings

        emb = aligned_embeddings(scan, store)
        if emb is None:
            log.warning("No embeddings: grouping uses exact + perceptual hashes only")
        manifest, dedup_summary = assign_groups(
            scan,
            phash_max_distance=cfg.phash_max_distance,
            embeddings=emb,
            embed_similarity=cfg.embed_similarity,
            excluded_sha={e.sha256 for e in cfg.exclude},
            exclude_name_patterns=cfg.exclude_name_patterns,
        )
        summary["dedup"] = dedup_summary
        summary["used_embeddings"] = emb is not None

    # 4. split
    if "split" in stages:
        manifest = assign_splits(manifest, cfg.split)
        frames = split_frames(manifest)
        check_no_leakage(frames)
        cfg.splits_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            frame.to_csv(cfg.splits_dir / f"{name}.csv", index=False)
        table = balance_table(frames, cfg.classes)
        summary["fingerprint"] = data_fingerprint(frames, cfg.classes)
        summary["balance"] = table.to_dict(orient="index")
        summary["invalid_issues"] = (
            manifest.loc[~manifest["is_valid"], "issues"]
            .str.split(";")
            .explode()
            .str.split(":")
            .str[0]
            .value_counts()
            .to_dict()
        )
        manifest.to_parquet(cfg.manifest_path, index=False)
        write_json(cfg.splits_dir / "split_info.json", summary)
        dump_config(cfg, cfg.splits_dir / "data_config.yaml")
        log.info("Split balance:\n%s", table.to_string())
        log.info("Data fingerprint: %s", summary["fingerprint"])
    return summary
