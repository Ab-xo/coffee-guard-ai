"""Download and extract the Kaggle dataset into ``data/raw``."""

from __future__ import annotations

import zipfile
from pathlib import Path

from coffeeguard.config import DataConfig
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)


def download_dataset(cfg: DataConfig, force: bool = False) -> Path:
    """Fetch the dataset zip (skipped if already present) and extract it.

    Authentication uses the Kaggle CLI's normal lookup: ``KAGGLE_API_TOKEN``,
    ``~/.kaggle/access_token(.txt)`` or ``~/.kaggle/kaggle.json``.
    """
    raw = cfg.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / f"{cfg.kaggle_dataset.split('/')[-1]}.zip"

    if force or not zip_path.exists():
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        log.info("Downloading %s -> %s", cfg.kaggle_dataset, raw)
        api.dataset_download_files(cfg.kaggle_dataset, path=str(raw), unzip=False, quiet=False)
    else:
        log.info("Zip already present: %s", zip_path)

    marker = raw / ".extracted"
    if force or not marker.exists():
        log.info("Extracting %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw)
        marker.write_text(zip_path.name, "utf-8")
    return raw
