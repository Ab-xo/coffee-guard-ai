"""Train on a free Kaggle GPU from the command line.

``coffeeguard remote upload-data`` — upload ``data/processed`` (the 384 px cache) as the
private dataset ``<user>/coffeeguard-processed`` (only needed when the data changes).

``coffeeguard remote train`` —
1. build the project wheel (``uv build``);
2. upload wheel + ``configs/`` + ``data/splits/`` as the private dataset
   ``<user>/coffeeguard-code`` (new version each time);
3. render ``kaggle/run_template.py`` with the job list and push it as a private GPU
   script kernel ``<user>/coffeeguard-train``;
4. optionally wait for it and download ``runs/`` into ``runs/``.

Nothing needs to be pushed to GitHub, and every run is still an ordinary
``coffeeguard train`` invocation, so results are identical in shape to local runs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from coffeeguard.config import DataConfig
from coffeeguard.utils.log import get_logger
from coffeeguard.utils.paths import project_root, resolve

log = get_logger(__name__)

BUILD = Path("kaggle/_build")
CODE_SLUG = "coffeeguard-code"
DATA_SLUG = "coffeeguard-processed"
KERNEL_SLUG = "coffeeguard-train"


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def kaggle_username(api=None) -> str:
    api = api or _api()
    user = api.get_config_value("username")
    if not user:
        raise RuntimeError("Could not determine Kaggle username from credentials")
    return user


def _dataset_exists(api, ref: str) -> bool:
    try:
        api.dataset_status(ref)
        return True
    except Exception:
        return False


def _publish_dataset(api, folder: Path, ref: str, title: str, message: str) -> None:
    meta = {"title": title, "id": ref, "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")
    if _dataset_exists(api, ref):
        log.info("Uploading new version of %s", ref)
        api.dataset_create_version(str(folder), version_notes=message, dir_mode="zip", quiet=False)
    else:
        log.info("Creating private dataset %s", ref)
        api.dataset_create_new(str(folder), public=False, dir_mode="zip", quiet=False)


def _wait_dataset_ready(api, ref: str, timeout: int = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            status = str(api.dataset_status(ref)).lower()
        except Exception as exc:  # just created; not visible yet
            status = f"unknown ({exc})"
        if "ready" in status:
            return
        log.info("Waiting for dataset %s (%s)", ref, status)
        time.sleep(15)
    raise TimeoutError(f"dataset {ref} not ready after {timeout}s")


def upload_data(data_cfg: DataConfig) -> str:
    """Upload the processed image cache as ``<user>/coffeeguard-processed``."""
    api = _api()
    ref = f"{kaggle_username(api)}/{DATA_SLUG}"
    stage = resolve(BUILD) / "data"
    if stage.exists():
        shutil.rmtree(stage)
    target = stage / "processed"
    shutil.copytree(data_cfg.processed_dir, target)
    (target / "PROCESSED_MARKER").write_text("coffeeguard processed cache\n", "utf-8")
    info = data_cfg.splits_dir / "split_info.json"
    fp = json.loads(info.read_text("utf-8")).get("fingerprint") if info.exists() else "unknown"
    _publish_dataset(
        api, stage, ref, "CoffeeGuard processed images", f"processed cache, data fingerprint {fp}"
    )
    return ref


def _build_code_bundle() -> Path:
    root = project_root()
    stage = resolve(BUILD) / "code"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    subprocess.run(["uv", "build", "--wheel", "-o", str(stage)], cwd=root, check=True)
    shutil.copytree(root / "configs", stage / "configs")
    shutil.copytree(root / "data" / "splits", stage / "data" / "splits")
    return stage


def push_training(jobs: list[dict], gpu: str = "NvidiaTeslaT4") -> str:
    """Upload code and push the GPU kernel. Returns the kernel ref."""
    api = _api()
    user = kaggle_username(api)
    code_ref, data_ref, kernel_ref = (
        f"{user}/{CODE_SLUG}",
        f"{user}/{DATA_SLUG}",
        f"{user}/{KERNEL_SLUG}",
    )
    if not _dataset_exists(api, data_ref):
        raise RuntimeError(f"{data_ref} missing: run `coffeeguard remote upload-data` first")

    code_stage = _build_code_bundle()
    _publish_dataset(
        api,
        code_stage,
        code_ref,
        "CoffeeGuard code",
        f"jobs: {', '.join(j['config'] for j in jobs)}",
    )
    _wait_dataset_ready(api, code_ref)

    kernel_dir = resolve(BUILD) / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True)
    template = (project_root() / "kaggle" / "run_template.py").read_text("utf-8")
    (kernel_dir / "run.py").write_text(template.replace("__JOBS__", json.dumps(jobs)), "utf-8")
    meta = {
        "id": kernel_ref,
        "title": KERNEL_SLUG,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": gpu,
        "dataset_sources": [code_ref, data_ref],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")
    log.info("Pushing kernel %s with %d job(s)", kernel_ref, len(jobs))
    api.kernels_push(str(kernel_dir))
    return kernel_ref


def wait_and_fetch(kernel_ref: str, poll: int = 60, timeout: int = 6 * 3600) -> Path:
    """Poll the kernel until it finishes, then download its output into runs/."""
    api = _api()
    t0 = time.time()
    while True:
        status = str(api.kernels_status(kernel_ref)).lower()
        log.info("Kernel %s: %s (%.0f min)", kernel_ref, status, (time.time() - t0) / 60)
        if "complete" in status:
            break
        if "error" in status or "cancel" in status:
            raise RuntimeError(
                f"Kernel {kernel_ref} ended with status {status}; "
                f"see https://www.kaggle.com/code/{kernel_ref}"
            )
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Kernel {kernel_ref} still running after {timeout}s")
        time.sleep(poll)
    out = resolve("runs") / "_kaggle" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel_ref, path=str(out))
    runs_src = out / "runs"
    if runs_src.exists():  # move run dirs next to local runs
        for run in runs_src.iterdir():
            shutil.move(str(run), str(resolve("runs") / run.name))
    log.info("Fetched kernel output into %s", out)
    return out
