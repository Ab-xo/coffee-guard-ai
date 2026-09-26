"""Run directories and environment capture, so every result traces back to code + data."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from coffeeguard.utils.paths import project_root, resolve

_TRACKED_PACKAGES = (
    "coffee-guard-ai",
    "numpy",
    "pillow",
    "torch",
    "torchvision",
    "timm",
    "scikit-learn",
    "onnx",
    "onnxruntime",
)


def create_run_dir(name: str, root: str | Path = "runs") -> Path:
    """Create ``runs/<YYYYmmdd-HHMMSS>-<name>`` and return it."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "run"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resolve(root) / f"{stamp}-{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def git_info() -> dict[str, Any]:
    """Current commit SHA and whether the working tree has uncommitted changes.

    Remote runs (Kaggle) have no git checkout; the launcher passes the local state in
    ``COFFEEGUARD_GIT_SHA`` / ``COFFEEGUARD_GIT_DIRTY`` instead.
    """
    if sha := os.environ.get("COFFEEGUARD_GIT_SHA"):
        return {"sha": sha, "dirty": os.environ.get("COFFEEGUARD_GIT_DIRTY") == "1"}

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=project_root(),
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def capture_env() -> dict[str, Any]:
    """Versions, platform, accelerator and git state for a run's ``env.json``."""
    packages = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            packages[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            continue
    env: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "git": git_info(),
    }
    try:
        import torch

        env["cuda"] = torch.cuda.is_available()
        if env["cuda"]:
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env
