"""Project-root discovery so relative paths in configs work from any working directory."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the repository root.

    Order: ``COFFEEGUARD_ROOT`` env var, then the nearest parent of the current working
    directory that contains ``pyproject.toml``, then the current working directory
    (e.g. on Kaggle, where the package is installed as a wheel).
    """
    env = os.environ.get("COFFEEGUARD_ROOT")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return cwd


def resolve(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def portable_path(path: str | Path) -> str:
    """Posix path relative to the project root when inside it, else absolute.

    Used when writing configs/reports to git so they don't contain machine paths.
    """
    p = Path(path)
    try:
        return p.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return p.as_posix()
