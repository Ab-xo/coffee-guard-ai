"""Seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    ``deterministic=True`` also forces deterministic cuDNN kernels, which is slower;
    use it for debugging, not for normal training runs.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the legacy global RNG other libraries use
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
