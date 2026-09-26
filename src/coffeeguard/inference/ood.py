"""Out-of-distribution scores from one ONNX forward pass (NumPy only).

Higher score = more likely out of distribution. All four use outputs the served model
already produces (logits, pooled embedding):

- ``msp``: 1 - max softmax probability (temperature-scaled)
- ``energy``: -T · logsumexp(logits / T) (Liu et al., 2020)
- ``mahalanobis``: min over classes of the Mahalanobis distance of the embedding to the
  class mean, shared (shrunk) covariance (Lee et al., 2018)
- ``knn``: 1 - cosine similarity to the k-th nearest training embedding (Sun et al., 2022)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coffeeguard.inference.predictor import softmax

SCORERS = ("msp", "energy", "mahalanobis", "knn")


def _l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


@dataclass
class OODState:
    """What the served scorer needs; saved in the bundle (``ood_*.npy``)."""

    temperature: float = 1.0
    knn_bank: np.ndarray | None = None  # (N, D) L2-normalised training embeddings
    knn_k: int = 10
    class_means: np.ndarray | None = None  # (C, D)
    precision: np.ndarray | None = None  # (D, D) inverse shared covariance


def score(name: str, logits: np.ndarray, emb: np.ndarray, st: OODState) -> np.ndarray:
    """``(N, C)`` logits and ``(N, D)`` embeddings → ``(N,)`` OOD scores."""
    if name == "msp":
        return 1.0 - softmax(logits, st.temperature).max(1)
    if name == "energy":
        z = logits / st.temperature
        m = z.max(1, keepdims=True)
        return -st.temperature * (m[:, 0] + np.log(np.exp(z - m).sum(1)))
    if name == "mahalanobis":
        assert st.class_means is not None and st.precision is not None
        d = emb[:, None, :] - st.class_means[None].astype(np.float32)  # (N, C, D)
        m = np.einsum("ncd,de,nce->nc", d, st.precision.astype(np.float32), d)
        return m.min(1)
    if name == "knn":
        assert st.knn_bank is not None
        sims = _l2n(emb.astype(np.float32)) @ st.knn_bank.astype(np.float32).T  # (N, M)
        k = min(st.knn_k, sims.shape[1])
        kth = -np.partition(-sims, k - 1, axis=1)[:, k - 1]
        return 1.0 - kth
    raise ValueError(f"unknown OOD scorer {name!r}; choose from {SCORERS}")


def load_state(bundle_dir, meta: dict) -> OODState:
    """Rebuild the fitted OOD state from a bundle (written by ``coffeeguard ood fit``)."""
    from pathlib import Path

    d = Path(bundle_dir)
    cfg = meta.get("ood") or {}
    st = OODState(
        temperature=float(meta.get("temperature") or 1.0), knn_k=int(cfg.get("knn_k", 10))
    )
    if (d / "ood_knn_bank.npy").exists():
        st.knn_bank = np.load(d / "ood_knn_bank.npy")
    if (d / "ood_class_means.npy").exists():
        st.class_means = np.load(d / "ood_class_means.npy")
        st.precision = np.load(d / "ood_precision.npy")
    return st
