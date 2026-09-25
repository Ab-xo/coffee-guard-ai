"""Stage 2 of ``data prepare``: duplicates, conflicts and leakage groups.

Three kinds of links between images are found:

1. **Exact** — identical SHA-256. Within one class, only the first copy is kept
   (``status=duplicate`` for the rest). Across classes, every copy is a
   ``conflict`` and removed (we cannot know which label is right).
2. **Near** — perceptual-hash Hamming distance ≤ ``phash_max_distance``: the same
   photo re-encoded, resized or lightly edited. Cross-class near-duplicates are
   also conflicts.
3. **Similar** — cosine similarity of DINOv2 embeddings ≥ ``embed_similarity``:
   typically a re-shot of the same leaf. These are *kept* but share a group.

Connected components over near + similar links become ``group_id``. The split keeps
each group inside one split, which is what prevents train/test leakage.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd

from coffeeguard.utils.log import get_logger

log = get_logger(__name__)


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def components(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))])


def phash_to_int(hex_hashes: Iterable[str]) -> np.ndarray:
    return np.array([int(h, 16) for h in hex_hashes], dtype=np.uint64)


_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):  # NumPy >= 2.0
        return np.bitwise_count(x)
    return _POPCOUNT8[x[..., None].view(np.uint8)].sum(axis=-1, dtype=np.uint8)


def phash_pairs(hashes: np.ndarray, max_distance: int, block: int = 512) -> np.ndarray:
    """All index pairs (i < j) whose 64-bit hashes differ in ≤ ``max_distance`` bits."""
    n = len(hashes)
    out: list[np.ndarray] = []
    for start in range(0, n, block):
        chunk = hashes[start : start + block]
        dist = _popcount64(chunk[:, None] ^ hashes[None, :])
        i, j = np.nonzero(dist <= max_distance)
        i = i + start
        keep = i < j
        out.append(np.stack([i[keep], j[keep]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 2), dtype=np.int64)


def phash_pairs_d4(
    identity: np.ndarray, variants: np.ndarray, max_distance: int, block: int = 128
) -> np.ndarray:
    """Pairs (i < j) where image i's hash is within ``max_distance`` bits of *any*
    rotation/flip variant of image j — catches mirrored / 90°-rotated copies.

    ``identity``: ``(N,)`` uint64; ``variants``: ``(N, 8)`` uint64.
    """
    n = len(identity)
    out: list[np.ndarray] = []
    for start in range(0, n, block):
        q = identity[start : start + block]
        dist = _popcount64(q[:, None, None] ^ variants[None, :, :]).min(axis=2)
        i, j = np.nonzero(dist <= max_distance)
        i = i + start
        keep = i != j
        pairs = np.stack([np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])], axis=1)
        out.append(pairs)
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(out), axis=0)


def embedding_pairs(emb: np.ndarray, threshold: float, block: int = 512) -> np.ndarray:
    """All index pairs (i < j) with cosine similarity ≥ ``threshold``."""
    emb = emb.astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
    n = len(emb)
    out: list[np.ndarray] = []
    for start in range(0, n, block):
        sim = emb[start : start + block] @ emb.T
        i, j = np.nonzero(sim >= threshold)
        i = i + start
        keep = i < j
        out.append(np.stack([i[keep], j[keep]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 2), dtype=np.int64)


def name_pattern_mask(df: pd.DataFrame, patterns: list[str]) -> pd.Series:
    """True for rows whose file name matches any of the regexes."""
    names = df["path"].str.rsplit("/", n=1).str[-1]
    mask = pd.Series(False, index=df.index)
    for pattern in patterns:
        mask |= names.str.contains(pattern, regex=True)
    return mask


def assign_groups(
    df: pd.DataFrame,
    phash_max_distance: int,
    embeddings: np.ndarray | None = None,
    embed_similarity: float = 0.95,
    excluded_sha: set[str] | None = None,
    exclude_name_patterns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Add ``status``, ``duplicate_of`` and ``group_id`` columns.

    ``embeddings`` (optional) must be row-aligned with ``df``. Returns the updated
    frame and a summary dict for the data-quality report.
    """
    df = df.copy()
    df["status"] = np.where(df["is_valid"], "ok", "invalid")
    df["duplicate_of"] = ""
    summary: dict = {}

    # --- file-name pattern exclusions (e.g. pre-generated augmentations)
    if exclude_name_patterns:
        mask = name_pattern_mask(df, exclude_name_patterns) & (df["status"] == "ok")
        df.loc[mask, "status"] = "excluded_pattern"
        summary["excluded_by_name_pattern"] = int(mask.sum())

    # --- manual exclusions from config
    if excluded_sha:
        mask = df["sha256"].isin(excluded_sha) & (df["status"] == "ok")
        df.loc[mask, "status"] = "excluded"
        summary["excluded_by_config"] = int(mask.sum())

    # --- 1. exact duplicates
    ok = df["status"] == "ok"
    exact_conflicts = 0
    exact_dupes = 0
    for _sha, idx in df[ok].groupby("sha256").groups.items():
        if len(idx) < 2:
            continue
        rows = df.loc[idx].sort_values("path")
        if rows["label"].nunique() > 1:
            df.loc[rows.index, "status"] = "conflict"
            exact_conflicts += len(rows)
        else:
            keeper = rows.index[0]
            dupes = rows.index[1:]
            df.loc[dupes, "status"] = "duplicate"
            df.loc[dupes, "duplicate_of"] = df.at[keeper, "path"]
            exact_dupes += len(dupes)
    summary["exact_duplicates_removed"] = exact_dupes
    summary["exact_conflicts_removed"] = exact_conflicts

    # --- 2 + 3. near duplicates and embedding similarity among kept images
    kept = df.index[df["status"] == "ok"].to_numpy()
    uf = UnionFind(len(kept))
    labels = df.loc[kept, "label"].to_numpy()

    if "phash_d4" in df.columns and df.loc[kept, "phash_d4"].notna().all():
        variants = np.stack([phash_to_int(v.split(",")) for v in df.loc[kept, "phash_d4"]])
        near = phash_pairs_d4(variants[:, 0], variants, phash_max_distance)
    else:
        near = phash_pairs(phash_to_int(df.loc[kept, "phash"]), phash_max_distance)
    cross = labels[near[:, 0]] != labels[near[:, 1]] if len(near) else np.array([], bool)
    conflict_pos = np.unique(near[cross].ravel()) if cross.any() else np.array([], int)
    for a, b in near[~cross] if len(near) else []:
        uf.union(int(a), int(b))
    summary["near_duplicate_pairs"] = len(near)
    summary["near_duplicate_cross_class_pairs"] = int(cross.sum()) if len(near) else 0

    if embeddings is not None:
        emb_kept = embeddings[kept]
        sim = embedding_pairs(emb_kept, embed_similarity)
        # Only same-class pairs are linked: one leaf cannot carry two labels, so a
        # cross-class match means "same photo setup", not "same leaf". Linking those
        # chains unrelated leaves into giant groups (observed: one group of 303).
        same = labels[sim[:, 0]] == labels[sim[:, 1]] if len(sim) else np.array([], bool)
        for a, b in sim[same] if len(sim) else []:
            uf.union(int(a), int(b))
        summary["embedding_similar_pairs"] = len(sim)
        summary["embedding_similar_cross_class_pairs_ignored"] = int((~same).sum())

    comp = uf.components()
    df["group_id"] = -1
    df.loc[kept, "group_id"] = comp

    if len(conflict_pos):
        conflict_idx = kept[conflict_pos]
        df.loc[conflict_idx, "status"] = "conflict"
        df.loc[conflict_idx, "group_id"] = -1
    summary["near_conflicts_removed"] = len(conflict_pos)

    # renumber groups 0..G-1 by first appearance for readability
    ok_groups = df.loc[df["status"] == "ok", "group_id"]
    mapping = {g: i for i, g in enumerate(pd.unique(ok_groups))}
    df.loc[df["status"] == "ok", "group_id"] = ok_groups.map(mapping)

    sizes = Counter(df.loc[df["status"] == "ok", "group_id"])
    size_hist = Counter(sizes.values())
    summary["groups"] = len(sizes)
    summary["largest_group"] = max(sizes.values()) if sizes else 0
    summary["group_size_histogram"] = {str(k): v for k, v in sorted(size_hist.items())}
    summary["status_counts"] = df["status"].value_counts().to_dict()
    log.info("Dedup summary: %s", {k: v for k, v in summary.items() if k != "group_size_histogram"})
    return df, summary
