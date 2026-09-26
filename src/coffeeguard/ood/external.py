"""Coffee leaves from *other* datasets: does the served pipeline still accept and classify them?

The OOD gate is fitted against other plants' leaves and non-leaf images, and its
in-distribution side is the Kaggle dataset itself - so a gate that only recognises
*this dataset's photos* would still score well. This check uses coffee-leaf photos
from two independent public datasets, used for evaluation only (never for fitting):

- **BRACOL** (Esgario et al., 2020; Brazil, arabica, one leaf on a white/grey background,
  2048×1024 phone photos, labelled with the predominant stress). The Kaggle dataset turned
  out to contain BRACOL photos, so every image with a perceptual-hash near-duplicate
  (any of the 8 flips/rotations, Hamming distance ≤ ``SEEN_MAX_DIST``) in our data is
  marked ``seen`` and left out of the "unseen" numbers.
- **RoCoLe** (Parraga-Alava et al., 2019; Ecuador, robusta, leaves on the plant). No
  disease labels are used - only whether the gate accepts them.

BRACOL is also re-shot under the conditions of photos people actually send: the leaf
standing vertically, and small re-compressed copies (from a web page or a chat app).
Stored in ``data/external/`` (git-ignored); the report goes to
``artifacts/ood/<bundle>/external.json``.
"""

from __future__ import annotations

import io
import os
import random
import time
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from coffeeguard.inference.ood import score
from coffeeguard.inference.pipeline import Pipeline
from coffeeguard.inference.predictor import softmax
from coffeeguard.inference.quality import to_rgb
from coffeeguard.utils.io import write_json
from coffeeguard.utils.log import get_logger

log = get_logger(__name__)
HF = "https://huggingface.co/datasets"
BRACOL_REPO = "luisangelico/bracol"  # mirror of github.com/esgario/lara2018
ROCOLE_REPO = "bhugxer/RoCoLe-Coffee"
BRACOL_CLASSES = {0: "Healthy", 2: "Leaf Rust", 3: "Phoma", 4: "Cercospora"}  # 1 miner, 5 other
SEEN_MAX_DIST = 8


def _download(url: str, dest: Path, tries: int = 4) -> bool:
    for attempt in range(tries):
        if dest.exists() and dest.stat().st_size > 1000:
            return True
        try:
            part = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, part)
            os.replace(part, dest)
        except OSError as e:
            log.warning("download failed (%s), retry %d: %s", e, attempt + 1, url)
            time.sleep(3 * (attempt + 1))
    return dest.exists()


def _fetch_all(jobs: list[tuple[str, Path]]) -> int:
    with ThreadPoolExecutor(4) as ex:  # more parallel requests time out on this connection
        return sum(ex.map(lambda j: _download(*j), jobs))


def _nearest_seen(paths: list[Path], manifest_path: Path) -> np.ndarray:
    """Smallest perceptual-hash distance of each image to any of our photos (8 variants)."""
    from coffeeguard.data.dedup import phash_to_int
    from coffeeguard.data.scan import dihedral_phashes

    m = pd.read_parquet(manifest_path, columns=["phash_d4"]).dropna()
    bank = np.stack([phash_to_int(v.split(",")) for v in m["phash_d4"]])  # (N, 8)
    popcount = np.vectorize(lambda v: int(v).bit_count())
    out = []
    for p in paths:
        with Image.open(p) as im:
            h = phash_to_int(dihedral_phashes(im.convert("RGB"))[:1])[0]
        out.append(int(popcount(np.bitwise_xor(bank, h)).min()))
    return np.array(out)


def collect_external(
    out: Path, manifest_path: Path, per_class: int = 80, rocole_n: int = 150, seed: int = 0
) -> dict:
    """Download a class-balanced BRACOL sample and a random RoCoLe sample (originals)."""
    bdir, rdir = out / "bracol", out / "rocole"
    (bdir / "images").mkdir(parents=True, exist_ok=True)
    (rdir / "images").mkdir(parents=True, exist_ok=True)

    _download(f"{HF}/{BRACOL_REPO}/resolve/main/dataset.csv", bdir / "dataset.csv")
    meta = pd.read_csv(bdir / "dataset.csv")
    meta = meta[meta["predominant_stress"].isin(BRACOL_CLASSES)]
    sample = pd.concat(
        g.sample(min(len(g), per_class), random_state=seed)
        for _, g in meta.groupby("predominant_stress")
    )
    sample["label"] = sample["predominant_stress"].map(BRACOL_CLASSES)
    paths = [bdir / "images" / f"{i}.jpg" for i in sample["id"]]
    got = _fetch_all([(f"{HF}/{BRACOL_REPO}/resolve/main/images/{p.name}", p) for p in paths])
    sample["nearest_seen_dist"] = _nearest_seen(paths, manifest_path)
    sample["seen"] = sample["nearest_seen_dist"] <= SEEN_MAX_DIST
    sample[["id", "label", "severity", "nearest_seen_dist", "seen"]].to_csv(
        bdir / "sample.csv", index=False
    )

    import json

    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{ROCOLE_REPO}") as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]]
    files = [f for f in files if f.lower().endswith(".jpg")]
    pick = random.Random(seed).sample(files, min(rocole_n, len(files)))
    got_r = _fetch_all(
        [(f"{HF}/{ROCOLE_REPO}/resolve/main/{f}", rdir / "images" / f) for f in pick]
    )
    summary = {
        "bracol": {
            "downloaded": got,
            "per_class": sample["label"].value_counts().to_dict(),
            "seen_per_class": sample[sample["seen"]]["label"].value_counts().to_dict(),
        },
        "rocole": {"downloaded": got_r},
    }
    log.info("external sets: %s", summary)
    return summary


def small_jpeg(img: Image.Image, side: int, quality: int = 35) -> Image.Image:
    """A small, heavily re-compressed copy (web page / chat app)."""
    im = img.copy()
    im.thumbnail((side, side), Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _vertical(im: Image.Image) -> Image.Image:
    return im.rotate(90, expand=True)


CONDITIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "as photographed": lambda im: im,
    "leaf vertical": _vertical,
    "300 px JPEG": lambda im: small_jpeg(im, 300),
    "150 px JPEG": lambda im: small_jpeg(im, 150),
    "vertical + 300 px JPEG": lambda im: small_jpeg(_vertical(im), 300),
}


def _scores(pipe: Pipeline, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
    """KNN/OOD scores and calibrated probabilities (no gates)."""
    lg, em = [], []
    for i in range(0, len(images), 32):
        x = np.stack([pipe.predictor.prepare(im)[0] for im in images[i : i + 32]])
        a, b, _ = pipe.predictor.run(x)
        lg.append(a)
        em.append(b)
    logits, emb = np.concatenate(lg), np.concatenate(em)
    return score(pipe.ood_scorer, logits, emb, pipe.ood_state), softmax(
        logits, pipe.predictor.temperature
    )


def _summary(
    pipe: Pipeline, images: list[Image.Image], labels: list[str] | None
) -> tuple[dict, np.ndarray]:
    decisions = [pipe.run(im).decision for im in images]
    s, probs = _scores(pipe, images)
    status = pd.Series([d.status if d.status != "rejected" else d.reason for d in decisions])
    out: dict = {
        "n": len(images),
        "passed_gates": float(np.mean([d.status != "rejected" for d in decisions])),
        "outcomes": {k: float(v) for k, v in status.value_counts(normalize=True).items()},
        "median_ood_score": float(np.median(s)),
    }
    if labels is not None:
        pred = np.array(pipe.classes)[probs.argmax(1)]
        y = np.array(labels)
        acc = [d.label == t for d, t in zip(decisions, y, strict=True) if d.status == "accepted"]
        out["top1_accuracy"] = float((pred == y).mean())
        out["accepted_accuracy"] = float(np.mean(acc)) if acc else None
        cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred"))
        out["confusion"] = {r: {c: int(v) for c, v in row.items()} for r, row in cm.iterrows()}
    return out, s


def evaluate_external(bundle_dir: Path, ext_root: Path, ood_root: Path, out_root: Path) -> dict:
    pipe = Pipeline(bundle_dir)
    sample = pd.read_csv(ext_root / "bracol" / "sample.csv")
    unseen = sample[~sample["seen"]]
    bracol = []
    for i in unseen["id"]:
        with Image.open(ext_root / "bracol" / "images" / f"{i}.jpg") as im:
            bracol.append(to_rgb(im))
    rocole = []
    for p in sorted((ext_root / "rocole" / "images").glob("*.jpg")):
        with Image.open(p) as im:
            rocole.append(to_rgb(im))
    near = []
    for p in sorted((ood_root / "test" / "near").glob("*/*.jpg")):
        with Image.open(p) as im:
            near.append(im.convert("RGB"))

    res: dict = {
        "bundle": bundle_dir.name,
        "tau_ood": pipe.tau_ood,
        "knn_bank": (pipe.meta.get("ood") or {}).get("knn_bank", ["original"]),
        "bracol_unseen_per_class": unseen["label"].value_counts().to_dict(),
        "bracol_seen_share": float(sample["seen"].mean()),
        "bracol_seen_per_class": sample[sample["seen"]]["label"].value_counts().to_dict(),
        "bracol": {},
    }
    coffee_scores = []
    for name, fn in CONDITIONS.items():
        res["bracol"][name], s = _summary(pipe, [fn(im) for im in bracol], list(unseen["label"]))
        coffee_scores.append(s)
        log.info("BRACOL %-24s passed %.3f", name, res["bracol"][name]["passed_gates"])
    res["rocole"], s = _summary(pipe, rocole, None)
    coffee_scores.append(s)
    s_near, _ = _scores(pipe, near)
    coffee = np.concatenate(coffee_scores)
    res["auroc_external_coffee_vs_other_plants"] = float(
        roc_auc_score(np.r_[np.zeros(len(coffee)), np.ones(len(s_near))], np.r_[coffee, s_near])
    )
    res["other_plants_passing_ood"] = float((s_near <= pipe.tau_ood).mean())
    write_json(out_root / bundle_dir.name / "external.json", res)
    log.info(
        "%s: external coffee vs other plants AUROC %.3f; RoCoLe passed %.3f",
        bundle_dir.name,
        res["auroc_external_coffee_vs_other_plants"],
        res["rocole"]["passed_gates"],
    )
    return res
