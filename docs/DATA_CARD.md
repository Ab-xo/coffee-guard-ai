# Data card — CoffeeGuard AI

## Source

| | |
|---|---|
| Dataset | [Ethiopian Coffee Leaf Disease](https://www.kaggle.com/datasets/biniyamyoseph/ethiopian-coffee-leaf-disease) by Biniyam Yoseph (Kaggle) |
| Licence | CC0 (public domain) |
| Download | ≈ 2.05 GB zip, 12,000 JPEG files (`coffeeguard data download`) |
| Classes | Healthy, Cercospora (folder spelled `Cerscospora`), Leaf Rust, Phoma |
| Author's layout | `train aug/` (2,700 per class) and `test/` (300 per class) |

## What the download really contains

| Finding | Consequence |
|---|---|
| 8,509 of 12,000 files are author-generated augmentations (`aug_*`), balancing classes to exactly 2,700 | Set aside; the model trains on original photos with its own augmentation |
| The author's test set overlaps the training folder (e.g. all 300 Cercospora test file names reappear in training) | The original split is unusable for honest evaluation; everything was re-split |
| 902 byte-identical duplicates; 69 photos appear under two different labels | Duplicates removed; cross-label copies removed as conflicts |
| Many photos exist in 4 rotations/flips; re-shots of the same leaf | Grouped (rotation-invariant perceptual hash + DINOv2 similarity ≥ 0.97 within a class) so they never cross splits |

**Clean dataset: 2,520 unique photos** — Healthy 715 · Cercospora 505 · Leaf Rust 837 · Phoma 463 (imbalance 1.81 : 1). Data fingerprint `53320bd30430396d`.

## Splits (committed in `data/splits/`)

Group-aware stratified split (`StratifiedGroupKFold`, 20 folds → 14/3/3), no group or image in two splits (asserted):

| | Healthy | Cercospora | Leaf Rust | Phoma | Total |
|---|---:|---:|---:|---:|---:|
| Train | 499 | 354 | 587 | 324 | 1,764 |
| Validation | 108 | 76 | 124 | 69 | 377 |
| Test | 108 | 75 | 126 | 70 | 379 |

**Cross-validation folds** (`data/splits_cv/`, `coffeeguard data cv-folds`): train + validation pooled (2,141 photos) into 5 group-stratified folds of ~428 photos; the test split is excluded.

## Preprocessing

Every photo is decoded fully, EXIF-rotated, converted to RGB and cached with its long side at 384 px (JPEG q95, content-addressed file names). Quality measures (brightness, RMS contrast, Laplacian sharpness, leaf-colour fraction) are computed at 384 px and stored in the manifest. The model sees the cache resized to 224 × 224.

## Known biases and confounds

- **Photo setup differs by class.** Median original resolution: Healthy and Phoma 2048 px, Cercospora and Leaf Rust 1024 px; median file size 238 KB (Healthy) vs. 38 KB (Cercospora); sharpness 457 vs. 79. Cercospora and Leaf Rust were mostly photographed on blue paper, Healthy and Phoma on white paper, some Healthy and Leaf Rust on the tree. File-name patterns (camera/photographer) are almost class-specific.
- **Mitigations:** quality-equalising augmentation (random down-scaling, JPEG), background-swap augmentation, and measurements (robustness sweep, shortcut test). Residual effect: with the leaf blanked out, blue-paper backgrounds still lean towards Cercospora/Leaf Rust (at ~0.54 confidence).
- **Severity:** no severity labels; the mildest photos still show visible lesions (≈ 1–3 % of the leaf). Very early infection is not represented.
- **Geography / season / cameras:** one dataset from Ethiopia; unknown collection conditions beyond what the files show.

## Label quality

An automatic audit (DINOv2 embeddings + logistic regression, 5-fold out-of-fold probabilities, cleanlab) flags **19 photos (0.75 %)**, mostly Cercospora ↔ Leaf Rust. None were removed — that needs an agronomist. List: `artifacts/reports/label_issues.csv`; gallery: `artifacts/figures/eda/label_issues.png`.

## Out-of-distribution evaluation data (`data/ood/`, git-ignored, `coffeeguard ood collect`)

Split **by source** so the gate is tested on image kinds it was never tuned on. Images resized to 384 px.

| Split | Source | Images | Licence |
|---|---|---:|---|
| cal · near | tomato leaves — Kaggle `kaustubhb999/tomatoleaf` | 50 | CC0 |
| cal · near | banana leaves — Kaggle `shifatearman/bananalsd` (original photos) | 50 | CC BY-SA 4.0 |
| cal · far | animals — HF `Francesco/animals-ij5d2` | 50 | CC BY 4.0 |
| cal · far | landscapes — Windows wallpapers on the development PC | 36 | used locally, not redistributed |
| test · near | bean leaves — HF `AI-Lab-Makerere/beans` | 50 | MIT |
| test · far | indoor scenes — HF `keremberke/indoor-scene-classification` | 42 | CC BY 4.0 |
| test · far | rendered text — HF `nateraw/rendered-sst2` | 50 | see dataset card |
| test · far | synthetic blank / noise / gradient / screenshot frames | 40 | generated |

## Intended use

Training and evaluating a four-class coffee-leaf classifier and its rejection gates for a research/educational prototype. Not suitable as the sole basis for treatment decisions.
