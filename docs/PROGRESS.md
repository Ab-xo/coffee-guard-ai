# CoffeeGuard AI — Progress Log

A running record of everything done to build the project, step by step: what was built, which files changed, commands run, results, decisions and why, and anything you need to do. The plan being followed is [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

**How to read this log:** newest phase at the bottom. Each entry has *What & why*, *Files*, *How to run / verify*, and *Notes & decisions*.

---

## Environment facts (discovered at setup)

| Item | Value | Consequence |
|---|---|---|
| Machine | Intel UHD 620 (no CUDA GPU), 7.8 GB RAM, 8 logical cores | GPU training runs on Kaggle; everything else runs locally on CPU |
| Drive C: | Full (≈1.4 GB free) | Project moved to **`E:\Projects\coffee-guard-ai`** (E: has ~60 GB free) |
| Drive E: | exFAT file system | No hard links or ownership records: uv copies files instead of linking (`UV_LINK_MODE=copy`), and git needed a `safe.directory` exception |
| Python | uv-managed CPython 3.12.13 (Anaconda 3.13 left untouched) | Pinned in `.python-version` |
| Kaggle CLI | New-style token auth | Token file `C:\Users\pc\.kaggle\access_token.txt` works |
| Dataset | `biniyamyoseph/ethiopian-coffee-leaf-disease`, CC0, ~2.1 GB | Already split by the author into `train/` and `test/` folders; class folder spelled `Cerscospora`; many files end with ` (2)` (likely copies). We ignore the author's split and re-split with duplicate grouping |

---

## Phase 0 — Foundation

### 0.1 Repository move and restructure

**What & why**
- Cloned `https://github.com/Ab-xo/coffee-guard-ai` (7 commits, scaffold only: empty packages + docs).
- C: ran out of space during the first dataset download (the download stopped at 1.06 GB of 2.1 GB). The whole repo was moved to `E:\Projects\coffee-guard-ai` with `robocopy /MOVE`; the half-downloaded zip was moved too and the download resumed there. The old folder `C:\Users\pc\Desktop\Qiyas last project\coffee-guard-ai` is now empty (it could not be deleted while VS Code had it open — delete it when convenient).
- Package moved from `ml/` to **`src/coffeeguard/`** (the modern "src layout": the package must be installed to be imported, so tests exercise what ships). The four empty `data/*` sub-packages and `preprocessing/` were removed; data code is flat modules in `coffeeguard/data/`. New packages: `utils/`, `embeddings/`, `export/`, `inference/`.

**Git config change:** `git config --global --add safe.directory E:/Projects/coffee-guard-ai` (needed on exFAT).

### 0.2 Packaging and environment — `pyproject.toml`, `uv.lock`, `.python-version`

**What & why**
- Build backend switched from setuptools to **`uv_build`**; package name `coffee-guard-ai`, import name `coffeeguard`.
- Dependencies split so each deployment installs only what it needs:
  - core (always): numpy, pillow, pydantic, pyyaml, typer
  - `[inference]`: onnxruntime — the slim serving path
  - `[ml]`: torch, torchvision, timm, scikit-learn, pandas, pyarrow, opencv, imagehash, cleanlab, onnx, grad-cam, matplotlib, seaborn, tqdm, kaggle
  - `[api]`: fastapi, uvicorn, python-multipart, pydantic-settings (+ inference)
  - `[web]`: streamlit, httpx
  - dev group: ruff, pytest, pytest-cov, mypy, pre-commit, nbstripout, ipykernel, httpx
- `coffeeguard` console script → `coffeeguard.cli:app`.
- **torch source:** on Linux (CI/Docker) torch comes from the PyTorch CPU index (avoids ~2 GB of CUDA libraries); on Windows it comes from PyPI, whose Windows wheels are already CPU-only. This also sidestepped a DNS failure on `download-r2.pytorch.org` during the first install.
- **ruff** replaces black + flake8 + isort (one fast tool). Rules: pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, numpy, ruff-specific.

**How to run**
```powershell
cd E:\Projects\coffee-guard-ai
$env:UV_CACHE_DIR = "E:\uv-cache"; $env:UV_LINK_MODE = "copy"   # keep uv's cache off the full C: drive
uv sync --all-extras
uv run coffeeguard --help
```
> Tip: make the cache setting permanent with `setx UV_CACHE_DIR E:\uv-cache` and `setx UV_LINK_MODE copy` (new terminals pick it up).

### 0.3 Core library — `src/coffeeguard/`

| File | Purpose |
|---|---|
| `config.py` | Pydantic config models. `StrictModel` forbids unknown keys, so a YAML typo fails loudly. `DataConfig` (classes, folder aliases, paths, validation limits, cache size, dedup thresholds, split), `SplitConfig` (validates fractions sum to 1 and map to whole folds). `load_config(model, path, overrides)` supports CLI overrides like `--set split.seed=7` (values parsed as YAML). Paths are resolved against the repo root when loading and written back *relative* when saving, so committed configs never contain machine paths. |
| `utils/paths.py` | `project_root()` (env `COFFEEGUARD_ROOT` → nearest folder with `pyproject.toml` → cwd), `resolve()`, `portable_path()`. |
| `utils/hashing.py` | `sha256_file`, `sha256_bytes`, order-independent `fingerprint()` (16 hex chars) used for the dataset fingerprint. |
| `utils/seed.py` | `set_seed()` for Python/NumPy/PyTorch, optional deterministic mode. |
| `utils/log.py` | One-time logging setup with rich formatting. (Named `log.py`, not `logging.py`, to avoid shadowing the standard library.) |
| `utils/io.py` | JSON read/write that understands NumPy types and paths. |
| `utils/runs.py` | `create_run_dir()` → `runs/<timestamp>-<name>`; `capture_env()` records Python, platform, package versions, git SHA + dirty flag, CUDA/GPU. |
| `cli.py` | Typer app. Heavy libraries are imported inside commands so `--help` is instant. Commands so far: `info`, `data download`, `data prepare`. |

### 0.4 Tooling — `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `.gitignore`

- **pre-commit:** ruff (lint + autofix), ruff-format, nbstripout (no notebook outputs in git), large-file guard (5 MB), YAML/TOML checks, whitespace and LF line endings.
- **CI (GitHub Actions):** `uv sync --all-extras --locked` → `ruff check` → `ruff format --check` → `pytest`. `--locked` fails the build if `uv.lock` is out of date.
- **.gitignore additions:** `.venv/`, `runs/`, `data/raw/*`, `data/processed/*`, `data/ood/`, manifests (`*.parquet`, `*.npz`, `*.npy`), ONNX/NumPy files in model bundles, `kaggle/_build/`, `.ruff_cache/`. The split CSVs in `data/splits/` **are** committed (small, and they define the experiment).

### 0.5 Test scaffolding — `tests/`

- `tests/fixtures/synthetic.py`: generates a Kaggle-shaped fake dataset (`train|test/<Folder>/*.jpg`, including the `Cerscospora` typo). Each image is a green "leaf" on a cluttered soil background with class-specific spots (none / small brown dots / orange blotches / large dark patches). Tests never need the real 2 GB dataset.
- `tests/conftest.py`: `synthetic_raw` fixture (4 classes × 6 images in a temp folder).
- `tests/unit/test_config.py`: the repo config is valid; typo aliases map correctly; unknown keys and bad split fractions are rejected; CLI overrides parse; YAML round-trip.
- `tests/unit/test_utils.py`: hashing, fingerprint order-independence, seeding, JSON with NumPy, run dirs, env capture.
- `tests/unit/test_cli.py`: `--help` and `info --set ...` work.

---

## Phase 1 — Data engineering (in progress)

### 1.0 What the raw dataset really contains (inspection before any processing)

The zip (2,048,300,030 bytes, integrity-checked with `zipfile.testzip()`) holds 12,003 files:

| Folder | Files | Size | Notes |
|---|---:|---:|---|
| `test/Cerscospora` | 300 | 12 MB | |
| `test/Healthy` | 300 | 215 MB | |
| `test/Leaf rust` | 300 | 85 MB | |
| `test/Phoma` | 300 | 50 MB | |
| `train aug/Cerscospora` | 2,700 | 104 MB | 2,164 are `aug_*.jpeg` |
| `train aug/Healthy` | 2,700 | 1,048 MB | 2,032 are `aug_*.jpeg` |
| `train aug/Leaf rust` | 2,700 | 281 MB | 1,902 are `aug_*.jpeg` |
| `train aug/Phoma` | 2,700 | 320 MB | 2,379 are `aug_*` (1,156 of them are ` (2)` copies of other aug files) |
| `.ipynb_checkpoints` | 3 | 1 MB | author's notebooks — ignored |

**Findings that change how we must use the data**

1. **The training folder is pre-augmented.** 70–88% of each class in `train aug/` are generated images (`aug_<n>_<m>.jpeg`), not photos. They were balanced to exactly 2,700 per class by generating more copies for the smaller classes.
2. **The author's test set leaks into their training set.** File names overlap: 300/300 Cercospora, 218/300 Healthy, 222/300 Leaf rust and 130/300 Phoma test file names also appear in `train aug/`. Any accuracy computed on the Kaggle split is inflated. We discard the author's split and re-split ourselves with duplicate grouping.
3. **Different classes come from different sources.** File-name styles differ by class: `C10P10H1.jpg` (Healthy, some Leaf rust), `20231123_102720.jpg` phone-camera names (Cercospora, Leaf rust), plain numbers `1244.jpg` (Phoma, Healthy). Average file size per image is ~390 KB for Healthy vs ~40 KB for Cercospora. A model could learn *source / resolution / compression* instead of disease. This is a shortcut risk we measure (EDA resolution-by-class figure, the Phase 5 shortcut test) rather than ignore.

**Decision:** only original photos form the dataset. `configs/data.yaml` gains
```yaml
exclude_name_patterns: ['^aug_']
```
which marks those files `excluded_pattern` (they are still scanned, so the report can count them). Training uses our own on-the-fly augmentation instead. Expected size after exclusion and deduplication: roughly 2,500–3,000 unique photos (exact numbers below after the run).

**Pipeline additions made because of these findings**
- `name_pattern` column in the manifest (`C10P10H1.jpg` → `C#P#H#`) and a pattern × class table in `eda_summary.json`.
- **Rotation/flip-invariant near-duplicate detection:** each image stores the pHash of all 8 rotations/flips (`phash_d4`); two images are near-duplicates if one's hash is within 5 bits of *any* variant of the other. This catches mirrored or 90°-rotated copies, a typical by-product of pre-augmentation.
- Resolution / file-size by class figure (`resolution_by_class.png`).

### 1.1 Download — `coffeeguard data download` (`src/coffeeguard/data/download.py`)

Downloads the zip with the Kaggle API (skipped if present) and extracts it into `data/raw/` once (a `.extracted` marker prevents re-extracting). Auth uses the standard Kaggle lookup (`KAGGLE_API_TOKEN`, `~/.kaggle/access_token(.txt)`, or `kaggle.json`).

### 1.2 The `data prepare` pipeline (code written; first real run pending the download)

`coffeeguard data prepare` runs four cached stages. Each can be re-run alone with `--stages`.

1. **scan** (`data/scan.py`) — one parallel pass per image (7 worker processes):
   - SHA-256 of the file bytes;
   - **full decode** (`Image.load()`, which catches truncated files that `verify()` misses), EXIF orientation fix, RGB conversion;
   - checks: empty file, unreadable, format not JPEG/PNG, short side < 100 px, unknown class folder (all → invalid); unusual aspect ratio (warning only);
   - quality measures (`inference/quality.py`, NumPy + Pillow only so the API can reuse the exact same code): brightness, RMS contrast, Laplacian-variance sharpness, green-pixel fraction, dark/bright pixel fractions — all computed at a fixed 384 px so they are comparable across camera resolutions;
   - 64-bit perceptual hash (pHash);
   - writes a resized copy (long side 384 px, JPEG q95) to `data/processed/<class>/<sha256[:20]>.jpg`. Content-addressed names make re-runs free and store identical files once.

   Output: `data/manifests/scan.parquet`.
2. **embed** (`embeddings/extract.py`) — DINOv2 ViT-S/14 image embeddings (384-d) for every valid image, keyed by SHA-256 in `data/manifests/embeddings.npz`; only new images are embedded on re-runs; checkpointed every 50 batches.
3. **group** (`data/dedup.py`):
   - exact duplicates (same SHA-256) in one class → keep the first, mark the rest `duplicate`; in different classes → all copies `conflict` (removed: we can't know the right label);
   - near duplicates (pHash differs in ≤ 5 of 64 bits): same class → same group; different classes → `conflict`;
   - similar images (embedding cosine ≥ 0.95, typically re-shots of the same leaf) → same group, kept;
   - groups are connected components (union-find). Memory-safe block computation (512 rows at a time) so ~50k images fit in 8 GB RAM.
4. **split** (`data/split.py`) — `StratifiedGroupKFold` with 20 folds: folds 0–2 → test, 3–5 → val, 6–19 → train (15/15/70). Groups never cross splits; `check_no_leakage()` asserts no SHA-256 or group appears in two splits. Writes `data/splits/{train,val,test}.csv` (columns: `image, label, class_id, group_id, sha256, path`), `split_info.json` (data fingerprint, class balance, dedup summary, invalid-file reasons) and `data_config.yaml` (the exact config used).

**Tests** (`tests/unit/test_data_pipeline.py`): bad files flagged correctly; notebook checkpoint folders skipped; cache resized and content-addressed; pHash and embedding pair search; exact duplicate / cross-class conflict / near-duplicate grouping; split proportions, stratification and no leakage on 2,000 synthetic rows; end-to-end `prepare` and idempotence (same fingerprint on re-run).

### 1.2b First real run — results and the fixes it forced

**Command:** `uv run coffeeguard data prepare` (scan ≈10 min with 7 workers, DINOv2 embeddings of 3,491 originals on CPU, grouping, split).

**Scan:** all 12,000 files decode as valid JPEGs (one is 108 megapixels, loaded with a decompression-bomb warning). 3,491 are original photos; only **2,571 are unique by SHA-256**.

**What happened to every file**

| Status | Files | Meaning |
|---|---:|---|
| `excluded_pattern` | 8,509 | author's generated `aug_*` images |
| `duplicate` | 902 | byte-identical copy of a kept photo (same class), mostly author's test images repeated in train |
| `conflict` | 69 | identical (36) or near-identical (33) image under two different labels; removed |
| **`ok`** | **2,520** | clean dataset |

**Fix 1 — giant groups from similarity chaining.** With embedding links at cosine ≥ 0.95 across all classes, one "group" grew to 303 images (Healthy + Leaf Rust + Phoma photos from the same source). The whole group landed in test, so Healthy's test share was 40% instead of 15%. Two changes:
- embedding links are now made **only within the same class** (a leaf can't be both Healthy and Phoma; cross-class similarity means "same photo setup", not "same leaf"). Cross-class *copies* are still caught as conflicts by the exact/pHash checks;
- threshold raised to **0.97** after comparing three values on the real data:

| Threshold | Largest group | Healthy test share | Phoma test share |
|---|---:|---:|---:|
| 0.93 | 316 | 44% | 71% |
| 0.95 | 238 | 40% | 17% |
| **0.97** | **30** | **15.1%** | **15.1%** |

**Fix 2** — `source_split` did not recognise the author's `train aug` folder; fixed (the cached scan column was recomputed instead of rescanning).

**Final grouping:** 1,728 groups; 1,457 singletons, **215 groups of exactly 4** (the same photo stored in 4 rotations/flips, caught by the rotation-invariant hash), a few larger same-leaf groups (max 30).

**Final split** (fingerprint `53320bd30430396d`, committed in `data/splits/`):

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Healthy | 499 | 108 | 108 | 715 |
| Cercospora | 354 | 76 | 75 | 505 |
| Leaf Rust | 587 | 124 | 126 | 837 |
| Phoma | 324 | 69 | 70 | 463 |
| **Total** | **1,764** | **377** | **379** | **2,520** |

Imbalance ratio 1.81 : 1 (< 3 : 1), so **no class weights** — macro-F1 is still the primary metric. Leakage check passed (no SHA-256 or group in two splits).

**EDA findings** (`artifacts/reports/eda_summary.json`, figures in `artifacts/figures/eda/`)

| Class | Original long side (median) | File size (median) | Sharpness (median) | Filename source patterns |
|---|---:|---:|---:|---|
| Healthy | 2048 px | 238 KB | 457 | `#` 312, `C#P#E#` 199, `C#P#H#` 204 |
| Cercospora | 1024 px | 38 KB | 79 | `#_#` 505 |
| Leaf Rust | 1024 px | 52 KB | 154 | `#_#` 496, `#` 219, `C#P#…` 122 |
| Phoma | 2048 px | 101 KB | 258 | `#` 456 |

**Serious confound:** resolution, compression, sharpness, photo source and background (Cercospora/Leaf Rust on white/blue paper; many Healthy leaves on the tree) all differ by class. A model can score well by learning *how* a photo was taken rather than the disease. Planned counter-measures:
- Phase 2: add **quality-equalising augmentation** (random downscale-upscale and JPEG re-compression) so image quality stops predicting class;
- Phase 5: the **shortcut test** (background-only / leaf-only images) and error analysis by quality measure quantify how much the final model relies on these cues.

**Label audit** (`coffeeguard embed audit`, `artifacts/reports/label_issues.csv`, gallery `artifacts/figures/eda/label_issues.png`): cleanlab flags **19 of 2,520 images (0.75%)** — 8 Cercospora, 8 Leaf Rust, 2 Phoma, 1 Healthy — almost all Cercospora ↔ Leaf Rust, the visually closest pair. **Nothing was excluded**: these look like genuinely ambiguous lesions, and removing them needs a domain expert's confirmation. To exclude one later, add its `sha256` + a reason under `exclude:` in `configs/data.yaml`.

**Foundation-model baseline** (`artifacts/metrics/baseline_dinov2_probe_val.json`): frozen DINOv2 ViT-S/14 embeddings + logistic regression, trained on train, scored on **val**:

| Accuracy | Macro-F1 | F1 Healthy | F1 Cercospora | F1 Leaf Rust | F1 Phoma |
|---:|---:|---:|---:|---:|---:|
| 0.958 | **0.961** | 0.964 | 0.949 | 0.932 | 1.000 |

This is the bar the fine-tuned CNNs must clear. (Cross-validated out-of-fold macro-F1 on all clean images: 0.976.)

**Housekeeping in this step:** replaced the unavailable "semibold" font weight with bold in figures; `ruff format` had also reformatted Python snippets *inside* `CONTRIBUTING.md` and `docs/PROJECT_SPECIFICATION.md` — those two files were restored from git and Markdown is now excluded from ruff; `pyproject.toml` license switched to the PEP 639 form (`license = "MIT"`, `license-files`). The CLI now keeps pretrained weights in the project's git-ignored `.cache/`.

### 1.3 Tests and lint status (synthetic data)

- `uv run ruff check .` and `uv run ruff format --check .` → clean (71 files).
- `uv run pytest -m "not slow"` → **28 passed** (config, utils, CLI, scan/validation, dedup incl. mirrored copies and `aug_` exclusion, group-aware split, end-to-end `prepare`, quality measures, preprocessing incl. EXIF rotation).
- `uv run pytest -m slow` → **1 passed**: synthetic data → `prepare` → two-stage training on CPU → ONNX export with parity check → prediction through the ONNX runtime.

Two test bugs were found and fixed while doing this (a glob that picked the wrong fixture file; pydantic's `==` also compares which fields were explicitly set, so the round-trip test now compares dumped values).

---

## Phase 2 — Training pipeline, baseline, walking skeleton (code written, smoke-tested)

Written ahead of time while the environment installed; each piece is covered by the smoke test above.

| File | What it does |
|---|---|
| `inference/preprocess.py` | **The single canonical eval preprocessing**: EXIF-orient → RGB → downscale long side to 384 (same as the training cache) → resize to 224×224 bicubic → normalise with the model's mean/std → CHW float32. Used by evaluation, ONNX parity checks *and* the API, so the served model sees exactly the pixels it was evaluated on. NumPy + Pillow only. |
| `data/transforms.py` | Train augmentation (torchvision v2): rotation ±20°, RandomResizedCrop(scale 0.6–1), horizontal + vertical flips, mild brightness/contrast/saturation jitter with **hue fixed at 0** (lesion colour is diagnostic), occasional light blur. Eval = the canonical preprocessing wrapped in a picklable class (Windows DataLoader workers use *spawn*, which cannot pickle closures). |
| `data/dataset.py` | `LeafDataset` over a split CSV; optional in-RAM preload for Kaggle. |
| `models/factory.py` | `build_model` (timm), `set_trainable(head / last_n / all)`, `freeze_frozen_batchnorm` (frozen layers must not update BatchNorm running stats), classifier-weight helper. |
| `evaluation/metrics.py` | accuracy, macro precision/recall/F1, per-class report, confusion matrix, one-vs-rest ROC-AUC. |
| `training/trainer.py` | Multi-stage trainer: per stage a fresh AdamW (timm `create_optimizer_v2`, optional **layer-wise LR decay**), linear warmup + cosine schedule, **EMA weights** with decay warm-up, label smoothing, gradient clipping, AMP + channels-last on CUDA, **early stopping on val macro-F1**. Both raw and EMA weights are scored every epoch; the better one is kept. The next stage starts from the previous stage's best weights. `best.pt` stores weights + model name, classes, image size, mean/std, data fingerprint and full config. Each run writes `config.yaml`, `env.json`, `history.jsonl`, `summary.json`. |
| `configs/train/*.yaml` | Recipes (see `configs/train/README.md`): `effnetv2_b0` (variant B: full fine-tune + layer decay 0.75), `effnetv2_b0_partial` (variant A, spec: last 3 stages), `mobilenetv3_small` (baseline), `mobilenetv3_large`, `efficientnet_b0` (comparison), `smoke` (CPU test). All share the same LP-FT schedule: 8 epochs head-only at LR 1e-3, then up to 25 fine-tuning epochs. |
| `export/onnx_export.py` | Wraps the model so one ONNX forward pass returns `logits`, `embedding` (pooled features, for OOD) and `feature_map` (for CAM). Verifies torch↔ONNX parity (max logit difference < 1e-3 and identical argmax) and records `cam_exact`: true when the head is pool→linear (EfficientNet family), so CAM = Grad-CAM with no torch at serving time. MobileNetV3 has an extra layer after pooling, so it is `false` there — found by the smoke test. Writes `bundle.json` with SHA-256 of every file. |
| `inference/predictor.py` | ONNX Runtime `Predictor` (numpy + pillow + onnxruntime only): preprocess → forward → temperature-scaled softmax → `Prediction` (label, confidence, probabilities, embedding, feature map, the resized image). |
| `training/remote.py`, `kaggle/run_template.py` | **Kaggle GPU runner.** `coffeeguard remote upload-data` uploads the 384 px cache as the private dataset `<user>/coffeeguard-processed`. `coffeeguard remote train -c <recipe> --seeds 0,1,2` builds the wheel, uploads wheel + configs + splits as `<user>/coffeeguard-code`, pushes a private GPU script kernel (T4) that installs the wheel and runs the normal `coffeeguard train` for every recipe × seed, then polls and downloads `runs/`. No GitHub push required. *(Not yet exercised live.)* |
| `cli.py` | New commands: `train`, `export`, `remote upload-data`, `remote train`, `remote fetch`, `embed audit`, `data report`. The CLI now keeps pretrained-weight downloads in the project's git-ignored `.cache/` folder so the full C: drive is never used. |

Note: PyTorch 2.14 warns that the TorchScript-based ONNX exporter is deprecated. It is still used (`dynamo=False`) because it is stable for these CNNs; switching to the new exporter only needs the `onnxscript` package and `dynamo=True`.

---

## Status at end of session 1 — Phase 1 complete ✅

**Phase 0 ✅ and Phase 1 ✅ done.** Phase 2 code is written and smoke-tested on CPU but no real model has been trained yet.

Checks at this point: `ruff check` clean, `ruff format --check` clean, `pytest` **29 passed** (28 unit + 1 train→export→predict smoke test).

**Phase 1 "Done when" gate:** `data prepare` idempotent ✅ · splits committed-ready ✅ · EDA + data-quality reports ✅ · probe baseline recorded ✅ · Kaggle upload of the processed dataset ⏳ (moved to the start of Phase 2, where the remote runner is first exercised).

**Next session (Phase 2):**
1. Add quality-equalising augmentation (random downscale + JPEG re-compression) — motivated by the confound above.
2. `coffeeguard remote upload-data`, then `coffeeguard remote train -c configs/train/mobilenetv3_small.yaml` — first live Kaggle GPU run (baseline).
3. Export the baseline to ONNX, then build the minimal FastAPI `/predict` + Streamlit page (walking skeleton).

**For you to decide / do**
- Review `artifacts/figures/eda/label_issues.png` if you (or someone with coffee-disease knowledge) can confirm any mislabels.
- The Kaggle token was pasted in chat; consider regenerating it at kaggle.com/settings/api when the project is done.

### Cleanup and first commit

**Removed (decided with you):** the team-process docs written for a 7-person team — `PHASE_1_CHECKLIST.md`, `docs/TEAM_ROLES.md`, `docs/GIT_WORKFLOW.md`, `docs/QUICK_START.md` — superseded by `IMPLEMENTATION_PLAN.md` + this log.

**Rewritten to match the real project:**
- `README.md` — structure (`src/coffeeguard/`), uv-based setup, the actual CLI pipeline, phase status table, links to plan/progress (the full results README comes in Phase 10);
- `CONTRIBUTING.md` — short solo workflow: uv, ruff, phase branches + conventional commits, test and results rules;
- `.env.example` — only variables the project uses (Kaggle token, uv settings, API/UI settings); the old `.pth` checkpoint path and Kaggle username/key pair were obsolete.

**Removed scaffold placeholders no longer in the plan:** `data/interim/`, `artifacts/checkpoints/` (replaced by `runs/` + `artifacts/models/`), `scripts/` (replaced by the CLI), `docs/architecture/`, `docs/experiments/`; `.gitkeep` files in folders that now hold real files. Kept placeholders for folders still empty in git (`data/raw`, `data/processed`, `data/manifests`, `apps/web`, `notebooks`, `docs/decisions`).

**Local junk deleted:** `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `logs/`, and the two empty leftover folders (`C:\…\Qiyas last project\coffee-guard-ai`, `E:\coffeeguard-staging`). The 2 GB dataset zip is kept (git-ignored) as the untouched original download.

**Git:** branch `phase/1-data`, one commit per phase, no AI-attribution trailers (this is an assignment):
1. `chore: phase 0 foundation …` — package layout, tooling, CI, config, CLI shell (`info` only), test scaffolding, plan, doc cleanup;
2. `feat: phase 1 data pipeline …` — data pipeline, EDA/audit, committed splits and reports, README update, this log, plus the smoke-tested Phase 2 groundwork.

Committed data products: `data/splits/*` (split CSVs, `split_info.json`, `data_config.yaml`), `artifacts/figures/eda/*`, `artifacts/reports/*`, `artifacts/metrics/baseline_dinov2_probe_val.json`. Not pushed.

---

## Session 2 — Phase 2: training pipeline, baseline, walking skeleton

Branch: `phase/2-training` (from `eleni-changes`, which holds the Phase 0 and Phase 1 commits).

### 2.0 CLI crash on Windows pipes (bug fix)

**What & why:** `coffeeguard --help` crashed with `UnicodeEncodeError` whenever its output went through a pipe or redirect (e.g. `coffeeguard --help | more`, CI logs): Windows then uses the ANSI code page (cp1252), which can't encode the `↔`/`×` characters in help text. `cli.py` now switches stdout/stderr to UTF-8 at import, before Typer prints anything.

### 2.1 Quality-equalising augmentation + augmentation preview

**What & why:** EDA showed photo quality predicts the class (Healthy/Phoma 2048 px and sharp; Cercospora/Leaf Rust 1024 px and heavily compressed). Two new picklable train-time transforms in `data/transforms.py`:

| Transform | What it does | Default |
|---|---|---|
| `QualityDegrade` | with p = 0.5 downscale to 35–100 % of the size and back up (loses fine detail); independently with p = 0.5 re-encode as JPEG at quality 30–95 | `augment.downscale_p/downscale_min/jpeg_p/jpeg_quality_min` in `config.py` |
| `RandomRotateFill` | ±20° rotation that fills the exposed corners with the image's mean border colour | replaces torchvision `RandomRotation` |

Degradation can only go one way (a blurry photo can't be sharpened), so it pulls the sharp classes towards the compressed ones — quality stops separating classes. Whether this worked is measured in Phase 5 (error rate by quality quantile, shortcut test).

**Found with the preview:** the first preview grid showed **black corner wedges** from torchvision's rotation. They never occur at eval time, so the model could use "has black corners" as a train-only cue. `RandomRotateFill` continues the background (paper, soil, foliage) instead.

**New command:** `uv run coffeeguard data aug-preview [-c configs/train/<recipe>.yaml]` → `artifacts/figures/augmentation_preview.png` (2 photos per class, original + 5 augmentations). Checked by eye: lesions (Cercospora spots with halos, rust pustules, Phoma dark patches) survive every augmentation.

**Tests:** `tests/unit/test_transforms.py` — degradation lowers sharpness and keeps the size; disabled = identity; rotation of a pale image has no dark pixels; the full train transform pickles (Windows spawn workers) and outputs `(3, S, S)`.

### 2.2 Kaggle runner fixes before the first live run

- **Upload only the clean images.** The local cache (`data/processed/`, 1.4 GB, 11,099 files) also holds the author's `aug_*` copies and duplicates. `remote upload-data` now stages only the 2,520 images referenced by the split CSVs → **59 MB** zip.
- **Git SHA on remote runs.** Kaggle has no git checkout, so `env.json` would have recorded `sha: null`. The launcher now embeds the local git state in the kernel script, which exports `COFFEEGUARD_GIT_SHA/DIRTY`; `git_info()` reads them first.

**Done:** `uv run coffeeguard remote upload-data` → private dataset **`eleniandualem/coffeeguard-processed`** (2,520 images, fingerprint `53320bd30430396d`).

### 2.3 Walking skeleton: FastAPI service + Streamlit UI

**API** (`apps/api/app/`, run with `uv run uvicorn app.main:app --app-dir apps/api`):

| File | Purpose |
|---|---|
| `settings.py` | `pydantic-settings`: `MODEL_BUNDLE`, `MAX_UPLOAD_MB` (10), `MAX_PIXELS` (100 MP), `ORT_THREADS`, `CORS_ORIGINS` — from env vars or `.env` |
| `main.py` | `create_app(settings)`: lifespan loads the ONNX bundle once (a missing bundle fails start-up loudly), CORS for the UI, one error shape `{"error": code, "detail": …}` for every failure incl. FastAPI's own 422 |
| `api/routes.py` | `GET /health`, `POST /predict` (sync handler → runs in FastAPI's threadpool so ONNX Runtime doesn't block the event loop; reads at most limit + 1 bytes) |
| `services/images.py` | upload validation: empty → 400, too big → 413, **magic-byte sniffing** (JPEG/PNG/WebP; the client's Content-Type is not trusted) → 415, pixel-count limit → 413, full decode (truncated files) → 400 |
| `schemas/prediction.py` | typed responses, shown in `/docs` |

**Bug found by the tests:** routes read the global settings instead of the ones passed to `create_app`, so the upload limits given in tests were ignored. Settings now live on `app.state`.

**Tests** (`tests/integration/test_api.py`, 9 tests, ~2 s): run against a **hand-built 4-class ONNX model** (`tests/fixtures/bundle.py`, made with `onnx.helper`: mean colour → fixed linear layer, same three outputs as a real export), so CI never needs a trained model. Covers health, green → Healthy / red → Cercospora predictions, probabilities summing to 1, fake image with an image content type, truncated JPEG, empty file, oversized file, too many pixels, missing field.

**UI** (`apps/web/streamlit_app.py`, run with `uv run streamlit run apps/web/streamlit_app.py`): upload or camera → `/predict` → predicted class, confidence and probability bar chart; clear message when the API is down. `API_URL` env var (default `http://localhost:8000`).

`pyproject.toml`: pytest `pythonpath = [".", "apps/api"]` so tests import the API package `app`.

### 2.4 First live Kaggle run — attempt 1 failed, runner hardened

`uv run coffeeguard remote train -c configs/train/mobilenetv3_small.yaml --seeds 0` built the wheel, uploaded `eleniandualem/coffeeguard-code`, pushed the private kernel `eleniandualem/coffeeguard-train` and polled it. After 2 minutes: status `error`, no message.

**Cause (from the downloaded kernel log):** the script called `nvidia-smi` to print the GPU, and Kaggle's current image (Python 3.12 at `/usr/bin/python3`) doesn't have it on `PATH` → `FileNotFoundError`. The kernel metadata on the server was correct (`enable_gpu: true`, `machine_shape: NvidiaTeslaT4`) and the account has a 6 h/week GPU quota (0 used).

**Fixes**
- `kaggle/run_template.py` checks the GPU through PyTorch (`torch.cuda.is_available()`, device name) and **stops with a clear message if no GPU is attached** (a silent CPU fallback would burn hours); `nvidia-smi` is only called if it exists.
- Jobs run as `python -m coffeeguard.cli train …` (no dependence on the console script being on `PATH`).
- `remote train` / `remote fetch`: when a kernel fails, the kernel log is downloaded and its last 40 lines are shown in the error, instead of a bare `status error`.

**Attempt 2** (with the fixes): the kernel stopped at the new GPU check — `torch 2.10.0+cpu | CUDA build None`. The session ran in Kaggle's **CPU image** (`gcr.io/kaggle-images/python`, not `kaggle-gpu-images`).

**Diagnosis** (a one-off diagnostic version of the same kernel, pushed with `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, `enable_internet: true`; the push response had no error): `cuda False`, no `/dev/nvidia*`, and **no internet** (`Temporary failure in name resolution`). GPU *and* internet silently missing is Kaggle's behaviour for accounts **without phone verification**. The weekly quota still shows 6 GPU-hours, but it can't be used until the phone number is verified at <https://www.kaggle.com/settings>. The kernel's no-GPU message now says this.

**CPU speed on this laptop** (8 threads, batch 32 at 224 px, forward + backward): MobileNetV3-Small 3.5 s/batch ≈ 3.2 min per training epoch; EfficientNetV2-B0 7.2 s/batch ≈ 6.6 min per epoch. The small baseline is feasible locally (~1–1.5 h); the Phase 3 runs (EffV2-B0 × ablation × 3 seeds + comparison models) are not.

**Decision:** train the MobileNetV3-Small baseline **locally on CPU** now so the walking skeleton can be finished; all Phase 3 training waits for the Kaggle GPU. The run's `env.json` records `device: cpu`.

### 2.5 Bug found by the first real run: classifier initialisation

**Symptom:** first linear-probe epoch of the CPU baseline: train loss **6.05** (chance level for 4 classes is ln 4 ≈ 1.39), train accuracy 0.26 (= chance), val macro-F1 0.36 — on ImageNet features that a linear probe normally separates immediately (the DINOv2 probe reaches 0.96).

**Diagnosis:** on 16 real training images, the pretrained features were normal (pre-logits std 0.72) but the **new classifier's weights had std 0.287** and the initial **logits std 6.3** (max 16.6). timm's EfficientNet/MobileNet init draws Linear weights from U(±1/√fan_out) with fan_out = number of classes — ±0.03 for ImageNet's 1,000 classes, but **±0.5 for our 4**. The probe stage spent its whole budget shrinking random weights instead of learning. The same init is used by EfficientNet-B0 and EfficientNetV2-B0, so every planned model was affected; the synthetic smoke test couldn't reveal it (it only checks that training runs).

**Fix:** `models/factory.py` → `zero_init_classifier()` after `timm.create_model`: weight and bias start at 0, i.e. a uniform prediction with loss exactly ln 4 — the usual LP-FT starting point. Gradients are still non-zero because the features differ between images. Test: `tests/unit/test_models.py` (zero logits and loss = ln 4 at start for MobileNetV3 and EfficientNet; `head` mode trains only the classifier).

The broken run was stopped and deleted; the baseline was restarted.

### 2.6 Baseline: MobileNetV3-Small (CPU, seed 0)

**Command:** `uv run coffeeguard train -c configs/train/mobilenetv3_small.yaml --seed 0 --set device=cpu --set num_workers=3 --set batch_size=32` (batch 32 instead of 64 to keep RAM use low on this laptop). Run dir `runs/20260925-211837-mobilenetv3_small-s0` (git `855755e` + uncommitted Phase 2 changes, i.e. `dirty: true`; data fingerprint `53320bd30430396d`). **27.7 min** on CPU.

| Stage | Epochs run | Best val macro-F1 |
|---|---:|---:|
| `head` (linear probe, backbone frozen, ~37 s/epoch) | 8 | 0.924 |
| `finetune` (all layers, layer decay 0.75, ~93 s/epoch) | 14 (early stop, patience 5) | **0.991** (epoch 9, EMA weights) |

**Best checkpoint (val, 377 images):** accuracy **0.992**, macro-F1 **0.991**, macro precision 0.991, macro recall 0.992 — above the DINOv2 linear-probe reference (0.961). EMA weights beat the raw weights at most fine-tuning epochs (they smooth out the epoch-to-epoch dips). Curves: `artifacts/figures/training/mobilenetv3_small-s0.png`; metrics: `artifacts/metrics/baseline_mobilenetv3_small_val.json`.

**Caution before reading too much into 0.99:** this is one seed on 377 val images (3 errors), and the EDA confound (photo source/quality differs by class) may be inflating it. The robustness sweep and shortcut test (Phase 5) exist to check exactly that; the test split stays sealed until Phase 4.

New command: `uv run coffeeguard curves --run runs/<run>` → `<run>/figures/training_curves.png` (loss and val macro-F1 per epoch across stages, raw vs. EMA, best point marked); covered by the smoke test.

### 2.7 Walking skeleton, end to end

1. **Export:** `uv run coffeeguard export --run runs/20260925-211837-mobilenetv3_small-s0 --name coffeeguard-mnv3s-baseline` → `artifacts/models/coffeeguard-mnv3s-baseline/` (`model.onnx` 6.1 MB). Parity torch ↔ ONNX: max |Δlogit| **1.8e-6**, argmax agreement **100%**. `cam_exact: false` for MobileNetV3 (extra layer after pooling), as expected. Only `bundle.json` is committed; the ONNX/NumPy files are git-ignored.
2. **API:** `MODEL_BUNDLE=artifacts/models/coffeeguard-mnv3s-baseline uv run uvicorn app.main:app --app-dir apps/api`. Sent 8 **original full-resolution** val photos (2 per class, from `data/raw/`, as a user would upload them): **8/8 correct**, confidence 0.75–0.95, **15–30 ms** model latency on this CPU (150 ms for the first request). A non-image file → `415 unsupported_media_type`. Confidences top out around 0.93–0.95 because of label smoothing (0.1); calibration (temperature scaling) is Phase 4.
3. **UI:** rendered headlessly with Streamlit's `AppTest` against the live API (model name in the sidebar, input choice shown) and with the API down.
   - **Bug found:** with the API down, the page still said the model was loaded — `api_health()` was cached with `st.cache_data` and had no arguments, so it returned the first result. The cache was removed (one tiny request per rerun).
   - Tests: `tests/integration/test_web.py` (API down → clear error and nothing else rendered; API up with `httpx` mocked → model name + input choice).

### Phase 2 status ✅ (with one open item)

| Gate item | Status |
|---|---|
| Transforms + augmentation preview checked by eye | ✅ |
| Dataset / DataLoader, model factory, multi-stage trainer | ✅ (classifier-init bug fixed) |
| Baseline MobileNetV3-Small trained, val metrics recorded | ✅ val macro-F1 0.991 — **trained on CPU**, not Kaggle |
| Export → Predictor → FastAPI → Streamlit | ✅ |
| CPU smoke test (train → export → predict) + API tests | ✅ |
| Remote runner (`remote train`) end to end | ⏳ upload + push work; **blocked on Kaggle phone verification** (no GPU/internet granted) |

Checks: `ruff check` / `ruff format --check` clean; `pytest` **47 passed** (incl. the slow train→export smoke test).

**For you to do:** verify a phone number at <https://www.kaggle.com/settings> so Kaggle kernels get a GPU and internet. Phase 3 (EfficientNetV2-B0 ablation + 3 seeds, comparison models) needs it: on this laptop one EffV2-B0 epoch takes ~6.6 min.

---

## Session 2 (continued) — Phase 3: EfficientNetV2-B0

Branch: `phase/3-effnetv2` (from the pushed Phase 2 commit `552851e`).

### 3.0 Kaggle account switch

Phone verification on the first account (`eleniandualem`) was blocked by Kaggle's rate limit ("You've sent too many requests"). You created a second account and replaced the token in `C:\Users\pc\.kaggle\access_token.txt`. Nothing in the code is tied to an account (the username comes from the token), so the only step was re-uploading the data:

- `uv run coffeeguard remote upload-data` → private dataset **`fevenabebe1616/coffeeguard-processed`** (2,520 images, 59 MB, fingerprint `53320bd30430396d`).
- GPU quota on the new account: 6 h/week.

**Why train on Kaggle at all when the data is local?** For the GPU, not the data. This laptop has no CUDA GPU; measured CPU speed is ~6.6 min per EfficientNetV2-B0 training epoch, so the ~7 Phase 3 runs would take ~18–20 hours on CPU versus ~1.5–2 hours on a Kaggle T4. Only the small 384 px image cache (59 MB) is uploaded; everything else (data prep, evaluation, API, UI) stays local.

### 3.1 Variant ablation A vs. B (seed 0)

Launched both variants in one GPU session (runs record git `552851e`, clean tree):

```bash
uv run coffeeguard remote train -c configs/train/effnetv2_b0_partial.yaml -c configs/train/effnetv2_b0.yaml --seeds 0
```

- **A (spec):** after the linear probe, fine-tune only the last 3 stages + `conv_head`, LR 1e-4.
- **B (modern):** after the linear probe, fine-tune everything, LR 3e-4 with layer-wise decay 0.75, weight decay 0.05.

**Results** (Kaggle **Tesla T4**, git `552851e`, clean tree; ~10 s per epoch; whole runs took **3.9 min (A)** and **3.0 min (B)** — versus ~2.5–3 h each projected on the laptop CPU):

| Variant | Trainable in fine-tune | Best val macro-F1 | Val accuracy | Best epoch | Early stop | s/epoch |
|---|---|---:|---:|---|---|---:|
| **A** — last 3 stages (spec) | 5.67 M (97%) | **0.9848** | 0.9841 | finetune 9 (raw) | finetune 14 | 10.1 |
| **B** — all layers, layer decay 0.75 | 5.86 M (100%) | 0.9839 | 0.9841 | finetune 3 (raw) | finetune 8 | 10.7 |

Curves: `artifacts/figures/training/ablation_A_effnetv2_b0_partial-s0.png`, `…/ablation_B_effnetv2_b0-s0.png`.

**What the numbers say**
- **Same accuracy (6 errors of 377).** The F1 difference is < 1 image — a tie on val.
- **"Partial" is not a small model.** EfficientNetV2-B0 keeps 72% of its parameters in the last stage and 97% in the last three, so variant A trains almost every weight. It doesn't meaningfully reduce overfitting capacity, and on the GPU it isn't faster either (data loading dominates: 10.1 vs 10.7 s/epoch).
- **No harmful overfitting in either run.** B peaks at fine-tune epoch 3 and its val F1 then dips to 0.976–0.979 while train accuracy reaches 0.996 — but that dip is only ~2 images and **B's val loss keeps falling** (0.20 → 0.12), so it is noise, not overfitting. A climbs more slowly and stays flat at 0.980–0.985 over epochs 5–14. In any case early stopping + keeping the best checkpoint means the saved model is the best val epoch, never the last.
- **The linear probe plateaus by epoch 5** (0.847 → 0.851 over epochs 5–8) and is identical in both runs to four decimals (same seed, same frozen backbone — the pipeline is reproducible).
- **EMA weights rarely won here** (decay 0.999 with only 27 steps/epoch at batch 64 → the average lags the fast-improving weights); the trainer picks raw or EMA per epoch, so nothing is lost.

**Decisions** (discussed with you before implementing)
1. **Variant A is the main recipe:** tied on accuracy, a flatter val curve (the result depends less on which epoch is picked), and it's the spec's choice.
2. **Shorter schedule** for every recipe (except the B record): linear probe 8 → **5** epochs (patience 3), fine-tuning 25 → **15** epochs (patience 4). The best epochs were 3 and 9, so nothing is lost; early stopping still applies.
3. **One seed** for all runs. Instead of mean ± std over 3 seeds, headline test metrics get a **bootstrap 95% CI** (Phase 4) — no extra training. (Deviation from `IMPLEMENTATION_PLAN.md` §6 Phase 3 step 3.)
4. The comparison models (MobileNetV3-Large, EfficientNet-B0) and a GPU re-run of MobileNetV3-Small use **the same recipe**, so the comparison differs only in the backbone.

### 3.2 Final Phase 3 batch (seed 0, one GPU session)

```bash
uv run coffeeguard remote train -c configs/train/effnetv2_b0_partial.yaml -c configs/train/mobilenetv3_large.yaml \
  -c configs/train/efficientnet_b0.yaml -c configs/train/mobilenetv3_small.yaml --seeds 0
```

Git state `552851e` + the uncommitted recipe changes above (`dirty: true`); each run's `config.yaml` holds the full resolved recipe, so the runs remain exactly reproducible.

**Results** (val, 377 images, seed 0, Kaggle Tesla T4; `artifacts/metrics/phase3_val_summary.json`):

| Model | Val macro-F1 | Val accuracy | Errors | Best epoch | Fine-tune epochs run | Run time | Checkpoint |
|---|---:|---:|---:|---|---:|---:|---:|
| **EfficientNetV2-B0** (main, variant A) | **0.9848** | 0.9841 | 6 | finetune 9 (raw) | 13 | 3.8 min | 23 MB |
| EfficientNet-B0 | 0.9843 | 0.9841 | 6 | finetune 4 (EMA) | 8 | 3.1 min | 16 MB |
| MobileNetV3-Small (GPU, same recipe) | 0.9843 | 0.9841 | 6 | finetune 5 (raw) | 9 | 3.0 min | 6 MB |
| MobileNetV3-Large | 0.9794 | 0.9788 | 8 | finetune 4 (raw) | 8 | 2.9 min | 17 MB |
| *MobileNetV3-Small — Phase 2 run (CPU, batch 32, all-layer schedule)* | *0.9913* | *0.9920* | *3* | *finetune 9 (EMA)* | *14* | *27.7 min* | *6 MB* |

Curves: `artifacts/figures/training/<model>-s0-gpu.png`.

**Reading the results**
- **The shorter schedule lost nothing:** the main model reached exactly the ablation's result (0.9848, best at fine-tune epoch 9) in 3.8 instead of 3.9 min, with early stopping at fine-tune epoch 13.
- **Val cannot separate the models.** Three models make exactly 6 errors of 377, the others 3 and 8. One image is 0.27 pp of accuracy; with one seed these differences are within noise. The Phase 2 CPU run's 0.991 is not evidence that MobileNetV3-Small is better — it differs from the GPU run in batch size (32 vs 64), schedule (all layers + layer decay) and seed-level randomness, and the gap is 3 images.
- The deployment choice therefore rests on Phases 4–7: **test-set macro-F1 with bootstrap CIs (and a paired bootstrap between the top two)**, calibration, robustness to image corruptions, the shortcut test, OOD detection, CPU latency and size — as the plan intended.

### Phase 3 status ✅ (one seed)

| Gate item | Status |
|---|---|
| Stage 1 linear probe + stage 2 fine-tune, variants A vs. B compared on val | ✅ tie on accuracy; **A chosen** (flatter val curve, spec's choice) |
| Winning variant trained | ✅ seed 0 only — 3 seeds replaced by bootstrap CIs in Phase 4 (decided with you) |
| Comparison candidates with the same recipe (MobileNetV3-Large, EfficientNet-B0) | ✅ (+ MobileNetV3-Small re-run on GPU) |
| Training-curve figures per run | ✅ |

GPU time used this week: about 25 min of 6 h (two kernels: 9 + 15 min).

---

## Phase 4 — Evaluation, calibration, uncertainty

### 4.1 What was built

| File | Purpose |
|---|---|
| `evaluation/evaluate.py` | `evaluate_bundle()`: predicts val + test **through the exported ONNX bundle** (exactly what the API serves), fits temperature and conformal q̂ on **val only**, reports test metrics, writes `artifacts/eval/<bundle>/metrics.json`, `predictions_{val,test}.parquet` (probabilities, set size, quality measures joined; git-ignored) and `embeddings_{val,test}.npy` (for the Phase 6 OOD gate; git-ignored), and **writes `temperature` + `conformal_qhat` into `bundle.json`** so the API serves calibrated probabilities. `compare()`: summary table + paired bootstrap of the main model vs. each alternative. |
| `evaluation/calibration.py` | temperature scaling (1-D bounded search on log T, minimising val NLL), reliability bins, ECE (15 bins) |
| `evaluation/conformal.py` | split conformal prediction, LAC score; sets never empty; coverage / set size overall and per class |
| `evaluation/metrics.py` | + `bootstrap_ci` (2,000 resamples, percentile 95% CI for accuracy and macro-F1), `paired_bootstrap_diff` |
| `evaluation/figures.py` | test confusion matrix (counts + row %), reliability diagram before/after (val, test), selective-accuracy curve, test-error gallery → `artifacts/eval/<bundle>/figures/` |
| `cli.py` | `coffeeguard evaluate -b <main bundle> -b <other> …` → also `artifacts/metrics/test_metrics.json` |

Tests: `tests/unit/test_evaluation.py` (temperature recovers a known ×2 over-confidence; ECE ≈ 0 on calibrated synthetic data and large when not; conformal coverage holds on fresh data at α 0.10 and 0.02; sets never empty; bootstrap CI brackets the estimate; paired diff of identical predictions is 0). The slow smoke test now also runs `evaluate_bundle` end to end on its exported bundle.

**Run:** the four Phase 3 models were exported (`cand-effv2b0`, `cand-effb0`, `cand-mnv3s`, `cand-mnv3l`; torch ↔ ONNX max |Δlogit| ≤ 5e-6, argmax agreement 100%, `cam_exact` true for both EfficientNets) and evaluated together with the Phase 2 baseline bundle — **2.5 min on CPU for all five**. This is the one time the test split was opened.

```bash
uv run coffeeguard evaluate -b artifacts/models/cand-effv2b0 -b artifacts/models/cand-effb0 \
  -b artifacts/models/cand-mnv3s -b artifacts/models/cand-mnv3l -b artifacts/models/coffeeguard-mnv3s-baseline
```

### 4.2 Decision: conformal level α = 0.02 (not the planned 0.10)

With α = 0.10 every prediction set had exactly one class (coverage = accuracy for all models): the models are ~98% accurate, so a 90% guarantee is met by the top class alone and the sets carry no information. Measured on the main model (q̂ fitted on val, evaluated on test):

| α (target coverage) | q̂ | Test coverage | Avg set size | Predictions with > 1 class | Test errors flagged by a set |
|---|---:|---:|---:|---:|---:|
| 0.10 (90%) | 0.035 | 0.979 | 1.000 | 0% | 0% |
| 0.05 (95%) | 0.139 | 0.979 | 1.000 | 0% | 0% |
| **0.02 (98%)** | **0.742** | **0.987** | **1.026** | **2.6%** | **38%** |
| 0.01 (99%) | 0.996 | 0.997 | 1.799 | 51% | 88% |

α = 0.02 is the useful operating point: coverage within ±2 pp of its target, only 2.6% of answers become "uncertain", and those catch over a third of the errors. α = 0.01 would make half of all answers uncertain, and its q̂ (0.996) rests on the ~4 hardest of 377 val images. The CLI default is now 0.02.

### 4.3 Results (test, 379 images, opened once)

`artifacts/metrics/test_metrics.json`; per-model details in `artifacts/eval/<bundle>/metrics.json`.

| Model | Val macro-F1 | **Test macro-F1 [95% CI]** | Test errors | ECE before → after | T | Conformal coverage (98%) | Avg set size |
|---|---:|---|---:|---|---:|---:|---:|
| **EfficientNetV2-B0** (main) | 0.983 | **0.979** [0.963, 0.993] | 8 | 0.098 → **0.012** | 0.55 | 0.987 | 1.026 |
| EfficientNet-B0 | 0.984 | 0.974 [0.957, 0.988] | 10 | 0.094 → 0.016 | 0.47 | 0.982 | 1.026 |
| MobileNetV3-Small (Phase 2, CPU) | 0.991 | 0.966 [0.947, 0.984] | 13 | 0.070 → 0.025 | 0.58 | 0.966 | 1.000 |
| MobileNetV3-Small (GPU) | 0.984 | 0.960 [0.938, 0.979] | 15 | 0.099 → 0.023 | 0.47 | 0.979 | 1.034 |
| MobileNetV3-Large | 0.979 | 0.958 [0.935, 0.977] | 16 | 0.087 → 0.018 | 0.49 | 0.984 | 1.061 |

(Val macro-F1 here is recomputed from the ONNX bundle in FP32; training scored val with FP16 autocast on the GPU, which flipped one borderline image for EfficientNetV2-B0: 0.9848 → 0.9826.)

**Paired bootstrap, main model minus each alternative** (same 379 test images, 2,000 resamples):

| vs. | Δ macro-F1 | 95% CI | P(main not better) |
|---|---:|---|---:|
| EfficientNet-B0 | +0.005 | [−0.008, +0.020] | 0.24 — a tie |
| MobileNetV3-Small (GPU) | +0.019 | [−0.001, +0.040] | 0.03 |
| MobileNetV3-Small (Phase 2 CPU) | +0.013 | [−0.004, +0.032] | 0.07 |
| MobileNetV3-Large | +0.022 | [+0.004, +0.041] | 0.007 — better |

**Main model in detail (EfficientNetV2-B0, test)**
- Per-class F1: Healthy 0.986 · Cercospora 0.974 · Leaf Rust 0.972 · Phoma 0.986; macro ROC-AUC 0.997.
- Confusion (rows = true Healthy/Cercospora/Leaf Rust/Phoma): `[[107,0,1,0],[0,74,1,0],[1,3,122,0],[1,0,1,68]]` — the main confusion is **Leaf Rust → Cercospora (3)**, the pair the label audit singled out.
- **Calibration:** T = 0.55 (< 1): the models were *under*-confident, as expected after training with label smoothing 0.1. Temperature scaling brings ECE from 0.098 to 0.012 (target ≤ 0.05 ✓). The reliability diagram is dominated by the top bin: only 14 of 379 test images have confidence < 0.8, so the low-confidence points are single images and look jumpy.
- **Confidence buckets (spec):** 0–0.4: none · 0.4–0.6: 5 images, 60% correct · 0.6–0.8: 9 images, 67% correct · 0.8–1.0: 365 images, **99.2% correct**. Answering only the most confident 95% (confidence ≥ 0.88) gives 99.2% accuracy; 90% (≥ 0.96) gives 99.1%.
- **Error rate by photo quality** (val + test, quartiles of 189 images): the lowest quartile of sharpness, brightness, contrast and green fraction each has 3.2% errors vs. 0.5–2.6% elsewhere — a mild trend resting on 6 vs. 1–5 errors, not evidence of a quality shortcut. Phase 5's corruption sweep and shortcut test measure this properly.

**Are the remaining errors label noise?** 4 of the 8 test errors — including the two most confident (0.997 and 0.983: "Leaf Rust" leaves with Cercospora-like brown spots and yellow halos) — are among the 19 images the Phase 1 cleanlab audit flagged independently, with a different model (DINOv2). Leaving out the 5 flagged test images: accuracy 0.989, macro-F1 0.990 (4 errors of 374). Nothing is excluded without an expert's check; `artifacts/eval/cand-effv2b0/figures/errors_test.png` is the gallery to review, together with `artifacts/figures/eda/label_issues.png`. Conformal sets flag 3 of the 8 errors as uncertain (set size 2).

### Phase 4 status ✅

| Gate item / success target | Result |
|---|---|
| Full metrics report for the main model and the baselines | ✅ incl. bootstrap CIs, per-class, confusion, ROC-AUC |
| Test macro-F1 ≥ 0.90, every class F1 ≥ 0.85 | ✅ 0.979; lowest class 0.972 |
| ECE ≤ 0.05 after temperature scaling | ✅ 0.012 |
| Conformal coverage within ±2 pp of target | ✅ 0.987 at the 98% target (α changed from 0.10, see 4.2) |
| Confidence buckets, selective curve, error galleries, error by quality | ✅ |
| Test set opened once | ✅ |

Workflow change recorded in `CONTRIBUTING.md` and the plan: all work is committed on `eleni-changes`, one commit per phase, no other branches.

---

## Phase 5 — Explainability and robustness

All on the laptop CPU through the ONNX bundles (calibrated temperature applied), on the test split. Commands:

```bash
uv run coffeeguard explain    -b artifacts/models/cand-effv2b0 -b artifacts/models/cand-effb0
uv run coffeeguard robustness -b artifacts/models/cand-effv2b0 -b artifacts/models/cand-effb0 \
  -b artifacts/models/cand-mnv3s -b artifacts/models/cand-mnv3l
```

### 5.1 What was built

| File | Purpose |
|---|---|
| `inference/cam.py` | **CAM from one ONNX forward pass** (NumPy + Pillow only): ReLU(Σ W_ck · A_k) on the 7×7 feature map, bilinear upsample, [0, 1]; plus a heat-map overlay without matplotlib. In the slim `inference` package so the API (Phase 8) can use it without torch. |
| `explainability/analysis.py` | per-class CAM galleries (most confident correct / errors and least confident), **leaf-focus score** (share of CAM mass inside the leaf mask), **deletion faithfulness** (hide 16 px patches hottest-first vs. random, track the calibrated probability of the predicted class) → `artifacts/explain/<bundle>/` |
| `robustness/corruptions.py` | 9 deterministic corruptions × 5 severities, seeded per image: brightness (darker), contrast, Gaussian blur, motion blur, Gaussian noise, JPEG, occlusion (grey patches), crop/zoom, rotation (border-colour fill) |
| `robustness/leafmask.py` | colour leaf segmentation (saturated non-blue or dark pixels → largest component → holes filled) and masking |
| `robustness/sweep.py`, `robustness/figures.py` | corruption sweep (accuracy, macro-F1, mean confidence, ECE, where errors go, confusion) + shortcut test + degradation-curve figure → `artifacts/robustness/<bundle>/`, `artifacts/robustness/summary.csv` |
| `cli.py` | `coffeeguard explain`, `coffeeguard robustness` |

**Tests** (`tests/unit/test_explain_robustness.py`, 12 tests): the NumPy CAM **matches `pytorch-grad-cam`'s Grad-CAM** on an EfficientNet (correlation > 0.99) — so the claim "CAM = Grad-CAM for a pool→linear head" is verified, not assumed; CAM range and overlay size; every corruption is deterministic for a given seed, keeps the size and changes the image; the leaf mask separates a synthetic leaf (with a dark lesion) from pale-blue paper.

**Leaf-mask check by eye** (`artifacts/robustness/cand-effv2b0/figures/shortcut_examples.png`): the first version missed most of the very dark Cercospora leaves on blue paper (barely saturated); dark pixels (V < 90) were added — paper is bright — and the masks then cover whole leaves including lesions. In **field photos** (leaf on the tree) the mask covers the whole frame because the surrounding foliage is green too; those images (51 Healthy, 30 Leaf Rust) are left out of the shortcut test and the leaf-focus score, which therefore use the **298 test photos with a separable background**.

### 5.2 Explainability (main model: EfficientNetV2-B0)

| | EfficientNetV2-B0 | EfficientNet-B0 |
|---|---:|---:|
| **Deletion AUC, CAM order** (lower = faster drop) | **0.450** | 0.620 |
| Deletion AUC, random order | 0.611 | 0.869 |
| Images where CAM order drops faster | **87%** | 89% |
| **Leaf focus** (CAM mass on the leaf; leaf = 24% of the image) | **0.61** | 0.62 |
| Leaf focus: correct / errors | 0.62 / 0.40 | 0.63 / 0.51 |
| Leaf focus per class: Healthy / Cercospora / Leaf Rust / Phoma | 0.68 / 0.40 / 0.68 / 0.69 | 0.75 / 0.44 / 0.68 / 0.65 |

- **The CAM is faithful:** hiding the hottest 5% of patches drops the predicted-class probability from 0.97 to 0.72, a random 5% only to 0.88.
- **The model looks at the leaf and its lesions:** 61% of the CAM mass falls on a leaf that covers 24% of the image (2.5× its area share); Cercospora leaves are small in the frame (14%), so 0.40 is ~3× their share. In the galleries (`artifacts/explain/cand-effv2b0/figures/cam_<class>.png`) the hot spots sit on the brown spots with yellow halos, rust pustules and Phoma patches. Most of the "off-leaf" mass is the halo of a 7×7 map upsampled to 224 px.
- **Errors look more at the background** (0.40 vs. 0.62). Example in the Cercospora gallery: a second leaf with orange spots in the frame corner drew the CAM and the answer "Leaf Rust" (0.98).

### 5.3 Robustness (test, 379 images, 9 corruptions × 5 severities)

**Relative robustness** = mean corrupted macro-F1 / clean macro-F1 (target ≥ 0.85 at severity ≤ 3):

| Model | Clean macro-F1 | Rel. robustness sev 1–3 | sev 1–5 | Worst at severity 5 |
|---|---:|---:|---:|---|
| **EfficientNetV2-B0** | 0.979 | **0.975** ✅ | 0.926 | noise 0.64, occlusion 0.65, JPEG 0.70 |
| EfficientNet-B0 | 0.974 | 0.980 | 0.931 | noise 0.45, JPEG 0.73 |
| MobileNetV3-Small | 0.960 | 0.991 | 0.963 | noise 0.73, blur 0.77 |
| MobileNetV3-Large | 0.958 | 0.983 | 0.934 | noise 0.37, contrast 0.76 |

EfficientNetV2-B0 macro-F1 by severity 1→5: brightness 0.979→0.880 · contrast 0.971→0.824 · Gaussian blur 0.975→0.804 · motion blur 0.979→0.828 · Gaussian noise 0.959→0.644 · JPEG 0.973→0.698 · occlusion 0.963→0.654 · crop/zoom 0.987→0.934 · rotation 0.970→0.869 (`artifacts/robustness/cand-effv2b0/figures/degradation.png`). All models are robust up to severity 3; the small MobileNetV3 degrades least in relative terms (from a lower start). Under strong corruption confidence falls (0.97 → ~0.75) but calibration loosens (ECE 0.06–0.12 at severity 5).

**Where the errors go — evidence about the EDA quality confound.** EDA found that low photo quality co-occurs with Cercospora/Leaf Rust. If the model had learned "low quality ⇒ Cercospora/Leaf Rust", degrading photos would push predictions *towards* those classes. The opposite happens: under blur, JPEG, noise, low contrast, darkness and occlusion almost all new errors are **"Healthy"** (e.g. JPEG severity 5: 84 of 110 errors; noise severity 5: 116 of 135). When lesions become invisible the model says "no disease" — the natural failure mode, not a quality shortcut. **Safety consequence:** a diseased leaf in a bad photo tends to be called Healthy. Phase 6's quality gate (reject too dark / blurry / low-contrast photos and ask for a retake) exists for exactly this.

### 5.4 Shortcut test (leaf only vs. background only)

298 test photos with a separable background; masked area filled with grey. Majority-class rate of this subset: 0.322.

| Model | Original | Leaf only | **Background only** (target ≤ 0.40) | Mean confidence on background only |
|---|---:|---:|---:|---:|
| **EfficientNetV2-B0** | 0.980 | 0.926 | **0.399** ✅ (just) | 0.83 |
| EfficientNet-B0 | — | 0.903 | 0.416 | 0.70 |
| MobileNetV3-Small | — | 0.956 | 0.453 | 0.89 |
| MobileNetV3-Large | — | 0.919 | 0.356 | 0.71 |

**The headline number hides a real finding.** Per class for EfficientNetV2-B0 (rows = true class; `artifacts/robustness/cand-effv2b0/shortcut_confusion.json`):

| Background only → | Healthy | Cercospora | Leaf Rust | Phoma |
|---|---:|---:|---:|---:|
| Healthy (white paper) | **57** | 0 | 0 | 0 |
| Cercospora (blue paper) | 11 | **21** | 43 | 0 |
| Leaf Rust (blue / white paper) | 48 | 7 | **41** | 0 |
| Phoma (white paper) | 70 | 0 | 0 | **0** |

- With the leaf blanked out, **64 of 75 Cercospora backgrounds are still called "Cercospora or Leaf Rust"** — the two classes photographed on blue paper — and every white-paper background defaults to Healthy. So the model has partly learned the **photo setup** (blue paper ⇒ a leaf-spot disease), a direct consequence of each class being photographed in its own setup. The ≤ 40% target is met only because Healthy is the default answer.
- With the leaf visible the leaf dominates: leaf-only accuracy 0.926 (Healthy 1.00, Cercospora 0.95, Leaf Rust 0.95), CAMs sit on lesions, and the corruption errors point away from the quality shortcut. The exception is **Phoma: with its white background removed, 11 of 70 Phoma leaves slip to Healthy** (leaf-only recall 0.81).
- The model is **confidently wrong on backgrounds** (mean confidence 0.83 with no leaf present) — input that isn't a leaf has to be caught by the Phase 6 OOD gate, not by the softmax.

**Limitation and possible fix:** in the field, photos won't come on blue paper, so the blue-paper cue can't help there and could hurt. The standard remedy is **background-swap augmentation** (paste masked leaves onto other classes' backgrounds / neutral colours during training) — about 4 GPU minutes to retrain; offered as an optional step, not done here. More field-style photos per class would fix it at the source.

### 5.5 Follow-up: background-swap augmentation (decided with you)

**Why:** the shortcut test (5.4) showed the model partly learned each class's photo setup. Field photos won't come on blue paper, so a background cue can only hurt there. The fix had to come before Phase 6 (OOD thresholds are fitted on the main model's outputs) and Phase 7 (export/comparison).

**What was built**
- `data/bgswap.py` → `BackgroundSwap`: with probability `augment.bg_swap_p` the training leaf is cut out with the Phase 5 colour mask and pasted (1.5 px soft edge) onto **another training photo's background** (its own leaf painted over with its median background colour and blurred; 60% of swaps) or a **plain background** (paper white, grey, pale blue, soil brown, foliage green, with a gentle gradient and fine noise; 40%). Field photos (mask > 80% of the frame) and empty masks are left unchanged. Donors: up to 400 training photos. Picklable (Windows workers).
- `AugmentConfig.bg_swap_p` (default 0 → all earlier recipes unchanged); `LeafDataset(pre_transform=…)`; the trainer builds the swap from the (preloaded) training images; `data aug-preview` shows it (`artifacts/figures/augmentation_preview_bgswap.png` — Cercospora on white paper and green, Leaf Rust on blue and maroon, Phoma on teal and blue; lesions intact).
- Recipe `configs/train/effnetv2_b0_bgswap.yaml` = the main recipe + `augment.bg_swap_p: 0.5` — the only difference.
- Test (`tests/unit/test_transforms.py`): leaf pixels kept, background replaced by the donor's, field photo unchanged, picklable.
- `coffeeguard robustness` now rebuilds `summary.csv` from every bundle evaluated so far (a single-bundle run used to overwrite it).

**Run:** Kaggle T4, seed 0, 6 min kernel (run `20260926-042311-effnetv2_b0_bgswap-s0`, git `2bdbee7` + the uncommitted swap code; `dirty: true`, committed right after). Exported as `artifacts/models/cand-effv2b0-bgswap` (parity max |Δlogit| 8e-7, argmax 100%). Val macro-F1 **0.9869** (5 errors) vs. 0.9848 (6) without the swap. Then the full Phase 4–5 checks.

**Results — same model and seed, with vs. without background swap**

| Check | Without | **With swap** | |
|---|---:|---:|---|
| Test macro-F1 [95% CI] | 0.979 [0.963, 0.993] | **0.974** [0.956, 0.990] | paired Δ −0.006 [−0.018, +0.006] — not significant |
| Test errors (of 379) | 8 | 10 | |
| ECE after temperature scaling | 0.012 | 0.014 | |
| Conformal coverage (98% target) / avg set size | 0.987 / 1.026 | 0.974 / — | within ±2 pp |
| **Leaf-only accuracy** (298 paper-background photos) | 0.926 | **0.963** | |
| ↳ Phoma leaf-only recall | 0.81 (57/70) | **0.96** (67/70) | the Phoma weakness is fixed |
| Background-only accuracy | 0.399 | 0.386 | |
| **Mean confidence on background-only images** | 0.83 | **0.54** | much less sure without a leaf |
| Blue-paper backgrounds (Cercospora + Leaf Rust) still called Cercospora/Leaf Rust | 65.5% | 65.5% | unchanged |
| Relative robustness, severity 1–3 / 1–5 | 0.975 / 0.926 | **0.979 / 0.938** | better at severity 5 for 7 of 9 corruptions |
| Leaf focus (CAM mass on the leaf) | 0.61 | **0.66** | Phoma 0.69 → 0.77, Healthy 0.68 → 0.74 |
| Deletion AUC, CAM vs. random | 0.450 vs. 0.611 | 0.445 vs. 0.630 | still faithful |

Background-only confusion with the swap (rows = true Healthy/Cercospora/Leaf Rust/Phoma; columns = predicted): `[[56,1,0,0],[11,26,37,1],[46,16,33,1],[53,17,0,0]]`.

**Decision: the background-swap model (`cand-effv2b0-bgswap`) becomes the main model.** It relies more on the leaf (leaf-only +3.7 pp, leaf focus +5 pp), fixes the Phoma-without-background weakness, is far less confident when no leaf is present, and is slightly more robust. The small test drop is not significant — and expected, because the test set shares the photo-setup bias that the swap removes.

**What the swap did not fix:** blue-paper backgrounds with the leaf blanked out still *lean* towards Cercospora/Leaf Rust, now at ~0.54 confidence. A likely reason is that the background-only test keeps a grey leaf-shaped silhouette with its paper shadows, so shape and setup cues remain. The real fix is field photos of every class; this stays a documented limitation (to go into the model card).

**Paired bootstrap vs. the new main model** (test; `artifacts/metrics/test_metrics.json`): EffV2-B0 without swap −0.006 (tie) · EfficientNet-B0 −0.000 (tie) · MobileNetV3-Small +0.014 [−0.009, +0.036] · MobileNetV3-Large +0.016 [−0.002, +0.036], P(not better) 0.04 · MobileNetV3-S baseline +0.008 (tie).

### Phase 5 status ✅

| Gate item / success target | Result |
|---|---|
| CAM (NumPy) verified against Grad-CAM | ✅ correlation > 0.99 (unit test) |
| CAM galleries per class, correct vs. incorrect, high vs. low confidence | ✅ `artifacts/explain/<bundle>/figures/` |
| Faithfulness (deletion) and leaf-focus score | ✅ AUC 0.45 vs. 0.61 random; leaf focus 0.61 |
| Robustness table/curves; relative robustness ≥ 0.85 at severity ≤ 3 | ✅ 0.975 (all four models ≥ 0.975) |
| Shortcut test: background-only accuracy ≤ 40% | ✅ 0.399 (0.386 with background swap) — blue-paper backgrounds still carry class information (5.4, 5.5) |
| Main model after Phase 5 | **`cand-effv2b0-bgswap`** (EfficientNetV2-B0 + background swap), see 5.5 |

Not done (scope, per plan §11): Grad-CAM++ / Eigen-CAM comparison (stretch item) and the narrative notebook.

---

## Phase 6 — Rejection gates: photo quality and out-of-distribution (OOD)

Main model: `cand-effv2b0-bgswap`. Everything the API needs is written into its bundle (`bundle.json` → `quality_thresholds`, `ood`, `tau_conf`; `ood_knn_bank.npy`). The plan put the thresholds in `configs/serve.yaml`; they live in the bundle instead, so a model and its thresholds can never get out of sync.

```bash
uv run coffeeguard ood collect                                   # OOD images -> data/ood (git-ignored)
uv run coffeeguard ood fit -b artifacts/models/cand-effv2b0-bgswap  # -> artifacts/ood/<bundle>/ood.json + bundle
```

### 6.1 OOD image set (split by source)

| Split | Near-OOD (other plant leaves) | Far-OOD (not a leaf) |
|---|---|---|
| **cal** — fits thresholds | tomato leaves 50 (Kaggle `kaustubhb999/tomatoleaf`, CC0) · banana leaves 50 (Kaggle `shifatearman/bananalsd`, CC BY-SA 4.0, original photos only) | animals 50 (HF `Francesco/animals-ij5d2`, CC BY 4.0) · landscapes 36 (Windows wallpapers on this PC; used locally, not redistributed) |
| **test** — reported only | bean leaves 50 (HF `AI-Lab-Makerere/beans`, MIT) | indoor scenes 42 (HF `keremberke/indoor-scene-classification`, CC BY 4.0) · rendered text/screenshots 50 (HF `nateraw/rendered-sst2`, see dataset card) · synthetic 40 (blank, noise, gradients, fake screenshots; generated) |

186 cal + 182 test images, resized to 384 px like the training cache. No source appears in both splits, so the test numbers show how the gate handles kinds of images it was never tuned on.

**Collection problems, and what changed:** per-file downloads from Kaggle were slow (~1 image / 3 s) and, parallelised, hit Kaggle's rate limit (HTTP 429). The collector now makes one request per dataset (zip → sample → delete zip) with exponential back-off on 429. The connection here is ~0.2–0.7 MB/s, so large Kaggle datasets (maize 169 MB, tea 776 MB, scenes 363 MB, …) were replaced by small Hugging Face files (≤ 15 MB each) and local images.

### 6.2 What was built

| File | Purpose |
|---|---|
| `ood/collect.py` | the source table above (Kaggle / HF / local / synthetic), reproducible sampling (seed 0) |
| `inference/ood.py` | **NumPy-only OOD scores** from the one ONNX pass: MSP, energy, Mahalanobis (class means + Ledoit-Wolf shared covariance), **KNN** (1 − cosine similarity to the 10th-nearest training embedding); `load_state()` rebuilds the fitted state from a bundle |
| `inference/decision.py` | **the serving decision**: quality gate → OOD gate → conformal set + confidence → `accepted` / `uncertain` (low_confidence) / `rejected` (low_quality or ood), with plain-language advice ("too dark — take it in daylight or open shade", …) |
| `ood/fit.py` | fits quality thresholds, compares scorers, sets τ_ood and τ_conf on train / val / OOD-cal; reports on test / OOD-test; writes the bundle; score-histogram figure |
| `cli.py` | `coffeeguard ood collect`, `coffeeguard ood fit` |

Tests (`tests/unit/test_ood_decision.py`): KNN and Mahalanobis score training-like embeddings lower than far ones; MSP and energy prefer confident logits; every decision path (accepted, uncertain with a 2-class set, rejected-ood, rejected-low_quality checked first); **the quality gate rejects a blank and a darkened image and passes a normal leaf** (the plan's gate tests).

### 6.3 Choosing the OOD scorer (on cal)

| Scorer | cal AUROC near / far | test AUROC near / far | test near-OOD accepted at 95% ID |
|---|---|---|---:|
| MSP | 0.978 / 0.989 | 0.944 / 0.990 | 20% |
| Energy | 0.986 / 0.994 | 0.958 / 0.994 | 18% |
| Mahalanobis | 0.997 / 0.987 | 0.990 / **0.469** | 2% (far: 54%!) |
| **KNN (chosen)** | **1.000 / 1.000** | **0.993 / 1.000** | **2%** (far: 0%) |

KNN wins on every split. Mahalanobis looks fine on cal but collapses on the far-OOD test sources (text, blank and noise frames sit close to the class means in its metric) — a reason to test on unseen sources. Softmax-based scores (MSP, energy) let 18–20% of other plant leaves through: the classifier is confidently wrong on them, as expected.

### 6.4 Quality gate — fitted to the model, not to the training photos

**First version:** thresholds at the 0.5th percentile of the training photos. It rejected 94% of photos darkened to half brightness and 100% of mildly blurred ones — but the robustness sweep (5.3) showed the model still scores ~0.95 F1 on exactly those. Every training photo is bright and sharp, so "unlike the training photos" is far stricter than "the model can't handle it". It also missed noise and JPEG artefacts, which do hurt the model.

**Final version:** for brightness, contrast and sharpness, corrupted **val** photos were run through the model at severities 1–5; the threshold sits where val accuracy drops below **90%** (midpoint between the last good and first bad severity's median quality), or beyond the worst severity if accuracy never drops that far. Over-exposure and leaf-colour fraction keep the training-percentile rule.

| Measure | Training 0.5th pct | Val accuracy by severity 1→5 | **Threshold** |
|---|---:|---|---:|
| brightness (mean, 0–255) | 89.7 | 0.984 · 0.984 · 0.981 · 0.958 · 0.934 (never < 0.90) | **41.3** |
| contrast (RMS) | 24.5 | 0.984 · 0.979 · 0.971 · 0.955 · **0.873** | **12.5** |
| sharpness (Laplacian var.) | 39.0 | 0.976 · 0.971 · 0.952 · 0.920 · **0.870** | **1.6** |
| max brightness / min leaf-colour fraction | 203.0 / 0.021 | — | 203.0 / 0.021 |

| Rejected by the quality gate | Training-percentile version | **Final** |
|---|---:|---:|
| genuine val / test photos | 1.3% / 1.6% | **0.3% / 0.5%** |
| darkened ×0.5 (sev 3) / ×0.25 (sev 5) | 94% / 100% | **7% / 76%** |
| contrast sev 3 / sev 5 | 98% / 100% | **26% / 100%** |
| Gaussian blur sev 3 / sev 5 | 100% / 100% | **14% / 51%** |
| Gaussian noise, JPEG (sev 5) | 0% / 4% | 0% / 4% — not caught |
| OOD test images | 56% | 50% |

Known gap: noise and heavy JPEG raise or keep the "sharpness" measure, so the gate can't see them; those photos reach the model, which then tends to answer Healthy (5.3). A noise/compression detector is future work.

### 6.5 OOD threshold and confidence threshold

- **τ_ood** — the first version used the val 95th percentile. On test it rejected 25 genuine photos (6.6%), mostly Cercospora (15%) and Leaf Rust from one phone-camera source on blue paper; field photos were fine (1 of 81 rejected). Per-class thresholds barely helped (Cercospora 85% → 88% accepted, Healthy 100% → 96%). Instead the acceptance level is now chosen **on cal data**: the highest val percentile (95–99.5%) at which ≤ 1% of OOD-cal images pass → **99.5th percentile, τ = 0.554** (OOD-cal images all score ≥ 0.534 near / ≥ 0.606 far).
- **τ_conf = 0.947** — the lowest confidence at which val predictions that pass both gates are ≥ 99% correct; below it (or when the 98% conformal set has > 1 class) the answer is `uncertain` with the top candidates.

### 6.6 End-to-end results (test sets; fitted on train / val / OOD-cal only)

| Genuine coffee-leaf test photos (379) | | OOD test images (182) | |
|---|---:|---|---:|
| **accepted** | **344 (90.8%)** — **99.4% correct** | rejected low_quality | 91 |
| uncertain (top candidates + retake advice) | 28 (7.4%) | rejected ood | 84 |
| rejected ood | 5 (1.3%) | uncertain | 4 |
| rejected low_quality | 2 (0.5%) | **accepted** | **3** (all bean leaves) |

OOD gate alone, per test source: indoor scenes, rendered text and synthetic frames 0% accepted; bean leaves 14% accepted (the hardest case: a different green leaf with spots). OOD AUROC on test: **near 0.993, far 1.000** (targets ≥ 0.85 / ≥ 0.95 ✅).

### Phase 6 status ✅

| Gate item / success target | Result |
|---|---|
| OOD data with sources and licences, split by source | ✅ 186 cal / 182 test, 8 sources (6.1) |
| Quality gate + OOD scorers compared, best chosen on cal | ✅ quality gate fitted to model accuracy; KNN chosen |
| AUROC ≥ 0.95 far-OOD, ≥ 0.85 near-OOD | ✅ 1.000 / 0.993 |
| Thresholds stored for serving | ✅ in `bundle.json` (not `configs/serve.yaml`, see top) |
| Decision function + gate tests (blank, dark, near-OOD) | ✅ `inference/decision.py`, `tests/unit/test_ood_decision.py`; near-OOD measured on the bean-leaf source |

---

## Phase 7 — Model comparison, decision and release bundle

### 7.1 What was built

| File | Purpose |
|---|---|
| `export/benchmark.py` → `coffeeguard benchmark -b …` | CPU latency with ONNX Runtime (batch 1, 4 threads, 200 timed runs after 20 warm-up): model only, predict from a decoded photo, and **end to end** (decode an original phone JPEG + predict); ONNX size and parameter count → `artifacts/benchmark/<bundle>.json` |
| `evaluation/compare.py` → `coffeeguard compare -b …` | decision matrix from the Phase 4–6 results + benchmarks → `artifacts/metrics/model_comparison.{json,md}` |
| `docs/decisions/001-deployment-model.md` | the decision record (context, options, paired bootstrap, reasons, consequences) |
| `docs/MODEL_CARD.md` | intended use, decision pipeline with every threshold, data, training, performance, robustness, OOD, limitations |
| `artifacts/models/coffeeguard-effv2b0-v1/` | **release bundle** (model.onnx 23.5 MB, KNN bank 4.5 MB, classifier weights, `bundle.json` with version 1.0.0, all thresholds, headline metrics and SHA-256 of every file) — the path the API already defaults to |

Tests: `tests/unit/test_benchmark_compare.py` (benchmark on the hand-built ONNX bundle reports sizes and p50 ≤ p95; compare tolerates missing results and renders the table).

For a fair OOD comparison, `coffeeguard ood fit` was also run on the four other candidates (~8–10 min each on the CPU; each fits its own quality gate, KNN bank and thresholds).

### 7.2 Results (test split; full table in the ADR)

| Model | Test macro-F1 [95% CI] | Leaf-only acc | OOD AUROC near / far | Answered (accuracy) | Params | ONNX | CPU p50 |
|---|---|---:|---|---|---:|---:|---:|
| **EfficientNetV2-B0 + bg swap** | 0.974 [0.956, 0.990] | **0.963** | 0.993 / 1.000 | 90.8% (99.4%) | 5.83 M | 23.5 MB | 15.5 ms |
| EfficientNetV2-B0 | 0.979 [0.963, 0.993] | 0.926 | 0.990 / 1.000 | 94.5% (99.4%) | 5.83 M | 23.5 MB | 22.3 ms |
| EfficientNet-B0 | 0.974 [0.957, 0.988] | 0.903 | 0.990 / 1.000 | 92.3% (98.6%) | 3.99 M | 16.0 MB | 17.0 ms |
| MobileNetV3-Small | 0.960 [0.938, 0.979] | 0.956 | 0.997 / 0.992 | 87.9% (99.1%) | 1.52 M | 6.1 MB | 2.7 ms |
| MobileNetV3-Large | 0.958 [0.935, 0.977] | 0.919 | 0.995 / 1.000 | 89.4% (99.4%) | 4.19 M | 16.8 MB | 6.5 ms |

- **Latency noise:** the two EfficientNetV2-B0 rows are the same network, yet measured 15.5 vs. 22.3 ms — laptop timing varies by about ±30%, so only large latency differences mean anything.
- **End to end** (decode a 2048 px phone photo + preprocess + model), chosen model: **63 ms p50 / 72 ms p95**; with a 1024 px photo 42 ms. Target ≤ 100 ms ✅.

### 7.3 Decision (ADR 001)

**EfficientNetV2-B0 with background swap.** The three EfficientNets tie on test accuracy (paired CIs include 0); the background-swap model depends least on the photo setup (leaf-only 0.963, confidence on leaf-less images 0.54), which matters most for field photos; speed and size are not limiting. Trade-off: it answers slightly fewer photos directly (90.8% vs. 94.5%) at the same 99.4% accuracy. MobileNetV3-Small is recorded as the candidate for a future on-device app (retrain with background swap first).

### 7.4 Release bundle and the "reproduces exactly" gate

- `artifacts/models/coffeeguard-effv2b0-v1` = the `cand-effv2b0-bgswap` bundle + version, recipe, source run and headline metrics in `bundle.json`; SHA-256 recomputed for every file.
- **Reproduction check:** `Predictor` on the release bundle reproduces the Phase 4 test predictions with **max |Δ probability| = 0.0** and macro-F1 0.97390 = 0.97390 ✅ (the plan's Phase 7 gate).
- **Live API check** (FastAPI `TestClient`, release bundle, original full-size test photos): `/health` reports the model; Cercospora / Healthy / Leaf Rust predicted correctly at calibrated confidence 0.996–0.999 in 29–49 ms; a Phoma photo (one of the 10 test errors) came back as Healthy at **0.68** — below τ_conf 0.947, so the full decision would answer *uncertain*. Wiring the gates into the API is Phase 8.
- **Committed in git:** only this release bundle's weights (gitignore exception + pre-commit large-file exclusion for this folder), so a fresh clone and the Phase 10 Docker image can serve without retraining; candidate bundles keep only `bundle.json`.
- Not done: INT8 quantisation (not needed for the latency target; stretch item).

### Phase 7 status ✅

| Gate item | Result |
|---|---|
| Per-candidate test macro-F1, params, ONNX size, CPU latency p50/p95, robustness, OOD AUROC | ✅ `artifacts/metrics/model_comparison.md` |
| Paired bootstrap between models | ✅ (vs. the chosen model) |
| Decision recorded | ✅ `docs/decisions/001-deployment-model.md` |
| ONNX export with three outputs + parity; bundle with SHA-256 | ✅ `artifacts/models/coffeeguard-effv2b0-v1` |
| Model card | ✅ `docs/MODEL_CARD.md` |
| Bundle loads in `Predictor` and reproduces the test metrics exactly | ✅ max |Δp| = 0.0 |
| CPU p50 ≤ 100 ms | ✅ 15.5 ms model, 63 ms end to end |

---

## Phase 8 — FastAPI service

Run: `uv run uvicorn app.main:app --app-dir apps/api` (bundle from `MODEL_BUNDLE`, default the release bundle `artifacts/models/coffeeguard-effv2b0-v1`); interactive docs at `http://localhost:8000/docs`.

**Wording fix (your question after Phase 7):** the ADR and model card now say the laptop stands in for a small cloud CPU server — in a web app the model runs on the server, and for users on rural mobile networks the photo upload takes far longer than the model, so model speed only matters for server cost and capacity. Keeping uploads small is a Phase 9 (UI) item; an on-phone offline app is where model speed/size would really matter (MobileNetV3-Small).

### 8.1 What was built

| File | Purpose |
|---|---|
| `src/coffeeguard/inference/pipeline.py` | **`Pipeline`**: the whole serving path in the slim library (NumPy + Pillow + ONNX Runtime): EXIF-orient → quality gate (the model is skipped for rejected photos) → one ONNX pass → KNN OOD score → calibrated probabilities + conformal set → decision → optional CAM overlay (PNG, base64); per-step timings. Refuses a bundle without fitted gates. |
| `apps/api/app/main.py` | app factory; lifespan loads the `Pipeline` once; **request-ID middleware** (`X-Request-ID` generated or echoed, exposed via CORS); **structured JSON logs** — one line per request with id, method, path, status code, ms, and the decision (status / reason / label) or error code; consistent error body `{"error", "detail"}` incl. FastAPI's own 422 |
| `apps/api/app/api/routes.py` | `GET /health` (liveness, model name + version), `GET /model-info` (architecture, classes, metrics, every threshold, data fingerprint, source run), `POST /predict` (decision: status, reason, label, calibrated confidence, prediction set, issues, advice, version, latency), `POST /analyze` (the same + all probabilities, OOD score and threshold, quality report, CAM overlay, timings) |
| `apps/api/app/schemas/prediction.py` | typed request/response models with descriptions (shown in `/docs`) |
| `apps/api/app/settings.py` | + `LOG_LEVEL`; default bundle = the release bundle |
| `apps/web/streamlit_app.py` | minimal update to the new API (`/analyze`; shows rejected / uncertain with advice) — the full UI is Phase 9 |

Hardening from Phase 2 stays: magic-byte sniffing (JPEG/PNG/WebP, the Content-Type header is not trusted), 10 MB limit (reads at most limit + 1 byte), 100 MP pixel limit, full decode (truncated files), EXIF transpose.

### 8.2 Tests (`tests/integration/test_api.py`, 23 tests, ~3 s, no trained model)

The hand-built ONNX fixture (`tests/fixtures/bundle.py`) now carries the Phase 6 gates too (permissive quality thresholds, a 3-colour KNN bank, τ values, version) and its classifier is ×10 so clear colours are confident. The old tests used flat single-colour images, which the quality gate now (correctly) rejects as blurry, so they use **textured** colours (`textured()`): green → Healthy, red → Cercospora, blue → out of distribution.

Covered: `/health`, `/model-info`, OpenAPI lists all endpoints; accepted green/red (PNG and JPEG); **blue → rejected `ood`**; **blank → rejected `low_quality` (too_blurry)**; **darkened → `too_dark`**; `/analyze` returns probabilities summing to 1, quality report, OOD score ≤ threshold, a decodable CAM PNG and timings; a rejected photo has no CAM/probabilities; request ID generated and echoed; the request log line is valid JSON with the decision; non-image with an image Content-Type → 415; truncated JPEG → 400; empty → 400; oversized → 413; too many pixels → 413; missing field → 422 with the same error shape; **EXIF-rotated photo** handled; a bundle without fitted gates fails at start-up.

### 8.3 Checks with the real release bundle

- **The API reproduces Phase 6 exactly:** all 379 test photos through `POST /predict` (in-process `TestClient`): 344 accepted / 28 uncertain / 5 rejected ood / 2 rejected low_quality, accepted answers 99.42% correct — identical to the offline evaluation (6.6). All 182 OOD test images: 84 ood / 91 low_quality / 4 uncertain / 3 accepted — identical.
- **Latency through the API:** p50 37 ms, p95 46 ms per `/predict` (384 px images); `/analyze` adds ~17 ms for the CAM.
- **Real server** (`uvicorn`, port 8000, `curl`): `/health` → model `coffeeguard-effv2b0-v1` v1.0.0; `/model-info` → T 0.52, q̂ 0.513, τ_conf 0.947, KNN τ 0.554; an original phone photo → accepted Cercospora 0.999 with an `X-Request-ID` header; a text file sent as `image/jpeg` → `415 unsupported_media_type`.

### Phase 8 status ✅

| Gate item | Result |
|---|---|
| `/health`, `/model-info`, `/predict`, `/analyze` | ✅ typed schemas in `/docs` |
| Lifespan loads the bundle once; sync handlers (threadpool) | ✅ |
| Hardening: type/magic bytes, size and pixel limits, EXIF, one error shape, request ID, JSON logs, CORS | ✅ |
| Tests against a generated ONNX fixture incl. OOD and blank images | ✅ 23 tests |
| Served decisions = offline evaluation | ✅ identical counts on 379 + 182 images |

---

## Gap check against the project goal, and a severity proxy (before Phase 9)

**Question from you:** does the system answer the stated goal — predict the disease from a photo that varies in lighting, blur, background, framing and disease severity, while exposing confidence, explanation, robustness and rejection?

| Goal element | Status | Evidence |
|---|---|---|
| Predict the disease condition | ✅ | test macro-F1 0.974 [0.956, 0.990]; served by `/predict` |
| Quickly | ✅ | 37–63 ms per photo on a CPU server (upload dominates for users) |
| Lighting | ✅ mostly | darkening to ¼ brightness → 0.92 F1; too dark / flat photos get a retake request; colour casts untested |
| Blur | ✅ mostly | moderate blur 0.95, heavy 0.82 F1; half of heavily blurred photos caught by the gate |
| Background | ⚠️ partly | background swap → leaf-only accuracy 0.963; blue paper still nudges towards Cercospora/Leaf Rust |
| Framing | ⚠️ partly | crop/rotation robust (0.93 / 0.89 at severity 5); 40% occlusion 0.67; one leaf per photo only |
| Disease severity | ⚠️ proxy | see below |
| Confidence / explanation / robustness / rejection | ✅ | ECE 0.014 + *uncertain*; faithful CAM; 9×5 corruption sweep; quality + OOD gates (AUROC 0.993 / 1.000) |

**Not only "is it a coffee leaf?":** the OOD check is a gatekeeper. Of the 379 genuine test photos, 344 (90.8%) get a disease answer (99.4% correct) and 28 (7.4%) an *uncertain* answer naming the likely diseases; only 7 (1.8%) are turned away (5 as not-a-leaf, 2 as poor quality).

**Biggest caveat:** all results come from one dataset (same photographers, papers, cameras); robustness was tested with synthetic corruptions. A field test set (30–50 farm phone photos per class, labelled by an agronomist) is the key next step — it needs data and domain expertise, so it is recorded as future work.

### Severity proxy — `coffeeguard severity` (`src/coffeeguard/evaluation/severity.py`)

**Why a proxy:** the dataset has no severity labels. Lesion coverage is estimated from colour alone, independent of the model: inside the leaf mask, pixels that are yellow/orange/brown (hue 5–30, saturation ≥ 70) or much darker than the leaf's median brightness (< 50%). Photos with a separable background only (596 of val + test).

**Sanity checks:** Healthy leaves score a median **0.1%** (IQR 0–0.7%) vs. Cercospora 1.4%, Phoma 5.5%, Leaf Rust 10.4%; the example gallery (`artifacts/severity/coffeeguard-effv2b0-v1/figures/severity_examples.png`, lesions in magenta) goes from a few pustules/spots to heavily covered leaves for Leaf Rust and Phoma. **Weak spot:** small dark spots on very dark Cercospora leaves are missed (several mild Cercospora read 0%), and one "51%" case measured a second leaf entering the frame.

**Model behaviour by lesion-coverage third** (release bundle, full decision; values for mild · moderate · severe):

| Class | Median coverage | n | Accepted | Uncertain | Rejected | Wrong top class | Called Healthy |
|---|---|---|---|---|---|---|---|
| Cercospora | 0.3% · 1.5% · 7.7% | 51 · 50 · 50 | 0.90 · 0.88 · 0.86 | 0.08 · 0.12 · 0.08 | 0.02 · 0 · 0.06 | 0 · 0.02 · 0 | 0 · 0 · 0 |
| Leaf Rust | 3.5% · 10.4% · 25.1% | 69 · 68 · 69 | 0.91 · 0.93 · 0.96 | 0.09 · 0.04 · 0.03 | 0 · 0.03 · 0.01 | 0.04 · 0 · 0.03 | 0 · 0 · 0 |
| Phoma | 3.1% · 5.5% · 10.8% | 47 · 46 · 46 | 0.91 · 0.98 · 0.93 | 0.09 · 0.02 · 0.02 | 0 · 0 · 0.04 | 0.02 · 0 · 0.02 | 0.02 · 0 · 0 |

**Reading:** no drop for the mildest third — wrong answers stay at 0–4% and only one diseased leaf (a mild Phoma) was called Healthy; for Leaf Rust and Phoma, mild cases are answered *uncertain* a bit more often (9% vs. 2–3%), i.e. the system hedges rather than errs. **Limit:** the mildest photos here still show visible lesions (about 1–3% of the leaf); leaves at a very early stage with little colour change aren't in the dataset, so we cannot claim the model catches them. Error counts per third are small (0–3 photos), so these are indications.

**Docs updated:** model card (new *Disease severity* section; limitations now list early infection, colour casts, several leaves per photo, undetected noise/JPEG, and the missing field validation), README (goal-coverage table).
