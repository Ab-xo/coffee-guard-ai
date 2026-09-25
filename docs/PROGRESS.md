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
