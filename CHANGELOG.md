# Changelog

All notable changes to the CoffeeGuard AI project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Phase 8 Completed - FastAPI Service ✅

#### Added
- `src/coffeeguard/inference/pipeline.py` - full serving pipeline (quality gate, ONNX pass, OOD score, decision, CAM)
- `GET /model-info`, `POST /analyze` (probabilities, quality report, OOD score, CAM overlay); `/predict` now returns the gated decision with advice
- Request-ID middleware (`X-Request-ID`) and JSON request logs; `LOG_LEVEL` setting
- 23 API tests; the test bundle fixture carries the Phase 6 gates

#### Changed
- `/predict` response: `status`, `reason`, `label`, `confidence`, `prediction_set`, `issues`, `advice`, `model_version`, `latency_ms` (probabilities moved to `/analyze`)
- Streamlit page uses `/analyze` and shows rejected / uncertain answers with advice
- ADR 001 and model card: laptop latency described as a stand-in for a small cloud CPU server

#### Results
- The API reproduces the offline decisions exactly (379 test photos, 182 OOD images); 37 ms p50 per request

### Phase 7 Completed - Model Comparison and Release Bundle ✅

#### Added
- `src/coffeeguard/export/benchmark.py` - CPU latency/size benchmark; CLI `coffeeguard benchmark`
- `src/coffeeguard/evaluation/compare.py` - decision matrix; CLI `coffeeguard compare`
- `docs/decisions/001-deployment-model.md` (ADR), `docs/MODEL_CARD.md`
- Release bundle `artifacts/models/coffeeguard-effv2b0-v1` (committed; weights exempted from the large-file rule)
- `tests/unit/test_benchmark_compare.py`

#### Results
- Chosen: EfficientNetV2-B0 + background swap - test macro-F1 0.974 [0.956, 0.990], leaf-only accuracy 0.963, OOD AUROC 0.993 / 1.000
- CPU latency 15.5 ms (model), 63 ms end to end for a 2048 px photo
- Release bundle reproduces the Phase 4 test predictions exactly (max |dp| = 0)

### Phase 6 Completed - Quality and OOD Gates ✅

#### Added
- `src/coffeeguard/ood/collect.py` - OOD image set (8 sources, split by source into cal/test); CLI `coffeeguard ood collect`
- `src/coffeeguard/inference/ood.py` - NumPy OOD scorers (MSP, energy, Mahalanobis, KNN) + bundle loader
- `src/coffeeguard/inference/decision.py` - accepted / uncertain / rejected decision with advice
- `src/coffeeguard/ood/fit.py` - fits quality gate, scorer, tau_ood and tau_conf; CLI `coffeeguard ood fit`
- `tests/unit/test_ood_decision.py`; report `artifacts/ood/<bundle>/ood.json`

#### Results
- KNN OOD scorer: test AUROC near 0.993 / far 1.000; tau_ood at the val 99.5th percentile (<= 1% of OOD-cal accepted)
- Quality gate fitted where model accuracy drops below 90%: 0.5% of genuine photos rejected
- End to end: 90.8% of genuine test photos accepted at 99.4% accuracy; 175/182 OOD test images rejected
- Thresholds stored in `bundle.json` (not a separate serve.yaml)

### Phase 5 Completed - Explainability and Robustness ✅

#### Added
- `src/coffeeguard/inference/cam.py` - torch-free CAM from one ONNX pass + overlay (verified equal to Grad-CAM)
- `src/coffeeguard/explainability/analysis.py` - CAM galleries, leaf-focus score, deletion faithfulness; CLI `coffeeguard explain`
- `src/coffeeguard/robustness/` - 9 corruptions x 5 severities, colour leaf mask, sweep + shortcut test; CLI `coffeeguard robustness`
- `tests/unit/test_explain_robustness.py`
- Results: `artifacts/explain/<bundle>/`, `artifacts/robustness/<bundle>/`, `artifacts/robustness/summary.csv`

#### Results
- EfficientNetV2-B0: deletion AUC 0.45 (CAM order) vs. 0.61 (random); leaf focus 0.61; relative robustness 0.975 (sev <= 3)
- Shortcut test: background-only accuracy 0.399, but blue-paper backgrounds are still classified as Cercospora/Leaf Rust (photo-setup confound)

#### Added - Background-swap augmentation (follow-up)
- `src/coffeeguard/data/bgswap.py`, `augment.bg_swap_p`, recipe `configs/train/effnetv2_b0_bgswap.yaml`
- New main model `cand-effv2b0-bgswap`: leaf-only accuracy 0.926 -> 0.963 (Phoma 0.81 -> 0.96), background-only confidence 0.83 -> 0.54, relative robustness 0.979, test macro-F1 0.974 (paired difference to the previous main model not significant)

### Phase 4 Completed - Evaluation, Calibration, Uncertainty ✅

#### Added
- `src/coffeeguard/evaluation/evaluate.py` - evaluates ONNX bundles on val + test; CLI `coffeeguard evaluate -b <main> -b <other> ...`
- `src/coffeeguard/evaluation/calibration.py` - temperature scaling, ECE, reliability bins
- `src/coffeeguard/evaluation/conformal.py` - split conformal prediction sets (LAC)
- `src/coffeeguard/evaluation/figures.py` - confusion matrix, reliability diagram, selective accuracy, error gallery
- Bootstrap CIs and paired bootstrap in `evaluation/metrics.py`; `tests/unit/test_evaluation.py`
- Results: `artifacts/metrics/test_metrics.json`, `artifacts/eval/<bundle>/metrics.json` + figures

#### Results
- EfficientNetV2-B0 test macro-F1 **0.979** [0.963, 0.993]; ECE 0.098 → 0.012 (T = 0.55); 98% conformal coverage 0.987
- Bundles now carry the fitted `temperature` and `conformal_qhat` in `bundle.json`

#### Changed
- Conformal level α 0.10 → 0.02 (at 0.10 every set had one class)
- Workflow: all work on the `eleni-changes` branch, one commit per phase (CONTRIBUTING, plan)

### Phase 3 Completed - Main Model (EfficientNetV2-B0) ✅

#### Added - Results
- Variant ablation (seed 0, Kaggle T4): A = fine-tune last 3 stages 0.9848 vs. B = all layers + layer decay 0.9839 val macro-F1 (same accuracy) → **A chosen**
- Main model EfficientNetV2-B0: val macro-F1 **0.9848**, accuracy 0.9841
- Comparison models, same recipe: EfficientNet-B0 0.9843, MobileNetV3-Small 0.9843, MobileNetV3-Large 0.9794
- `artifacts/metrics/phase3_val_summary.json`, training curves in `artifacts/figures/training/`

#### Changed - Training Recipes
- All recipes (except the variant-B record) use the chosen schedule: 5-epoch linear probe, then ≤ 15 epochs fine-tuning the last 3 stages at LR 1e-4, early stopping (patience 4)
- One seed per run; headline metrics will use bootstrap confidence intervals (plan updated)

### Phase 2 Completed - Training Pipeline, Baseline, Walking Skeleton ✅

#### Added - Training
- `src/coffeeguard/data/transforms.py` - `QualityDegrade` (random downscale/upscale + JPEG re-compression, against the class/photo-quality confound) and `RandomRotateFill` (rotation without black corners)
- `src/coffeeguard/training/curves.py` - training-curve figure; CLI `coffeeguard curves --run <dir>`
- CLI `coffeeguard data aug-preview` - augmentation preview grid (`artifacts/figures/augmentation_preview.png`)

#### Added - Baseline
- MobileNetV3-Small (LP-FT, seed 0, trained on CPU): val macro-F1 **0.991**, accuracy 0.992
  - `artifacts/metrics/baseline_mobilenetv3_small_val.json`, `artifacts/figures/training/mobilenetv3_small-s0.png`
  - `artifacts/models/coffeeguard-mnv3s-baseline/bundle.json` (ONNX weights are git-ignored)

#### Added - Walking Skeleton
- `apps/api/app/` - FastAPI service: `GET /health`, `POST /predict`, upload validation (magic bytes, size, pixel count, full decode), one error format
- `apps/web/streamlit_app.py` - upload or camera → prediction + probability chart
- `tests/fixtures/bundle.py` - hand-built ONNX bundle so API tests need no trained model
- `tests/integration/test_api.py`, `tests/integration/test_web.py`, `tests/unit/test_models.py`, `tests/unit/test_transforms.py`

#### Changed - Kaggle Runner
- `remote upload-data` uploads only the 2,520 clean images (59 MB)
- remote runs record the local git SHA; the kernel checks the GPU via PyTorch and fails fast; failures show the kernel log

#### Fixed
- New classifier is zero-initialised: timm's EfficientNet-family init gave logits with std ~6 for 4 classes and a linear probe stuck at chance
- `coffeeguard --help` crashed with `UnicodeEncodeError` in Windows pipes
- Streamlit page reported a loaded model while the API was down (cached health check)

### Phase 1 Completed - Data Pipeline ✅

#### Added - Data Engineering
- **Complete data pipeline implementation**
  - `src/coffeeguard/data/download.py` - Kaggle dataset downloader with integrity checks
  - `src/coffeeguard/data/scan.py` - Image validation (format, size, corruption detection)
  - `src/coffeeguard/data/dedup.py` - Exact, near-duplicate, and rotated-copy detection using perceptual hashing
  - `src/coffeeguard/data/split.py` - Group-aware stratified splitting (70/15/15) preventing data leakage
  - `src/coffeeguard/data/report.py` - Automated EDA with visualizations and JSON summaries
  - `src/coffeeguard/data/dataset.py` - PyTorch Dataset implementation with augmentation
  - `src/coffeeguard/data/transforms.py` - Training and inference transforms

#### Added - ML Infrastructure
- **Model and training framework**
  - `src/coffeeguard/models/factory.py` - timm model factory with EfficientNet support
  - `src/coffeeguard/training/trainer.py` - Multi-stage trainer (LP → FT)
  - `src/coffeeguard/training/remote.py` - Kaggle GPU kernel integration
  - `src/coffeeguard/evaluation/metrics.py` - Classification metrics calculation
  - `src/coffeeguard/export/onnx_export.py` - ONNX export with parity checking

#### Added - Embeddings & Analysis
- **DINOv2 embeddings pipeline**
  - `src/coffeeguard/embeddings/extract.py` - Feature extraction for label auditing
  - `src/coffeeguard/embeddings/audit.py` - Cleanlab-based label issue detection
  - Baseline probe classifier using DINOv2 features

#### Added - Inference Runtime
- **Production-ready inference module**
  - `src/coffeeguard/inference/predictor.py` - ONNX Runtime predictor
  - `src/coffeeguard/inference/preprocess.py` - Inference preprocessing pipeline
  - `src/coffeeguard/inference/quality.py` - Input quality checks

#### Added - Testing
- **Comprehensive test suite**
  - `tests/fixtures/synthetic.py` - Synthetic dataset generator for testing
  - `tests/unit/test_cli.py` - CLI command tests
  - `tests/unit/test_config.py` - Configuration validation tests
  - `tests/unit/test_data_pipeline.py` - Data pipeline tests
  - `tests/unit/test_inference_core.py` - Inference tests
  - `tests/unit/test_utils.py` - Utility function tests
  - `tests/integration/test_training_smoke.py` - End-to-end training smoke test

#### Added - Configuration
- **Configuration files**
  - `configs/data.yaml` - Data pipeline configuration
  - `configs/train/*.yaml` - Training recipes (EfficientNet, MobileNet variants, smoke test)
  - `.pre-commit-config.yaml` - Pre-commit hooks (ruff, nbstripout, file checks)
  - `.github/workflows/ci.yml` - CI/CD pipeline
  - `.python-version` - Python 3.12 pinned
  - `.gitattributes` - Git LFS configuration

#### Added - Documentation
- `docs/IMPLEMENTATION_PLAN.md` - Detailed phase-by-phase implementation roadmap
- `docs/PROGRESS.md` - Running progress log with decisions and results
- `kaggle/run_template.py` - Template for remote GPU training

#### Added - Artifacts
- **Data quality reports**
  - `artifacts/reports/eda_summary.json` - Dataset statistics
  - `artifacts/reports/label_audit.json` - Label quality analysis
  - `artifacts/reports/label_issues.csv` - Detected label problems
  - `artifacts/reports/data_quality_removed.csv` - Removed problematic samples
  
- **EDA Visualizations**
  - `artifacts/figures/eda/class_distribution.png` - Class balance visualization
  - `artifacts/figures/eda/image_sizes.png` - Image dimension distribution
  - `artifacts/figures/eda/resolution_by_class.png` - Resolution analysis per class
  - `artifacts/figures/eda/quality_by_class.png` - Quality metrics by class
  - `artifacts/figures/eda/duplicate_groups.png` - Duplicate detection results
  - `artifacts/figures/eda/label_issues.png` - Label audit visualization
  - `artifacts/figures/eda/samples_*.png` - Sample images from each class

- **Baseline metrics**
  - `artifacts/metrics/baseline_dinov2_probe_val.json` - DINOv2 probe baseline results

#### Added - Data Splits
- `data/splits/train.csv` - Training set (70%)
- `data/splits/val.csv` - Validation set (15%)
- `data/splits/test.csv` - Test set (15%)
- `data/splits/split_info.json` - Split metadata and statistics
- `data/splits/data_config.yaml` - Data configuration snapshot

### Phase 0 Completed - Foundation ✅

#### Changed - Project Structure
- **Package reorganization**
  - Moved `ml/` → `src/coffeeguard/` (modern src layout)
  - Removed empty placeholder packages
  - Added new modules: `utils/`, `embeddings/`, `export/`, `inference/`

#### Changed - Build System
- **Modern Python tooling**
  - Switched to `uv` package manager for fast, reliable dependency management
  - Updated `pyproject.toml` with `uv_build` backend
  - Added optional dependency groups:
    - `[inference]` - ONNX Runtime (slim deployment)
    - `[ml]` - Full ML stack (torch, timm, scikit-learn, etc.)
    - `[api]` - FastAPI service dependencies
    - `[web]` - Streamlit UI dependencies
  - Dev dependencies: ruff, pytest, pytest-cov, mypy, pre-commit

#### Changed - Core Utilities
- **Configuration system** (`src/coffeeguard/config.py`)
  - Pydantic-based type-safe configuration
  - CLI override support (`--set key=value`)
  - Automatic path resolution and validation

- **Utility modules** (`src/coffeeguard/utils/`)
  - `paths.py` - Project root detection and path resolution
  - `hashing.py` - SHA256 hashing and fingerprinting
  - `seed.py` - Reproducible random seed management
  - `log.py` - Rich logging setup
  - `io.py` - JSON I/O with NumPy/Path support
  - `runs.py` - Experiment run directory management
  - `plotting.py` - Visualization utilities

#### Changed - CLI
- **Command-line interface** (`src/coffeeguard/cli.py`)
  - `coffeeguard info` - System and config information
  - `coffeeguard data download` - Download Kaggle dataset
  - `coffeeguard data prepare` - Run full data pipeline
  - `coffeeguard data report` - Generate EDA
  - `coffeeguard embed audit` - Label quality audit
  - `coffeeguard train` - Local training
  - `coffeeguard remote train` - Kaggle GPU training
  - `coffeeguard export` - ONNX export

#### Changed - Documentation
- Updated `README.md` with modern structure and getting started guide
- Updated `CONTRIBUTING.md` with development workflow
- Updated `.env.example` with comprehensive environment variables

#### Changed - Git Configuration
- Enhanced `.gitignore` for ML project patterns
  - Virtual environments, cache directories
  - Large model files and artifacts
  - Data directories (raw/processed)
  - Build artifacts and logs

#### Removed
- `PHASE_1_CHECKLIST.md` - Consolidated into PROGRESS.md
- `docs/GIT_WORKFLOW.md` - Merged into CONTRIBUTING.md
- `docs/QUICK_START.md` - Merged into README.md
- `docs/TEAM_ROLES.md` - No longer needed
- `docs/architecture/.gitkeep` - Empty placeholder
- `docs/decisions/.gitkeep` - Empty placeholder
- `docs/experiments/.gitkeep` - Empty placeholder
- `scripts/.gitkeep` - Empty placeholder
- `configs/.gitkeep` - Replaced with actual configs
- Various `.gitkeep` files in `artifacts/` - Replaced with actual artifacts

---

## [0.1.0] - Initial Scaffold

### Added
- Initial project structure with empty package directories
- Basic documentation (PROJECT_SPECIFICATION.md, initial README)
- MIT License
- GitHub issue and PR templates
- Initial pyproject.toml with setuptools

---

## Summary of Changes by Phase

| Phase | Description | Status | Files Changed |
|-------|-------------|--------|---------------|
| **Phase 0** | Foundation - tooling, structure, CI | ✅ Complete | ~25 files |
| **Phase 1** | Data engineering - validation, dedup, split, EDA | ✅ Complete | ~82 files |
| **Phase 2** | Training pipeline, baseline, walking skeleton | ✅ Complete | ~30 files |
| **Phase 3** | EfficientNetV2-B0 training (LP-FT, 1 seed) + comparison runs | ✅ Complete | ~15 files |
| **Phase 4** | Evaluation, calibration, conformal prediction | ✅ Complete | ~20 files |
| **Phase 5** | Explainability and robustness | ✅ Complete | ~20 files |
| **Phase 6** | Quality and OOD gates | ✅ Complete | ~10 files |
| **Phase 7** | Model comparison and export | ✅ Complete | ~15 files |
| **Phase 8** | FastAPI service | ✅ Complete | ~10 files |
| **Phase 9** | Streamlit UI | ⏳ Planned | - |
| **Phase 10** | Docker, documentation, final polish | ⏳ Planned | - |

---

## How to Read This Changelog

- **Added** - New features, files, or capabilities
- **Changed** - Changes to existing functionality
- **Deprecated** - Features that will be removed in future versions
- **Removed** - Deleted features or files
- **Fixed** - Bug fixes
- **Security** - Security-related changes

For detailed technical decisions and step-by-step progress, see [docs/PROGRESS.md](docs/PROGRESS.md).
